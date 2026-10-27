"""Strict JSON boundary shared by always-on transports."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import types
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from . import models
from .models import WaitForChange, WaitForSlot, WaitUntil


def loads_strict(raw: str | bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key.")
            result[key] = value
        return result

    def invalid(_: str) -> object:
        raise ValueError("Non-finite JSON number.")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def canonical_json(value: object) -> str:
    return json.dumps(
        encode_always_on(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def encode_always_on(value: object) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timezone-aware datetime required.")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {
            field.name: encode_always_on(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        kinds = {WaitUntil: "until", WaitForSlot: "slot", WaitForChange: "change"}
        if type(value) in kinds:
            result = {"kind": kinds[type(value)], **result}
        return result
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON keys must be strings.")
        return {key: encode_always_on(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_always_on(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Invalid JSON value.")


def decode_value(value: object, target: Any) -> Any:
    """Decode a typed value, rejecting coercion and unknown typed fields."""
    origin, args = get_origin(target), get_args(target)
    if target is Any or target is object:
        return encode_always_on(value)
    if target is type(None):
        if value is not None:
            raise ValueError("Expected null.")
        return None
    if origin in (Union, types.UnionType):
        if isinstance(value, dict) and "kind" in value:
            selected = {"until": WaitUntil, "slot": WaitForSlot, "change": WaitForChange}.get(
                value["kind"]
            )
            if selected not in args:
                raise ValueError("Unknown wait condition.")
            return decode_value(value, selected)
        for item in args:
            try:
                return decode_value(value, item)
            except (ValueError, TypeError):
                pass
        raise ValueError("Value does not match expected type.")
    if origin is Literal:
        if value not in args or isinstance(value, bool):
            raise ValueError("Invalid choice.")
        return value
    if target is datetime:
        if not isinstance(value, str):
            raise ValueError("Expected datetime string.")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Timezone-aware datetime required.")
        return parsed.astimezone(UTC)
    if isinstance(target, type) and issubclass(target, Enum):
        if not isinstance(value, str):
            raise ValueError("Expected enum string.")
        return target(value)
    if isinstance(target, type) and dataclasses.is_dataclass(target):
        if not isinstance(value, dict):
            raise ValueError("Expected object.")
        payload = dict(value)
        kinds = {WaitUntil: "until", WaitForSlot: "slot", WaitForChange: "change"}
        if target in kinds and payload.pop("kind", None) != kinds[target]:
            raise ValueError("Invalid condition kind.")
        fields = {field.name: field for field in dataclasses.fields(target)}
        if payload.keys() - fields.keys():
            raise ValueError("Unknown object field.")
        hints = get_type_hints(target)
        decoded = {key: decode_value(item, hints[key]) for key, item in payload.items()}
        for key in ("progress", "request", "result", "evidence", "selector"):
            if key in decoded and len(canonical_json(decoded[key]).encode("utf-8")) > 65536:
                raise ValueError("JSON field exceeds 65536 bytes.")
        return target(**decoded)
    if origin in (tuple, list):
        if not isinstance(value, list):
            raise ValueError("Expected array.")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ValueError("Wrong tuple size.")
            return tuple(decode_value(item, hint) for item, hint in zip(value, args, strict=True))
        decoded_items = [decode_value(item, args[0] if args else Any) for item in value]
        return tuple(decoded_items) if origin is tuple else decoded_items
    if origin is dict:
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError("Expected JSON object.")
        return encode_always_on(value)
    if target is bool:
        if type(value) is not bool:
            raise ValueError("Expected boolean.")
    elif target is int:
        if type(value) is not int:
            raise ValueError("Expected integer.")
    elif target is float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Expected finite number.")
        return float(value)
    elif target is str:
        if not isinstance(value, str):
            raise ValueError("Expected string.")
    else:
        raise ValueError("Unsupported input type.")
    return value


def decode_arguments(method: Any, payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected request object.")
    signature = inspect.signature(method)
    if payload.keys() - signature.parameters.keys():
        raise ValueError("Unknown request field.")
    hints = get_type_hints(method, localns=vars(models))
    result = {key: decode_value(value, hints[key]) for key, value in payload.items()}
    signature.bind(**result)
    for key in ("progress", "request", "result", "evidence", "selector"):
        if key in result and len(canonical_json(result[key]).encode("utf-8")) > 65536:
            raise ValueError("JSON field exceeds 65536 bytes.")
    return result


def error_response(exc: Exception) -> tuple[int, dict[str, object]]:
    code = type(exc).__name__
    statuses = {
        "AccessDenied": 403,
        "CursorExpired": 410,
        "UnsupportedAlwaysOnBackend": 501,
        "RevisionConflict": 409,
        "LeaseLost": 409,
        "IdempotencyConflict": 409,
        "StaleEvidence": 409,
        "InvalidDelivery": 409,
    }
    status = statuses.get(code, 422)
    if code not in statuses:
        code = "InvalidInput"
    messages = {
        "LeaseLost": "The lease is no longer current.",
        "AccessDenied": "The requested object is not accessible.",
        "InvalidInput": "The request is invalid.",
        "UnsupportedAlwaysOnBackend": "Always-on memory requires SQLite or PostgreSQL.",
    }
    message = messages.get(code, "The operation conflicts with current memory state.")
    return status, {"error": {"code": code, "message": message}}
