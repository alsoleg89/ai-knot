"""Content-free memory changes and deterministic evidence fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ai_knot.types import Fact

EVIDENCE_FIELDS = (
    "content",
    "type",
    "slot_key",
    "entity",
    "attribute",
    "value_text",
    "canonical_surface",
    "witness_surface",
    "prompt_surface",
    "qualifiers",
    "source_snippets",
    "source_spans",
    "source_verbatim",
    "supported",
    "support_confidence",
    "verification_source",
    "state_confidence",
    "valid_from",
    "valid_until",
    "event_time",
    "origin_agent_id",
    "visibility",
    "visibility_scope",
    "claim_key",
    "topic_channel",
    "memory_tier",
)


def canonical(value: Any) -> str:
    def encode(obj: Any) -> str:
        if isinstance(obj, datetime):
            if obj.tzinfo is None or obj.utcoffset() is None:
                raise ValueError("timezone-aware datetime required")
            return obj.astimezone(UTC).isoformat(timespec="microseconds")
        if isinstance(obj, Enum):
            return str(obj.value)
        raise TypeError(f"unsupported JSON value: {type(obj).__name__}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=encode,
    )


def evidence_revision(fact: Fact) -> str:
    return hashlib.sha256(
        canonical({key: getattr(fact, key) for key in EVIDENCE_FIELDS}).encode()
    ).hexdigest()


def access_projection(fact: Fact) -> str:
    return canonical(
        {
            key: getattr(fact, key)
            for key in (
                "origin_agent_id",
                "visibility_scope",
                "visibility",
                "topic_channel",
                "memory_tier",
            )
        }
    )


def visible(projection: str | None, owner: str, grants: set[str]) -> bool:
    if projection is None:
        return False
    data = json.loads(projection)
    scope = data.get("visibility_scope") or "global"
    return scope == "global" or data.get("origin_agent_id") == owner or scope in grants


@dataclass(frozen=True)
class FactChange:
    namespace: str
    seq: int
    kind: str
    fact_id: str
    slot_key: str
    before_revision: str | None
    after_revision: str | None


@dataclass(frozen=True)
class ChangeBatch:
    delivery_id: str | None
    events: tuple[FactChange, ...]
    resync_required: bool = False
