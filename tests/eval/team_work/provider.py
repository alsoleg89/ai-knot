"""Study-only exact provider recording and replay; no transport or key lookup."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from typing import Any

from tests.eval.team_decisions.openai import BudgetExceeded, EvaluationError


def transport_failure(call):
    return (
        call.get("kind") == "complete"
        and call.get("status") == "error"
        and call.get("http_status") is None
        and call.get("usage") is None
    )


def truncated_text(snapshot, meter_index, limit=None):
    calls = snapshot["meter_calls"]
    if not 0 <= meter_index < len(calls):
        return None
    call = calls[meter_index]
    if (
        call.get("kind") != "complete"
        or call.get("status") != "error"
        or call.get("http_status") != 200
        or not isinstance(call.get("usage"), dict)
        or not call.get("returned_model")
        or (limit is not None and call["usage"].get("completion_tokens", limit + 1) > limit)
    ):
        return None
    return next(
        (
            r["content"]
            for r in snapshot.get("raw_completions", [])
            if r["meter_call_index"] == meter_index
            and r["finish_reason"] == "length"
            and isinstance(r["content"], str)
        ),
        None,
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class RecordingClient:
    """Persist via a synchronous callback before I/O and after every outcome.

    The reservation matches team_decisions.openai.MeteredOpenAI, not a tokenizer:
    canonical UTF-8 input/schema bytes + 256 framing + output ceiling. The wrapped
    meter owns actual/unknown charges. Only an unentered pending request adds a
    reservation to accounted_tokens; a terminal zero-meter refusal adds nothing.
    """

    def __init__(
        self,
        client: Any,
        *,
        on_update: Callable[[dict[str, Any]], None],
        boundary: Callable[[], bool] = lambda: True,
        retry_transport: bool = False,
        allow_truncated: bool = False,
    ) -> None:
        self.client = client
        self._on_update = on_update
        self._boundary = boundary
        self._requests: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._retry_transport = retry_transport
        self._allow_truncated = allow_truncated

    @property
    def calls(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.client.calls)

    @property
    def completion_truncated(self):
        return bool(self._requests and self._requests[-1]["status"] == "truncated")

    @property
    def spent_tokens(self) -> int:
        return self.client.spent_tokens

    @property
    def actual_provider_calls(self) -> int:
        return getattr(self.client, "actual_provider_calls", len(self.calls))

    def snapshot(self) -> dict[str, Any]:
        calls = self.calls
        unentered = sum(
            request["reserved_tokens"]
            for request in self._requests
            if request["status"] == "pending" and len(calls) == request["meter_start"]
        )
        return copy.deepcopy(
            {
                "requests": self._requests,
                "meter_calls": calls,
                "spent_tokens": self.spent_tokens,
                "accounted_tokens": self.spent_tokens + unentered,
                "actual_provider_calls": self.actual_provider_calls,
                "raw_completions": getattr(self.client, "raw_completions", []),
            }
        )

    async def __aenter__(self) -> RecordingClient:
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        return await self.client.__aexit__(*args)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        return await self._call("complete", messages, max_output_tokens, response_schema)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._call("embed", texts, 0, None)

    async def _call(self, kind: str, inputs: Any, limit: int, schema: Any) -> Any:
        start = len(self.calls)
        try:
            return await self._call_once(kind, inputs, limit, schema)
        except EvaluationError:
            calls = self.calls
            if not (
                self._retry_transport
                and kind == "complete"
                and len(calls) == start + 1
                and transport_failure(calls[-1])
            ):
                raise
        # A missing HTTP response is not a returned proposal. Retry once with the
        # identical input; the wrapped meter still enforces every original cap.
        return await self._call_once(kind, inputs, limit, schema)

    async def _call_once(self, kind: str, inputs: Any, limit: int, schema: Any) -> Any:
        inputs, schema = copy.deepcopy(inputs), copy.deepcopy(schema)
        reservation = len(_json_bytes(inputs)) + 256 + limit
        if schema is not None:
            reservation += len(_json_bytes(schema))
        async with self._lock:
            if self._boundary() is not True:
                raise RuntimeError("SQL must be closed at the provider boundary")
            entry = {
                "kind": kind,
                "inputs": inputs,
                "response_schema": schema,
                "max_output_tokens": limit,
                "reserved_tokens": reservation,
                "status": "pending",
                "meter_start": len(self.calls),
                "meter_call_indexes": [],
            }
            self._requests.append(entry)
            try:
                self._on_update(self.snapshot())
                if self._boundary() is not True:
                    raise RuntimeError("SQL must be closed at the provider boundary")
                if kind == "complete":
                    result = await self.client.complete(
                        copy.deepcopy(inputs),
                        max_output_tokens=limit,
                        response_schema=copy.deepcopy(schema),
                    )
                else:
                    result = await self.client.embed(copy.deepcopy(inputs))
                entry["raw_output" if kind == "complete" else "vectors"] = copy.deepcopy(result)
                partial = truncated_text(self.snapshot(), entry["meter_start"], limit)
                entry["status"] = (
                    "truncated" if self._allow_truncated and partial is not None else "ok"
                )
                return result
            except BaseException as exc:
                partial = truncated_text(self.snapshot(), entry["meter_start"], limit)
                if (
                    self._allow_truncated
                    and isinstance(exc, EvaluationError)
                    and not isinstance(exc, BudgetExceeded)
                    and partial is not None
                ):
                    entry.update(status="truncated", raw_output=partial)
                    return partial
                entry["status"] = (
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                )
                entry["error_type"] = type(exc).__name__
                raise
            finally:
                entry["meter_call_indexes"] = list(range(entry["meter_start"], len(self.calls)))
                self._on_update(self.snapshot())


class ReplayClient:
    """Exact ordered replay with original logical meter usage and zero provider I/O.

    Eligible sources have only successful requests and successful meter calls,
    contiguous complete meter coverage, and consistent final accounting. Pending,
    cancelled, provider-failed, and zero-charge refused requests are ineligible.
    A later local parse failure does not change a successful recorded request.
    Empty embedding requests may have no meter call. Transport latency is copied
    as source metadata, never incurred. Every other request field must match.
    """

    actual_provider_calls = 0

    def __init__(
        self,
        recording: dict[str, Any],
        *,
        allow_transport_failure=False,
        allow_truncated=False,
        allow_budget_refusal=False,
    ) -> None:
        self._source = copy.deepcopy(recording)
        self._position = 0
        self._calls: list[dict[str, Any]] = []
        self._spent_tokens = 0
        self._allow_transport_failure = allow_transport_failure
        self._allow_truncated = allow_truncated
        self._allow_budget_refusal = allow_budget_refusal
        try:
            _json_bytes(self._source)
            self._validate_source()
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise ValueError("Ineligible replay recording") from None

    def _validate_source(self) -> None:
        requests, calls = self._source["requests"], self._source["meter_calls"]
        if not isinstance(requests, list) or not isinstance(calls, list):
            raise ValueError
        cursor = 0
        failed_indexes = set()
        for request in requests:
            kind = request["kind"]
            failed = request["status"] == "failed"
            refused = (
                failed
                and request.get("error_type") == "BudgetExceeded"
                and self._allow_budget_refusal
            )
            indexes = request["meter_call_indexes"]
            partial = (
                truncated_text(self._source, indexes[0], request["max_output_tokens"])
                if len(indexes) == 1
                else None
            )
            truncated = self._allow_truncated and partial is not None
            if kind not in ("complete", "embed") or (
                request["status"] != "ok"
                and not (
                    refused
                    or (truncated and request["status"] in ("failed", "truncated"))
                    or (
                        self._allow_transport_failure
                        and failed
                        and kind == "complete"
                        and request.get("error_type") == "EvaluationError"
                    )
                )
            ):
                raise ValueError
            inputs, schema = request["inputs"], request["response_schema"]
            limit = request["max_output_tokens"]
            if type(limit) is not int or limit < 0 or not isinstance(inputs, list):
                raise ValueError
            if kind == "complete":
                if (
                    (not failed and not isinstance(request["raw_output"], str))
                    or not inputs
                    or limit == 0
                ):
                    raise ValueError
                if schema is not None and not isinstance(schema, dict):
                    raise ValueError
            elif (
                schema is not None
                or limit != 0
                or not isinstance(request["vectors"], list)
                or len(request["vectors"]) != len(inputs)
            ):
                raise ValueError
            expected_reservation = len(_json_bytes(inputs)) + 256 + limit
            if schema is not None:
                expected_reservation += len(_json_bytes(schema))
            if request["reserved_tokens"] != expected_reservation:
                raise ValueError
            indexes = request["meter_call_indexes"]
            if not isinstance(indexes, list) or request["meter_start"] != cursor:
                raise ValueError
            if indexes != list(range(cursor, cursor + len(indexes))):
                raise ValueError
            if not indexes and (kind == "complete" or inputs) and not refused:
                raise ValueError
            for index in indexes:
                if index >= len(calls) or calls[index]["kind"] != kind:
                    raise ValueError
            if refused:
                if indexes:
                    raise ValueError
            elif truncated:
                failed_indexes.update(indexes)
            elif failed:
                if len(indexes) != 1 or not transport_failure(calls[indexes[0]]):
                    raise ValueError
                call = calls[indexes[0]]
                if (
                    call["charged_tokens"] != call["reserved_tokens"]
                    or call["reserved_tokens"] != expected_reservation
                ):
                    raise ValueError
                failed_indexes.update(indexes)
            cursor += len(indexes)
        if cursor != len(calls):
            raise ValueError
        for index, call in enumerate(calls):
            if (
                (call["status"] != "ok" and index not in failed_indexes)
                or type(call["charged_tokens"]) is not int
                or call["charged_tokens"] < 0
            ):
                raise ValueError
        total = sum(call["charged_tokens"] for call in calls)
        if self._source["spent_tokens"] != total or self._source["accounted_tokens"] != total:
            raise ValueError

    @property
    def calls(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._calls)

    @property
    def spent_tokens(self) -> int:
        return self._spent_tokens

    @property
    def raw_completions(self) -> list[dict[str, Any]]:
        return copy.deepcopy(
            [
                row
                for row in self._source.get("raw_completions", [])
                if row["meter_call_index"] < len(self._calls)
            ]
        )

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "requests": self._source["requests"][: self._position],
                "meter_calls": self._calls,
                "spent_tokens": self.spent_tokens,
                "accounted_tokens": self.spent_tokens,
                "actual_provider_calls": 0,
                "raw_completions": self.raw_completions,
            }
        )

    async def __aenter__(self) -> ReplayClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        return self._consume("complete", messages, max_output_tokens, response_schema)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return self._consume("embed", texts, 0, None)

    def _consume(self, kind: str, inputs: Any, limit: int, schema: Any) -> Any:
        if self._position >= len(self._source["requests"]):
            raise ValueError("Replay received an extra request")
        expected = self._source["requests"][self._position]
        actual = {
            "kind": kind,
            "inputs": inputs,
            "max_output_tokens": limit,
            "response_schema": schema,
        }
        try:
            matches = _json_bytes(actual) == _json_bytes({key: expected[key] for key in actual})
        except (TypeError, ValueError, UnicodeError):
            matches = False
        if not matches:
            raise ValueError("Replay request does not match the recording")
        for index in expected["meter_call_indexes"]:
            call = copy.deepcopy(self._source["meter_calls"][index])
            self._calls.append(call)
            self._spent_tokens += call["charged_tokens"]
        self._position += 1
        if expected["status"] == "failed":
            if expected.get("error_type") == "BudgetExceeded":
                raise BudgetExceeded("Recorded budget refusal")
            indexes = expected["meter_call_indexes"]
            partial = truncated_text(self._source, indexes[0], limit) if len(indexes) == 1 else None
            if self._allow_truncated and partial is not None:
                return partial
            raise EvaluationError("Recorded transport failure")
        return copy.deepcopy(expected["raw_output" if kind == "complete" else "vectors"])

    @property
    def has_remaining(self):
        return self._position < len(self._source["requests"])

    def assert_consumed(self) -> None:
        if self._position != len(self._source["requests"]):
            raise ValueError("Replay has missing requests")


class PrefixClient:
    """Consume an exact recorded prefix before lazily opening paid transport.

    This never substitutes a live response for a mismatching recorded request.
    All original known/unknown charges reduce the remaining live authority.
    """

    def __init__(self, recording, factory, **limits):
        self.prefix = ReplayClient(
            recording, allow_transport_failure=True, allow_truncated=True, allow_budget_refusal=True
        )
        self.factory, self.limits, self.live = factory, limits, None

    @property
    def calls(self):
        return self.prefix.calls + (self.live.calls if self.live else [])

    @property
    def spent_tokens(self):
        return self.prefix.spent_tokens + (self.live.spent_tokens if self.live else 0)

    @property
    def actual_provider_calls(self):
        return len(self.live.calls) if self.live else 0

    @property
    def raw_completions(self):
        extra = copy.deepcopy(getattr(self.live, "raw_completions", []))
        for row in extra:
            row["meter_call_index"] += len(self.prefix.calls)
        return self.prefix.raw_completions + extra

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        try:
            self.prefix.assert_consumed()
        finally:
            if self.live:
                await self.live.__aexit__(*args)

    async def complete(self, messages, *, max_output_tokens, response_schema=None):
        return await self._call("complete", messages, max_output_tokens, response_schema)

    async def embed(self, texts):
        return await self._call("embed", texts, 0, None)

    async def _call(self, kind, inputs, limit, schema):
        if self.prefix.has_remaining:
            target = self.prefix
        else:
            if self.live is None:
                limits = dict(self.limits)
                limits["max_calls"] -= len(self.prefix.calls)
                limits["max_tokens"] -= self.prefix.spent_tokens
                if limits["max_calls"] < 0 or limits["max_tokens"] < 0:
                    raise EvaluationError("Recorded prefix exceeds authority")
                self.live = self.factory(**limits)
                await self.live.__aenter__()
            target = self.live
        if kind == "complete":
            return await target.complete(inputs, max_output_tokens=limit, response_schema=schema)
        return await target.embed(inputs)
