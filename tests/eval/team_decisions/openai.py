"""Strict, budgeted OpenAI calls; provider failures invalidate an evaluation."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
import time
from typing import Any

import httpx

_READER = "gpt-4.1-mini"
_EMBEDDER = "text-embedding-3-small"
_BASE_URL = "https://api.openai.com"
_FRAMING_TOKENS = 256
_DIMENSIONS = 1536


class EvaluationError(RuntimeError):
    """A sanitized provider or evaluation configuration failure."""


class BudgetExceeded(EvaluationError):
    """A call cannot fit the remaining evaluation budget."""


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or "\r" in value or "\n" in value:
        raise EvaluationError("Invalid OpenAI API key configuration")
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise EvaluationError("Invalid OpenAI request input") from None


def _usage(body: dict[str, Any], kind: str) -> dict[str, int]:
    usage = body.get("usage")
    names = ["prompt_tokens", "total_tokens"]
    if kind == "complete":
        names.append("completion_tokens")
    if not isinstance(usage, dict) or any(not _integer(usage.get(name)) for name in names):
        raise EvaluationError("Invalid OpenAI token usage")
    result = {name: usage[name] for name in names}
    if result["total_tokens"] != result["prompt_tokens"] + result.get("completion_tokens", 0):
        raise EvaluationError("Inconsistent OpenAI token usage")
    return result


def _completion(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise EvaluationError("Invalid OpenAI completion choices")
    choice = choices[0]
    message = choice.get("message")
    if choice.get("finish_reason") != "stop":
        raise EvaluationError("OpenAI completion did not finish normally")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise EvaluationError("Invalid OpenAI completion content")
    if not message["content"].strip():
        raise EvaluationError("Empty OpenAI completion content")
    return message["content"]


def _embeddings(body: dict[str, Any], count: int) -> list[list[float]]:
    data = body.get("data")
    if not isinstance(data, list) or len(data) != count:
        raise EvaluationError("Invalid OpenAI embedding count")
    ordered: dict[int, list[float]] = {}
    for item in data:
        if not isinstance(item, dict):
            raise EvaluationError("Invalid OpenAI embedding item")
        index, vector = item.get("index"), item.get("embedding")
        if not _integer(index) or index >= count or index in ordered:
            raise EvaluationError("Invalid OpenAI embedding indices")
        if not isinstance(vector, list) or len(vector) != _DIMENSIONS:
            raise EvaluationError("Invalid OpenAI embedding dimensions")
        if any(type(number) not in (float, int) or not math.isfinite(number) for number in vector):
            raise EvaluationError("Invalid OpenAI embedding values")
        ordered[index] = [float(number) for number in vector]
    return [ordered[index] for index in range(count)]


class MeteredOpenAI:
    """One ledger for reader and embedding requests, used within one event loop.

    Reserve UTF-8 input/schema bytes, 256 framing tokens and the output ceiling
    before sending. Valid provider usage replaces that reservation even when the
    result is unusable. Unknown billing keeps the reservation. No retry is automatic.
    """

    def __init__(
        self,
        api_key: str,
        *,
        max_calls: int = 20,
        max_tokens: int = 20000,
        max_output_tokens_per_call: int = 128,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = _key(api_key)
        if not _integer(max_calls) or not _integer(max_tokens):
            raise EvaluationError("Invalid OpenAI evaluation budget")
        if not _integer(max_output_tokens_per_call) or not 1 <= max_output_tokens_per_call <= 4096:
            raise EvaluationError("Invalid OpenAI output authority")
        self._max_calls = max_calls
        self._max_tokens = max_tokens
        self._max_output_tokens_per_call = max_output_tokens_per_call
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._calls: list[dict[str, Any]] = []
        self._spent_tokens = 0
        self._budget_failed = False

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Return report-safe snapshots without credentials or source content."""
        return copy.deepcopy(self._calls)

    @property
    def spent_tokens(self) -> int:
        """Actual tokens plus conservative reservations for pending/unknown calls."""
        return self._spent_tokens

    async def __aenter__(self) -> MeteredOpenAI:
        if self._client is not None:
            raise EvaluationError("OpenAI evaluation client is already open")
        self._client = httpx.AsyncClient(
            transport=self._transport, timeout=30, follow_redirects=False, trust_env=False
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int = 128,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        if (
            not _integer(max_output_tokens)
            or not 1 <= max_output_tokens <= self._max_output_tokens_per_call
        ):
            raise EvaluationError("Invalid OpenAI output token ceiling")
        if response_schema is not None and not isinstance(response_schema, dict):
            raise EvaluationError("Invalid OpenAI response schema")
        if not isinstance(messages, list) or not messages:
            raise EvaluationError("Invalid OpenAI messages")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in ("system", "user", "assistant")
                or not isinstance(message.get("content"), str)
                or set(message) != {"role", "content"}
            ):
                raise EvaluationError("Invalid OpenAI messages")
        return await self._request(
            "complete",
            messages,
            max_output_tokens=max_output_tokens,
            response_schema=response_schema,
        )

    async def embed(
        self,
        texts: list[str],
        *,
        base_url: str = _BASE_URL,
        model: str = _EMBEDDER,
        api_key: str | None = None,
        timeout: float = 30,
    ) -> list[list[float]]:
        if not isinstance(base_url, str) or base_url.rstrip("/") != _BASE_URL:
            raise EvaluationError("Only the official OpenAI endpoint is allowed")
        if model != _EMBEDDER:
            raise EvaluationError("Unexpected OpenAI embedding model configuration")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise EvaluationError("Invalid OpenAI request timeout")
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise EvaluationError("Invalid OpenAI embedding input")
        bearer = self._api_key if api_key is None else _key(api_key)
        if not texts:
            return []
        return await self._request("embed", texts, api_key=bearer, timeout=timeout)

    async def _request(
        self,
        kind: str,
        inputs: Any,
        *,
        max_output_tokens: int = 0,
        response_schema: dict[str, Any] | None = None,
        api_key: str | None = None,
        timeout: float = 30,
    ) -> Any:
        if self._client is None:
            raise EvaluationError("OpenAI evaluation client is not open")
        model = _READER if kind == "complete" else _EMBEDDER
        input_bytes = _json_bytes(inputs)
        schema_bytes = _json_bytes(response_schema) if response_schema is not None else b""
        payload = {"model": model, "messages" if kind == "complete" else "input": inputs}
        if kind == "complete":
            payload["max_completion_tokens"] = max_output_tokens
            payload["temperature"] = 0
            payload["response_format"] = {"type": "json_object"}
            if response_schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "memory_decision",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
        request_bytes = _json_bytes(payload)
        reservation = len(input_bytes) + len(schema_bytes) + _FRAMING_TOKENS + max_output_tokens
        # No await between checking and reserving: concurrent tasks see this charge.
        if (
            self._budget_failed
            or len(self._calls) >= self._max_calls
            or self._spent_tokens + reservation > self._max_tokens
        ):
            self._budget_failed = True
            raise BudgetExceeded("OpenAI evaluation budget exhausted")
        self._spent_tokens += reservation
        call = {
            "kind": kind,
            "requested_model": model,
            "returned_model": None,
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "status": "pending",
            "http_status": None,
            "latency_seconds": 0.0,
            "reserved_tokens": reservation,
            "charged_tokens": reservation,
            "usage": None,
        }
        self._calls.append(call)
        started = time.perf_counter()
        error: BaseException | None = None
        result: Any = None
        try:
            path = "chat/completions" if kind == "complete" else "embeddings"
            response = await self._client.post(
                f"{_BASE_URL}/v1/{path}",
                headers={
                    "Authorization": f"Bearer {api_key or self._api_key}",
                    "Content-Type": "application/json",
                },
                content=request_bytes,
                timeout=timeout,
            )
            call["http_status"] = response.status_code
            if not response.is_success:
                raise EvaluationError("OpenAI HTTP request failed")
            body = response.json()
            if not isinstance(body, dict):
                raise EvaluationError("Invalid OpenAI response")
            returned_model = body.get("model")
            valid_model = isinstance(returned_model, str) and (
                returned_model == model
                or (
                    kind == "complete"
                    and re.fullmatch(r"gpt-4\.1-mini-\d{4}-\d{2}-\d{2}", returned_model)
                )
            )
            if valid_model:
                call["returned_model"] = returned_model
            raw_usage = body.get("usage")
            if isinstance(raw_usage, dict):
                # Inconsistent billing cannot erase a larger reported charge.
                counts = {
                    name: value
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if _integer(value := raw_usage.get(name))
                }
                conservative = max(
                    reservation,
                    counts.get("total_tokens", 0),
                    counts.get("prompt_tokens", 0) + counts.get("completion_tokens", 0),
                )
                self._spent_tokens += conservative - reservation
                call["charged_tokens"] = conservative
            usage = _usage(body, kind)
            call["usage"] = usage
            self._spent_tokens += usage["total_tokens"] - call["charged_tokens"]
            call["charged_tokens"] = usage["total_tokens"]
            if not valid_model:
                raise EvaluationError("Unexpected OpenAI response model")
            if self._spent_tokens > self._max_tokens:
                raise BudgetExceeded("OpenAI reported usage beyond the evaluation budget")
            if kind == "complete":
                if usage["completion_tokens"] > max_output_tokens:
                    raise EvaluationError("OpenAI exceeded the requested output token ceiling")
                result = _completion(body)
            else:
                result = _embeddings(body, len(inputs))
            call["status"] = "ok"
        except asyncio.CancelledError:
            call["status"] = "cancelled"
            error = asyncio.CancelledError()
        except EvaluationError as exc:
            call["status"] = "budget_exceeded" if isinstance(exc, BudgetExceeded) else "error"
            if isinstance(exc, BudgetExceeded):
                self._budget_failed = True
            error = exc
        except Exception:
            call["status"] = "error"
            error = EvaluationError("OpenAI request or response processing failed")
        finally:
            call["latency_seconds"] = time.perf_counter() - started
        # Raise outside the handler, without retaining an HTTP exception's secrets.
        if error is not None:
            raise error from None
        return result
