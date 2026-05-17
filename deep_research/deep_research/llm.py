from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(ABC):
    @abstractmethod
    def chat(self, system: str, user: str) -> LLMResponse: ...

    @property
    @abstractmethod
    def model(self) -> str: ...


class MockLLMClient(LLMClient):
    """Deterministic mock — no network calls, no API key required. For offline CI."""

    def __init__(self, response: str = "[MOCK]", tokens: int = 10) -> None:
        self._response = response
        self._tokens = tokens

    @property
    def model(self) -> str:
        return "mock"

    def chat(self, system: str, user: str) -> LLMResponse:
        content = f"{self._response} sys={system[:40]!r} usr={user[:40]!r}"
        return LLMResponse(
            content=content,
            input_tokens=self._tokens,
            output_tokens=self._tokens,
        )


_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "429", "too many requests", "overloaded")

_BACKOFF_DELAYS = (30, 60, 120, 240)  # seconds between retries


def _is_rate_limit(stderr: str, stdout: str) -> bool:
    combined = (stderr + stdout).lower()
    return any(m in combined for m in _RATE_LIMIT_MARKERS)


class ClaudeCliLLMClient(LLMClient):
    """Uses the local `claude` CLI binary (Claude Max subscription, no API key required).

    Token counts are unavailable via CLI — always returns 0, so the token budget
    check never triggers. Use wall-clock and tick budgets to bound the campaign.

    Rate-limit errors trigger exponential backoff (30→60→120→240s) before raising.
    """

    def __init__(self, model: str = "claude-opus-4-7", timeout: int = 120) -> None:
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def chat(self, system: str, user: str) -> LLMResponse:
        last_exc: RuntimeError | None = None
        for attempt, delay in enumerate([0] + list(_BACKOFF_DELAYS)):
            if delay:
                time.sleep(delay)
            try:
                result = subprocess.run(
                    [
                        "claude",
                        "--print",
                        "--system-prompt",
                        system,
                        "--model",
                        self._model,
                        "--output-format",
                        "text",
                        "--no-session-persistence",
                        "--tools",
                        "",
                        "--",
                        user,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"claude CLI timed out after {self._timeout}s") from exc
            except FileNotFoundError as exc:
                raise RuntimeError("claude CLI not found — is Claude Code installed?") from exc

            if result.returncode == 0:
                return LLMResponse(content=result.stdout.strip(), input_tokens=0, output_tokens=0)

            if _is_rate_limit(result.stderr, result.stdout):
                last_exc = RuntimeError(
                    f"claude CLI rate limited (attempt {attempt + 1}): {result.stderr[:200]}"
                )
                continue  # retry with backoff

            raise RuntimeError(f"claude CLI exit {result.returncode}: {result.stderr[:300]}")

        assert last_exc is not None
        raise last_exc


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str = "claude-opus-4-7", max_retries: int = 3) -> None:
        import anthropic

        self._client = anthropic.Anthropic(max_retries=max_retries)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def chat(self, system: str, user: str) -> LLMResponse:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        return LLMResponse(
            content=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
