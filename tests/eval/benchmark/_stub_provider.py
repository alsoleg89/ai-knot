"""Stub LLM provider for offline / mock benchmark runs.

Returns ``"[]"`` for every call so the Extractor produces zero facts.
This satisfies the LLMProvider Protocol without touching the network.
"""

from __future__ import annotations


class StubProvider:
    """Minimal LLMProvider that always returns an empty JSON array."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def default_model(self) -> str:
        return "stub"

    def call(
        self,
        system_prompt: str,
        user_content: str,
        model: str,
        *,
        timeout: float | None = None,
    ) -> str:
        return "[]"
