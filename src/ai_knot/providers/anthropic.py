"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

import httpx


class AnthropicProvider:
    """Provider for the Anthropic Messages API.

    Args:
        api_key: Anthropic API key.
        default_model: Model ID used when none is specified.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        *,
        default_model: str = "claude-haiku-4-5-20251001",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self._default_model

    def call(self, system_prompt: str, user_content: str, model: str) -> str:
        """Send a message to the Anthropic API and return the text response."""
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return str(response.json()["content"][0]["text"])
