"""OpenAI-compatible LLM provider.

Covers OpenAI, GigaChat (Sber), Qwen (Alibaba), and any API
that speaks the OpenAI chat completions format.
"""

from __future__ import annotations

import httpx


class OpenAICompatProvider:
    """Provider for any OpenAI-compatible chat completions API.

    Args:
        api_key: Bearer token for authentication.
        base_url: API base URL (without ``/chat/completions``).
        default_model: Model ID used when none is specified.
        extra_headers: Additional headers merged into every request.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o-mini",
        extra_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._extra_headers = extra_headers or {}
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "openai-compat"

    @property
    def default_model(self) -> str:
        return self._default_model

    def call(self, system_prompt: str, user_content: str, model: str) -> str:
        """Send a chat completion request and return the assistant message."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])
