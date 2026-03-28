"""Yandex GPT LLM provider."""

from __future__ import annotations

import httpx


class YandexGPTProvider:
    """Provider for the Yandex Foundation Models API.

    Args:
        api_key: Yandex Cloud IAM token or API key.
        folder_id: Yandex Cloud folder ID (required for model URI).
        default_model: Model ID used when none is specified.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        *,
        folder_id: str,
        default_model: str = "yandexgpt-lite",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._folder_id = folder_id
        self._default_model = default_model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "yandex"

    @property
    def default_model(self) -> str:
        return self._default_model

    def call(self, system_prompt: str, user_content: str, model: str) -> str:
        """Send a completion request to Yandex GPT and return the text response."""
        response = httpx.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers={
                "Authorization": f"Api-Key {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "modelUri": f"gpt://{self._folder_id}/{model}",
                "completionOptions": {
                    "temperature": 0.0,
                    "maxTokens": 2048,
                },
                "messages": [
                    {"role": "system", "text": system_prompt},
                    {"role": "user", "text": user_content},
                ],
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return str(response.json()["result"]["alternatives"][0]["message"]["text"])
