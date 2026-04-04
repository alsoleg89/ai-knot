"""Ollama LLM provider — thin wrapper over OpenAICompatProvider.

Points at the local Ollama OpenAI-compat endpoint so that
``create_provider("ollama")`` works without any environment variable.
Ollama does not validate API keys; ``"ollama"`` is used as a sentinel.
"""

from __future__ import annotations

from ai_knot.providers.openai_compat import OpenAICompatProvider

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_EMBED_URL = "http://localhost:11434/v1/embeddings"


class OllamaProvider(OpenAICompatProvider):
    """Provider for local Ollama at http://localhost:11434/v1.

    No API key required — Ollama is always local.

    Args:
        model: Model to use (default: ``llama3.2:3b``).
        timeout: Request timeout in seconds. Local models can be slow
            on first token generation; default is 120 s.
    """

    def __init__(
        self,
        *,
        model: str = OLLAMA_DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(
            "ollama",  # Ollama ignores the key value
            base_url=OLLAMA_BASE_URL,
            default_model=model,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return "ollama"
