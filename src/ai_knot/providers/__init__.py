"""LLM providers for ai-knot fact extraction."""

from __future__ import annotations

import os

from ai_knot.providers.base import LLMProvider, call_with_retry

__all__ = ["LLMProvider", "call_with_retry", "create_provider"]

# Map of env var names per provider for automatic API key lookup.
_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gigachat": "GIGACHAT_API_KEY",
    "yandex": "YANDEX_API_KEY",
    "qwen": "QWEN_API_KEY",
    "openai-compat": "OPENAI_API_KEY",
    "ollama": "",  # no env var needed — Ollama is always local
}


def create_provider(
    provider: str,
    api_key: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    **kwargs: str,
) -> LLMProvider:
    """Create an LLM provider by name.

    Args:
        provider: Provider name (openai, anthropic, gigachat, yandex, qwen, openai-compat).
        api_key: API key. If ``None``, reads from the provider-specific env var
            or ``LLM_API_KEY`` as fallback.
        model: Override the default model for this provider.
        base_url: Custom base URL (required for ``openai-compat``, ignored by others).
        **kwargs: Extra arguments passed to the provider constructor
            (e.g. ``folder_id`` for Yandex).

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If the provider name is unknown or required parameters are missing.
    """
    resolved_key = api_key or _resolve_api_key(provider)
    if not resolved_key:
        env_name = _ENV_VARS.get(provider, "LLM_API_KEY")
        raise ValueError(
            f"No API key for provider {provider!r}. "
            f"Pass api_key= or set {env_name} environment variable."
        )

    if provider in ("openai", "gigachat", "qwen", "openai-compat"):
        return _create_openai_compat(provider, resolved_key, model=model, base_url=base_url)
    if provider == "anthropic":
        return _create_anthropic(resolved_key, model=model)
    if provider == "yandex":
        return _create_yandex(resolved_key, model=model, **kwargs)
    if provider == "ollama":
        return _create_ollama(model=model)

    supported = "openai, anthropic, gigachat, yandex, qwen, openai-compat, ollama"
    raise ValueError(f"Unknown provider {provider!r}. Choose from: {supported}")


def _resolve_api_key(provider: str) -> str | None:
    """Try provider-specific env var, then LLM_API_KEY fallback."""
    if provider == "ollama":
        return "ollama"  # Ollama ignores the key; return sentinel to pass validation
    env_var = _ENV_VARS.get(provider)
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val
    return os.environ.get("LLM_API_KEY")


def _create_openai_compat(
    provider: str, api_key: str, *, model: str | None, base_url: str | None
) -> LLMProvider:
    from ai_knot.providers.openai_compat import OpenAICompatProvider

    defaults: dict[str, dict[str, str]] = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
        },
        "gigachat": {
            "base_url": "https://gigachat.devices.sberbank.ru/api/v1",
            "default_model": "GigaChat",
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "default_model": "qwen-turbo",
        },
        "openai-compat": {
            "base_url": base_url or "",
            "default_model": "gpt-4o-mini",
        },
    }
    cfg = defaults.get(provider, defaults["openai"])
    resolved_url = base_url or cfg["base_url"]
    if not resolved_url:
        raise ValueError("openai-compat provider requires base_url=")
    resolved_model = model or cfg["default_model"]
    return OpenAICompatProvider(api_key, base_url=resolved_url, default_model=resolved_model)


def _create_anthropic(api_key: str, *, model: str | None) -> LLMProvider:
    from ai_knot.providers.anthropic import AnthropicProvider

    if model:
        return AnthropicProvider(api_key, default_model=model)
    return AnthropicProvider(api_key)


def _create_ollama(*, model: str | None) -> LLMProvider:
    from ai_knot.providers.ollama import OLLAMA_DEFAULT_MODEL, OllamaProvider

    return OllamaProvider(model=model or OLLAMA_DEFAULT_MODEL)


def _create_yandex(api_key: str, *, model: str | None, **kwargs: str) -> LLMProvider:
    from ai_knot.providers.yandex import YandexGPTProvider

    folder_id = kwargs.get("folder_id") or os.environ.get("YANDEX_FOLDER_ID")
    if not folder_id:
        raise ValueError(
            "Yandex provider requires folder_id. "
            "Pass folder_id= or set YANDEX_FOLDER_ID environment variable."
        )
    if model:
        return YandexGPTProvider(api_key, folder_id=folder_id, default_model=model)
    return YandexGPTProvider(api_key, folder_id=folder_id)
