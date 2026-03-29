"""Tests for LLM providers: factory, retry wrapper, and individual provider call()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from ai_knot.providers import create_provider
from ai_knot.providers.base import call_with_retry


def _make_http_error(status: int) -> httpx.HTTPStatusError:
    mock_resp = MagicMock()
    mock_resp.status_code = status
    return httpx.HTTPStatusError("error", request=MagicMock(), response=mock_resp)


class TestCallWithRetry:
    """call_with_retry(): centralized retry logic."""

    def test_success_returns_content(self) -> None:
        provider = MagicMock()
        provider.name = "test"
        provider.call.return_value = "hello"
        result = call_with_retry(provider, "sys", "user", "model")
        assert result == "hello"
        assert provider.call.call_count == 1

    def test_timeout_retries(self) -> None:
        provider = MagicMock()
        provider.name = "test"
        provider.call.side_effect = httpx.TimeoutException("timeout")
        with patch("time.sleep"):
            result = call_with_retry(provider, "sys", "user", "model", max_retries=3)
        assert result == ""
        assert provider.call.call_count == 3

    def test_http_429_retries(self) -> None:
        provider = MagicMock()
        provider.name = "test"
        provider.call.side_effect = _make_http_error(429)
        with patch("time.sleep"):
            result = call_with_retry(provider, "sys", "user", "model", max_retries=3)
        assert result == ""
        assert provider.call.call_count == 3

    def test_http_401_no_retry(self) -> None:
        provider = MagicMock()
        provider.name = "test"
        provider.call.side_effect = _make_http_error(401)
        result = call_with_retry(provider, "sys", "user", "model", max_retries=3)
        assert result == ""
        assert provider.call.call_count == 1

    def test_network_error_retries(self) -> None:
        provider = MagicMock()
        provider.name = "test"
        provider.call.side_effect = httpx.ConnectError("refused")
        with patch("time.sleep"):
            result = call_with_retry(provider, "sys", "user", "model", max_retries=2)
        assert result == ""
        assert provider.call.call_count == 2

    def test_key_error_no_retry(self) -> None:
        provider = MagicMock()
        provider.name = "test"
        provider.call.side_effect = KeyError("choices")
        result = call_with_retry(provider, "sys", "user", "model", max_retries=3)
        assert result == ""
        assert provider.call.call_count == 1

    def test_success_after_retry(self) -> None:
        provider = MagicMock()
        provider.name = "test"
        provider.call.side_effect = [_make_http_error(500), "recovered"]
        with patch("time.sleep"):
            result = call_with_retry(provider, "sys", "user", "model", max_retries=3)
        assert result == "recovered"
        assert provider.call.call_count == 2


class TestCreateProvider:
    """create_provider() factory and env var fallback."""

    def test_openai_provider(self) -> None:
        p = create_provider("openai", "test-key")
        assert p.name == "openai-compat"
        assert p.default_model == "gpt-4o-mini"

    def test_anthropic_provider(self) -> None:
        p = create_provider("anthropic", "test-key")
        assert p.name == "anthropic"
        assert "claude" in p.default_model

    def test_gigachat_provider(self) -> None:
        p = create_provider("gigachat", "test-key")
        assert p.name == "openai-compat"
        assert p.default_model == "GigaChat"

    def test_qwen_provider(self) -> None:
        p = create_provider("qwen", "test-key")
        assert p.name == "openai-compat"
        assert "qwen" in p.default_model

    def test_yandex_provider(self) -> None:
        p = create_provider("yandex", "test-key", folder_id="b1g12345")
        assert p.name == "yandex"
        assert "yandex" in p.default_model

    def test_yandex_without_folder_id_raises(self) -> None:
        with pytest.raises(ValueError, match="folder_id"):
            create_provider("yandex", "test-key")

    def test_openai_compat_with_base_url(self) -> None:
        p = create_provider("openai-compat", "test-key", base_url="https://my-llm.example.com/v1")
        assert p.name == "openai-compat"

    def test_openai_compat_without_base_url_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            create_provider("openai-compat", "test-key")

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("gemini", "test-key")

    def test_no_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="No API key"):
            create_provider("openai")

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        p = create_provider("openai")
        assert p.default_model == "gpt-4o-mini"

    def test_llm_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_API_KEY", "fallback-key")
        p = create_provider("openai")
        assert p.default_model == "gpt-4o-mini"

    def test_custom_model_override(self) -> None:
        p = create_provider("openai", "key", model="gpt-4o")
        assert p.default_model == "gpt-4o"

    def test_yandex_folder_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YANDEX_FOLDER_ID", "env-folder")
        p = create_provider("yandex", "test-key")
        assert p.name == "yandex"


class TestOpenAICompatProviderCall:
    """OpenAICompatProvider.call() with mocked httpx."""

    def test_successful_call(self) -> None:
        p = create_provider("openai", "test-key")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": "extracted facts"}}]}
        with patch("httpx.post", return_value=mock_resp):
            result = p.call("system prompt", "user message", "gpt-4o-mini")
        assert result == "extracted facts"


class TestAnthropicProviderCall:
    """AnthropicProvider.call() with mocked httpx."""

    def test_successful_call(self) -> None:
        p = create_provider("anthropic", "test-key")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"content": [{"text": "extracted facts"}]}
        with patch("httpx.post", return_value=mock_resp):
            result = p.call("system prompt", "user message", "claude-haiku-4-5-20251001")
        assert result == "extracted facts"


class TestYandexProviderCall:
    """YandexGPTProvider.call() with mocked httpx."""

    def test_successful_call(self) -> None:
        p = create_provider("yandex", "test-key", folder_id="b1g12345")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "result": {"alternatives": [{"message": {"text": "extracted facts"}}]}
        }
        with patch("httpx.post", return_value=mock_resp):
            result = p.call("system prompt", "user message", "yandexgpt-lite")
        assert result == "extracted facts"
