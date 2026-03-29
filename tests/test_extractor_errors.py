"""Tests for Extractor error handling — network failures, HTTP errors, bad responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from agentmemo.extractor import Extractor
from agentmemo.types import ConversationTurn


@pytest.fixture
def turns() -> list[ConversationTurn]:
    return [ConversationTurn(role="user", content="I prefer Python")]


def _make_http_error(status: int) -> httpx.HTTPStatusError:
    mock_resp = MagicMock()
    mock_resp.status_code = status
    return httpx.HTTPStatusError("error", request=MagicMock(), response=mock_resp)


class TestOpenAIErrors:
    """OpenAI provider error paths."""

    def test_timeout_returns_empty(self, turns: list[ConversationTurn]) -> None:
        with (
            patch("httpx.post", side_effect=httpx.TimeoutException("timeout")),
            patch("agentmemo.providers.base.time.sleep"),
        ):
            result = Extractor(api_key="key", provider="openai").extract(turns)
        assert result == []

    def test_connection_error_returns_empty(self, turns: list[ConversationTurn]) -> None:
        with (
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
            patch("agentmemo.providers.base.time.sleep"),
        ):
            result = Extractor(api_key="key", provider="openai").extract(turns)
        assert result == []

    def test_http_401_no_retry(self, turns: list[ConversationTurn]) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _make_http_error(401)
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = Extractor(api_key="bad-key", provider="openai").extract(turns)
        assert result == []
        assert mock_post.call_count == 1  # no retry on 401

    def test_http_429_retries_three_times(self, turns: list[ConversationTurn]) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _make_http_error(429)
        with patch("httpx.post", return_value=mock_resp) as mock_post, patch("time.sleep"):
            result = Extractor(api_key="key", provider="openai").extract(turns)
        assert result == []
        assert mock_post.call_count == 3

    def test_http_500_retries_three_times(self, turns: list[ConversationTurn]) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _make_http_error(500)
        with patch("httpx.post", return_value=mock_resp) as mock_post, patch("time.sleep"):
            result = Extractor(api_key="key", provider="openai").extract(turns)
        assert result == []
        assert mock_post.call_count == 3

    def test_malformed_json_returns_empty(self, turns: list[ConversationTurn]) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"choices": [{"message": {"content": "not json {"}}]}
        with patch("httpx.post", return_value=mock_resp):
            result = Extractor(api_key="key", provider="openai").extract(turns)
        assert result == []

    def test_missing_choices_key_returns_empty(self, turns: list[ConversationTurn]) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"unexpected": "structure"}
        with patch("httpx.post", return_value=mock_resp):
            result = Extractor(api_key="key", provider="openai").extract(turns)
        assert result == []

    def test_success_after_retry(self, turns: list[ConversationTurn]) -> None:
        """Returns facts if a later attempt succeeds."""
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = _make_http_error(429)

        ok_resp = MagicMock()
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '[{"content": "User likes Python",'
                            ' "type": "procedural", "importance": 0.9}]'
                        )
                    }
                }
            ]
        }

        with patch("httpx.post", side_effect=[fail_resp, ok_resp]), patch("time.sleep"):
            result = Extractor(api_key="key", provider="openai").extract(turns)
        assert len(result) == 1
        assert result[0].content == "User likes Python"


class TestAnthropicErrors:
    """Anthropic provider error paths."""

    def test_timeout_returns_empty(self, turns: list[ConversationTurn]) -> None:
        with (
            patch("httpx.post", side_effect=httpx.TimeoutException("timeout")),
            patch("agentmemo.providers.base.time.sleep"),
        ):
            result = Extractor(api_key="key", provider="anthropic").extract(turns)
        assert result == []

    def test_http_500_retries(self, turns: list[ConversationTurn]) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _make_http_error(500)
        with patch("httpx.post", return_value=mock_resp) as mock_post, patch("time.sleep"):
            result = Extractor(api_key="key", provider="anthropic").extract(turns)
        assert result == []
        assert mock_post.call_count == 3

    def test_http_401_no_retry(self, turns: list[ConversationTurn]) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _make_http_error(401)
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = Extractor(api_key="bad-key", provider="anthropic").extract(turns)
        assert result == []
        assert mock_post.call_count == 1


class TestImportanceClamping:
    """LLM may return importance outside [0, 1] — should be clamped."""

    def test_importance_above_1_clamped(self, turns: list[ConversationTurn]) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '[{"content": "test", "type": "semantic", "importance": 1.5}]'
                    }
                }
            ]
        }
        with patch("httpx.post", return_value=mock_resp):
            result = Extractor(api_key="key", provider="openai").extract(turns)
        assert len(result) == 1
        assert result[0].importance == 1.0

    def test_importance_below_0_clamped(self, turns: list[ConversationTurn]) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '[{"content": "test", "type": "semantic", "importance": -0.5}]'
                    }
                }
            ]
        }
        with patch("httpx.post", return_value=mock_resp):
            result = Extractor(api_key="key", provider="openai").extract(turns)
        assert len(result) == 1
        assert result[0].importance == 0.0
