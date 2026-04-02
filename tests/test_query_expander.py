"""Tests for ai_knot.query_expander — LLM-based query expansion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_knot.query_expander import _MAX_CACHE_SIZE, LLMQueryExpander


def _make_provider(call_return: str = "database PostgreSQL SQL storage") -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.name = "mock"
    provider.default_model = "gpt-4o"
    provider.call.return_value = call_return
    return provider


class TestExpand:
    """LLMQueryExpander.expand() behaviour."""

    @patch("ai_knot.query_expander.call_with_retry")
    def test_expand_calls_provider(self, mock_retry: MagicMock) -> None:
        """expand() delegates to call_with_retry and returns its result."""
        mock_retry.return_value = "database PostgreSQL SQL storage"
        provider = _make_provider()
        expander = LLMQueryExpander(provider)

        result = expander.expand("what database?")

        assert result == "database PostgreSQL SQL storage"
        mock_retry.assert_called_once()

    @patch("ai_knot.query_expander.call_with_retry")
    def test_expand_caches_result(self, mock_retry: MagicMock) -> None:
        """Second call with same query returns cached result, no extra LLM call."""
        mock_retry.return_value = "expanded query"
        provider = _make_provider()
        expander = LLMQueryExpander(provider)

        expander.expand("query")
        expander.expand("query")  # second call — should hit cache

        assert mock_retry.call_count == 1

    @patch("ai_knot.query_expander.call_with_retry")
    def test_expand_fallback_on_empty(self, mock_retry: MagicMock) -> None:
        """When LLM returns empty string, fall back to original query."""
        mock_retry.return_value = ""
        provider = _make_provider()
        expander = LLMQueryExpander(provider)

        result = expander.expand("original query")

        assert result == "original query"

    @patch("ai_knot.query_expander.call_with_retry")
    def test_expand_strips_whitespace(self, mock_retry: MagicMock) -> None:
        """LLM response is stripped of surrounding whitespace."""
        mock_retry.return_value = "  expanded query  "
        provider = _make_provider()
        expander = LLMQueryExpander(provider)

        result = expander.expand("original query")

        assert result == "expanded query"

    @patch("ai_knot.query_expander.call_with_retry")
    def test_expand_cache_eviction(self, mock_retry: MagicMock) -> None:
        """When cache exceeds _MAX_CACHE_SIZE, oldest entry is evicted."""
        mock_retry.side_effect = lambda _prov, _sys, query, _model: f"expanded_{query}"
        provider = _make_provider()
        expander = LLMQueryExpander(provider)

        # Fill cache to max size
        for i in range(_MAX_CACHE_SIZE):
            expander.expand(f"query_{i}")

        assert len(expander._cache) == _MAX_CACHE_SIZE
        assert "query_0" in expander._cache  # oldest still present

        # One more entry should evict the oldest
        expander.expand("overflow_query")

        assert len(expander._cache) == _MAX_CACHE_SIZE
        assert "query_0" not in expander._cache  # evicted
        assert "overflow_query" in expander._cache
