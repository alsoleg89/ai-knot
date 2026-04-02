"""Optional LLM-based query expansion for improved recall.

When an LLM provider is configured and ``llm_recall=True``, queries are
expanded with synonyms/related terms before BM25 search.  This bridges
vocabulary gaps (e.g. "what database?" → "database PostgreSQL SQL storage").

Without an LLM the raw query is passed through unchanged.
"""

from __future__ import annotations

from ai_knot.providers import LLMProvider, call_with_retry

_EXPAND_PROMPT = """Given a search query, add 2-4 synonyms or closely related terms.
IMPORTANT: Keep the SAME LANGUAGE as the input query.
Include morphological variants (different word forms of the same root).
Return ONLY the expanded query as a single line. Do not explain.

Example (English):
Input: "what database?"
Output: "what database PostgreSQL SQL storage relational"

Example (Russian):
Input: "запрещённые слова"
Output: "запрещённые запретить слова слово стоп-слова ограничения"
"""

_MAX_CACHE_SIZE = 128


class LLMQueryExpander:
    """Expand search queries using an LLM for better BM25 recall.

    Results are cached (up to 128 entries) to avoid repeated LLM calls
    for the same query.

    Args:
        provider: An LLM provider instance.
        model: Model name override (defaults to provider's default).
    """

    def __init__(self, provider: LLMProvider, model: str | None = None) -> None:
        self._provider = provider
        self._model = model or provider.default_model
        self._cache: dict[str, str] = {}

    def expand(self, query: str) -> str:
        """Expand a query with LLM-generated synonyms.

        Args:
            query: The original search query.

        Returns:
            Expanded query string, or the original query on LLM failure.
        """
        if query in self._cache:
            return self._cache[query]

        result = call_with_retry(self._provider, _EXPAND_PROMPT, query, self._model)
        expanded = result.strip() if result else query

        # Evict oldest entry when cache is full.
        if len(self._cache) >= _MAX_CACHE_SIZE:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[query] = expanded
        return expanded
