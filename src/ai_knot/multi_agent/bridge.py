"""Controlled two-hop bridge retrieval for ASSEMBLY and INTEGRATION queries.

When the direct connection between two concepts is not expressed in a single
fact, bridge retrieval finds it via:
  1. Extract high-IDF tokens from top-N first-pass BM25 results.
  2. Run a second BM25 pass per bridge term over the full active pool.
  3. RRF-merge the per-term result lists, excluding first-pass facts.

This is a controlled second hop, not a graph walk: the bridge terms are
bounded by first-pass top-N and the second-hop depth is always 1.

Activated only for RetrievalIntent.ASSEMBLY and RetrievalIntent.INTEGRATION.
"""

from __future__ import annotations

import math

from ai_knot.retriever import BM25Retriever
from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# Smoothing constant for IDF computation over the first-pass corpus.
_IDF_SMOOTH = 1.0
# Default RRF rank constant (standard k=60).
_RRF_K = 60


def _build_idf_for_texts(texts: list[str]) -> dict[str, float]:
    """IDF over a small corpus of text strings (smoothed)."""
    n = len(texts)
    df: dict[str, int] = {}
    token_sets = [set(_tokenize(t)) for t in texts]
    for ts in token_sets:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + _IDF_SMOOTH) / (cnt + _IDF_SMOOTH)) + 1.0 for t, cnt in df.items()}


class BridgeRetriever:
    """Two-hop retrieval for integration and assembly queries.

    Usage::

        bridge = BridgeRetriever()
        first_pass = bm25.search(query, facts, top_k=15)
        terms = bridge.extract_bridge_terms(first_pass, top_n=3)
        extra = bridge.second_hop(terms, facts, top_k=10,
                                  exclude_ids={f.id for f, _ in first_pass})
    """

    def __init__(self) -> None:
        self._bm25 = BM25Retriever(skip_prf=True)

    def extract_bridge_terms(
        self,
        first_pass: list[tuple[Fact, float]],
        *,
        top_n: int = 3,
        min_idf: float = 1.5,
    ) -> list[str]:
        """Extract high-IDF content tokens from the top-N first-pass facts.

        Tokens that are rare in the first-pass corpus (high IDF) are specific
        entity identifiers or domain concepts useful as bridge anchors.
        Common vocabulary ("service", "agent", "fact") has low IDF and is
        filtered out.

        Args:
            first_pass: (Fact, score) pairs from the first BM25 pass, sorted
                by score descending.
            top_n: Consider only the top-N scored facts.
            min_idf: Minimum IDF threshold for a token to be included.

        Returns:
            List of high-IDF token strings (may be empty).
        """
        if not first_pass:
            return []

        top_facts = [f for f, _ in first_pass[:top_n]]
        texts = [f.content for f in top_facts]
        idf = _build_idf_for_texts(texts)

        # Collect all tokens from top facts, keep those above min_idf.
        seen: set[str] = set()
        bridge: list[str] = []
        for fact in top_facts:
            for tok in _tokenize(fact.content):
                if tok not in seen and idf.get(tok, 0.0) >= min_idf:
                    bridge.append(tok)
                    seen.add(tok)

        return bridge

    def second_hop(
        self,
        bridge_terms: list[str],
        active_facts: list[Fact],
        *,
        top_k: int = 5,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[Fact, float]]:
        """Run one BM25 search per bridge term; RRF-merge results.

        Each bridge term is used as a standalone BM25 query over the active
        pool, excluding first-pass facts.  The per-term ranked lists are
        merged via Reciprocal Rank Fusion.

        Args:
            bridge_terms: Tokens to use as second-hop queries.
            active_facts: Full active pool to search.
            top_k: Number of results to return.
            exclude_ids: Fact IDs from the first pass to exclude.

        Returns:
            List of (Fact, score) pairs after RRF merge, top-k.
        """
        if not bridge_terms or not active_facts:
            return []

        excluded = exclude_ids or set()
        search_facts = [f for f in active_facts if f.id not in excluded]
        if not search_facts:
            return []

        # Per-term BM25 results → RRF merge.
        # rrf_scores[fact_id] accumulates 1/(k + rank) contributions.
        rrf_scores: dict[str, float] = {}
        fact_by_id: dict[str, Fact] = {}

        fetch_k = min(top_k * 2, len(search_facts))
        for term in bridge_terms:
            results = self._bm25.search(term, search_facts, top_k=fetch_k)
            for rank, (fact, score) in enumerate(results):
                if score <= 0:
                    continue
                fact_by_id[fact.id] = fact
                rrf_scores[fact.id] = rrf_scores.get(fact.id, 0.0) + 1.0 / (_RRF_K + rank + 1)

        if not rrf_scores:
            return []

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(fact_by_id[fid], score) for fid, score in ranked[:top_k]]
