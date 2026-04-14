"""BM25Retriever — BM25F + PRF + RRF retriever (zero deps)."""

from __future__ import annotations

from ai_knot._inverted_index import (
    _PRF_ALPHA,
    _PRF_EXPANSION_TERMS,
    _PRF_TOP_K,
    _RRF_K,
    InvertedIndex,
    _char_trigrams,
)
from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact


def _prf_expand(
    index: InvertedIndex,
    query: str,
    raw_scores: dict[str, float],
) -> dict[str, float]:
    """Pseudo-relevance feedback: extract expansion terms from top-k docs.

    Implements simplified RM3 (Lavrenko & Croft, 2001):
    1. Take top-k documents by BM25 score.
    2. Score each term: fb(t) = Σ bm25(d) × tf(t,d) / |d|.
    3. Return top expansion terms (excluding original query terms).
    """
    if not raw_scores:
        return {}

    query_tokens = set(_tokenize(query))

    # Get top-k feedback documents.
    top_docs = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)[:_PRF_TOP_K]

    # Score expansion terms.
    fb_scores: dict[str, float] = {}
    for doc_id, bm25_score in top_docs:
        dl = index.content_lengths.get(doc_id, 1)
        if dl == 0:
            continue
        # Gather content term frequencies for this doc.
        for term, postings in index._content_postings.items():
            if doc_id in postings:
                tf = postings[doc_id]
                fb_scores[term] = fb_scores.get(term, 0.0) + bm25_score * tf / dl

    # Filter: remove original query terms and zero-score terms.
    expansion: dict[str, float] = {}
    for term, score in fb_scores.items():
        if term in query_tokens or score <= 0:
            continue
        expansion[term] = score

    # Take top expansion terms, apply alpha discount.
    sorted_terms = sorted(expansion.items(), key=lambda x: x[1], reverse=True)[
        :_PRF_EXPANSION_TERMS
    ]
    return {t: _PRF_ALPHA for t, _ in sorted_terms}


def _rrf_fuse(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    *,
    k: int = _RRF_K,
) -> dict[str, float]:
    """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

    Combines multiple ranked lists into a single fused score:
        RRF(d) = Σ_{r} w_r / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked document ID lists (best first).
        weights: Optional per-ranker weights (default: all 1.0).
        k: Smoothing constant (default 60).

    Returns:
        Dict mapping document id to fused RRF score.
    """
    fused: dict[str, float] = {}
    for i, ranked in enumerate(ranked_lists):
        w = weights[i] if weights else 1.0
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return fused


class BM25Retriever:
    """Zero-dependency BM25F retriever with PRF and RRF.

    Scoring pipeline:
      1. BM25F (content + canonical_surface + tags fields) with IDF high-DF filtering.
      2. Pseudo-relevance feedback expands query with top-doc terms.
      3. Reciprocal Rank Fusion combines six signals: BM25F, slot-exact,
         char-trigram, importance, retention, and recency.
    """

    def __init__(
        self,
        *,
        rrf_weights: tuple[float, ...] = (5.0, 3.0, 2.0, 1.5, 1.5, 1.0),
        skip_prf: bool = False,
    ) -> None:
        self._rrf_weights = rrf_weights
        self._skip_prf = skip_prf

    def prf_expand(self, query: str, facts: list[Fact]) -> dict[str, float]:
        """Compute PRF expansion terms for a query against facts.

        Builds a temporary inverted index, runs initial BM25 scoring, and
        extracts expansion terms from the top feedback documents using
        simplified RM3 (Lavrenko & Croft, 2001).

        Returns expansion_weights dict suitable for passing to search().
        Returns empty dict if insufficient facts for PRF.
        """
        if not facts or not query.strip() or len(facts) < 4:
            return {}
        index = InvertedIndex(facts)
        raw_scores = index.score(query)
        return _prf_expand(index, query, raw_scores)

    def search(
        self,
        query: str,
        facts: list[Fact],
        *,
        top_k: int = 5,
        faithfulness_k: int | None = None,
        expansion_weights: dict[str, float] | None = None,
        rrf_weights: tuple[float, ...] | None = None,
        bm25f_only: bool = False,
        skip_prf: bool = False,
    ) -> list[tuple[Fact, float]]:
        """Find the most relevant facts for a query.

        Args:
            query: The search query string.
            facts: Facts to search through.
            top_k: Maximum number of results to return.
            faithfulness_k: Threshold for the faithfulness floor check.  When
                ``None`` (default), uses ``top_k``.  Set to the *original*
                ``top_k`` when overfetching to prevent the larger ``top_k``
                from disabling the floor.
            expansion_weights: Optional LLM expansion terms with weights.
                Merged with PRF expansion; original query terms keep weight 1.0.
            bm25f_only: When True, return raw BM25F scores without RRF fusion.
                Used by HybridRetriever to avoid double-RRF.
            skip_prf: When True, skip pseudo-relevance feedback for this search.
                Useful for aggregation queries where PRF reinforces top matches
                instead of broadening coverage.

        Returns:
            List of (Fact, score) pairs sorted by relevance (most relevant first).
        """
        if not facts or not query.strip():
            return [(f, 0.0) for f in facts[:top_k]] if facts else []

        # 1. Build inverted index and get initial BM25F scores.
        index = InvertedIndex(facts)
        raw_scores = index.score(query)

        # 2. Query expansion: LLM (explicit) OR PRF (statistical fallback).
        # When LLM provides semantic expansion, skip PRF to avoid
        # reinforcing initial retrieval bias (Xu & Croft 1996).
        if expansion_weights:
            raw_scores = index.score(query, expansion_weights=expansion_weights)
        elif not (self._skip_prf or skip_prf) and len(facts) >= 4:
            prf = _prf_expand(index, query, raw_scores)
            if prf:
                raw_scores = index.score(query, expansion_weights=prf)

        # Early return: raw BM25F scores without RRF (used by HybridRetriever
        # to avoid double-RRF — hybrid does its own fusion of BM25F + dense).
        if bm25f_only:
            bm25f_results = [(f, raw_scores.get(f.id, 0.0)) for f in facts]
            bm25f_results.sort(key=lambda x: x[1], reverse=True)
            return bm25f_results[:top_k]

        # 3. Build six ranked lists for RRF.
        # Use BM25 score as secondary sort key to break ties deterministically.
        all_ids = [f.id for f in facts]
        query_tokens = frozenset(_tokenize(query))

        # Ranker 1: BM25F score (descending), importance tie-break.
        bm25_ranked = sorted(
            all_ids,
            key=lambda i: (raw_scores.get(i, 0.0), index.facts[i].importance),
            reverse=True,
        )

        # Precompute per-fact slot and trigram scores using index caches — avoids
        # recomputing _char_trigrams / slot tokenisation on every search call.
        query_trigrams = _char_trigrams(query)

        def _cached_slot_score(fid: str) -> float:
            slot_toks = index.slot_tokens[fid]
            if not slot_toks:
                return 0.0
            return len(query_tokens & slot_toks) / len(slot_toks)

        slot_scores: dict[str, float] = {fid: _cached_slot_score(fid) for fid in all_ids}

        def _cached_trigram_score(fid: str) -> float:
            ct = index.content_trigrams[fid]
            kt = index.canonical_trigrams[fid]
            et = index.evidence_trigrams[fid]
            qt = query_trigrams
            if not qt:
                return 0.0
            s_content = len(qt & ct) / len(qt | ct) if ct else 0.0
            s_canon = len(qt & kt) / len(qt | kt) if kt else 0.0
            s_evidence = len(qt & et) / len(qt | et) if et else 0.0
            return max(s_content, s_canon, s_evidence)

        trigram_scores: dict[str, float] = {fid: _cached_trigram_score(fid) for fid in all_ids}

        # Ranker 2: slot-exact — fraction of slot address tokens covered by the query.
        slot_ranked = sorted(
            all_ids,
            key=lambda i: (slot_scores[i], raw_scores.get(i, 0.0)),
            reverse=True,
        )

        # Ranker 3: char-trigram Jaccard — surface-level similarity between query
        # and fact content/canonical_surface.
        trigram_ranked = sorted(
            all_ids,
            key=lambda i: (trigram_scores[i], raw_scores.get(i, 0.0)),
            reverse=True,
        )

        # Ranker 4: importance (descending), BM25 tie-break.
        importance_ranked = sorted(
            all_ids,
            key=lambda i: (index.facts[i].importance, raw_scores.get(i, 0.0)),
            reverse=True,
        )

        # Ranker 5: retention (descending), BM25 tie-break.
        retention_ranked = sorted(
            all_ids,
            key=lambda i: (index.facts[i].retention_score, raw_scores.get(i, 0.0)),
            reverse=True,
        )

        # Ranker 6: recency — created_at (newest first), BM25 tie-break.
        # Use created_at, not last_accessed: last_accessed reflects when a fact was
        # *read*, which is identical for all facts inserted in a batch (S7 scenario).
        # created_at is immutable and correctly orders v1 → v5 temporal versions.
        recency_ranked = sorted(
            all_ids,
            key=lambda i: (index.facts[i].created_at, raw_scores.get(i, 0.0)),
            reverse=True,
        )

        # 4. RRF fusion — default weights (5,3,2,1.5,1.5,1) give BM25 ~37% influence,
        #    slot-exact ~22%, trigram ~15%, importance+retention ~22%, recency ~7%.
        fused = _rrf_fuse(
            [
                bm25_ranked,
                slot_ranked,
                trigram_ranked,
                importance_ranked,
                retention_ranked,
                recency_ranked,
            ],
            weights=list(rrf_weights or self._rrf_weights),
        )

        # 5. Faithfulness floor: when there are enough BM25 matches to fill
        # top_k, only return facts that have lexical relevance.  This prevents
        # importance/recency from pushing unrelated facts into results (S1/S3).
        # Slot-matched facts bypass this floor — they are deterministically relevant.
        # When there aren't enough matches, fall back to full RRF ranking.
        scored_ids = set(raw_scores.keys())
        slot_matched_ids = {fid for fid, s in slot_scores.items() if s > 0}

        results: list[tuple[Fact, float]] = []
        eligible_ids = scored_ids | slot_matched_ids
        floor_k = faithfulness_k if faithfulness_k is not None else top_k
        if len(eligible_ids) >= floor_k:
            for fact in facts:
                if fact.id in eligible_ids:
                    results.append((fact, fused.get(fact.id, 0.0)))
        else:
            for fact in facts:
                results.append((fact, fused.get(fact.id, 0.0)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Backward compatibility alias.
TFIDFRetriever = BM25Retriever
