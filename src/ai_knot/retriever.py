"""Hybrid retriever: BM25F + PRF + RRF (zero deps).

The retriever uses a pure-Python BM25F (Robertson et al., 2004) implementation
with structured field weighting (content + tags). Pseudo-relevance feedback
(Rocchio 1971 / RM3) expands queries via top-k feedback documents. Final ranking
uses Reciprocal Rank Fusion (Cormack et al., 2009) over four signals: BM25F,
importance, retention, and recency.
"""

from __future__ import annotations

import math

from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# BM25F parameters.
_BM25_K1: float = 1.5  # Term saturation parameter.
_BM25_B_CONTENT: float = 0.75  # Length normalization for content field.
_BM25_B_TAGS: float = 0.3  # Length normalization for tags field.
_W_CONTENT: float = 1.0  # Content field weight.
_W_TAGS: float = 2.0  # Tags field weight (more specific, higher boost).

# IDF high-DF threshold: terms in >70% of docs get zero IDF weight.
_IDF_HIGH_DF_RATIO: float = 0.7

# PRF parameters.
_PRF_TOP_K: int = 3  # Number of feedback documents.
_PRF_EXPANSION_TERMS: int = 5  # Max expansion terms.
_PRF_ALPHA: float = 0.5  # PRF expansion term weight (statistical fallback).
# When LLM expansion is active, PRF is skipped entirely (see search()).

# RRF parameter.
_RRF_K: int = 60  # Rank smoothing constant (Cormack et al., 2009).


class InvertedIndex:
    """Pre-computed inverted index for BM25F scoring.

    Indexes both ``fact.content`` and ``fact.tags`` as separate fields,
    combining them via BM25F (Robertson, Zaragoza & Taylor 2004).
    Terms appearing in >70% of documents are auto-filtered (IDF zeroed).
    """

    def __init__(self, facts: list[Fact]) -> None:
        self._facts: dict[str, Fact] = {}  # id -> Fact
        self._content_lengths: dict[str, int] = {}  # id -> content token count
        self._tags_lengths: dict[str, int] = {}  # id -> tags token count
        self._content_postings: dict[str, dict[str, int]] = {}  # term -> {id: tf}
        self._tags_postings: dict[str, dict[str, int]] = {}  # term -> {id: tf}
        self._doc_count: int = 0
        self._avg_content_dl: float = 0.0
        self._avg_tags_dl: float = 0.0
        self._build(facts)

    def _build(self, facts: list[Fact]) -> None:
        total_content_len = 0
        total_tags_len = 0

        for fact in facts:
            self._facts[fact.id] = fact

            # Content field.
            content_tokens = _tokenize(fact.content)
            self._content_lengths[fact.id] = len(content_tokens)
            total_content_len += len(content_tokens)

            content_tf: dict[str, int] = {}
            for token in content_tokens:
                content_tf[token] = content_tf.get(token, 0) + 1
            for term, tf in content_tf.items():
                if term not in self._content_postings:
                    self._content_postings[term] = {}
                self._content_postings[term][fact.id] = tf

            # Tags field.
            tags_text = " ".join(fact.tags)
            tags_tokens = _tokenize(tags_text) if tags_text else []
            self._tags_lengths[fact.id] = len(tags_tokens)
            total_tags_len += len(tags_tokens)

            tags_tf: dict[str, int] = {}
            for token in tags_tokens:
                tags_tf[token] = tags_tf.get(token, 0) + 1
            for term, tf in tags_tf.items():
                if term not in self._tags_postings:
                    self._tags_postings[term] = {}
                self._tags_postings[term][fact.id] = tf

        self._doc_count = len(facts)
        self._avg_content_dl = total_content_len / self._doc_count if self._doc_count else 1.0
        self._avg_tags_dl = total_tags_len / self._doc_count if self._doc_count else 1.0

    def _combined_df(self, term: str) -> int:
        """Number of documents containing *term* in any field."""
        content_docs = set(self._content_postings.get(term, {}).keys())
        tags_docs = set(self._tags_postings.get(term, {}).keys())
        return len(content_docs | tags_docs)

    def score(
        self,
        query: str,
        *,
        k1: float = _BM25_K1,
        b: float = _BM25_B_CONTENT,
        expansion_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Return BM25F scores for all documents matching any query term.

        Args:
            query: Search query string.
            k1: BM25 saturation parameter.
            b: Length normalization parameter (used for content field).
            expansion_weights: Optional PRF expansion terms with weights.

        Returns:
            Dict mapping document id to BM25F score.
        """
        query_tokens = _tokenize(query)
        # Build term weights: original query terms get weight 1.0.
        term_weights: dict[str, float] = {}
        for t in query_tokens:
            term_weights[t] = 1.0

        # Merge expansion terms.
        if expansion_weights:
            for t, w in expansion_weights.items():
                if t not in term_weights:
                    term_weights[t] = w

        scores: dict[str, float] = {}
        n = self._doc_count
        high_df_threshold = _IDF_HIGH_DF_RATIO * n

        for term, q_weight in term_weights.items():
            df = self._combined_df(term)
            if not df:
                continue
            # IDF auto-filter: zero weight for terms in >70% of docs.
            # Only apply when corpus is large enough (>= 5 docs) to avoid
            # filtering legitimate terms in tiny document sets.
            if n >= 5 and df > high_df_threshold:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            # Collect all doc_ids that have this term in either field.
            content_posting = self._content_postings.get(term, {})
            tags_posting = self._tags_postings.get(term, {})
            doc_ids = set(content_posting.keys()) | set(tags_posting.keys())

            for doc_id in doc_ids:
                # BM25F: weighted sum of normalized tf across fields.
                tf_content = content_posting.get(doc_id, 0)
                dl_c = self._content_lengths[doc_id]
                norm_tf_c = tf_content / (1.0 + b * (dl_c / self._avg_content_dl - 1.0))

                tf_tags = tags_posting.get(doc_id, 0)
                dl_t = self._tags_lengths[doc_id]
                avg_t = self._avg_tags_dl if self._avg_tags_dl > 0 else 1.0
                norm_tf_t = tf_tags / (1.0 + _BM25_B_TAGS * (dl_t / avg_t - 1.0))

                tf_bm25f = _W_CONTENT * norm_tf_c + _W_TAGS * norm_tf_t
                tf_score = (k1 + 1.0) * tf_bm25f / (k1 + tf_bm25f)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_score * q_weight

        return scores

    @property
    def facts(self) -> dict[str, Fact]:
        """Return the id -> Fact mapping."""
        return self._facts

    @property
    def content_lengths(self) -> dict[str, int]:
        """Return document content lengths for PRF."""
        return self._content_lengths


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
      1. BM25F (content + tags fields) with IDF high-DF filtering.
      2. Pseudo-relevance feedback expands query with top-doc terms.
      3. Reciprocal Rank Fusion combines BM25F, importance, retention,
         and recency into a single ranking.
    """

    def __init__(
        self,
        *,
        rrf_weights: tuple[float, ...] = (5.0, 1.0, 1.0, 1.0),
    ) -> None:
        self._rrf_weights = rrf_weights

    def search(
        self,
        query: str,
        facts: list[Fact],
        *,
        top_k: int = 5,
        expansion_weights: dict[str, float] | None = None,
    ) -> list[tuple[Fact, float]]:
        """Find the most relevant facts for a query.

        Args:
            query: The search query string.
            facts: Facts to search through.
            top_k: Maximum number of results to return.
            expansion_weights: Optional LLM expansion terms with weights.
                Merged with PRF expansion; original query terms keep weight 1.0.

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
        elif len(facts) >= 4:
            prf = _prf_expand(index, query, raw_scores)
            if prf:
                raw_scores = index.score(query, expansion_weights=prf)

        # 3. Build four ranked lists for RRF.
        # Use BM25 score as secondary sort key to break ties deterministically.
        all_ids = [f.id for f in facts]

        # Ranker 1: BM25F score (descending), importance tie-break.
        bm25_ranked = sorted(
            all_ids,
            key=lambda i: (raw_scores.get(i, 0.0), index.facts[i].importance),
            reverse=True,
        )

        # Ranker 2: importance (descending), BM25 tie-break.
        importance_ranked = sorted(
            all_ids,
            key=lambda i: (index.facts[i].importance, raw_scores.get(i, 0.0)),
            reverse=True,
        )

        # Ranker 3: retention (descending), BM25 tie-break.
        retention_ranked = sorted(
            all_ids,
            key=lambda i: (index.facts[i].retention_score, raw_scores.get(i, 0.0)),
            reverse=True,
        )

        # Ranker 4: recency — last_accessed (newest first), BM25 tie-break.
        recency_ranked = sorted(
            all_ids,
            key=lambda i: (index.facts[i].last_accessed, raw_scores.get(i, 0.0)),
            reverse=True,
        )

        # 4. RRF fusion — default weights give BM25 5x weight (≈62% influence).
        fused = _rrf_fuse(
            [bm25_ranked, importance_ranked, retention_ranked, recency_ranked],
            weights=list(self._rrf_weights),
        )

        results: list[tuple[Fact, float]] = []
        for fact in facts:
            results.append((fact, fused.get(fact.id, 0.0)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Backward compatibility alias.
TFIDFRetriever = BM25Retriever
