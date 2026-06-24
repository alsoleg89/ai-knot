"""Hybrid retriever: BM25F + PRF + RRF (zero deps).

The retriever uses a pure-Python BM25F (Robertson et al., 2004) implementation
with structured field weighting (content + canonical_surface + tags).
Pseudo-relevance feedback (Rocchio 1971 / RM3) expands queries via top-k feedback
documents. Final ranking uses Reciprocal Rank Fusion (Cormack et al., 2009) over
six signals: BM25F, slot-exact, char-trigram, importance, retention, and recency.
"""

from __future__ import annotations

import math

from ai_knot._bm25 import BM25Retriever, TFIDFRetriever, _prf_expand, _rrf_fuse
from ai_knot._inverted_index import (
    InvertedIndex,
    _char_trigram_jaccard,
    _char_trigrams,
    _slot_exact_score,
    _trigram_jaccard_against,
)
from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# Default RRF weights for hybrid fusion: BM25 gets 2x the dense weight,
# reflecting that lexical match is more reliable when embeddings are noisy
# or the model hasn't seen the domain vocabulary.
_HYBRID_BM25_WEIGHT: float = 2.0
_HYBRID_DENSE_WEIGHT: float = 1.0

__all__ = [
    "BM25Retriever",
    "TFIDFRetriever",
    "InvertedIndex",
    "DenseRetriever",
    "HybridRetriever",
    "_tokenize",
    "_prf_expand",
    "_rrf_fuse",
    "_char_trigrams",
    "_char_trigram_jaccard",
    "_trigram_jaccard_against",
    "_slot_exact_score",
]


class DenseRetriever:
    """Cosine-similarity retriever using precomputed embedding vectors.

    Embeddings must be provided via ``set_embeddings()`` before search.
    When no embeddings are available, ``search()`` returns empty results
    so callers can fall back to BM25.

    This retriever is synchronous — call ``embed_texts()`` from the async
    embedder externally and feed the results in via ``set_embeddings()``.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}  # fact_id -> embedding

    def set_embeddings(self, vectors: dict[str, list[float]]) -> None:
        """Load precomputed embeddings (fact_id → vector), replacing existing."""
        self._vectors = vectors

    def add_embeddings(self, vectors: dict[str, list[float]]) -> None:
        """Merge new embeddings into the existing set without rebuilding."""
        self._vectors.update(vectors)

    def has_embeddings(self) -> bool:
        """Return True if at least one embedding is available."""
        return bool(self._vectors)

    def search(
        self,
        query_vector: list[float],
        facts: list[Fact],
        *,
        top_k: int = 5,
    ) -> list[tuple[Fact, float]]:
        """Rank facts by cosine similarity to *query_vector*.

        Facts without a precomputed embedding are scored 0.0.

        Returns:
            (Fact, score) pairs sorted by descending cosine similarity.
        """
        if not query_vector or not facts:
            return []

        results: list[tuple[Fact, float]] = []
        for fact in facts:
            vec = self._vectors.get(fact.id)
            if vec is None:
                results.append((fact, 0.0))
                continue
            results.append((fact, _cosine(query_vector, vec)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class HybridRetriever:
    """Fuses BM25 and dense (embedding) retrieval via Reciprocal Rank Fusion.

    Falls back to BM25-only when no embeddings are available, providing a
    seamless upgrade path: existing code keeps working, and embeddings
    improve results when present.

    Args:
        bm25: The BM25 retriever instance (with its own RRF/PRF config).
        dense: The dense retriever instance (pre-loaded with embeddings).
        bm25_weight: RRF fusion weight for BM25 ranking.
        dense_weight: RRF fusion weight for dense ranking.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        *,
        bm25_weight: float = _HYBRID_BM25_WEIGHT,
        dense_weight: float = _HYBRID_DENSE_WEIGHT,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight

    def search(
        self,
        query: str,
        facts: list[Fact],
        *,
        top_k: int = 5,
        query_vector: list[float] | None = None,
        expansion_weights: dict[str, float] | None = None,
        rrf_weights: tuple[float, ...] | None = None,
    ) -> list[tuple[Fact, float]]:
        """Search using BM25 + dense fusion.

        When *query_vector* is ``None`` or the dense retriever has no
        embeddings, falls back to BM25-only (pure lexical).

        Args:
            query: Text query for BM25.
            facts: Facts to search through.
            top_k: Maximum results to return.
            query_vector: Optional embedding vector for the query.
            expansion_weights: Optional LLM expansion terms for BM25.
            rrf_weights: Optional per-call BM25 RRF weight override.

        Returns:
            (Fact, score) pairs sorted by fused relevance.
        """
        # Always run BM25.  Use bm25f_only=True to get raw BM25F scores
        # without internal 6-signal RRF — hybrid does its own fusion and
        # double-RRF would amplify noise from non-relevance signals.
        bm25_results = self._bm25.search(
            query,
            facts,
            top_k=len(facts),  # full ranking for fusion
            expansion_weights=expansion_weights,
            rrf_weights=rrf_weights,
            bm25f_only=True,
        )

        # If no dense signal available, return BM25 directly.
        # Filter zero-scored facts: without dense signal to rescue them,
        # they would pollute top-k and inflate used_count for trust.
        if query_vector is None or not self._dense.has_embeddings():
            scored = [(f, s) for f, s in bm25_results if s > 0.0]
            return scored[:top_k] if scored else bm25_results[:top_k]

        # Run dense retrieval.
        dense_results = self._dense.search(
            query_vector,
            facts,
            top_k=len(facts),
        )

        # Build ranked lists for RRF fusion.
        bm25_ranked = [f.id for f, _ in bm25_results]
        dense_ranked = [f.id for f, _ in dense_results]

        fused = _rrf_fuse(
            [bm25_ranked, dense_ranked],
            weights=[self._bm25_weight, self._dense_weight],
        )

        fact_map = {f.id: f for f in facts}
        results = [(fact_map[fid], score) for fid, score in fused.items() if fid in fact_map]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
