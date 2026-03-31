"""Hybrid retriever: BM25 (zero deps) + optional embeddings.

The default retriever uses a pure-Python BM25 (Okapi BM25) implementation with
no external dependencies. Results are boosted by retention_score and
importance to favor fresh, important facts.
"""

from __future__ import annotations

import math

from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# Weight multipliers for the hybrid score.
_TFIDF_WEIGHT: float = 0.6
_RETENTION_WEIGHT: float = 0.2
_IMPORTANCE_WEIGHT: float = 0.2

# BM25 parameters.
_BM25_K1: float = 1.5  # Term saturation parameter.
_BM25_B: float = 0.75  # Length normalization parameter.


class InvertedIndex:
    """Pre-computed inverted index for BM25 scoring.

    Builds once from a list of facts, then supports fast query lookups.
    Only the posting lists for query terms are traversed — O(Q * avg_postings)
    instead of O(N * Q).
    """

    def __init__(self, facts: list[Fact]) -> None:
        self._facts: dict[str, Fact] = {}  # id -> Fact
        self._doc_lengths: dict[str, int] = {}  # id -> token count
        self._postings: dict[str, dict[str, int]] = {}  # term -> {doc_id: tf}
        self._doc_count: int = 0
        self._avg_dl: float = 0.0
        self._build(facts)

    def _build(self, facts: list[Fact]) -> None:
        total_length = 0
        for fact in facts:
            self._facts[fact.id] = fact
            tokens = _tokenize(fact.content)
            self._doc_lengths[fact.id] = len(tokens)
            total_length += len(tokens)

            # Build term frequencies.
            tf_map: dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            # Update postings.
            for term, tf in tf_map.items():
                if term not in self._postings:
                    self._postings[term] = {}
                self._postings[term][fact.id] = tf

        self._doc_count = len(facts)
        self._avg_dl = total_length / self._doc_count if self._doc_count else 1.0

    def score(self, query: str, *, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        """Return BM25 scores for all documents matching any query term."""
        query_tokens = _tokenize(query)
        scores: dict[str, float] = {}
        n = self._doc_count

        for term in query_tokens:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in postings.items():
                dl = self._doc_lengths[doc_id]
                tf_score = (k1 + 1.0) * tf / (tf + k1 * (1.0 - b + b * dl / self._avg_dl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_score

        return scores

    @property
    def facts(self) -> dict[str, Fact]:
        """Return the id -> Fact mapping."""
        return self._facts


class BM25Retriever:
    """Zero-dependency BM25 (Okapi BM25) search over a list of Facts.

    Scoring uses BM25 formula with p95-clip normalization:
      BM25(q, d) = Σ_t  IDF(t) * (k1+1)*tf / (k1*(1-b+b*dl/avgdl) + tf)

    The normalized BM25 score is combined with retention and importance:
      hybrid = w1*bm25_normalized + w2*retention + w3*importance
    """

    def search(
        self,
        query: str,
        facts: list[Fact],
        *,
        top_k: int = 5,
    ) -> list[tuple[Fact, float]]:
        """Find the most relevant facts for a query.

        Args:
            query: The search query string.
            facts: Facts to search through.
            top_k: Maximum number of results to return.

        Returns:
            List of (Fact, score) pairs sorted by relevance (most relevant first).
            Scores are hybrid values combining BM25, retention, and importance.
        """
        if not facts or not query.strip():
            return [(f, 0.0) for f in facts[:top_k]] if facts else []

        # Build inverted index (could be cached in future).
        index = InvertedIndex(facts)
        raw_scores = index.score(query, k1=_BM25_K1, b=_BM25_B)

        # p95-clip normalization: clip to 95th percentile then normalize to [0, 1].
        if raw_scores:
            sorted_vals = sorted(raw_scores.values())
            n = len(sorted_vals)
            p95_idx = min(int(0.95 * n), n - 1)
            p95 = sorted_vals[p95_idx] if sorted_vals else 1.0
        else:
            p95 = 1.0

        results: list[tuple[Fact, float]] = []
        for fact in facts:
            bm25_raw = raw_scores.get(fact.id, 0.0)
            bm25_norm = min(bm25_raw / p95, 1.0) if p95 > 0 else 0.0
            hybrid = (
                _TFIDF_WEIGHT * bm25_norm
                + _RETENTION_WEIGHT * fact.retention_score
                + _IMPORTANCE_WEIGHT * fact.importance
            )
            results.append((fact, hybrid))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Backward compatibility alias.
TFIDFRetriever = BM25Retriever
