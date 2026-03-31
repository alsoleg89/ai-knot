"""Hybrid retriever: BM25 (zero deps) + optional embeddings.

The default retriever uses a pure-Python BM25 (Okapi BM25) implementation with
no external dependencies. Results are boosted by retention_score and
importance to favor fresh, important facts.
"""

from __future__ import annotations

import math
from collections import Counter

from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# Weight multipliers for the hybrid score.
_TFIDF_WEIGHT: float = 0.6
_RETENTION_WEIGHT: float = 0.2
_IMPORTANCE_WEIGHT: float = 0.2

# BM25 parameters.
_BM25_K1: float = 1.5  # Term saturation parameter.
_BM25_B: float = 0.75  # Length normalization parameter.


class BM25Retriever:
    """Zero-dependency BM25 (Okapi BM25) search over a list of Facts.

    Scoring uses BM25 formula with p95-clip normalization:
      BM25(q, d) = Σ_t  IDF(t) * (k1+1)*tf / (k1*(1-b+b*dl/avgdl) + tf)

    The normalized BM25 score is combined with retention and importance:
      hybrid = w1*bm25_normalized + w2*retention + w3*importance
    """

    def search(self, query: str, facts: list[Fact], *, top_k: int = 5) -> list[tuple[Fact, float]]:
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

        query_tokens = _tokenize(query)
        if not query_tokens:
            return [(f, 0.0) for f in facts[:top_k]]

        # Tokenize all documents.
        doc_tokens_list: list[list[str]] = [_tokenize(fact.content) for fact in facts]

        num_docs = len(facts)

        # Compute average document length.
        doc_lengths = [len(tokens) for tokens in doc_tokens_list]
        avgdl = sum(doc_lengths) / num_docs if num_docs > 0 else 1.0

        # Count how many documents contain each token (document frequency).
        doc_freq: Counter[str] = Counter()
        for doc_tokens in doc_tokens_list:
            for token in set(doc_tokens):
                doc_freq[token] += 1

        # Compute raw BM25 scores.
        raw_bm25: list[float] = []
        for doc_tokens, dl in zip(doc_tokens_list, doc_lengths, strict=True):
            if not doc_tokens:
                raw_bm25.append(0.0)
                continue

            tf_counts = Counter(doc_tokens)
            bm25_score = 0.0

            for qt in query_tokens:
                tf = tf_counts.get(qt, 0)
                df = doc_freq.get(qt, 0)
                # IDF with smoothing to avoid log(0).
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
                # BM25 term score with length normalization.
                denom = tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / avgdl)
                bm25_score += idf * (_BM25_K1 + 1) * tf / denom if denom > 0 else 0.0

            raw_bm25.append(bm25_score)

        # p95-clip normalization: clip to 95th percentile then normalize to [0, 1].
        sorted_scores = sorted(raw_bm25)
        p95_idx = int(0.95 * len(sorted_scores))
        # If few docs, use max; otherwise use p95 value.
        p95 = sorted_scores[p95_idx] if p95_idx < len(sorted_scores) else sorted_scores[-1]

        normalized_bm25: list[float] = []
        for raw in raw_bm25:
            if p95 > 0.0:
                normalized_bm25.append(min(raw / p95, 1.0))
            else:
                normalized_bm25.append(0.0)

        # Compute hybrid scores and collect (score, idx) pairs.
        scored: list[tuple[float, int]] = []
        for idx, (fact, norm_score) in enumerate(zip(facts, normalized_bm25, strict=True)):
            hybrid = (
                _TFIDF_WEIGHT * norm_score
                + _RETENTION_WEIGHT * fact.retention_score
                + _IMPORTANCE_WEIGHT * fact.importance
            )
            scored.append((hybrid, idx))

        # Sort descending by score.
        scored.sort(key=lambda x: x[0], reverse=True)

        return [(facts[idx], score) for score, idx in scored[:top_k]]


# Backward compatibility alias.
TFIDFRetriever = BM25Retriever
