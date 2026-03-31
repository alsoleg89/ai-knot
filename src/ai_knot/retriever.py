"""Hybrid retriever: TF-IDF (zero deps) + optional embeddings.

The default retriever uses a pure-Python TF-IDF implementation with
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


class TFIDFRetriever:
    """Zero-dependency TF-IDF search over a list of Facts.

    Scoring: hybrid_score = w1*tfidf + w2*retention + w3*importance
    """

    def search(self, query: str, facts: list[Fact], *, top_k: int = 5) -> list[tuple[Fact, float]]:
        """Find the most relevant facts for a query.

        Args:
            query: The search query string.
            facts: Facts to search through.
            top_k: Maximum number of results to return.

        Returns:
            List of (Fact, score) pairs sorted by relevance (most relevant first).
            Scores are hybrid values combining TF-IDF, retention, and importance.
        """
        if not facts or not query.strip():
            return [(f, 0.0) for f in facts[:top_k]] if facts else []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return [(f, 0.0) for f in facts[:top_k]]

        # Build document frequency map.
        doc_tokens_list: list[list[str]] = []
        for fact in facts:
            doc_tokens_list.append(_tokenize(fact.content))

        num_docs = len(facts)
        # Count how many documents contain each token.
        doc_freq: Counter[str] = Counter()
        for doc_tokens in doc_tokens_list:
            for token in set(doc_tokens):
                doc_freq[token] += 1

        # Score each fact.
        scored: list[tuple[float, int]] = []
        for idx, (fact, doc_tokens) in enumerate(zip(facts, doc_tokens_list, strict=True)):
            if not doc_tokens:
                scored.append((0.0, idx))
                continue

            # TF-IDF score for this document against the query.
            tf_counts = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            tfidf_score = 0.0

            for qt in query_tokens:
                tf = tf_counts.get(qt, 0) / doc_len if doc_len > 0 else 0.0
                df = doc_freq.get(qt, 0)
                idf = math.log(1.0 + num_docs / (1.0 + df))
                tfidf_score += tf * idf

            # Hybrid score: combine TF-IDF with retention and importance.
            hybrid = (
                _TFIDF_WEIGHT * tfidf_score
                + _RETENTION_WEIGHT * fact.retention_score
                + _IMPORTANCE_WEIGHT * fact.importance
            )
            scored.append((hybrid, idx))

        # Sort descending by score.
        scored.sort(key=lambda x: x[0], reverse=True)

        return [(facts[idx], score) for score, idx in scored[:top_k]]
