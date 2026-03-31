"""Retrieval quality metrics for ai-knot eval."""

from __future__ import annotations


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant retrieved in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for r in top_k if r in relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """MRR: reciprocal of rank of first relevant result."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    import math

    top_k = retrieved_ids[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(top_k) if rid in relevant_ids)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(
    scores: list[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for mean of scores.
    Uses stdlib random.choices — no numpy required.
    """
    import random

    rng = random.Random(seed)
    if not scores:
        return (0.0, 0.0)
    means = []
    for _ in range(n_resamples):
        sample = rng.choices(scores, k=len(scores))
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = int((1 - confidence) / 2 * n_resamples)
    hi = int((1 + confidence) / 2 * n_resamples)
    return (means[lo], means[min(hi, len(means) - 1)])
