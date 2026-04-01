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


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse standard normal (rational approximation, Abramowitz & Stegun)."""
    import math

    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        return -_normal_ppf(1.0 - p)

    # Rational approximation for 0.5 < p < 1
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def bootstrap_ci(
    scores: list[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """BCa bootstrap confidence interval for mean of scores (Efron, 1987).

    Bias-corrected and accelerated — adjusts percentile bounds for
    skewness and median-bias. Uses stdlib only, no numpy/scipy.
    """
    import random

    if not scores:
        return (0.0, 0.0)
    n = len(scores)
    if n < 2:
        v = scores[0]
        return (v, v)

    observed_mean = sum(scores) / n
    rng = random.Random(seed)

    # Bootstrap resampling
    means: list[float] = []
    for _ in range(n_resamples):
        sample = rng.choices(scores, k=n)
        means.append(sum(sample) / n)
    means.sort()

    # Bias correction: z0 = Φ⁻¹(fraction of means < observed)
    below = sum(1 for m in means if m < observed_mean)
    prop = below / n_resamples
    prop = max(1e-10, min(1.0 - 1e-10, prop))
    z0 = _normal_ppf(prop)

    # Acceleration via jackknife
    jackknife_means: list[float] = []
    for i in range(n):
        jk_sum = sum(scores) - scores[i]
        jackknife_means.append(jk_sum / (n - 1))
    jk_bar = sum(jackknife_means) / n

    num = 0.0
    den = 0.0
    for jm in jackknife_means:
        diff = jk_bar - jm
        num += diff * diff * diff
        den += diff * diff

    acc = num / (6.0 * max(den, 1e-30) ** 1.5) if den > 1e-30 else 0.0

    # Adjusted percentiles
    alpha = (1.0 - confidence) / 2.0
    z_alpha = _normal_ppf(alpha)
    z_1alpha = _normal_ppf(1.0 - alpha)

    def _adjusted_pct(z_a: float) -> float:
        numer = z0 + z_a
        denom = 1.0 - acc * numer
        if abs(denom) < 1e-10:
            return _normal_cdf(z0 + z_a)
        return _normal_cdf(z0 + numer / denom)

    p_lo = _adjusted_pct(z_alpha)
    p_hi = _adjusted_pct(z_1alpha)

    idx_lo = max(0, min(int(p_lo * n_resamples), n_resamples - 1))
    idx_hi = max(0, min(int(p_hi * n_resamples), n_resamples - 1))

    return (means[idx_lo], means[idx_hi])
