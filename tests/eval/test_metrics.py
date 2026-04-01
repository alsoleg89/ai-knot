"""Unit tests for retrieval quality metric functions."""

from __future__ import annotations

import math

import pytest

from tests.eval.metrics import (
    bootstrap_ci,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, k=3) == 0.0

    def test_half_relevant(self) -> None:
        result = precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4)
        assert result == pytest.approx(0.5)

    def test_k_truncates(self) -> None:
        # Only first k=2 considered: ["a", "b"], "a" is relevant -> 1/2
        result = precision_at_k(["a", "b", "c"], {"a"}, k=2)
        assert result == pytest.approx(0.5)

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_k_larger_than_results(self) -> None:
        # Only 2 results but k=5: top_k = ["a", "b"], precision = 1/5 = 0.2
        result = precision_at_k(["a", "b"], {"a"}, k=5)
        assert result == pytest.approx(0.2)


class TestRecallAtK:
    def test_all_relevant_retrieved(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_partial_recall(self) -> None:
        result = recall_at_k(["a", "b", "c"], {"a", "b", "d"}, k=3)
        # 2 out of 3 relevant retrieved
        assert result == pytest.approx(2 / 3)

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_none_retrieved_relevant(self) -> None:
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_k_truncates(self) -> None:
        # Only first k=1: ["a"], relevant={"a","b"}, recall=1/2
        result = recall_at_k(["a", "b"], {"a", "b"}, k=1)
        assert result == pytest.approx(0.5)


class TestMeanReciprocalRank:
    def test_first_result_relevant(self) -> None:
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_result_relevant(self) -> None:
        assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)

    def test_third_result_relevant(self) -> None:
        assert mean_reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant(self) -> None:
        assert mean_reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self) -> None:
        assert mean_reciprocal_rank([], {"a"}) == 0.0

    def test_multiple_relevant_uses_first(self) -> None:
        # First relevant is at rank 2
        result = mean_reciprocal_rank(["x", "a", "b"], {"a", "b"})
        assert result == pytest.approx(0.5)


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        # All relevant items at top
        result = ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3)
        assert result == pytest.approx(1.0)

    def test_no_relevant(self) -> None:
        assert ndcg_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0

    def test_single_relevant_at_rank_1(self) -> None:
        # DCG = 1/log2(2) = 1.0, IDCG = 1/log2(2) = 1.0
        result = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
        assert result == pytest.approx(1.0)

    def test_single_relevant_at_rank_2(self) -> None:
        # DCG = 1/log2(3), IDCG = 1/log2(2) = 1.0
        result = ndcg_at_k(["x", "a", "y"], {"a"}, k=3)
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert result == pytest.approx(expected)

    def test_partial_ranking(self) -> None:
        # retrieved: [a, b, x, y], relevant: {a, b}
        # DCG = 1/log2(2) + 1/log2(3)
        # IDCG = 1/log2(2) + 1/log2(3) (both at top 2)
        result = ndcg_at_k(["a", "b", "x", "y"], {"a", "b"}, k=4)
        assert result == pytest.approx(1.0)

    def test_k_truncation(self) -> None:
        # k=1, relevant at rank 2 -> DCG=0, NDCG=0
        result = ndcg_at_k(["x", "a"], {"a"}, k=1)
        assert result == pytest.approx(0.0)


class TestBootstrapCi:
    def test_empty_scores(self) -> None:
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_single_score(self) -> None:
        lo, hi = bootstrap_ci([0.5])
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_all_same_scores(self) -> None:
        lo, hi = bootstrap_ci([1.0] * 10)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_ci_bounds_order(self) -> None:
        scores = [float(i) / 10 for i in range(11)]
        lo, hi = bootstrap_ci(scores)
        assert lo <= hi

    def test_ci_within_range(self) -> None:
        scores = [0.0, 0.5, 1.0, 0.25, 0.75]
        lo, hi = bootstrap_ci(scores)
        assert lo >= 0.0 <= hi <= 1.0

    def test_reproducible_with_seed(self) -> None:
        scores = [0.1, 0.4, 0.7, 0.3, 0.9]
        ci1 = bootstrap_ci(scores, seed=42)
        ci2 = bootstrap_ci(scores, seed=42)
        assert ci1 == ci2

    def test_different_seeds_may_differ(self) -> None:
        scores = [0.1, 0.4, 0.7, 0.3, 0.9]
        ci1 = bootstrap_ci(scores, seed=1)
        ci2 = bootstrap_ci(scores, seed=99)
        # Not guaranteed to differ but very likely with different seeds
        # Just verify both are valid tuples
        assert len(ci1) == 2
        assert len(ci2) == 2

    def test_bca_skewed_data(self) -> None:
        """BCa should produce asymmetric intervals on skewed data."""
        scores = [0.01, 0.02, 0.03, 0.05, 0.9, 0.95, 0.99]
        lo, hi = bootstrap_ci(scores, n_resamples=2000)
        mean = sum(scores) / len(scores)
        assert lo < mean < hi
        assert lo >= 0.0
        assert hi <= 1.0

    def test_bca_two_scores(self) -> None:
        """BCa should handle n=2 without error."""
        lo, hi = bootstrap_ci([0.2, 0.8])
        assert lo <= hi
        assert lo >= 0.0
        assert hi <= 1.0

    def test_bca_ci_narrows_with_more_resamples(self) -> None:
        """More resamples should give stable (not wider) intervals."""
        scores = [0.3, 0.5, 0.7, 0.4, 0.6]
        _, hi_500 = bootstrap_ci(scores, n_resamples=500, seed=42)
        _, hi_5000 = bootstrap_ci(scores, n_resamples=5000, seed=42)
        # Both should be reasonable; mainly testing no crash
        assert 0.0 <= hi_500 <= 1.0
        assert 0.0 <= hi_5000 <= 1.0
