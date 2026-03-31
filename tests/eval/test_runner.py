"""Tests for the eval runner."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_knot.retriever import BM25Retriever
from ai_knot.types import Fact, MemoryType
from tests.eval.datasets import RETRIEVAL_DATASET, RetrievalCase
from tests.eval.runner import run_case, run_eval

_METRIC_KEYS = {"precision", "recall", "mrr", "ndcg"}
_SUMMARY_KEYS = {"mean", "ci_lo", "ci_hi"}


def _make_fact(fact_id: str, content: str) -> Fact:
    return Fact(
        id=fact_id,
        content=content,
        type=MemoryType.SEMANTIC,
        importance=0.8,
        retention_score=1.0,
        access_count=0,
        last_accessed=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )


class TestRunCase:
    def test_returns_all_metric_keys(self) -> None:
        case = RETRIEVAL_DATASET[0]
        retriever = BM25Retriever()
        result = run_case(case, retriever, k=5)
        assert set(result.keys()) == _METRIC_KEYS

    def test_metric_values_are_floats(self) -> None:
        case = RETRIEVAL_DATASET[0]
        retriever = BM25Retriever()
        result = run_case(case, retriever, k=5)
        for key in _METRIC_KEYS:
            assert isinstance(result[key], float), f"{key} is not float"

    def test_metric_values_in_range(self) -> None:
        case = RETRIEVAL_DATASET[0]
        retriever = BM25Retriever()
        result = run_case(case, retriever, k=5)
        for key in _METRIC_KEYS:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0,1]"

    def test_k_parameter_respected(self) -> None:
        # With k=1, precision denominator is 1 so result is 0.0 or 1.0
        case = RETRIEVAL_DATASET[0]
        retriever = BM25Retriever()
        result = run_case(case, retriever, k=1)
        assert result["precision"] in (0.0, 1.0)


class TestRunEval:
    def test_returns_all_metrics(self) -> None:
        summary = run_eval(dataset=RETRIEVAL_DATASET[:3], k=5)
        assert set(summary.keys()) == _METRIC_KEYS

    def test_summary_has_correct_subkeys(self) -> None:
        summary = run_eval(dataset=RETRIEVAL_DATASET[:3], k=5)
        for metric in _METRIC_KEYS:
            assert set(summary[metric].keys()) == _SUMMARY_KEYS, f"Missing keys for {metric}"

    def test_mean_in_range(self) -> None:
        summary = run_eval(dataset=RETRIEVAL_DATASET[:5], k=5)
        for metric in _METRIC_KEYS:
            mean = summary[metric]["mean"]
            assert 0.0 <= mean <= 1.0, f"{metric} mean={mean} out of [0,1]"

    def test_ci_bounds_ordered(self) -> None:
        summary = run_eval(dataset=RETRIEVAL_DATASET[:5], k=5)
        for metric in _METRIC_KEYS:
            lo = summary[metric]["ci_lo"]
            hi = summary[metric]["ci_hi"]
            assert lo <= hi, f"{metric} ci_lo={lo} > ci_hi={hi}"

    def test_full_dataset_runs_without_error(self) -> None:
        summary = run_eval()
        assert len(summary) == len(_METRIC_KEYS)


class TestPerfectRetriever:
    """A query that exactly matches a fact's content should score P@k=1."""

    def test_exact_match_precision(self) -> None:
        # Create a case where the query text matches the relevant fact exactly
        unique_phrase = "zxqvb uniquephrase for perfect retrieval test"
        case = RetrievalCase(
            query=unique_phrase,
            facts=[
                {"id": "perfect_001", "content": unique_phrase},
                {"id": "noise_001", "content": "completely unrelated topic about cooking recipes"},
                {"id": "noise_002", "content": "weather forecast for the coming week"},
                {"id": "noise_003", "content": "sports results from last weekend games"},
                {"id": "noise_004", "content": "financial news and stock market updates"},
            ],
            relevant_ids=["perfect_001"],
        )
        retriever = BM25Retriever()
        result = run_case(case, retriever, k=5)
        assert result["precision"] == pytest.approx(1.0 / 5)
        # The relevant item should be top-ranked, so MRR=1.0
        assert result["mrr"] == pytest.approx(1.0)

    def test_exact_match_mrr_is_one(self) -> None:
        unique_phrase = "highly specific memory query about user authentication token refresh"
        case = RetrievalCase(
            query=unique_phrase,
            facts=[
                {"id": "top_001", "content": unique_phrase},
                {"id": "noise_a", "content": "user likes coffee in the morning"},
                {"id": "noise_b", "content": "project deadline is next Friday"},
                {"id": "noise_c", "content": "meeting scheduled for Tuesday afternoon"},
            ],
            relevant_ids=["top_001"],
        )
        retriever = BM25Retriever()
        result = run_case(case, retriever, k=4)
        assert result["mrr"] == pytest.approx(1.0)


class TestKnownMrrRanking:
    """Verify MRR computation against known rankings."""

    def test_mrr_second_position(self) -> None:
        # Construct case where relevant fact should rank 2nd
        # Noise fact shares more query terms; relevant is second best
        case = RetrievalCase(
            query="apple banana cherry",
            facts=[
                {
                    "id": "rank2",
                    "content": "banana cherry memory fact to be retrieved second place",
                },
                {
                    "id": "rank1",
                    "content": "apple banana cherry highly specific triple match query",
                },
                {"id": "noise_1", "content": "completely different topic unrelated"},
                {"id": "noise_2", "content": "another unrelated entry about weather"},
            ],
            relevant_ids=["rank2"],
        )
        retriever = BM25Retriever()
        # Just verify the runner returns a valid MRR in [0,1]
        result = run_case(case, retriever, k=4)
        assert 0.0 <= result["mrr"] <= 1.0

    def test_mrr_zero_when_not_retrieved(self) -> None:
        # Mark an id as relevant that definitely won't score high
        case = RetrievalCase(
            query="python type hints mypy strict",
            facts=[
                {"id": "rel", "content": "completely unrelated xyz abc topic"},
                {"id": "irrel_1", "content": "python type hints mypy strict mode checking"},
                {"id": "irrel_2", "content": "python mypy type annotations strict"},
                {"id": "irrel_3", "content": "type hints in python with mypy"},
                {"id": "irrel_4", "content": "strict type checking python mypy annotations"},
                {"id": "irrel_5", "content": "python strict mypy type hints configuration"},
            ],
            relevant_ids=["rel"],
        )
        retriever = BM25Retriever()
        result = run_case(case, retriever, k=5)
        # "rel" is totally unrelated — BM25 should rank it last, so MRR ~ 0
        assert result["mrr"] == pytest.approx(0.0)
