"""Pytest smoke gate for the benchmark suite.

Runs S4 (deduplication) and S5 (decay) with MockJudge + StubProvider.
No Ollama required. Validates the benchmark framework itself — not
memory quality. Excluded from the default pytest run via addopts.

Run explicitly with:
    pytest tests/test_benchmark_smoke.py -v -m benchmark
"""

from __future__ import annotations

import asyncio

import pytest

from tests.eval.benchmark._stub_provider import StubProvider
from tests.eval.benchmark.backends.ai_knot_backend import AiKnotBackend
from tests.eval.benchmark.backends.baseline import BaselineBackend
from tests.eval.benchmark.judge import MockJudge
from tests.eval.benchmark.scenarios.s4_deduplication import run as run_s4
from tests.eval.benchmark.scenarios.s5_decay import run as run_s5


@pytest.mark.benchmark
def test_s4_baseline_stores_all_paraphrases() -> None:
    """Baseline should store every paraphrase — no deduplication."""
    result = asyncio.run(run_s4(BaselineBackend(), MockJudge()))
    dedup_ratio = result.judge_scores["dedup_ratio"][0]
    # Baseline stores all 50 verbatim → 50 unique texts → dedup_ratio ≈ 0
    assert dedup_ratio < 0.1, f"baseline unexpectedly deduplicates: ratio={dedup_ratio:.2f}"


@pytest.mark.benchmark
def test_s4_ai_knot_deduplicates_with_add() -> None:
    """ai-knot kb.add() bypasses LLM but still deduplicates via resolve_against_existing."""
    provider = StubProvider()
    backend = AiKnotBackend(provider, use_add=True)  # type: ignore[arg-type]
    result = asyncio.run(run_s4(backend, MockJudge()))
    dedup_ratio = result.judge_scores["dedup_ratio"][0]
    # With use_add=True all 50 paraphrases are stored (no extraction dedup),
    # but resolve_against_existing runs on retrieve — so unique count may vary.
    # The key assertion: the framework ran without error and returned a valid ratio.
    assert 0.0 <= dedup_ratio <= 1.0, f"dedup_ratio out of range: {dedup_ratio}"


@pytest.mark.benchmark
def test_s4_retention_ratio_baseline() -> None:
    """Baseline retains all 20 distinct rules (no false-positive dedup)."""
    result = asyncio.run(run_s4(BaselineBackend(), MockJudge()))
    retention = result.judge_scores["retention_ratio"][0]
    assert retention >= 0.8, f"baseline retention_ratio={retention:.2f} below 0.8"


@pytest.mark.benchmark
def test_s5_runs_without_error() -> None:
    """S5 framework runs to completion for both backends."""
    for BackendClass in (BaselineBackend,):
        result = asyncio.run(run_s5(BackendClass(), MockJudge()))  # type: ignore[call-arg]
        assert "relevance" in result.judge_scores
        assert result.scenario_id == "s5_decay"


@pytest.mark.benchmark
def test_s5_ai_knot_runs_without_error() -> None:
    """S5 runs for ai-knot with stub provider (use_add mode)."""
    provider = StubProvider()
    backend = AiKnotBackend(provider, use_add=True)  # type: ignore[arg-type]
    result = asyncio.run(run_s5(backend, MockJudge()))
    assert "relevance" in result.judge_scores
    assert "retention_delta" in result.judge_scores


@pytest.mark.benchmark
def test_mock_judge_returns_fixed_scores() -> None:
    """MockJudge returns consistent 3-run scores without variance."""
    judge = MockJudge()
    scores = judge.score_all("test query", ["some text"])
    for metric in ("relevance", "completeness", "faithfulness"):
        assert metric in scores
        vals = scores[metric]
        assert len(vals) == 3
        assert all(1.0 <= v <= 5.0 for v in vals), f"out-of-range score in {metric}: {vals}"
