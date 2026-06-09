"""Tests for the multi-agent acceptance gate and scorecard (PR A / A2).

Locks the pass/fail semantics of ``ma_gate.GATE``: the locked baseline numbers
(research/ma_baseline_20260609.md) must FAIL the gate, target numbers must PASS,
absent metrics are non-applicable (a partial run is not penalised), and a
protocol-correctness regression fails the gate.
"""

from __future__ import annotations

from tests.eval.benchmark.base import BenchmarkMetrics, ScenarioResult
from tests.eval.benchmark.ma_gate import GATE, evaluate_gate, gate_passed
from tests.eval.benchmark.report import _ma_scorecard


def _metrics(backend: str, scores: dict[str, dict[str, float]]) -> BenchmarkMetrics:
    """Build a BenchmarkMetrics from ``{scenario_id: {metric: value}}``."""
    m = BenchmarkMetrics(backend_name=backend)
    for sid, metric_map in scores.items():
        m.scenario_results.append(
            ScenarioResult(
                scenario_id=sid,
                backend_name=backend,
                judge_scores={k: [v] for k, v in metric_map.items()},
                insert_result=None,
                retrieval_result=None,
            )
        )
    return m


_PROTOCOL = {
    "s10_ma_mesi_cas": {"cas_correctness": 1.0},
    "s11_ma_mesi_sync": {"delta_correctness": 1.0},
    "s13_concurrent_writers": {"no_lost_updates": 1.0},
    "s17_self_correction": {"correction_surfaced": 1.0},
    "s20_belief_revision": {"final_consensus": 1.0},
    "s25_conflict_resolution": {"resolution_correctness": 1.0},
}

# Baseline numbers from research/ma_baseline_20260609.md — must FAIL the gate.
_BASELINE = {
    **_PROTOCOL,
    "s9_ma_pool_publish": {"conflict_resolution": 0.0, "precision_at_3": 0.44},
    "s19_incident_reconstruction": {"evidence_precision": 0.57},
    "s23_adversarial_noise": {"free_standing_suppression": 0.60, "trust_penalty": 0.0},
    "s26_sparse_assembly": {
        "target_shard_recall_at_1000": 0.13,
        "distractor_rate_at_1000": 0.96,
        "p95_retrieve_ms_at_1000": 83.61,
    },
}

# Targets reached — must PASS the gate.
_PASSING = {
    **_PROTOCOL,
    "s9_ma_pool_publish": {"conflict_resolution": 0.85, "precision_at_3": 0.72},
    "s19_incident_reconstruction": {"evidence_precision": 0.75},
    "s23_adversarial_noise": {"free_standing_suppression": 0.90, "trust_penalty": 0.6},
    "s26_sparse_assembly": {
        "target_shard_recall_at_1000": 0.65,
        "distractor_rate_at_1000": 0.40,
        "p95_retrieve_ms_at_1000": 90.0,
    },
}


def test_baseline_fails_gate() -> None:
    results = evaluate_gate(_metrics("ai_knot_multi_agent", _BASELINE))
    assert not gate_passed(results)
    failed = {
        (r.threshold.scenario_id, r.threshold.metric)
        for r in results
        if r.applicable and not r.passed
    }
    assert ("s9_ma_pool_publish", "conflict_resolution") in failed
    assert ("s23_adversarial_noise", "trust_penalty") in failed
    assert ("s26_sparse_assembly", "target_shard_recall_at_1000") in failed
    # Protocol correctness stays green even at baseline.
    assert all(r.passed for r in results if r.threshold.group == "protocol")


def test_passing_metrics_pass_gate() -> None:
    results = evaluate_gate(_metrics("ai_knot_multi_agent", _PASSING))
    assert gate_passed(results)
    assert all(r.passed for r in results if r.applicable)


def test_absent_metric_not_applicable() -> None:
    # Only protocol scenarios present — conflict/evidence/scale thresholds are n/a.
    results = evaluate_gate(_metrics("ai_knot_multi_agent", _PROTOCOL))
    s9 = next(r for r in results if r.threshold.scenario_id == "s9_ma_pool_publish")
    assert not s9.applicable
    assert not s9.passed
    # Gate still passes — only applicable (protocol) thresholds count.
    assert gate_passed(results)


def test_protocol_regression_fails_gate() -> None:
    regressed = {**_PASSING, "s10_ma_mesi_cas": {"cas_correctness": 0.75}}
    results = evaluate_gate(_metrics("ai_knot_multi_agent", regressed))
    assert not gate_passed(results)


def test_empty_metrics_not_passed() -> None:
    # No applicable thresholds → nothing was verified → not a pass.
    results = evaluate_gate(_metrics("ai_knot_multi_agent", {}))
    assert not gate_passed(results)


def test_scorecard_renders_pass_and_fail() -> None:
    md_fail = "\n".join(_ma_scorecard([_metrics("ai_knot_multi_agent", _BASELINE)]))
    assert "Acceptance Scorecard" in md_fail
    assert "❌ FAIL" in md_fail
    assert "conflict_resolution" in md_fail

    md_pass = "\n".join(_ma_scorecard([_metrics("ai_knot_multi_agent", _PASSING)]))
    assert "✅ PASS" in md_pass


def test_gate_covers_documented_failures() -> None:
    pairs = {(t.scenario_id, t.metric) for t in GATE}
    assert ("s9_ma_pool_publish", "conflict_resolution") in pairs
    assert ("s19_incident_reconstruction", "evidence_precision") in pairs
    assert ("s23_adversarial_noise", "free_standing_suppression") in pairs
    assert ("s23_adversarial_noise", "trust_penalty") in pairs
    assert ("s26_sparse_assembly", "target_shard_recall_at_1000") in pairs
