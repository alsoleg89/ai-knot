"""Pass/fail gate for the multi-agent benchmark suite.

Encodes the acceptance thresholds locked in ``research/ma_baseline_20260609.md``.
Each threshold reads one ``(scenario, metric)`` median from a backend's
``BenchmarkMetrics`` and compares it against a target with a comparison
operator.  Thresholds whose metric is absent from a run are marked
non-applicable so a partial run (a subset of scenarios) is not penalised for
what it did not measure.  Pure logic — markdown rendering lives in ``report.py``.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass

from tests.eval.benchmark.base import BenchmarkMetrics

# Comparison operators keyed by their symbol for readable threshold declarations.
# Protocol "no regression" guards use ">=" 1.0 (the metrics cannot exceed 1.0,
# so ">=" 1.0 is an exact-1.0 check that is also float-dust safe on the high side).
_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
}


@dataclass(frozen=True)
class GateThreshold:
    """One acceptance threshold: ``(scenario_id, metric) op target``."""

    scenario_id: str
    metric: str
    op: str
    target: float
    group: str  # "protocol" | "conflict" | "evidence" | "adversarial" | "scale"


@dataclass(frozen=True)
class GateResult:
    """Outcome of evaluating one threshold against a backend's metrics."""

    threshold: GateThreshold
    value: float
    applicable: bool
    passed: bool


# Acceptance thresholds — locked in research/ma_baseline_20260609.md.
# Protocol-correctness entries are no-regression guards (must stay 1.0).
GATE: tuple[GateThreshold, ...] = (
    # Protocol correctness — no regression (currently all 1.00).
    GateThreshold("s10_ma_mesi_cas", "cas_correctness", ">=", 1.0, "protocol"),
    GateThreshold("s11_ma_mesi_sync", "delta_correctness", ">=", 1.0, "protocol"),
    GateThreshold("s13_concurrent_writers", "no_lost_updates", ">=", 1.0, "protocol"),
    GateThreshold("s17_self_correction", "correction_surfaced", ">=", 1.0, "protocol"),
    GateThreshold("s20_belief_revision", "final_consensus", ">=", 1.0, "protocol"),
    GateThreshold("s25_conflict_resolution", "resolution_correctness", ">=", 1.0, "protocol"),
    # Conflict safety (S9) — baseline 0.00 / 0.44.
    GateThreshold("s9_ma_pool_publish", "conflict_resolution", ">=", 0.80, "conflict"),
    GateThreshold("s9_ma_pool_publish", "precision_at_3", ">=", 0.70, "conflict"),
    # Evidence precision (S19) — baseline 0.57.
    GateThreshold("s19_incident_reconstruction", "evidence_precision", ">=", 0.70, "evidence"),
    # Adversarial suppression + trust penalty (S23) — baseline 0.60 / 0.00.
    GateThreshold("s23_adversarial_noise", "free_standing_suppression", ">=", 0.85, "adversarial"),
    GateThreshold("s23_adversarial_noise", "trust_penalty", ">", 0.50, "adversarial"),
    # Sparse-assembly scale tail (S26) — baseline 0.13 / 0.96 / 84ms.
    GateThreshold("s26_sparse_assembly", "target_shard_recall_at_1000", ">=", 0.60, "scale"),
    GateThreshold("s26_sparse_assembly", "distractor_rate_at_1000", "<=", 0.50, "scale"),
    GateThreshold("s26_sparse_assembly", "p95_retrieve_ms_at_1000", "<=", 150.0, "scale"),
)


def _metric_present(metrics: BenchmarkMetrics, scenario_id: str, metric: str) -> bool:
    """Return True if *metrics* actually carries scores for *scenario_id*/*metric*."""
    for r in metrics.scenario_results:
        if r.scenario_id == scenario_id:
            return bool(r.judge_scores.get(metric))
    return False


def evaluate_gate(
    metrics: BenchmarkMetrics, thresholds: tuple[GateThreshold, ...] = GATE
) -> list[GateResult]:
    """Evaluate every threshold against *metrics*.

    A threshold whose metric is absent from this run is marked non-applicable
    (``applicable=False``) and does not count toward pass/fail — a partial run
    is not penalised for scenarios it did not execute.
    """
    out: list[GateResult] = []
    for t in thresholds:
        present = _metric_present(metrics, t.scenario_id, t.metric)
        value = metrics.median_score(t.scenario_id, t.metric) if present else 0.0
        passed = present and _OPS[t.op](value, t.target)
        out.append(GateResult(threshold=t, value=value, applicable=present, passed=passed))
    return out


def gate_passed(results: list[GateResult]) -> bool:
    """True if at least one threshold was applicable and every applicable one passed."""
    applicable = [r for r in results if r.applicable]
    return bool(applicable) and all(r.passed for r in applicable)
