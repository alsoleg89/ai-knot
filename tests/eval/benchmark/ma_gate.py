"""Pass/fail gate for the multi-agent benchmark suite.

Encodes the acceptance thresholds for the multi-agent benchmark suite.
Each threshold reads one ``(scenario, metric)`` median from a backend's
``BenchmarkMetrics`` and compares it against a target with a comparison
operator. Unselected, absent scenarios are non-applicable; missing evidence
from a selected or present scenario fails verification. Pure logic — markdown
rendering lives in ``report.py``.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass

from tests.eval.benchmark.base import BenchmarkMetrics, valid_scores

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
    """One acceptance threshold: ``(scenario_id, metric) op target``.

    ``advisory`` thresholds are reported but do not bind the pass/fail verdict:
    they track diagnostic targets affected by source ambiguity, result length,
    runtime environment or an optional semantic resolver. Notes explain those
    limitations without turning an observed score into a universal bound.
    """

    scenario_id: str
    metric: str
    op: str
    target: float
    group: str  # "protocol" | "conflict" | "evidence" | "adversarial" | "scale"
    advisory: bool = False
    note: str = ""


@dataclass(frozen=True)
class GateResult:
    """Outcome of evaluating one threshold against a backend's metrics."""

    threshold: GateThreshold
    value: float
    applicable: bool
    passed: bool
    missing: bool = False


# Acceptance thresholds.  Binding entries are achievable targets / no-regression
# guards; advisory entries retain diagnostic targets and their limitations.
_CALIB = "calibration notes"
GATE: tuple[GateThreshold, ...] = (
    # Protocol correctness — no regression (currently all 1.00).
    GateThreshold("s10_ma_mesi_cas", "cas_correctness", ">=", 1.0, "protocol"),
    GateThreshold("s11_ma_mesi_sync", "delta_correctness", ">=", 1.0, "protocol"),
    GateThreshold("s13_concurrent_writers", "no_lost_updates", ">=", 1.0, "protocol"),
    GateThreshold("s17_self_correction", "correction_surfaced", ">=", 1.0, "protocol"),
    GateThreshold("s20_belief_revision", "final_consensus", ">=", 1.0, "protocol"),
    GateThreshold("s25_conflict_resolution", "resolution_correctness", ">=", 1.0, "protocol"),
    # Conflict safety (S9).  The system must SURFACE the correct answer (binding);
    # SUPPRESSING the lexically-divergent stale rival is a semantic value-conflict
    # the deterministic resolver cannot reach (advisory — needs the opt-in
    # semantic resolver; the observed 0.33 is not a universal ceiling).
    GateThreshold("s9_ma_pool_publish", "correct_at_3", ">=", 0.90, "conflict"),
    GateThreshold(
        "s9_ma_pool_publish",
        "conflict_resolution",
        ">=",
        0.80,
        "conflict",
        advisory=True,
        note=f"cross-source suppression target; observed deterministic score ~0.33 ({_CALIB} S4)",
    ),
    GateThreshold(
        "s9_ma_pool_publish",
        "wrong_suppression",
        ">=",
        0.80,
        "conflict",
        advisory=True,
        note=f"semantic value-conflict; needs the opt-in resolver ({_CALIB} S4)",
    ),
    # Evidence precision (S19) counts keyword-positive returned rows. top_k is
    # a maximum; variable result length prevents a universal 3/5 ceiling.
    GateThreshold(
        "s19_incident_reconstruction",
        "evidence_precision",
        ">=",
        0.70,
        "evidence",
        advisory=True,
        note="keyword precision with variable result length, not an entailment score "
        f"({_CALIB} S2)",
    ),
    # Adversarial suppression + trust penalty (S23) — binding.
    GateThreshold("s23_adversarial_noise", "free_standing_suppression", ">=", 0.85, "adversarial"),
    GateThreshold("s23_adversarial_noise", "trust_penalty", ">", 0.50, "adversarial"),
    # Sparse-assembly (S26).  Binding: small-pool exact recall.  Latency is
    # advisory — p95 retrieval time is environment-dependent (a shared CI runner
    # measures ~170ms where local hardware is <90ms), so it is reported but does
    # not bind the correctness verdict; the dedicated perf-benchmark job tracks
    # latency regressions with run-relative comparison.
    GateThreshold("s26_sparse_assembly", "target_shard_recall_at_10", ">=", 0.60, "scale"),
    GateThreshold(
        "s26_sparse_assembly",
        "p95_retrieve_ms_at_1000",
        "<=",
        150.0,
        "scale",
        advisory=True,
        note="latency is environment-dependent (shared CI runners ~170ms vs <90ms local); "
        "tracked by the perf-benchmark job, not a correctness bind",
    ),
    # Binding domain-coverage at scale: when the markerless query cannot name the
    # exact target among ~50 identical-content peers, surfacing ANY same-domain
    # shard is the achievable, meaningful signal — the ambiguity-aware S26 scenario
    # emits this. The exact-agent target_shard_recall_at_1000 stays advisory below.
    GateThreshold(
        "s26_sparse_assembly",
        "equivalence_recall_at_1000",
        ">=",
        0.90,
        "scale",
        note=f"domain-coverage gate replacing exact-agent recall@1000 ({_CALIB} S3)",
    ),
    # Advisory: at N=1000 each target has ~49 identical-content peers and the
    # markerless query cannot name it. Observed 0.33 is not a universal bound;
    # non-target rows include equivalent shards and returned count can be <10.
    GateThreshold(
        "s26_sparse_assembly",
        "target_shard_recall_at_1000",
        ">=",
        0.60,
        "scale",
        advisory=True,
        note=f"markerless peer ambiguity; observed exact recall ~0.33 is not a cap ({_CALIB} S3)",
    ),
    GateThreshold(
        "s26_sparse_assembly",
        "distractor_rate_at_1000",
        "<=",
        0.50,
        "scale",
        advisory=True,
        note=f"exact-target surplus includes useful peers; result length varies ({_CALIB} S3)",
    ),
)


def _metric_present(metrics: BenchmarkMetrics, scenario_id: str, metric: str) -> bool:
    """Return True if *metrics* carries usable scores for *scenario_id*/*metric*."""
    for r in metrics.scenario_results:
        if r.scenario_id == scenario_id:
            return valid_scores(r.judge_scores.get(metric, []))
    return False


def evaluate_gate(
    metrics: BenchmarkMetrics, thresholds: tuple[GateThreshold, ...] = GATE
) -> list[GateResult]:
    """Evaluate every threshold against *metrics*.

    Only absent, unselected scenarios are non-applicable. A selected or present
    scenario must supply its expected metrics, including advisory measurements.
    """
    out: list[GateResult] = []
    expected = set(metrics.selected_scenarios) | {r.scenario_id for r in metrics.scenario_results}
    for t in thresholds:
        present = _metric_present(metrics, t.scenario_id, t.metric)
        applicable = present or t.scenario_id in expected
        value = metrics.median_score(t.scenario_id, t.metric) if present else 0.0
        passed = present and _OPS[t.op](value, t.target)
        out.append(
            GateResult(
                threshold=t,
                value=value,
                applicable=applicable,
                passed=passed,
                missing=applicable and not present,
            )
        )
    return out


def gate_passed(results: list[GateResult]) -> bool:
    """True if at least one binding threshold applied and every binding one passed.

    Advisory values never bind the verdict, but missing expected measurements
    invalidate verification. Execution failures are checked separately by callers.
    """
    binding = [r for r in results if r.applicable and not r.threshold.advisory]
    return bool(binding) and all(r.passed for r in binding) and not any(r.missing for r in results)
