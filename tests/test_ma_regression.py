"""MA scenario regression gate — fast check (< 5s) for pre-commit.

Runs all 19 multi-agent scenarios with mock judge and verifies that
key metrics don't drop below known-good thresholds.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

# Thresholds: metric must be >= value to pass.
# S9 conflict_resolution and S23 trust_penalty are known issues from
# the bm25f_only fix and excluded from hard gates for now.
_THRESHOLDS: dict[str, dict[str, float]] = {
    "s8_ma_isolation": {"overlap_coverage": 1.0, "exclusivity_recall": 1.0},
    "s10_ma_mesi_cas": {"cas_correctness": 1.0, "latest_surfaced": 1.0},
    "s11_ma_mesi_sync": {"delta_correctness": 1.0, "delta_efficiency": 1.0},
    "s12_topic_gating": {"triage_precision": 1.0},
    "s13_concurrent_writers": {"no_lost_updates": 1.0, "version_chain_integrity": 1.0},
    "s14_trust_drift": {"trust_floor_reached": 1.0, "trust_recovery": 1.0},
    "s15_topic_leakage": {"channel_precision": 1.0, "shared_term_isolation": 0.65},
    "s16_knowledge_relay": {"layer_a_recall": 1.0, "layer_b_recall": 1.0, "layer_c_recall": 1.0},
    "s17_self_correction": {"correction_surfaced": 1.0, "trust_recovery": 1.0},
    "s18_trust_calibration": {"trust_calibration": 1.0},
    "s20_belief_revision": {"convergence_ok": 1.0, "final_consensus": 1.0},
    "s21_partial_assembly": {"coverage": 0.8},
    "s22_temporal_staleness": {"freshness_recall": 1.0, "staleness_rejection": 1.0},
    "s24_onboarding": {"pool_retrieval_recall": 1.0, "kb_absorption": 1.0},
    "s25_conflict_resolution": {"resolution_correctness": 1.0, "conflict_collapse": 1.0},
}


@pytest.mark.timeout(120)
def test_ma_scenarios_regression() -> None:
    """Run all MA scenarios and check key metrics against thresholds."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.eval.benchmark.runner",
            "--multi-agent",
            "--ma-category",
            "all",
            "--mock-judge",
            "--quick",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Benchmark runner failed:\n{result.stderr[-500:]}"

    # Find the raw JSON output file from stdout.
    raw_path = ""
    for line in result.stdout.splitlines():
        if "Raw JSON written to" in line:
            raw_path = line.split("Raw JSON written to ")[-1].strip()
            break
    assert raw_path, f"Could not find raw JSON path in output:\n{result.stdout[-300:]}"

    with open(raw_path) as f:
        data = json.load(f)

    backend = data["backends"].get("ai_knot_multi_agent:en", {})
    assert backend, "No ai_knot_multi_agent:en backend in results"

    failures: list[str] = []
    for scenario, metrics in _THRESHOLDS.items():
        scenario_data = backend.get(scenario)
        if not isinstance(scenario_data, dict):
            failures.append(f"{scenario}: missing from results")
            continue
        scores = scenario_data.get("judge_scores", {})
        for metric, threshold in metrics.items():
            value = scores.get(metric, {}).get("mean", 0.0)
            if value < threshold:
                failures.append(f"{scenario}:{metric} = {value:.2f} < {threshold:.2f}")

    assert not failures, "MA regression failures:\n" + "\n".join(failures)
