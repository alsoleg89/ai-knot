"""Benchmark report renderer.

Generates a single Markdown document with:
  1. Summary table (single-agent backends × key metrics)
  2. Multi-agent summary table (MA backends × MA scenario metrics)
  3. Per-scenario detail sections (professional S1–S9, MA S8–S25, then legacy if present)
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime

from tests.eval.benchmark.base import BenchmarkMetrics

# SA scenario IDs — used to classify backends as single-agent vs multi-agent.
_SA_SCENARIO_IDS = frozenset(
    {
        "s1_mrr",
        "s2_semantic_gap",
        "s3_staleness",
        "s4_compression_f1",
        "s5_noise",
        "s6_token_economy",
        "s7_grounding",
        "s8_throughput",
        "s9_scale",
        # legacy
        "s1_profile_retrieval",
        "s2_avoid_repeats",
        "s3_feedback_learning",
        "s4_deduplication",
        "s5_decay",
        "s6_load",
        "s7_consolidation",
        "s16_update_correctness",
        "s_locomo",
    }
)

_MA_SCENARIO_IDS = frozenset(
    {
        "s8_ma_isolation",
        "s9_ma_pool_publish",
        "s10_ma_mesi_cas",
        "s11_ma_mesi_sync",
        "s12_topic_gating",
        "s13_concurrent_writers",
        "s14_trust_drift",
        "s15_topic_leakage",
        "s16_knowledge_relay",
        "s17_self_correction",
        "s18_trust_calibration",
        "s19_incident_reconstruction",
        "s20_belief_revision",
        "s21_partial_assembly",
        "s22_temporal_staleness",
        "s23_adversarial_noise",
        "s24_onboarding",
        "s25_conflict_resolution",
        "s26_sparse_assembly",
    }
)

# Display name overrides: backend_name → label shown in report tables.
_DISPLAY_NAMES: dict[str, str] = {
    "ai_knot_no_llm": "ai_knot (no-LLM control)",
}


def _display(backend_name: str) -> str:
    return _DISPLAY_NAMES.get(backend_name, backend_name)


def render_markdown(results: list[BenchmarkMetrics]) -> str:
    lines: list[str] = []

    lines.append("# ai-knot Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("\n---\n")

    sa_results = [
        m for m in results if any(sr.scenario_id in _SA_SCENARIO_IDS for sr in m.scenario_results)
    ]
    ma_results = [
        m for m in results if any(sr.scenario_id in _MA_SCENARIO_IDS for sr in m.scenario_results)
    ]

    lines.append("## Summary — Single-Agent\n")
    if sa_results:
        lines += _summary_table(sa_results)
    else:
        lines.append("_No single-agent results._\n")

    if ma_results:
        lines.append("\n## Summary — Multi-Agent\n")
        lines += _ma_summary_table(ma_results)

    lines.append("\n---\n")
    lines.append("## Per-Scenario Results\n")

    # Retrieval Metrics
    lines.append("### Retrieval Metrics\n")
    retrieval_scenarios = [
        ("s1_mrr", "S1 — MRR & Precision@k"),
        ("s2_semantic_gap", "S2 — Semantic Gap"),
        ("s5_noise", "S5 — Noise Tolerance"),
        ("s6_token_economy", "S6 — Context Economy"),
    ]
    for sid, title in retrieval_scenarios:
        section = _scenario_section(results, sid, title)
        if section:
            lines += section
            lines.append("")

    # State Metrics
    lines.append("### State Metrics\n")
    state_scenarios = [
        ("s3_staleness", "S3 — Staleness Resistance"),
        ("s4_compression_f1", "S4 — Memory Compression F1"),
        ("s7_grounding", "S7 — Grounding Rate"),
    ]
    for sid, title in state_scenarios:
        section = _scenario_section(results, sid, title)
        if section:
            lines += section
            lines.append("")

    # Sync & Performance Metrics
    lines.append("### Sync & Performance Metrics\n")
    perf_scenarios = [
        ("s8_throughput", "S8 — Latency & Throughput"),
        ("s9_scale", "S9 — Scale Sensitivity"),
        ("s16_update_correctness", "S16 — Update Semantics"),
        ("s_locomo", "S-LoCoMo — Long-Context Memory QA"),
    ]
    for sid, title in perf_scenarios:
        section = _scenario_section(results, sid, title)
        if section:
            lines += section
            lines.append("")

    # Multi-Agent Scenarios
    lines.append("### Multi-Agent Scenarios\n")
    ma_scenarios = [
        ("s8_ma_isolation", "S8-MA — Multi-Agent Isolation"),
        ("s9_ma_pool_publish", "S9 — Pool Publish"),
        ("s10_ma_mesi_cas", "S10 — MESI CAS"),
        ("s11_ma_mesi_sync", "S11 — MESI Sync"),
        ("s12_topic_gating", "S12 — Topic Gating"),
        ("s13_concurrent_writers", "S13 — Concurrent Writers"),
        ("s14_trust_drift", "S14 — Trust Drift"),
        ("s15_topic_leakage", "S15 — Topic Leakage"),
        ("s16_knowledge_relay", "S16 — Knowledge Relay"),
        ("s17_self_correction", "S17 — Self-Correction"),
        ("s18_trust_calibration", "S18 — Trust Calibration"),
        ("s19_incident_reconstruction", "S19 — Incident Reconstruction"),
        ("s20_belief_revision", "S20 — Belief Revision"),
        ("s21_partial_assembly", "S21 — Partial Assembly"),
        ("s22_temporal_staleness", "S22 — Temporal Staleness"),
        ("s23_adversarial_noise", "S23 — Adversarial Noise"),
        ("s24_onboarding", "S24 — Onboarding"),
        ("s25_conflict_resolution", "S25 — Conflict Resolution"),
        ("s26_sparse_assembly", "S26 — Sparse Assembly"),
    ]
    for sid, title in ma_scenarios:
        section = _scenario_section(results, sid, title)
        if section:
            lines += section
            lines.append("")

    # Legacy scenarios (shown if present)
    legacy_scenarios = [
        ("s1_profile_retrieval", "S1-legacy — Profile Retrieval"),
        ("s2_avoid_repeats", "S2-legacy — Avoid Repeats"),
        ("s3_feedback_learning", "S3-legacy — Feedback Learning"),
        ("s4_deduplication", "S4-legacy — Deduplication"),
        ("s5_decay", "S5-legacy — Decay"),
        ("s6_load", "S6-legacy — Load & Reliability"),
        ("s7_consolidation", "S7-legacy — Temporal Consolidation"),
    ]
    present_ids = {sr.scenario_id for m in results for sr in m.scenario_results}
    if present_ids & {sid for sid, _ in legacy_scenarios}:
        lines.append("### Legacy Scenarios\n")
    for sid, title in legacy_scenarios:
        section = _scenario_section(results, sid, title)
        if section:
            lines += section
            lines.append("")

    return "\n".join(lines)


def _fmt(val: float, *, pct: bool = False, ms: bool = False) -> str:
    if pct:
        return f"{val:.1%}"
    if ms:
        return f"{val:.0f}ms"
    return f"{val:.2f}"


def _has_metric(metrics: BenchmarkMetrics, sid: str, metric: str) -> bool:
    """Return True if *metrics* actually contains scores for *sid*/*metric*."""
    for r in metrics.scenario_results:
        if r.scenario_id == sid:
            return bool(r.judge_scores.get(metric))
    return False


def _cell(metrics: BenchmarkMetrics, sid: str, metric: str, **fmt_kwargs: bool) -> str:
    if not _has_metric(metrics, sid, metric):
        return "N/A"
    med = metrics.median_score(sid, metric)
    std = metrics.stdev_score(sid, metric)
    val_str = _fmt(med, **fmt_kwargs)
    if std > 0.01:
        return f"{val_str} (±{std:.2f})"
    return val_str


def _summary_table(results: list[BenchmarkMetrics]) -> list[str]:
    lines: list[str] = []
    lines.append(
        "| Backend | Lang"
        " | S1 LexMRR | S1 SemMRR | S1 P@3"
        " | S2 SemGap"
        " | S3 StateAcc | S3 OverCon | S3 SlotDedup"
        " | S4 Dedup% | S4 Retain% | S4 F1"
        " | S5 Signal@3 | S5 SNR"
        " | S6 TokComp | S6 Q/Tok"
        " | S7 Grounding | S7 HalluRate"
        " | S8 P95ms | S8 QPS"
        " | S9 MRR@0 | S9 MRR@1k | S9 Degrad"
        " | S16 Del | S16 Noop | S16 Upd"
        " | LoCoMo F1 | LoCoMo 1-hop | LoCoMo Multi | LoCoMo Temp |"
    )
    lines.append(
        "|---------|-----"
        "|-----------|-----------|-------"
        "|----------"
        "|-------------|------------|------------"
        "|-----------|------------|-------"
        "|------------|--------"
        "|------------|----------"
        "|--------------|-------------"
        "|----------|--------"
        "|----------|-----------|-----------|"
        "---------|----------|---------|"
        "-----------|-------------|-------------|------------|"
    )

    for m in results:
        lang = getattr(m, "language", "en")
        row = (
            f"| {_display(m.backend_name)} | {lang}"
            f" | {_cell(m, 's1_mrr', 'lexical_mrr')}"
            f" | {_cell(m, 's1_mrr', 'semantic_mrr')}"
            f" | {_cell(m, 's1_mrr', 'p_at_3')}"
            f" | {_cell(m, 's2_semantic_gap', 'semantic_gap')}"
            f" | {_cell(m, 's3_staleness', 'latest_state_accuracy')}"
            f" | {_cell(m, 's3_staleness', 'overconsolidation_rate')}"
            f" | {_cell(m, 's3_staleness', 'slot_dedup_ratio', pct=True)}"
            f" | {_cell(m, 's4_compression_f1', 'dedup_ratio', pct=True)}"
            f" | {_cell(m, 's4_compression_f1', 'retention_ratio', pct=True)}"
            f" | {_cell(m, 's4_compression_f1', 'compression_f1')}"
            f" | {_cell(m, 's5_noise', 'signal_recall_at3')}"
            f" | {_cell(m, 's5_noise', 'snr')}"
            f" | {_cell(m, 's6_token_economy', 'token_compression', pct=True)}"
            f" | {_cell(m, 's6_token_economy', 'quality_per_token')}"
            f" | {_cell(m, 's7_grounding', 'mean_grounding')}"
            f" | {_cell(m, 's7_grounding', 'hallucination_rate')}"
            f" | {_cell(m, 's8_throughput', 'p95_ms', ms=True)}"
            f" | {_cell(m, 's8_throughput', 'throughput')}"
            f" | {_cell(m, 's9_scale', 'mrr_at_0')}"
            f" | {_cell(m, 's9_scale', 'mrr_at_1000')}"
            f" | {_cell(m, 's9_scale', 'mrr_degradation')}"
            f" | {_cell(m, 's16_update_correctness', 'delete_correctness', pct=True)}"
            f" | {_cell(m, 's16_update_correctness', 'noop_correctness', pct=True)}"
            f" | {_cell(m, 's16_update_correctness', 'update_correctness', pct=True)}"
            f" | {_cell(m, 's_locomo', 'overall_f1')}"
            f" | {_cell(m, 's_locomo', 'single_hop_f1')}"
            f" | {_cell(m, 's_locomo', 'multi_hop_f1')}"
            f" | {_cell(m, 's_locomo', 'temporal_f1')} |"
        )
        lines.append(row)

    return lines


def _ma_summary_table(results: list[BenchmarkMetrics]) -> list[str]:
    """Compact MA summary split into Protocol Correctness and Retrieval & Behavior."""

    _PROTOCOL_COLS: list[tuple[str, str, str]] = [
        ("s10_ma_mesi_cas", "S10 CAS", "cas_correctness"),
        ("s11_ma_mesi_sync", "S11 Delta", "delta_correctness"),
        ("s13_concurrent_writers", "S13 NoLost", "no_lost_updates"),
        ("s17_self_correction", "S17 CorrSurf", "correction_surfaced"),
        ("s20_belief_revision", "S20 Consens", "final_consensus"),
        ("s25_conflict_resolution", "S25 Canon", "canonical_coverage"),
    ]

    _RETRIEVAL_COLS: list[tuple[str, str, str]] = [
        ("s8_ma_isolation", "S8 Overlap", "overlap_coverage"),
        ("s9_ma_pool_publish", "S9 Conflict", "conflict_resolution"),
        ("s12_topic_gating", "S12 Triage", "triage_precision"),
        ("s14_trust_drift", "S14 Floor", "trust_floor_reached"),
        ("s15_topic_leakage", "S15 ChPrec", "channel_precision"),
        ("s16_knowledge_relay", "S16 Chain", "chain_depth"),
        ("s18_trust_calibration", "S18 TrCal", "trust_calibration"),
        ("s19_incident_reconstruction", "S19 EvPrec", "evidence_precision"),
        ("s21_partial_assembly", "S21 Cover", "coverage"),
        ("s22_temporal_staleness", "S22 Fresh", "freshness_recall"),
        ("s23_adversarial_noise", "S23 SlotSup", "slot_suppression"),
        ("s24_onboarding", "S24 PoolRec", "pool_retrieval_recall"),
        ("s26_sparse_assembly", "S26 Recall", "target_shard_recall_at_10"),
    ]

    def _render_group(title: str, cols: list[tuple[str, str, str]]) -> list[str]:
        lines: list[str] = [f"#### {title}\n"]
        header = "| Backend | " + " | ".join(label for _, label, _ in cols) + " |"
        sep = "|---------|" + "|".join("-" * (len(label) + 2) for _, label, _ in cols) + "|"
        lines.append(header)
        lines.append(sep)
        for m in results:
            cells = " | ".join(_cell(m, sid, metric) for sid, _, metric in cols)
            lines.append(f"| {_display(m.backend_name)} | {cells} |")
        return lines

    out: list[str] = []
    out += _render_group("Protocol Correctness", _PROTOCOL_COLS)
    out.append("")
    out += _render_group("Retrieval & Behavior", _RETRIEVAL_COLS)
    return out


def _scenario_section(results: list[BenchmarkMetrics], sid: str, title: str) -> list[str] | None:
    # Single pass: collect matching results and their metrics
    from tests.eval.benchmark.base import ScenarioResult

    matching: list[tuple[BenchmarkMetrics, ScenarioResult]] = []
    all_metrics: set[str] = set()
    for m in results:
        for sr in m.scenario_results:
            if sr.scenario_id == sid:
                matching.append((m, sr))
                all_metrics.update(sr.judge_scores.keys())

    if not matching:
        return None

    lines: list[str] = [f"### {title}\n"]
    metric_list = sorted(all_metrics)
    header = "| Backend | " + " | ".join(metric_list) + " | Notes |"
    sep = "|---------|" + "|".join("-" * (len(k) + 2) for k in metric_list) + "|-------|"
    lines.append(header)
    lines.append(sep)

    for m, sr in matching:
        cells = []
        for metric in metric_list:
            vals = sr.judge_scores.get(metric, [])
            if vals:
                med = statistics.median(vals)
                cells.append(_fmt(med))
            else:
                cells.append("—")
        notes_short = sr.notes[:80] + "…" if len(sr.notes) > 80 else sr.notes
        lines.append(f"| {_display(m.backend_name)} | " + " | ".join(cells) + f" | {notes_short} |")

    return lines
