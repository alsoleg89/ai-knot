"""Benchmark report renderer.

Generates a single Markdown document with:
  1. Summary table (all backends × key metrics)
  2. Per-scenario detail sections (professional S1–S8, then legacy if present)
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime

from tests.eval.benchmark.base import BenchmarkMetrics


def render_markdown(results: list[BenchmarkMetrics]) -> str:
    lines: list[str] = []

    lines.append("# ai-knot Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("\n---\n")

    lines.append("## Summary\n")
    lines += _summary_table(results)

    lines.append("\n---\n")
    lines.append("## Per-Scenario Results\n")

    # Professional S1–S8 scenarios
    pro_scenarios = [
        ("s1_mrr", "S1 — MRR & Precision@k"),
        ("s2_semantic_gap", "S2 — Semantic Gap"),
        ("s3_staleness", "S3 — Staleness Resistance"),
        ("s4_compression_f1", "S4 — Memory Compression F1"),
        ("s5_noise", "S5 — Noise Tolerance"),
        ("s6_token_economy", "S6 — Context Economy"),
        ("s7_grounding", "S7 — Grounding Rate"),
        ("s8_throughput", "S8 — Latency & Throughput"),
    ]
    # Legacy S1–S7 scenarios (shown if present)
    legacy_scenarios = [
        ("s1_profile_retrieval", "S1-legacy — Profile Retrieval"),
        ("s2_avoid_repeats", "S2-legacy — Avoid Repeats"),
        ("s3_feedback_learning", "S3-legacy — Feedback Learning"),
        ("s4_deduplication", "S4-legacy — Deduplication"),
        ("s5_decay", "S5-legacy — Decay"),
        ("s6_load", "S6-legacy — Load & Reliability"),
        ("s7_consolidation", "S7-legacy — Temporal Consolidation"),
    ]

    for sid, title in pro_scenarios + legacy_scenarios:
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


def _cell(metrics: BenchmarkMetrics, sid: str, metric: str, **fmt_kwargs: bool) -> str:
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
        " | S1 MRR | S1 P@3"
        " | S2 SemGap"
        " | S3 Fresh@1 | S3 Stale"
        " | S4 Dedup% | S4 Retain% | S4 F1"
        " | S5 Signal@3 | S5 SNR"
        " | S6 TokComp | S6 Q/Tok"
        " | S7 Grounding | S7 HalluRate"
        " | S8 P95ms | S8 QPS |"
    )
    lines.append(
        "|---------|-----"
        "|--------|-------"
        "|----------"
        "|------------|----------"
        "|-----------|------------|-------"
        "|------------|--------"
        "|------------|----------"
        "|--------------|-------------"
        "|----------|--------|"
    )

    for m in results:
        lang = getattr(m, "language", "en")
        row = (
            f"| {m.backend_name} | {lang}"
            f" | {_cell(m, 's1_mrr', 'mrr')}"
            f" | {_cell(m, 's1_mrr', 'p_at_3')}"
            f" | {_cell(m, 's2_semantic_gap', 'semantic_gap')}"
            f" | {_cell(m, 's3_staleness', 'freshness_at1')}"
            f" | {_cell(m, 's3_staleness', 'staleness_rate')}"
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
            f" | {_cell(m, 's8_throughput', 'throughput')} |"
        )
        lines.append(row)

    return lines


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
        lines.append(f"| {m.backend_name} | " + " | ".join(cells) + f" | {notes_short} |")

    return lines
