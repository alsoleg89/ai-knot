"""Benchmark report renderer.

Generates a single Markdown document with:
  1. Summary table (all backends × key metrics)
  2. Per-scenario detail sections
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

    for sid, title in [
        ("s1_profile_retrieval", "S1 — Profile Retrieval"),
        ("s2_avoid_repeats", "S2 — Avoid Repeats"),
        ("s3_feedback_learning", "S3 — Feedback Learning"),
        ("s4_deduplication", "S4 — Deduplication"),
        ("s5_decay", "S5 — Decay"),
        ("s6_load", "S6 — Load & Reliability"),
    ]:
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
        "| Backend | S1 Relevance | S1 Token↓ | S4 Dedup% | S4 Retain% | S5 Δ Retain | S6 P95ms |"
    )
    lines.append(
        "|---------|-------------|-----------|-----------|------------|-------------|----------|"
    )

    for m in results:
        row = (
            f"| {m.backend_name} "
            f"| {_cell(m, 's1_profile_retrieval', 'relevance')} "
            f"| {_cell(m, 's1_profile_retrieval', 'token_reduction', pct=True)} "
            f"| {_cell(m, 's4_deduplication', 'dedup_ratio', pct=True)} "
            f"| {_cell(m, 's4_deduplication', 'retention_ratio', pct=True)} "
            f"| {_cell(m, 's5_decay', 'retention_delta')} "
            f"| {_cell(m, 's6_load', 'p95_latency_ms', ms=True)} |"
        )
        lines.append(row)

    return lines


def _scenario_section(results: list[BenchmarkMetrics], sid: str, title: str) -> list[str] | None:
    # Only render if at least one backend has results for this scenario
    has_data = any(any(sr.scenario_id == sid for sr in m.scenario_results) for m in results)
    if not has_data:
        return None

    lines: list[str] = [f"### {title}\n"]

    # Collect all metrics for this scenario
    all_metrics: set[str] = set()
    for m in results:
        for sr in m.scenario_results:
            if sr.scenario_id == sid:
                all_metrics.update(sr.judge_scores.keys())

    metric_list = sorted(all_metrics)
    header = "| Backend | " + " | ".join(metric_list) + " | Notes |"
    sep = "|---------|" + "|".join("-" * (len(k) + 2) for k in metric_list) + "|-------|"
    lines.append(header)
    lines.append(sep)

    for m in results:
        for sr in m.scenario_results:
            if sr.scenario_id == sid:
                cells = []
                for metric in metric_list:
                    vals = sr.judge_scores.get(metric, [])
                    if vals:
                        med = statistics.median(vals)
                        cells.append(_fmt(med))
                    else:
                        cells.append("—")
                notes_short = sr.notes[:60] + "…" if len(sr.notes) > 60 else sr.notes
                lines.append(f"| {m.backend_name} | " + " | ".join(cells) + f" | {notes_short} |")

    return lines
