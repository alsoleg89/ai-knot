"""Build a publish-ready competitor benchmark pack.

The benchmark runner already supports a wide backend matrix. This script wraps
it into a shareable artifact set:

- a full runner report,
- raw JSON,
- a curated scorecard markdown,
- a compact summary JSON,
- metadata with the exact command and profile.

Use this when you want a controlled side-by-side pack for launch threads,
benchmark discussions, or release notes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    scenario_id: str
    metric: str
    label: str
    direction: str
    fmt: str


@dataclass(frozen=True)
class BenchProfile:
    name: str
    description: str
    backends: tuple[str, ...]
    scenarios: tuple[str, ...]
    metric_specs: tuple[MetricSpec, ...]
    notes: tuple[str, ...]
    env: dict[str, str]
    mock_judge: bool = False
    skip_multi_agent: bool = True


_BENCH_SCENARIOS_OFFLINE = (
    "s1_mrr",
    "s5_noise",
    "s6_token_economy",
    "s9_scale",
    "s16_update_correctness",
)

_BENCH_SCENARIOS_LIVE = (
    "s1_mrr",
    "s3_staleness",
    "s4_compression_f1",
    "s5_noise",
    "s6_token_economy",
    "s9_scale",
    "s16_update_correctness",
)

_OFFLINE_METRICS = (
    MetricSpec("s1_mrr", "lexical_mrr", "S1 LexMRR", "up", "score"),
    MetricSpec("s5_noise", "signal_recall_at3", "S5 Signal@3", "up", "score"),
    MetricSpec("s6_token_economy", "token_compression", "S6 TokComp", "up", "pct"),
    MetricSpec("s6_token_economy", "quality_per_token", "S6 Q/Tok", "up", "score"),
    MetricSpec("s9_scale", "mrr_degradation", "S9 Degrad", "down", "pct"),
    MetricSpec("s16_update_correctness", "update_correctness", "S16 Update", "up", "pct"),
)

_LIVE_METRICS = (
    MetricSpec("s1_mrr", "lexical_mrr", "S1 LexMRR", "up", "score"),
    MetricSpec("s3_staleness", "latest_state_accuracy", "S3 Latest", "up", "score"),
    MetricSpec("s4_compression_f1", "compression_f1", "S4 CompF1", "up", "score"),
    MetricSpec("s5_noise", "signal_recall_at3", "S5 Signal@3", "up", "score"),
    MetricSpec("s6_token_economy", "token_compression", "S6 TokComp", "up", "pct"),
    MetricSpec("s6_token_economy", "quality_per_token", "S6 Q/Tok", "up", "score"),
    MetricSpec("s9_scale", "mrr_degradation", "S9 Degrad", "down", "pct"),
    MetricSpec("s16_update_correctness", "update_correctness", "S16 Update", "up", "pct"),
)

PROFILES: dict[str, BenchProfile] = {
    "offline": BenchProfile(
        name="offline",
        description=(
            "Zero-network control pack: MockJudge + no embeddings. This isolates "
            "deterministic storage and retrieval behavior without local Ollama."
        ),
        backends=("baseline", "ai_knot_no_llm", "qdrant", "mem0"),
        scenarios=_BENCH_SCENARIOS_OFFLINE,
        metric_specs=_OFFLINE_METRICS,
        notes=(
            "This profile sets AI_KNOT_BENCH_DISABLE_EMBED=1, so qdrant/mem0 "
            "emulators stay fully offline even if Ollama is running locally.",
            "Use this for reproducible architecture comparisons and launch threads "
            "where you want the exact raw JSON to be re-runnable anywhere.",
            "It compares deterministic control surfaces, not full extraction quality. "
            "Use local-llm or real before making claims about learned compression.",
        ),
        env={"AI_KNOT_BENCH_DISABLE_EMBED": "1"},
        mock_judge=True,
    ),
    "local-llm": BenchProfile(
        name="local-llm",
        description=(
            "Local Ollama extraction pack: compares ai-knot against vector-store "
            "controls with LLM extraction enabled, but without external hosted APIs."
        ),
        backends=("baseline", "ai_knot", "qdrant_extraction", "mem0"),
        scenarios=_BENCH_SCENARIOS_LIVE,
        metric_specs=_LIVE_METRICS,
        notes=(
            "Requires a local Ollama-compatible endpoint for extraction/embeddings.",
            "This is the most useful pre-launch apples-to-apples pack when you want "
            "to compare retrieval architecture and compression behavior.",
            "Do not present these numbers as a category leaderboard; attach the raw "
            "JSON and exact command whenever you cite them.",
        ),
        env={},
    ),
    "real": BenchProfile(
        name="real",
        description=(
            "Live services pack: real qdrant + mem0ai surfaces plus ai-knot. Use "
            "this before publishing a fresh public competitor scorecard."
        ),
        backends=("baseline", "ai_knot", "qdrant_real", "mem0_real"),
        scenarios=_BENCH_SCENARIOS_LIVE,
        metric_specs=_LIVE_METRICS,
        notes=(
            "Requires qdrant-client, mem0ai, chromadb, and a running local Ollama.",
            "Use this profile when you want the closest in-repo comparison to the "
            "real external stacks, not only emulator controls.",
            "Pair any public claim with the generated metadata.json and runner_raw.json.",
        ),
        env={},
    ),
}

_DISPLAY_NAMES = {
    "baseline": "naive log",
    "ai_knot_no_llm": "ai-knot (deterministic control)",
    "ai_knot": "ai-knot",
    "qdrant_emulator": "qdrant emulator",
    "qdrant_extraction": "qdrant + extraction",
    "qdrant_real": "qdrant",
    "mem0_emulator": "mem0-style emulator",
    "mem0_real": "mem0",
    "memvid": "memvid",
}

_SURFACES = {
    "baseline": "FIFO log / control",
    "ai_knot_no_llm": "facts + deterministic recall",
    "ai_knot": "extract -> facts -> deterministic recall",
    "qdrant_emulator": "dense vector control",
    "qdrant_extraction": "extract -> dense vector control",
    "qdrant_real": "real vector DB",
    "mem0_emulator": "extract -> dense vector control",
    "mem0_real": "real mem0 library",
    "memvid": "video-encoded memory",
}

_ARTIFACT_NAMES = {
    "report": "runner_report.md",
    "raw": "runner_raw.json",
    "jsonl": "runner_live.jsonl",
    "scorecard": "scorecard.md",
    "summary": "summary.json",
    "metadata": "metadata.json",
    "command": "command.sh",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="offline",
        help="Benchmark pack profile to run or render.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for artifacts. Defaults to "
            "benchmark_results/competitor_pack_<profile>_<stamp>/"
        ),
    )
    parser.add_argument(
        "--from-raw",
        default=None,
        help="Skip the runner and render the scorecard from an existing raw JSON file.",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="Override backends with a comma-separated list.",
    )
    parser.add_argument(
        "--include-memvid",
        action="store_true",
        help="Append memvid when building the runner command.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used for the benchmark runner.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact runner command and exit without running it.",
    )
    return parser.parse_args(argv)


def _default_output_dir(profile_name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path("benchmark_results") / f"competitor_pack_{profile_name}_{stamp}"


def _resolve_profile(name: str) -> BenchProfile:
    return PROFILES[name]


def _resolve_backends(
    profile: BenchProfile,
    *,
    backends_override: str | None,
    include_memvid: bool,
) -> list[str]:
    if backends_override:
        backends = [part.strip() for part in backends_override.split(",") if part.strip()]
    else:
        backends = list(profile.backends)
    if include_memvid and "memvid" not in backends:
        backends.append("memvid")
    return backends


def _build_runner_command(
    *,
    python_bin: str,
    profile: BenchProfile,
    output_dir: Path,
    backends: list[str],
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "tests.eval.benchmark.runner",
        "--backends",
        ",".join(backends),
        "--scenarios",
        ",".join(profile.scenarios),
        "--output",
        str(output_dir / _ARTIFACT_NAMES["report"]),
        "--raw-output",
        str(output_dir / _ARTIFACT_NAMES["raw"]),
        "--jsonl-output",
        str(output_dir / _ARTIFACT_NAMES["jsonl"]),
    ]
    if profile.mock_judge:
        cmd.append("--mock-judge")
    if profile.skip_multi_agent:
        cmd.append("--skip-multi-agent")
    return cmd


def _shell_command(cmd: list[str], env_overrides: dict[str, str]) -> str:
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_overrides.items())
    cmd_text = " ".join(shlex.quote(part) for part in cmd)
    return f"{env_prefix} {cmd_text}".strip()


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "backends" not in data:
        raise ValueError(f"unexpected raw benchmark schema in {path}")
    return data


def _metric_value(backend_data: dict[str, Any], scenario_id: str, metric: str) -> float | None:
    scenario = backend_data.get(scenario_id)
    if not isinstance(scenario, dict):
        return None
    judge_scores = scenario.get("judge_scores")
    if not isinstance(judge_scores, dict):
        return None
    metric_obj = judge_scores.get(metric)
    if not isinstance(metric_obj, dict):
        return None
    mean = metric_obj.get("mean")
    if isinstance(mean, int | float):
        return float(mean)
    return None


def _backend_rows(
    raw: dict[str, Any], *, preferred_order: list[str] | None
) -> list[dict[str, Any]]:
    backends_obj = raw.get("backends")
    if not isinstance(backends_obj, dict):
        raise ValueError("raw benchmark JSON is missing the 'backends' mapping")

    rows: list[dict[str, Any]] = []
    for key, value in backends_obj.items():
        if not isinstance(value, dict):
            continue
        backend_id, _, language = key.partition(":")
        rows.append(
            {
                "backend_key": key,
                "backend_id": backend_id,
                "language": language or value.get("language", "en"),
                "data": value,
            }
        )

    if preferred_order is None:
        return rows

    order_index = {name: idx for idx, name in enumerate(preferred_order)}
    rows.sort(key=lambda row: (order_index.get(row["backend_id"], 999), row["backend_key"]))
    return rows


def _format_value(value: float | None, fmt: str) -> str:
    if value is None:
        return "—"
    if fmt == "pct":
        return f"{value:.1%}"
    return f"{value:.2f}"


def _best_metric_values(
    rows: list[dict[str, Any]],
    metric_specs: tuple[MetricSpec, ...],
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for spec in metric_specs:
        values = [
            _metric_value(row["data"], spec.scenario_id, spec.metric)
            for row in rows
        ]
        usable = [value for value in values if value is not None]
        if not usable:
            continue
        key = (spec.scenario_id, spec.metric)
        out[key] = max(usable) if spec.direction == "up" else min(usable)
    return out


def _display_name(backend_id: str) -> str:
    return _DISPLAY_NAMES.get(backend_id, backend_id)


def _surface_name(backend_id: str) -> str:
    return _SURFACES.get(backend_id, "custom")


def render_scorecard(
    raw: dict[str, Any],
    *,
    profile: BenchProfile,
    output_dir: Path,
    runner_command: str,
    backends: list[str] | None,
    env_overrides: dict[str, str],
) -> str:
    rows = _backend_rows(raw, preferred_order=backends)
    best_values = _best_metric_values(rows, profile.metric_specs)

    lines: list[str] = []
    lines.append(f"# ai-knot Competitor Bench Pack — {profile.name}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(profile.description)
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(
        f"- `{_ARTIFACT_NAMES['scorecard']}` — curated summary for launch threads and docs"
    )
    lines.append(f"- `{_ARTIFACT_NAMES['report']}` — full benchmark runner markdown")
    lines.append(f"- `{_ARTIFACT_NAMES['raw']}` — raw schema-v2 JSON")
    lines.append(f"- `{_ARTIFACT_NAMES['jsonl']}` — per-scenario streaming JSONL from the run")
    lines.append(f"- `{_ARTIFACT_NAMES['metadata']}` — profile, command, and artifact metadata")
    lines.append("")
    lines.append("## Exact command")
    lines.append("")
    lines.append("```bash")
    lines.append(runner_command)
    lines.append("```")
    lines.append("")
    if env_overrides:
        lines.append("## Environment overrides")
        lines.append("")
        for key, value in env_overrides.items():
            lines.append(f"- `{key}={value}`")
        lines.append("")
    lines.append("## Backend surfaces")
    lines.append("")
    lines.append("| Backend | Surface |")
    lines.append("|---|---|")
    for row in rows:
        lines.append(f"| {_display_name(row['backend_id'])} | {_surface_name(row['backend_id'])} |")
    lines.append("")
    lines.append("## Key metrics")
    lines.append("")
    header = ["Backend", "Surface"]
    header.extend(
        f"{spec.label} {'↑' if spec.direction == 'up' else '↓'}"
        for spec in profile.metric_specs
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        cells = [_display_name(row["backend_id"]), _surface_name(row["backend_id"])]
        for spec in profile.metric_specs:
            value = _metric_value(row["data"], spec.scenario_id, spec.metric)
            rendered = _format_value(value, spec.fmt)
            best = best_values.get((spec.scenario_id, spec.metric))
            if value is not None and best is not None and abs(value - best) <= 1e-9:
                rendered = f"**{rendered}**"
            cells.append(rendered)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Read this run correctly")
    lines.append("")
    for note in profile.notes:
        lines.append(f"- {note}")
    lines.append(
        "- A strong result here means the backend did well inside this exact harness, "
        "with this exact profile and fixture set."
    )
    lines.append(
        "- When you cite numbers publicly, cite the profile name, date, exact "
        "command, and attach `runner_raw.json`."
    )
    lines.append(f"- Artifacts directory: `{output_dir}`")
    return "\n".join(lines) + "\n"


def build_summary(
    raw: dict[str, Any],
    *,
    profile: BenchProfile,
    backends: list[str] | None,
    runner_command: str,
) -> dict[str, Any]:
    rows = _backend_rows(raw, preferred_order=backends)
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        metric_map: dict[str, float | None] = {}
        for spec in profile.metric_specs:
            key = f"{spec.scenario_id}.{spec.metric}"
            metric_map[key] = _metric_value(row["data"], spec.scenario_id, spec.metric)
        summary_rows.append(
            {
                "backend": row["backend_id"],
                "display_name": _display_name(row["backend_id"]),
                "surface": _surface_name(row["backend_id"]),
                "language": row["language"],
                "metrics": metric_map,
            }
        )
    return {
        "profile": profile.name,
        "description": profile.description,
        "generated_at": datetime.now(UTC).isoformat(),
        "runner_command": runner_command,
        "rows": summary_rows,
    }


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = _resolve_profile(args.profile)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(profile.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_overrides = dict(profile.env)
    backends = _resolve_backends(
        profile,
        backends_override=args.backends,
        include_memvid=args.include_memvid,
    )
    runner_cmd = _build_runner_command(
        python_bin=args.python_bin,
        profile=profile,
        output_dir=output_dir,
        backends=backends,
    )
    shell_cmd = _shell_command(runner_cmd, env_overrides)

    raw_path = Path(args.from_raw) if args.from_raw else output_dir / _ARTIFACT_NAMES["raw"]

    if args.dry_run:
        print(shell_cmd)
        return 0

    if args.from_raw is None:
        full_env = os.environ.copy()
        full_env.update(env_overrides)
        subprocess.run(runner_cmd, check=True, env=full_env)
        _write_text(output_dir / _ARTIFACT_NAMES["command"], shell_cmd + "\n")

    raw = _load_raw(raw_path)
    scorecard = render_scorecard(
        raw,
        profile=profile,
        output_dir=output_dir,
        runner_command=shell_cmd,
        backends=backends,
        env_overrides=env_overrides,
    )
    summary = build_summary(raw, profile=profile, backends=backends, runner_command=shell_cmd)
    metadata = {
        "profile": profile.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "description": profile.description,
        "backends": backends,
        "scenarios": list(profile.scenarios),
        "env_overrides": env_overrides,
        "runner_command": runner_cmd,
        "shell_command": shell_cmd,
        "artifacts": {
            name: str(output_dir / filename) for name, filename in _ARTIFACT_NAMES.items()
        },
    }

    _write_text(output_dir / _ARTIFACT_NAMES["scorecard"], scorecard)
    _write_json(output_dir / _ARTIFACT_NAMES["summary"], summary)
    _write_json(output_dir / _ARTIFACT_NAMES["metadata"], metadata)

    print(f"Scorecard written to {output_dir / _ARTIFACT_NAMES['scorecard']}")
    print(f"Summary JSON written to {output_dir / _ARTIFACT_NAMES['summary']}")
    print(f"Metadata written to {output_dir / _ARTIFACT_NAMES['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
