"""Benchmark runner CLI.

Usage (from the ai-knot project root):

    # Auto-detect mode: extended if Qdrant + mem0ai installed, else basic
    python -m tests.eval.benchmark.runner

    # Explicit mode
    python -m tests.eval.benchmark.runner --mode basic
    python -m tests.eval.benchmark.runner --mode extended

    # Specific backends and scenarios
    python -m tests.eval.benchmark.runner --backends ai_knot,qdrant --scenarios s1,s4

    # Offline run (no Ollama required)
    python -m tests.eval.benchmark.runner --mock-judge

    # CI / low-resource (reduces S6 concurrency from 50 → 20)
    python -m tests.eval.benchmark.runner --quick

Modes:
  basic    — baseline + ai_knot + qdrant_emulator + mem0_emulator (no extra deps)
  extended — baseline + ai_knot + qdrant_real + mem0_real (needs services + packages)
  auto     — detect: if Qdrant running + qdrant-client + mem0ai installed → extended

Backends run in parallel (asyncio.gather) since they use separate stores/temp dirs.
"""

from __future__ import annotations

import asyncio
import functools
import json
import statistics
import sys
import time
from datetime import UTC, datetime

import click

from tests.eval.benchmark._check_ollama import check_ollama_available
from tests.eval.benchmark.base import (
    BenchmarkMetrics,
    LongRunStats,
    MemoryBackend,
    MultiAgentMemoryBackend,
    ScenarioResult,
)
from tests.eval.benchmark.report import render_markdown

_BACKEND_CHOICES = (
    "ai_knot",
    "ai_knot_no_llm",
    "qdrant",
    "qdrant_extraction",
    "mem0",
    "baseline",
    "qdrant_real",
    "mem0_real",
    "memvid",
)
_MA_BACKEND_CHOICES = ("ai_knot_multi_agent",)
_SCENARIO_CHOICES = (
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    # legacy (prefix-selectable)
    "s1_profile_retrieval",
    "s2_avoid_repeats",
    "s3_feedback_learning",
    "s4_deduplication",
    "s5_decay",
    "s6_load",
    "s7_consolidation",
)
_MA_SCENARIO_CHOICES = ("s8_ma", "s9", "s10", "s11", "s12", "s13", "s14", "s15")
_MODE_CHOICES = ("basic", "extended", "auto")
_LANGUAGE_CHOICES = ("en", "ru", "both")

# Default output filenames — used to detect whether the user overrode them.
_DEFAULT_OUTPUT = "benchmark_report.md"
_DEFAULT_RAW = "benchmark_raw.json"
_DEFAULT_JSONL = "benchmark_live.jsonl"


def _stamp_path(path: str, stamp: str) -> str:
    """Insert ``_<stamp>`` before the file extension.

    >>> _stamp_path("benchmark_report.md", "20260408_153012")
    'benchmark_report_20260408_153012.md'
    """
    dot = path.rfind(".")
    if dot == -1:
        return f"{path}_{stamp}"
    return f"{path[:dot]}_{stamp}{path[dot:]}"


@click.command()
@click.option(
    "--mode",
    default="auto",
    type=click.Choice(_MODE_CHOICES),
    show_default=True,
    help=("basic=emulators only, extended=real services, auto=detect services and packages"),
)
@click.option(
    "--backends",
    default=None,
    help=(
        f"Override: comma-separated backends. Options: {', '.join(_BACKEND_CHOICES)}. "
        "Overrides --mode."
    ),
)
@click.option(
    "--scenarios",
    default="all",
    help=(
        f"Comma-separated scenario prefixes. Options: {', '.join(_SCENARIO_CHOICES[:9])}, all. "
        "Use '--scenarios legacy' to run the original S1–S7 scenarios."
    ),
)
@click.option(
    "--output",
    default=_DEFAULT_OUTPUT,
    show_default=True,
    help="Path for the Markdown report.",
)
@click.option(
    "--raw-output",
    default=_DEFAULT_RAW,
    show_default=True,
    help="Path for raw JSON results.",
)
@click.option(
    "--mock-judge",
    is_flag=True,
    default=False,
    help="Use MockJudge (deterministic, no Ollama). Also disables LLM extraction.",
)
@click.option(
    "--quick",
    is_flag=True,
    default=False,
    help="Reduce S6 concurrent tasks from 50 → 20. Useful for CI / low-resource envs.",
)
@click.option(
    "--language",
    default="en",
    type=click.Choice(_LANGUAGE_CHOICES),
    show_default=True,
    help=(
        "Fixture language: 'en' (Alex Chen / FinServe Capital), "
        "'ru' (Максим Петров / Яндекс), or 'both' to run both sequentially."
    ),
)
@click.option(
    "--fast",
    is_flag=True,
    default=False,
    help=(
        "Fast dev mode: mini fixtures (S2×15, S7×9), "
        "2 backends (ai_knot+baseline), mock judge. Target ≤5 min."
    ),
)
@click.option(
    "--multi-agent",
    "multi_agent",
    is_flag=True,
    default=False,
    help=("Run ONLY multi-agent scenarios (S8–S11). Uses AiKnotMultiAgentBackend only."),
)
@click.option(
    "--skip-multi-agent",
    "skip_multi_agent",
    is_flag=True,
    default=False,
    help=(
        "Skip multi-agent scenarios (S8-MA, S9, S10, S11) that are "
        "normally appended to a standard run."
    ),
)
@click.option(
    "--long-run",
    "long_run",
    is_flag=True,
    default=False,
    help=(
        "Run each MA scenario in a timed loop for --duration seconds. "
        "Reports aggregate metrics across all iterations."
    ),
)
@click.option(
    "--duration",
    default=60,
    show_default=True,
    type=int,
    help="Seconds per scenario in --long-run mode.",
)
@click.option(
    "--ma-storage",
    "ma_storage",
    default="sqlite",
    show_default=True,
    help=(
        "Comma-separated storage backends for multi-agent scenarios. "
        "Options: sqlite, yaml, postgres. "
        "Example: --ma-storage sqlite,yaml,postgres"
    ),
)
@click.option(
    "--ma-postgres-dsn",
    "ma_postgres_dsn",
    default="",
    help=(
        "PostgreSQL DSN for multi-agent postgres storage. "
        "Overrides AI_KNOT_DSN env var. "
        "Example: postgresql://user:pass@localhost:5432/bench"
    ),
)
@click.option(
    "--jsonl-output",
    "jsonl_output",
    default=_DEFAULT_JSONL,
    show_default=True,
    help=(
        "Path for incremental JSONL output. Each scenario result is appended "
        "as a single JSON line immediately after completion, so you can "
        "analyze results while the benchmark is still running."
    ),
)
@click.option(
    "--ma-category",
    "ma_category",
    default="all",
    type=click.Choice(["all", "protocol", "retrieval"]),
    show_default=True,
    help=(
        "Filter multi-agent scenarios by category: "
        "'protocol' (CAS, sync, concurrency), "
        "'retrieval' (ranking, trust, assembly), or 'all'."
    ),
)
def main(
    mode: str,
    backends: str | None,
    scenarios: str,
    output: str,
    raw_output: str,
    mock_judge: bool,
    quick: bool,
    language: str,
    fast: bool,
    multi_agent: bool,
    skip_multi_agent: bool,
    long_run: bool,
    duration: int,
    ma_storage: str,
    ma_postgres_dsn: str,
    jsonl_output: str,
    ma_category: str,
) -> None:
    """Run the ai-knot benchmark suite."""
    asyncio.run(
        _run(
            mode,
            backends,
            scenarios,
            output,
            raw_output,
            mock_judge,
            quick,
            language,
            fast,
            multi_agent=multi_agent,
            skip_multi_agent=skip_multi_agent,
            long_run=long_run,
            duration=duration,
            ma_storage=ma_storage,
            ma_postgres_dsn=ma_postgres_dsn,
            jsonl_output=jsonl_output,
            ma_category=ma_category,
        )
    )


async def _run(
    mode: str,
    backends_override: str | None,
    scenarios_arg: str,
    output: str,
    raw_output: str,
    mock_judge: bool,
    quick: bool,
    language: str = "en",
    fast: bool = False,
    multi_agent: bool = False,
    skip_multi_agent: bool = False,
    long_run: bool = False,
    duration: int = 60,
    ma_storage: str = "sqlite",
    ma_postgres_dsn: str = "",
    jsonl_output: str = "benchmark_live.jsonl",
    ma_category: str = "all",
) -> None:
    # Stamp default filenames with current datetime so runs don't overwrite each other.
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if output == _DEFAULT_OUTPUT:
        output = _stamp_path(output, stamp)
    if raw_output == _DEFAULT_RAW:
        raw_output = _stamp_path(raw_output, stamp)
    if jsonl_output == _DEFAULT_JSONL:
        jsonl_output = _stamp_path(jsonl_output, stamp)

    # --fast: mini fixtures + 2 backends. Real LLM still used (tests actual system).
    # Add --mock-judge explicitly if you need an offline/instant run (~20s, no LLM).
    if fast:
        if scenarios_arg == "all":
            scenarios_arg = "s1,s4,s7"
        if backends_override is None:
            backends_override = "ai_knot,baseline,qdrant"
        if output == _stamp_path(_DEFAULT_OUTPUT, stamp):
            output = _stamp_path("benchmark_fast.md", stamp)
        if raw_output == _stamp_path(_DEFAULT_RAW, stamp):
            raw_output = _stamp_path("benchmark_fast_raw.json", stamp)

    _fast_bundle = None  # professional scenarios use standalone fixtures, not LanguageBundle

    if not mock_judge and not check_ollama_available():
        click.echo(
            "ERROR: Ollama not running at http://localhost:11434. "
            "Start Ollama or use --mock-judge for an offline run.",
            err=True,
        )
        sys.exit(1)

    from tests.eval.benchmark.judge import MockJudge, OllamaJudge

    judge = MockJudge() if mock_judge else OllamaJudge()

    if mock_judge:
        from tests.eval.benchmark._stub_provider import StubProvider

        provider: object = StubProvider()
    else:
        from ai_knot.providers.ollama import OllamaProvider

        provider = OllamaProvider()

    # Clear JSONL file at start of run.
    with open(jsonl_output, "w", encoding="utf-8") as f:
        f.write("")
    click.echo(f"Live JSONL output → {jsonl_output}")

    # --- Multi-agent path ---
    if multi_agent:
        await _run_multi_agent(
            scenarios_arg=scenarios_arg,
            output=output,
            raw_output=raw_output,
            judge=judge,
            long_run=long_run,
            duration=duration,
            ma_storage=ma_storage,
            ma_postgres_dsn=ma_postgres_dsn,
            jsonl_output=jsonl_output,
            ma_category=ma_category,
        )
        return

    # Resolve effective mode when backends are not explicitly overridden
    effective_mode = "basic"
    if backends_override is None:
        effective_mode = _resolve_mode(mode)
        selected_backends = _build_backends_for_mode(
            effective_mode, provider, mock_judge=mock_judge
        )
        click.echo(f"Mode: {effective_mode}")
    else:
        selected_backends = _build_backends_from_names(
            backends_override, provider, mock_judge=mock_judge
        )

    base_scenarios = _build_scenarios(scenarios_arg, quick=quick)
    bundles = [_fast_bundle] if _fast_bundle is not None else _build_bundles(language)

    if not selected_backends:
        click.echo("No backends selected.", err=True)
        sys.exit(1)

    click.echo(
        f"Running {len(base_scenarios)} scenario(s) × {len(selected_backends)} backend(s)"
        f" × {len(bundles)} language(s)"
        f" [{', '.join(b.name for b in selected_backends)}]"
    )

    all_metrics: list[BenchmarkMetrics] = []
    for bundle in bundles:
        # Bind bundle to every scenario that accepts it; S6 has no bundle parameter.
        bound_scenarios = _bind_bundle(base_scenarios, bundle)
        click.echo(f"\n=== Language: {bundle.language.upper()} ===")
        lang_metrics = await asyncio.gather(
            *[
                _run_backend(
                    backend,
                    bound_scenarios,
                    judge,
                    language=bundle.language,
                    jsonl_output=jsonl_output,
                )
                for backend in selected_backends
            ]
        )
        all_metrics.extend(lang_metrics)

    # Also run multi-agent scenarios unless skipped.
    # mem0_multi_agent is only added when Ollama is confirmed available
    # (effective_mode=="extended" or explicit override containing real backends).
    if not skip_multi_agent:
        extra_ma: list[MultiAgentMemoryBackend] = []
        if effective_mode == "extended" or (backends_override is not None and not mock_judge):
            from tests.eval.benchmark.backends.mem0_ma_backend import Mem0MultiAgentBackend

            extra_ma = [Mem0MultiAgentBackend()]
        ma_metrics_list = await _run_multi_agent_inline(
            judge=judge,
            extra_ma_backends=extra_ma,
            ma_storage=ma_storage,
            ma_postgres_dsn=ma_postgres_dsn,
            jsonl_output=jsonl_output,
            ma_category=ma_category,
        )
        all_metrics.extend(ma_metrics_list)

    _write_reports(all_metrics, output, raw_output)


async def _run_backend(
    backend: MemoryBackend,
    scenarios: list[object],
    judge: object,
    *,
    language: str = "en",
    jsonl_output: str = "",
) -> BenchmarkMetrics:
    click.echo(f"\n>>> Backend: {backend.name} [{language}]")
    metrics = BenchmarkMetrics(backend_name=backend.name, language=language)

    for sid, scenario_fn in scenarios:  # type: ignore[misc]
        click.echo(f"    {sid} ...", nl=False)
        try:
            result = await scenario_fn(backend, judge)
            metrics.scenario_results.append(result)
            click.echo(f" done  ({result.notes[:60]})" if result.notes else " done")
            if jsonl_output:
                _append_jsonl(jsonl_output, result, backend.name, language=language)
        except Exception as exc:
            click.echo(f" ERROR: {exc}")

    return metrics


async def _run_timed_scenario(
    backend: MultiAgentMemoryBackend,
    scenario_fn: object,
    sid: str,
    judge: object,
    duration_s: int,
) -> ScenarioResult:
    """Loop a scenario for *duration_s* seconds, return aggregate ScenarioResult."""
    deadline = time.monotonic() + duration_s
    iteration = 0
    all_scores: dict[str, list[float]] = {}
    iter_times: list[float] = []
    last_notes = ""
    next_print = time.monotonic() + 10.0

    while time.monotonic() < deadline:
        t0 = time.monotonic()
        result: ScenarioResult = await scenario_fn(backend, judge)  # type: ignore[call-arg]
        elapsed = time.monotonic() - t0
        iter_times.append(elapsed)
        iteration += 1
        last_notes = result.notes or ""
        for k, v in result.judge_scores.items():
            all_scores.setdefault(k, []).extend(v)

        now = time.monotonic()
        if now >= next_print:
            remaining = max(0.0, deadline - now)
            click.echo(
                f"\r      [{sid}] iter={iteration}  "
                f"avg={sum(iter_times) / iteration:.3f}s/iter  "
                f"remaining={remaining:.0f}s      ",
                nl=False,
            )
            next_print = now + 10.0

    total_s = sum(iter_times)
    avg_s = total_s / max(iteration, 1)

    # Build aggregate scores: mean per metric + stdev annotation in notes.
    agg_scores: dict[str, list[float]] = {}
    metric_stdev: dict[str, float] = {}
    metric_summary_parts: list[str] = []
    for k, vals in all_scores.items():
        mean = sum(vals) / len(vals)
        agg_scores[k] = [mean]
        if len(vals) > 1:
            std = statistics.stdev(vals)
            metric_stdev[k] = std
            metric_summary_parts.append(f"{k}={mean:.3f}±{std:.3f}")
        else:
            metric_summary_parts.append(f"{k}={mean:.3f}")

    notes = (
        f"iters={iteration}, total={total_s:.1f}s, avg={avg_s:.3f}s/iter"
        + (f" | {', '.join(metric_summary_parts)}" if metric_summary_parts else "")
        + (f" | last: {last_notes}" if last_notes else "")
    )

    return ScenarioResult(
        scenario_id=sid,
        backend_name=backend.name,
        judge_scores=agg_scores,
        insert_result=None,
        retrieval_result=None,
        notes=notes,
        long_run_stats=LongRunStats(
            iterations=iteration,
            wall_time_s=total_s,
            avg_iter_s=avg_s,
            metric_stdev=metric_stdev,
        ),
    )


async def _run_multi_agent_inline(
    *,
    judge: object,
    scenarios_arg: str = "all",
    extra_ma_backends: list[MultiAgentMemoryBackend] | None = None,
    long_run: bool = False,
    duration: int = 60,
    ma_storage: str = "sqlite",
    ma_postgres_dsn: str = "",
    jsonl_output: str = "",
    ma_category: str = "all",
) -> list[BenchmarkMetrics]:
    """Run multi-agent scenarios and return metrics per backend.

    Used both by the inline path (appended to standard runs) and by
    ``_run_multi_agent()`` (standalone ``--multi-agent`` mode).
    Returns empty list when no runners match.
    """
    from tests.eval.benchmark.backends.ai_knot_multi_agent_backend import (
        AiKnotMultiAgentBackend,
    )
    from tests.eval.benchmark.scenarios import get_ma_scenario_runners

    if scenarios_arg in ("all", "ma"):
        runners = get_ma_scenario_runners(category=ma_category)
    else:
        names = [s.strip() for s in scenarios_arg.split(",")]
        runners = get_ma_scenario_runners(names=names, category=ma_category)
        if not runners:
            return []

    storage_types = [s.strip() for s in ma_storage.split(",") if s.strip()]
    backends: list[MultiAgentMemoryBackend] = [
        AiKnotMultiAgentBackend(st, postgres_dsn=ma_postgres_dsn) for st in storage_types
    ]
    backends.extend(extra_ma_backends or [])

    mode_label = f"long-run {duration}s/scenario" if long_run else "single-pass"
    click.echo(
        f"\nRunning {len(runners)} multi-agent scenario(s) [{mode_label}]"
        f" [{', '.join(sid for sid, _ in runners)}]"
    )

    all_ma_metrics: list[BenchmarkMetrics] = []
    for backend in backends:
        click.echo(f"\n>>> Backend: {backend.name}")
        metrics = BenchmarkMetrics(backend_name=backend.name, language="en")
        for sid, scenario_fn in runners:
            click.echo(f"    {sid} ...", nl=False)
            try:
                if long_run:
                    click.echo(f" running for {duration}s ...")
                    result = await _run_timed_scenario(backend, scenario_fn, sid, judge, duration)
                    click.echo(f"\r    {sid} done  ({result.notes[:100]})")
                else:
                    result = await scenario_fn(backend, judge)  # type: ignore[call-arg]
                    click.echo(f" done  ({result.notes[:80]})" if result.notes else " done")
                metrics.scenario_results.append(result)
                if jsonl_output:
                    _append_jsonl(jsonl_output, result, backend.name, language="en")
            except Exception as exc:
                click.echo(f" ERROR: {exc}")
        all_ma_metrics.append(metrics)

    return all_ma_metrics


async def _run_multi_agent(
    *,
    scenarios_arg: str,
    output: str,
    raw_output: str,
    judge: object,
    long_run: bool = False,
    duration: int = 60,
    ma_storage: str = "sqlite",
    ma_postgres_dsn: str = "",
    jsonl_output: str = "",
    ma_category: str = "all",
) -> None:
    """Run multi-agent scenarios standalone (``--multi-agent`` flag)."""
    metrics_list = await _run_multi_agent_inline(
        judge=judge,
        scenarios_arg=scenarios_arg,
        long_run=long_run,
        duration=duration,
        ma_storage=ma_storage,
        ma_postgres_dsn=ma_postgres_dsn,
        jsonl_output=jsonl_output,
        ma_category=ma_category,
    )
    if not metrics_list:
        click.echo(
            f"No multi-agent scenarios matched {scenarios_arg!r}. "
            "Valid prefixes: s8, s9, s10, s11, s12.",
            err=True,
        )
        sys.exit(1)

    _write_reports(metrics_list, output, raw_output)


def _append_jsonl(
    path: str, result: ScenarioResult, backend_name: str, language: str = "en"
) -> None:
    """Append a single scenario result as one JSON line."""
    record: dict[str, object] = {
        "ts": datetime.now(UTC).isoformat(),
        "backend": backend_name,
        "language": language,
        "scenario": result.scenario_id,
        "scores": {
            k: v[0] if isinstance(v, list) and len(v) == 1 else v
            for k, v in result.judge_scores.items()
        },
        "notes": result.notes or "",
    }
    if result.long_run_stats:
        record["long_run"] = {
            "iterations": result.long_run_stats.iterations,
            "wall_time_s": result.long_run_stats.wall_time_s,
            "avg_iter_s": result.long_run_stats.avg_iter_s,
            "metric_stdev": result.long_run_stats.metric_stdev,
        }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_reports(metrics: list[BenchmarkMetrics], output: str, raw_output: str) -> None:
    md = render_markdown(metrics)
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)
    click.echo(f"\nReport written to {output}")

    raw = _to_raw_json(metrics)
    with open(raw_output, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    click.echo(f"Raw JSON written to {raw_output}")


def _resolve_mode(mode: str) -> str:
    """Resolve 'auto' to 'basic' or 'extended' based on what's available."""
    if mode != "auto":
        return mode

    from tests.eval.benchmark._check_services import (
        check_mem0_available,
        check_qdrant_available,
        check_qdrant_client_installed,
    )

    if check_qdrant_available() and check_qdrant_client_installed() and check_mem0_available():
        return "extended"
    return "basic"


def _build_backends_for_mode(
    mode: str, provider: object, *, mock_judge: bool
) -> list[MemoryBackend]:
    from tests.eval.benchmark.backends.ai_knot_backend import AiKnotBackend, AiKnotNoLlmBackend
    from tests.eval.benchmark.backends.baseline import BaselineBackend

    base: list[MemoryBackend] = [
        BaselineBackend(),
        AiKnotNoLlmBackend(),
    ]
    if not mock_judge:
        base.append(AiKnotBackend(provider, use_add=False))  # type: ignore[arg-type]

    if mode == "extended":
        from tests.eval.benchmark.backends.mem0_real import Mem0RealBackend
        from tests.eval.benchmark.backends.qdrant_real import QdrantRealBackend

        return base + [QdrantRealBackend(), Mem0RealBackend()]

    # basic
    from tests.eval.benchmark.backends.mem0_emulator import Mem0Emulator
    from tests.eval.benchmark.backends.memvid_backend import MemvidBackend
    from tests.eval.benchmark.backends.qdrant_emulator import QdrantEmulator
    from tests.eval.benchmark.backends.qdrant_extraction_backend import QdrantWithExtractionBackend

    return base + [
        QdrantEmulator(),
        QdrantWithExtractionBackend(provider),
        Mem0Emulator(provider),
        MemvidBackend(),
    ]  # type: ignore[arg-type]


def _build_backends_from_names(
    backends_arg: str, provider: object, *, mock_judge: bool
) -> list[MemoryBackend]:
    from tests.eval.benchmark.backends.ai_knot_backend import AiKnotBackend, AiKnotNoLlmBackend
    from tests.eval.benchmark.backends.baseline import BaselineBackend
    from tests.eval.benchmark.backends.mem0_emulator import Mem0Emulator
    from tests.eval.benchmark.backends.qdrant_emulator import QdrantEmulator
    from tests.eval.benchmark.backends.qdrant_extraction_backend import QdrantWithExtractionBackend

    all_map: dict[str, MemoryBackend] = {
        "baseline": BaselineBackend(),
        "ai_knot": AiKnotBackend(provider, use_add=mock_judge),  # type: ignore[arg-type]
        "ai_knot_no_llm": AiKnotNoLlmBackend(),
        "qdrant": QdrantEmulator(),
        "qdrant_extraction": QdrantWithExtractionBackend(provider),  # type: ignore[arg-type]
        "mem0": Mem0Emulator(provider),  # type: ignore[arg-type]
    }

    result: list[MemoryBackend] = []
    for name in backends_arg.split(","):
        name = name.strip()
        if name == "ai_knot" and mock_judge:
            click.echo(
                "WARNING: ai_knot with --mock-judge uses kb.add() (identical to ai_knot_no_llm). "
                "Including it anyway since explicitly requested.",
                err=True,
            )
        if name in all_map:
            result.append(all_map[name])
        elif name == "qdrant_real":
            from tests.eval.benchmark.backends.qdrant_real import QdrantRealBackend

            result.append(QdrantRealBackend())
        elif name == "mem0_real":
            from tests.eval.benchmark.backends.mem0_real import Mem0RealBackend

            result.append(Mem0RealBackend())
        elif name == "memvid":
            from tests.eval.benchmark.backends.memvid_backend import MemvidBackend

            result.append(MemvidBackend())
        else:
            click.echo(f"Unknown backend {name!r}, skipping.", err=True)
    return result


def _build_bundles(language: str) -> list[object]:
    """Return list of LanguageBundle instances for the given language flag."""
    from tests.eval.benchmark.fixtures import BUNDLE_EN, BUNDLE_RU

    if language == "en":
        return [BUNDLE_EN]
    if language == "ru":
        return [BUNDLE_RU]
    return [BUNDLE_EN, BUNDLE_RU]  # "both"


def _bind_bundle(scenarios: list[object], bundle: object) -> list[object]:
    """Return scenarios with bundle bound via functools.partial.

    Scenarios that don't accept a ``bundle`` keyword (e.g. S6 load) are
    passed through unchanged; others get the bundle pre-filled.
    """
    import inspect

    bound: list[object] = []
    for sid, fn in scenarios:  # type: ignore[misc]
        sig = inspect.signature(fn)
        if "bundle" in sig.parameters:
            bound.append((sid, functools.partial(fn, bundle=bundle)))
        else:
            bound.append((sid, fn))
    return bound


def _build_scenarios(scenarios_arg: str, *, quick: bool = False) -> list[object]:
    from tests.eval.benchmark.scenarios import get_scenario_runners

    if scenarios_arg == "all":
        runners = get_scenario_runners()
    elif scenarios_arg == "legacy":
        runners = get_scenario_runners(legacy=True)
    else:
        names = [s.strip() for s in scenarios_arg.split(",")]
        # Try professional scenarios first, fall back to legacy if needed
        runners = get_scenario_runners(names=names)
        if not runners:
            runners = get_scenario_runners(names=names, legacy=True)

    if not quick:
        return runners

    # Wrap legacy S6 run() to inject quick=True
    wrapped = []
    for sid, fn in runners:  # type: ignore[misc]
        if sid == "s6_load":
            wrapped.append((sid, functools.partial(fn, quick=True)))
        else:
            wrapped.append((sid, fn))
    return wrapped


def _to_raw_json(metrics: list[BenchmarkMetrics]) -> dict[str, object]:
    out: dict[str, object] = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "backends": {},
    }
    for m in metrics:
        key = f"{m.backend_name}:{m.language}"
        backend_data: dict[str, object] = {"language": m.language}
        for sr in m.scenario_results:
            # Unified schema (v2): always {"mean": X, "stdev": Y} per metric.
            enriched_scores: dict[str, object] = {}
            for k, v in sr.judge_scores.items():
                mean_val = sum(v) / len(v) if v else 0.0
                stdev_val = (
                    sr.long_run_stats.metric_stdev.get(k, 0.0)
                    if sr.long_run_stats
                    else (statistics.stdev(v) if len(v) > 1 else 0.0)
                )
                enriched_scores[k] = {"mean": mean_val, "stdev": stdev_val}

            scenario_data: dict[str, object] = {
                "judge_scores": enriched_scores,
                "notes": sr.notes,
                "insert": {
                    "facts_stored": sr.insert_result.facts_stored,
                    "facts_extracted": sr.insert_result.facts_extracted,
                    "insert_ms": sr.insert_result.insert_ms,
                }
                if sr.insert_result
                else None,
            }
            if sr.long_run_stats:
                scenario_data["long_run"] = {
                    "iterations": sr.long_run_stats.iterations,
                    "wall_time_s": sr.long_run_stats.wall_time_s,
                    "avg_iter_s": sr.long_run_stats.avg_iter_s,
                }
            backend_data[sr.scenario_id] = scenario_data
        out["backends"][key] = backend_data  # type: ignore[index]
    return out


if __name__ == "__main__":
    main()
