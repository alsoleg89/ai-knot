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
import sys
from datetime import UTC, datetime

import click

from tests.eval.benchmark._check_ollama import check_ollama_available
from tests.eval.benchmark.base import BenchmarkMetrics, MemoryBackend
from tests.eval.benchmark.report import render_markdown

_BACKEND_CHOICES = (
    "ai_knot",
    "ai_knot_no_llm",
    "qdrant",
    "mem0",
    "baseline",
    "qdrant_real",
    "mem0_real",
    "memvid",
)
_MA_BACKEND_CHOICES = ("ai_knot_multi_agent",)
_SCENARIO_CHOICES = ("s1", "s2", "s3", "s4", "s5", "s6", "s7")
_MA_SCENARIO_CHOICES = ("s8", "s9", "s10", "s11")
_MODE_CHOICES = ("basic", "extended", "auto")
_LANGUAGE_CHOICES = ("en", "ru", "both")


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
    help=f"Comma-separated scenario prefixes. Options: {', '.join(_SCENARIO_CHOICES)}, all",
)
@click.option(
    "--output",
    default="benchmark_report.md",
    show_default=True,
    help="Path for the Markdown report.",
)
@click.option(
    "--raw-output",
    default="benchmark_raw.json",
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
    help=(
        "Run multi-agent scenarios (S8–S11) instead of standard scenarios. "
        "Uses AiKnotMultiAgentBackend only."
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
) -> None:
    # --fast: mini fixtures + 2 backends. Real LLM still used (tests actual system).
    # Add --mock-judge explicitly if you need an offline/instant run (~20s, no LLM).
    if fast:
        from tests.eval.benchmark.fixtures import BUNDLE_EN_FAST

        if scenarios_arg == "all":
            scenarios_arg = "s2,s7"
        if backends_override is None:
            backends_override = "ai_knot,baseline"
        if output == "benchmark_report.md":
            output = "benchmark_fast.md"
        if raw_output == "benchmark_raw.json":
            raw_output = "benchmark_fast_raw.json"
        # Replace bundle resolution: always use BUNDLE_EN_FAST regardless of --language
        _fast_bundle = BUNDLE_EN_FAST
    else:
        _fast_bundle = None

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

    # --- Multi-agent path ---
    if multi_agent:
        await _run_multi_agent(
            scenarios_arg=scenarios_arg,
            output=output,
            raw_output=raw_output,
            judge=judge,
        )
        return

    # Resolve effective mode when backends are not explicitly overridden
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
                _run_backend(backend, bound_scenarios, judge, language=bundle.language)
                for backend in selected_backends
            ]
        )
        all_metrics.extend(lang_metrics)

    _write_reports(all_metrics, output, raw_output)


async def _run_backend(
    backend: MemoryBackend,
    scenarios: list[object],
    judge: object,
    *,
    language: str = "en",
) -> BenchmarkMetrics:
    click.echo(f"\n>>> Backend: {backend.name} [{language}]")
    metrics = BenchmarkMetrics(backend_name=backend.name, language=language)

    for sid, scenario_fn in scenarios:  # type: ignore[misc]
        click.echo(f"    {sid} ...", nl=False)
        try:
            result = await scenario_fn(backend, judge)
            metrics.scenario_results.append(result)
            click.echo(f" done  ({result.notes[:60]})" if result.notes else " done")
        except Exception as exc:
            click.echo(f" ERROR: {exc}")

    return metrics


async def _run_multi_agent(
    *,
    scenarios_arg: str,
    output: str,
    raw_output: str,
    judge: object,
) -> None:
    """Run multi-agent scenarios (S8–S11) against AiKnotMultiAgentBackend."""
    from tests.eval.benchmark.backends.ai_knot_multi_agent_backend import (
        AiKnotMultiAgentBackend,
    )
    from tests.eval.benchmark.base import BenchmarkMetrics
    from tests.eval.benchmark.scenarios import get_ma_scenario_runners

    backend = AiKnotMultiAgentBackend()

    if scenarios_arg in ("all", "ma"):
        runners = get_ma_scenario_runners()
    else:
        names = [s.strip() for s in scenarios_arg.split(",")]
        runners = get_ma_scenario_runners(names=names)
        if not runners:
            click.echo(
                f"No multi-agent scenarios matched {names!r}. Valid prefixes: s8, s9, s10, s11.",
                err=True,
            )
            sys.exit(1)

    click.echo(
        f"Running {len(runners)} multi-agent scenario(s) [{', '.join(sid for sid, _ in runners)}]"
    )
    click.echo(f"\n>>> Backend: {backend.name}")

    metrics = BenchmarkMetrics(backend_name=backend.name, language="en")
    for sid, scenario_fn in runners:
        click.echo(f"    {sid} ...", nl=False)
        try:
            result = await scenario_fn(backend, judge)
            metrics.scenario_results.append(result)
            click.echo(f" done  ({result.notes[:80]})" if result.notes else " done")
        except Exception as exc:
            click.echo(f" ERROR: {exc}")

    _write_reports([metrics], output, raw_output)


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
        AiKnotBackend(provider, use_add=mock_judge),  # type: ignore[arg-type]
    ]

    if mode == "extended":
        from tests.eval.benchmark.backends.mem0_real import Mem0RealBackend
        from tests.eval.benchmark.backends.qdrant_real import QdrantRealBackend

        return base + [QdrantRealBackend(), Mem0RealBackend()]

    # basic
    from tests.eval.benchmark.backends.mem0_emulator import Mem0Emulator
    from tests.eval.benchmark.backends.memvid_backend import MemvidBackend
    from tests.eval.benchmark.backends.qdrant_emulator import QdrantEmulator

    return base + [QdrantEmulator(), Mem0Emulator(provider), MemvidBackend()]  # type: ignore[arg-type]


def _build_backends_from_names(
    backends_arg: str, provider: object, *, mock_judge: bool
) -> list[MemoryBackend]:
    from tests.eval.benchmark.backends.ai_knot_backend import AiKnotBackend, AiKnotNoLlmBackend
    from tests.eval.benchmark.backends.baseline import BaselineBackend
    from tests.eval.benchmark.backends.mem0_emulator import Mem0Emulator
    from tests.eval.benchmark.backends.qdrant_emulator import QdrantEmulator

    all_map: dict[str, MemoryBackend] = {
        "baseline": BaselineBackend(),
        "ai_knot": AiKnotBackend(provider, use_add=mock_judge),  # type: ignore[arg-type]
        "ai_knot_no_llm": AiKnotNoLlmBackend(),
        "qdrant": QdrantEmulator(),
        "mem0": Mem0Emulator(provider),  # type: ignore[arg-type]
    }

    result: list[MemoryBackend] = []
    for name in backends_arg.split(","):
        name = name.strip()
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
    else:
        names = [s.strip() for s in scenarios_arg.split(",")]
        runners = get_scenario_runners(names=names)

    if not quick:
        return runners

    # Wrap S6 run() to inject quick=True
    wrapped = []
    for sid, fn in runners:  # type: ignore[misc]
        if sid == "s6_load":
            wrapped.append((sid, functools.partial(fn, quick=True)))
        else:
            wrapped.append((sid, fn))
    return wrapped


def _to_raw_json(metrics: list[BenchmarkMetrics]) -> dict[str, object]:
    out: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "backends": {},
    }
    for m in metrics:
        key = f"{m.backend_name}:{m.language}"
        backend_data: dict[str, object] = {"language": m.language}
        for sr in m.scenario_results:
            backend_data[sr.scenario_id] = {
                "judge_scores": sr.judge_scores,
                "notes": sr.notes,
                "insert": {
                    "facts_stored": sr.insert_result.facts_stored,
                    "facts_extracted": sr.insert_result.facts_extracted,
                    "insert_ms": sr.insert_result.insert_ms,
                }
                if sr.insert_result
                else None,
            }
        out["backends"][key] = backend_data  # type: ignore[index]
    return out


if __name__ == "__main__":
    main()
