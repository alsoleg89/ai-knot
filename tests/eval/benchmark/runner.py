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
import json
import sys
from datetime import UTC, datetime

import click

from tests.eval.benchmark._check_ollama import check_ollama_available
from tests.eval.benchmark.base import BenchmarkMetrics, MemoryBackend
from tests.eval.benchmark.report import render_markdown

_BACKEND_CHOICES = ("ai_knot", "qdrant", "mem0", "baseline", "qdrant_real", "mem0_real")
_SCENARIO_CHOICES = ("s1", "s2", "s3", "s4", "s5", "s6")
_MODE_CHOICES = ("basic", "extended", "auto")


@click.command()
@click.option(
    "--mode",
    default="auto",
    type=click.Choice(_MODE_CHOICES),
    show_default=True,
    help=(
        "basic=emulators only, extended=real services, "
        "auto=detect services and packages"
    ),
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
def main(
    mode: str,
    backends: str | None,
    scenarios: str,
    output: str,
    raw_output: str,
    mock_judge: bool,
    quick: bool,
) -> None:
    """Run the ai-knot benchmark suite."""
    asyncio.run(_run(mode, backends, scenarios, output, raw_output, mock_judge, quick))


async def _run(
    mode: str,
    backends_override: str | None,
    scenarios_arg: str,
    output: str,
    raw_output: str,
    mock_judge: bool,
    quick: bool,
) -> None:
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

    selected_scenarios = _build_scenarios(scenarios_arg, quick=quick)

    if not selected_backends:
        click.echo("No backends selected.", err=True)
        sys.exit(1)

    click.echo(
        f"Running {len(selected_scenarios)} scenario(s) × {len(selected_backends)} backend(s)"
        f" [{', '.join(b.name for b in selected_backends)}]"
    )

    # Run all backends in parallel — each uses an isolated store (temp dir / collection)
    all_metrics = await asyncio.gather(
        *[_run_backend(backend, selected_scenarios, judge) for backend in selected_backends]
    )

    md = render_markdown(list(all_metrics))
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)
    click.echo(f"\nReport written to {output}")

    raw = _to_raw_json(list(all_metrics))
    with open(raw_output, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    click.echo(f"Raw JSON written to {raw_output}")


async def _run_backend(
    backend: MemoryBackend,
    scenarios: list[object],
    judge: object,
) -> BenchmarkMetrics:
    click.echo(f"\n>>> Backend: {backend.name}")
    metrics = BenchmarkMetrics(backend_name=backend.name)

    for sid, scenario_fn in scenarios:  # type: ignore[misc]
        click.echo(f"    {sid} ...", nl=False)
        try:
            result = await scenario_fn(backend, judge)
            metrics.scenario_results.append(result)
            click.echo(f" done  ({result.notes[:60]})" if result.notes else " done")
        except Exception as exc:
            click.echo(f" ERROR: {exc}")

    return metrics


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
    from tests.eval.benchmark.backends.ai_knot_backend import AiKnotBackend
    from tests.eval.benchmark.backends.baseline import BaselineBackend

    base = [
        BaselineBackend(),
        AiKnotBackend(provider, use_add=mock_judge),  # type: ignore[arg-type]
    ]

    if mode == "extended":
        from tests.eval.benchmark.backends.mem0_real import Mem0RealBackend
        from tests.eval.benchmark.backends.qdrant_real import QdrantRealBackend

        return base + [QdrantRealBackend(), Mem0RealBackend()]

    # basic
    from tests.eval.benchmark.backends.mem0_emulator import Mem0Emulator
    from tests.eval.benchmark.backends.qdrant_emulator import QdrantEmulator

    return base + [QdrantEmulator(), Mem0Emulator(provider)]  # type: ignore[arg-type]


def _build_backends_from_names(
    backends_arg: str, provider: object, *, mock_judge: bool
) -> list[MemoryBackend]:
    from tests.eval.benchmark.backends.ai_knot_backend import AiKnotBackend
    from tests.eval.benchmark.backends.baseline import BaselineBackend
    from tests.eval.benchmark.backends.mem0_emulator import Mem0Emulator
    from tests.eval.benchmark.backends.qdrant_emulator import QdrantEmulator

    all_map: dict[str, MemoryBackend] = {
        "baseline": BaselineBackend(),
        "ai_knot": AiKnotBackend(provider, use_add=mock_judge),  # type: ignore[arg-type]
        "qdrant": QdrantEmulator(),
        "mem0": Mem0Emulator(provider),  # type: ignore[arg-type]
    }

    # Lazy-load real backends only if requested
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
        else:
            click.echo(f"Unknown backend {name!r}, skipping.", err=True)
    return result


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
            import functools

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
        backend_data: dict[str, object] = {}
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
        out["backends"][m.backend_name] = backend_data  # type: ignore[index]
    return out


if __name__ == "__main__":
    main()
