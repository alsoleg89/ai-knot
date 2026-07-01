"""Tests for the competitor bench-pack wrapper script."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from tests.eval.benchmark.backends import qdrant_emulator as qdrant_module

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_competitor_bench_pack.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_competitor_bench_pack", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_offline_profile_builds_expected_runner_command() -> None:
    module = _load_module()
    profile = module.PROFILES["offline"]
    output_dir = Path("/tmp/bench-pack")
    backends = module._resolve_backends(
        profile,  # noqa: SLF001 - exercising script internals directly
        backends_override=None,
        include_memvid=False,
    )
    cmd = module._build_runner_command(  # noqa: SLF001
        python_bin="python",
        profile=profile,
        output_dir=output_dir,
        backends=backends,
    )

    assert "--mock-judge" in cmd
    assert "--skip-multi-agent" in cmd
    assert "s1_mrr,s5_noise,s6_token_economy,s9_scale,s16_update_correctness" in cmd
    assert "baseline,ai_knot_no_llm,qdrant,mem0" in cmd


def test_render_scorecard_includes_profile_artifacts_and_metrics() -> None:
    module = _load_module()
    profile = module.PROFILES["offline"]
    raw = {
        "schema_version": 2,
        "generated_at": "2026-07-01T00:00:00+00:00",
        "backends": {
            "baseline:en": {
                "language": "en",
                "s1_mrr": {"judge_scores": {"lexical_mrr": {"mean": 0.2, "stdev": 0.0}}},
                "s5_noise": {
                    "judge_scores": {"signal_recall_at3": {"mean": 0.6, "stdev": 0.0}}
                },
                "s6_token_economy": {
                    "judge_scores": {
                        "token_compression": {"mean": 0.67, "stdev": 0.0},
                        "quality_per_token": {"mean": 0.61, "stdev": 0.0},
                    }
                },
                "s9_scale": {
                    "judge_scores": {"mrr_degradation": {"mean": 0.0, "stdev": 0.0}}
                },
                "s16_update_correctness": {"judge_scores": {}},
            },
            "ai_knot_no_llm:en": {
                "language": "en",
                "s1_mrr": {"judge_scores": {"lexical_mrr": {"mean": 0.9, "stdev": 0.0}}},
                "s5_noise": {
                    "judge_scores": {"signal_recall_at3": {"mean": 0.8, "stdev": 0.0}}
                },
                "s6_token_economy": {
                    "judge_scores": {
                        "token_compression": {"mean": 0.72, "stdev": 0.0},
                        "quality_per_token": {"mean": 3.68, "stdev": 0.0},
                    }
                },
                "s9_scale": {
                    "judge_scores": {"mrr_degradation": {"mean": 0.37, "stdev": 0.0}}
                },
                "s16_update_correctness": {
                    "judge_scores": {"update_correctness": {"mean": 1.0, "stdev": 0.0}}
                },
            },
        },
    }

    rendered = module.render_scorecard(
        raw,
        profile=profile,
        output_dir=Path("benchmark_results/competitor_pack_offline"),
        runner_command="AI_KNOT_BENCH_DISABLE_EMBED=1 python -m tests.eval.benchmark.runner ...",
        backends=["baseline", "ai_knot_no_llm"],
        env_overrides={"AI_KNOT_BENCH_DISABLE_EMBED": "1"},
    )

    assert "Competitor Bench Pack — offline" in rendered
    assert "runner_raw.json" in rendered
    assert "Environment overrides" in rendered
    assert "naive log" in rendered
    assert "ai-knot (deterministic control)" in rendered
    assert "**0.90**" in rendered
    assert "**100.0%**" in rendered


def test_qdrant_emulator_can_be_forced_offline(monkeypatch) -> None:
    monkeypatch.setenv("AI_KNOT_BENCH_DISABLE_EMBED", "1")

    async def boom(text: str) -> list[float]:
        raise AssertionError(f"unexpected embedding call for: {text}")

    monkeypatch.setattr(qdrant_module, "embed_text", boom)

    backend = qdrant_module.QdrantEmulator()
    asyncio.run(backend.reset())
    asyncio.run(backend.insert("User prefers Python"))
    result = asyncio.run(backend.retrieve("python", top_k=1))

    assert result.texts == ["User prefers Python"]
