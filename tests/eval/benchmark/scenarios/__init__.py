"""Benchmark scenario registry."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from tests.eval.benchmark.scenarios.s1_profile_retrieval import run as s1
from tests.eval.benchmark.scenarios.s2_avoid_repeats import run as s2
from tests.eval.benchmark.scenarios.s3_feedback_learning import run as s3
from tests.eval.benchmark.scenarios.s4_deduplication import run as s4
from tests.eval.benchmark.scenarios.s5_decay import run as s5
from tests.eval.benchmark.scenarios.s6_load import run as s6
from tests.eval.benchmark.scenarios.s7_consolidation import run as s7
from tests.eval.benchmark.scenarios.s8_ma_isolation import run as s8
from tests.eval.benchmark.scenarios.s9_ma_pool_publish import run as s9
from tests.eval.benchmark.scenarios.s10_ma_mesi_cas import run as s10
from tests.eval.benchmark.scenarios.s11_ma_mesi_sync import run as s11

ScenarioFn = Callable[..., Coroutine[Any, Any, Any]]

_ALL: list[tuple[str, ScenarioFn]] = [
    ("s1", s1),
    ("s2", s2),
    ("s3", s3),
    ("s4", s4),
    ("s5", s5),
    ("s6", s6),
    ("s7", s7),
]

# Multi-agent scenarios (S8–S11) require MultiAgentMemoryBackend.
# Run with: --scenarios ma  or  --multi-agent
_MA: list[tuple[str, ScenarioFn]] = [
    ("s8_ma_isolation", s8),
    ("s9_ma_pool_publish", s9),
    ("s10_ma_mesi_cas", s10),
    ("s11_ma_mesi_sync", s11),
]


def get_scenario_runners(
    names: list[str] | None = None,
) -> list[tuple[str, ScenarioFn]]:
    """Return list of (id, coroutine_fn) pairs, optionally filtered by name prefix."""
    if names is None:
        return list(_ALL)
    return [(sid, fn) for sid, fn in _ALL if any(sid.startswith(n) for n in names)]


def get_ma_scenario_runners(
    names: list[str] | None = None,
) -> list[tuple[str, ScenarioFn]]:
    """Return multi-agent scenario runners, optionally filtered by name prefix."""
    if names is None:
        return list(_MA)
    return [(sid, fn) for sid, fn in _MA if any(sid.startswith(n) for n in names)]
