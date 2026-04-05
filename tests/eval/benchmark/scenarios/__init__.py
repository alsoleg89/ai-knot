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


def get_scenario_runners(
    names: list[str] | None = None,
) -> list[tuple[str, ScenarioFn]]:
    """Return list of (id, coroutine_fn) pairs, optionally filtered by name prefix."""
    if names is None:
        return list(_ALL)
    return [(sid, fn) for sid, fn in _ALL if any(sid.startswith(n) for n in names)]
