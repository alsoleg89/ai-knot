"""Benchmark scenario registry.

_ALL  — professional S1–S9 scenarios (community-recognized metrics, no LLM judge).
_MA   — multi-agent S8–S11 scenarios (require MultiAgentMemoryBackend).
_LEGACY — original S1–S7 scenarios (kept for backward compatibility, not in default run).
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

# Professional S1–S9 (new default)
from tests.eval.benchmark.scenarios.s1_mrr import run as s1_mrr

# Legacy scenarios (not in default run, still importable)
from tests.eval.benchmark.scenarios.s1_profile_retrieval import run as _s1_legacy
from tests.eval.benchmark.scenarios.s2_avoid_repeats import run as _s2_legacy
from tests.eval.benchmark.scenarios.s2_semantic_gap import run as s2_semantic_gap
from tests.eval.benchmark.scenarios.s3_feedback_learning import run as _s3_legacy
from tests.eval.benchmark.scenarios.s3_staleness import run as s3_staleness
from tests.eval.benchmark.scenarios.s4_compression_f1 import run as s4_compression_f1
from tests.eval.benchmark.scenarios.s4_deduplication import run as _s4_legacy
from tests.eval.benchmark.scenarios.s5_decay import run as _s5_legacy
from tests.eval.benchmark.scenarios.s5_noise import run as s5_noise
from tests.eval.benchmark.scenarios.s6_load import run as _s6_legacy
from tests.eval.benchmark.scenarios.s6_token_economy import run as s6_token_economy
from tests.eval.benchmark.scenarios.s7_consolidation import run as _s7_legacy
from tests.eval.benchmark.scenarios.s7_grounding import run as s7_grounding

# Multi-agent scenarios
from tests.eval.benchmark.scenarios.s8_ma_isolation import run as s8_ma
from tests.eval.benchmark.scenarios.s8_throughput import run as s8_throughput
from tests.eval.benchmark.scenarios.s9_ma_pool_publish import run as s9_ma
from tests.eval.benchmark.scenarios.s9_scale import run as s9_scale
from tests.eval.benchmark.scenarios.s10_ma_mesi_cas import run as s10_ma
from tests.eval.benchmark.scenarios.s11_ma_mesi_sync import run as s11_ma
from tests.eval.benchmark.scenarios.s12_topic_gating import run as s12_topic_gating
from tests.eval.benchmark.scenarios.s13_concurrent_writers import run as s13_concurrent_writers
from tests.eval.benchmark.scenarios.s14_trust_drift import run as s14_trust_drift
from tests.eval.benchmark.scenarios.s15_topic_leakage import run as s15_topic_leakage

ScenarioFn = Callable[..., Coroutine[Any, Any, Any]]

_ALL: list[tuple[str, ScenarioFn]] = [
    ("s1_mrr", s1_mrr),
    ("s2_semantic_gap", s2_semantic_gap),
    ("s3_staleness", s3_staleness),
    ("s4_compression_f1", s4_compression_f1),
    ("s5_noise", s5_noise),
    ("s6_token_economy", s6_token_economy),
    ("s7_grounding", s7_grounding),
    ("s8_throughput", s8_throughput),
    ("s9_scale", s9_scale),
]

_LEGACY: list[tuple[str, ScenarioFn]] = [
    ("s1_profile_retrieval", _s1_legacy),
    ("s2_avoid_repeats", _s2_legacy),
    ("s3_feedback_learning", _s3_legacy),
    ("s4_deduplication", _s4_legacy),
    ("s5_decay", _s5_legacy),
    ("s6_load", _s6_legacy),
    ("s7_consolidation", _s7_legacy),
]

# Multi-agent scenarios (S8–S15) require MultiAgentMemoryBackend.
# Run with: --multi-agent
_MA: list[tuple[str, ScenarioFn]] = [
    ("s8_ma_isolation", s8_ma),
    ("s9_ma_pool_publish", s9_ma),
    ("s10_ma_mesi_cas", s10_ma),
    ("s11_ma_mesi_sync", s11_ma),
    ("s12_topic_gating", s12_topic_gating),
    ("s13_concurrent_writers", s13_concurrent_writers),
    ("s14_trust_drift", s14_trust_drift),
    ("s15_topic_leakage", s15_topic_leakage),
]


def get_scenario_runners(
    names: list[str] | None = None,
    *,
    legacy: bool = False,
) -> list[tuple[str, ScenarioFn]]:
    """Return list of (id, coroutine_fn) pairs, optionally filtered by name prefix.

    Args:
        names: Optional list of name prefixes to filter (e.g. ["s1", "s3"]).
        legacy: If True, search _LEGACY instead of _ALL.
    """
    pool = _LEGACY if legacy else _ALL
    if names is None:
        return list(pool)
    return [(sid, fn) for sid, fn in pool if any(sid.startswith(n) for n in names)]


def get_ma_scenario_runners(
    names: list[str] | None = None,
) -> list[tuple[str, ScenarioFn]]:
    """Return multi-agent scenario runners, optionally filtered by name prefix."""
    if names is None:
        return list(_MA)
    return [(sid, fn) for sid, fn in _MA if any(sid.startswith(n) for n in names)]
