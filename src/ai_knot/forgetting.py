"""Ebbinghaus forgetting curve implementation.

Core formula:
    retention(t) = (1 + t / (c * S))^(-decay_exponent)
    S = BASE_STABILITY_HOURS * importance * type_mult * count_factor * spacing_factor

FSRS-inspired spacing effect (Ye, 2022-2024): well-spaced accesses
give stronger reinforcement than cramped ones.

Type-aware decay (Tulving, 1972): episodic memory fades fastest,
semantic slowest.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from ai_knot.types import Fact

# Base stability in hours (2 weeks). A fact with importance=1.0 and
# access_count=0 will retain ~37% after 2 weeks.
BASE_STABILITY_HOURS: float = 336.0

_POWER_LAW_FACTOR: float = 9.0

# FSRS-inspired type-aware decay exponents (Ye, 2022-2024).
# In FSRS these are trainable per-item; here we use fixed per-type values
# calibrated to Tulving (1972): episodic memory decays fastest,
# semantic knowledge persists longest.
_TYPE_DECAY_EXPONENT: dict[str, float] = {
    "semantic": 0.8,  # core facts: slower power-law decay
    "procedural": 1.0,  # preferences/rules: baseline
    "episodic": 1.3,  # events: steeper decay curve
}
_DEFAULT_DECAY_EXPONENT: float = 1.0

# Tulving (1972): different memory types decay at different rates.
_TYPE_STABILITY_MULTIPLIER: dict[str, float] = {
    "semantic": 1.5,  # core facts persist longest (months)
    "procedural": 1.0,  # preferences/rules: baseline (weeks-months)
    "episodic": 0.5,  # events fade fastest (days-weeks)
}


def calculate_stability(
    importance: float,
    access_count: int,
    access_intervals: list[float] | None = None,
    memory_type: str = "semantic",
) -> float:
    """Compute how long a fact resists forgetting (in hours).

    FSRS-inspired: spaced repetition gives stronger reinforcement.
    Five accesses spread over a month > five accesses in one minute.

    Args:
        importance: Fact importance (0.0-1.0).
        access_count: Number of times the fact has been recalled.
        access_intervals: Hours between consecutive accesses.
        memory_type: Memory type for type-aware decay multiplier.

    Returns:
        Stability in hours. Higher = slower decay.
    """
    type_mult = _TYPE_STABILITY_MULTIPLIER.get(memory_type, 1.0)
    base = BASE_STABILITY_HOURS * importance * type_mult

    if access_count <= 0:
        return base

    # Logarithmic access count factor (unchanged)
    count_factor = 1.0 + math.log(1.0 + access_count)

    # Spacing factor: mean interval normalized to stability scale
    # Well-spaced accesses (mean interval > 24h) get bonus
    # Cramped accesses (mean interval < 1h) get penalty
    spacing_factor = 1.0
    if access_intervals and len(access_intervals) >= 1:
        mean_interval = sum(access_intervals) / len(access_intervals)
        # Logarithmic scaling: diminishing returns for very long intervals
        # 1h -> ~0.7, 24h -> ~1.0, 168h (1w) -> ~1.15, 720h (1mo) -> ~1.3
        spacing_factor = 0.7 + 0.3 * math.log(1.0 + mean_interval / 24.0)
        spacing_factor = max(spacing_factor, 0.5)  # floor

    return base * count_factor * spacing_factor


def calculate_retention(
    fact: Fact,
    *,
    now: datetime | None = None,
    type_exponents: dict[str, float] | None = None,
) -> float:
    """Compute current retention score for a fact.

    Uses a power-law forgetting curve (Wixted & Ebbesen, 1997):
        retention = (1 + t / (c * S)) ** -decay_exponent
    where t is hours since last access, S is stability, and c is
    _POWER_LAW_FACTOR.

    Args:
        fact: The fact to evaluate.
        now: Current time (defaults to UTC now).
        type_exponents: Optional per-type decay exponent overrides.
            When ``None``, uses the built-in defaults.

    Returns:
        Retention score between 0.0 and 1.0.
    """
    now = now or datetime.now(UTC)
    time_hours = (now - fact.last_accessed).total_seconds() / 3600.0

    if time_hours <= 0:
        return 1.0

    stability = calculate_stability(
        fact.importance, fact.access_count, fact.access_intervals, fact.type.value
    )
    if stability <= 0:
        return 0.0

    exponents = type_exponents or _TYPE_DECAY_EXPONENT
    decay_exp = exponents.get(fact.type.value, _DEFAULT_DECAY_EXPONENT)
    return float((1.0 + time_hours / (_POWER_LAW_FACTOR * stability)) ** (-decay_exp))


def apply_decay(
    facts: list[Fact],
    *,
    now: datetime | None = None,
    type_exponents: dict[str, float] | None = None,
) -> list[Fact]:
    """Update retention_score on all facts using the forgetting curve.

    This is a bulk operation — call it periodically or before recall.

    Args:
        facts: Facts to update (modified in place).
        now: Current time (defaults to UTC now).
        type_exponents: Optional per-type decay exponent overrides.
            When ``None``, uses the built-in defaults.

    Returns:
        The same list of facts with updated retention_score values.
    """
    now = now or datetime.now(UTC)
    for fact in facts:
        fact.retention_score = calculate_retention(fact, now=now, type_exponents=type_exponents)
    return facts
