"""Tests for ai_knot.forgetting — Ebbinghaus curve core math."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from ai_knot.forgetting import (
    _TYPE_STABILITY_MULTIPLIER,
    BASE_STABILITY_HOURS,
    apply_decay,
    calculate_retention,
    calculate_stability,
)
from ai_knot.types import Fact, MemoryType


class TestCalculateStability:
    """stability = base_hours * importance * type_mult * count_factor * spacing_factor."""

    def test_default_importance_no_access(self) -> None:
        stability = calculate_stability(importance=0.8, access_count=0, memory_type="semantic")
        expected = BASE_STABILITY_HOURS * 0.8 * _TYPE_STABILITY_MULTIPLIER["semantic"]
        assert stability == pytest.approx(expected, rel=1e-6)

    def test_high_importance_many_accesses(self) -> None:
        stability = calculate_stability(importance=1.0, access_count=100, memory_type="semantic")
        expected = (
            BASE_STABILITY_HOURS
            * 1.0
            * _TYPE_STABILITY_MULTIPLIER["semantic"]
            * (1.0 + math.log(101))
        )
        assert stability == pytest.approx(expected, rel=1e-6)

    def test_zero_importance(self) -> None:
        stability = calculate_stability(importance=0.0, access_count=10)
        assert stability == 0.0

    def test_more_accesses_means_higher_stability(self) -> None:
        s1 = calculate_stability(importance=0.8, access_count=0)
        s2 = calculate_stability(importance=0.8, access_count=10)
        s3 = calculate_stability(importance=0.8, access_count=100)
        assert s1 < s2 < s3

    def test_higher_importance_means_higher_stability(self) -> None:
        s1 = calculate_stability(importance=0.3, access_count=5)
        s2 = calculate_stability(importance=0.9, access_count=5)
        assert s1 < s2

    def test_spacing_effect_spaced_beats_cramped(self) -> None:
        """Well-spaced accesses give higher stability than cramped ones."""
        cramped = calculate_stability(
            importance=0.8,
            access_count=5,
            access_intervals=[0.1, 0.1, 0.1, 0.1, 0.1],  # 6 minutes apart
        )
        spaced = calculate_stability(
            importance=0.8,
            access_count=5,
            access_intervals=[48.0, 48.0, 48.0, 48.0, 48.0],  # 2 days apart
        )
        assert spaced > cramped

    def test_spacing_factor_floor(self) -> None:
        """Spacing factor never goes below 0.5."""
        # Very tiny intervals
        stability = calculate_stability(
            importance=0.8,
            access_count=5,
            access_intervals=[0.001, 0.001, 0.001],
        )
        # Should still be positive and > 0
        assert stability > 0

    def test_no_intervals_same_as_no_spacing(self) -> None:
        """Without intervals, spacing_factor defaults to 1.0."""
        s1 = calculate_stability(importance=0.8, access_count=5, access_intervals=None)
        s2 = calculate_stability(importance=0.8, access_count=5, access_intervals=[])
        assert s1 == s2

    def test_type_aware_stability(self) -> None:
        """Semantic > procedural > episodic for same importance/access."""
        sem = calculate_stability(importance=0.8, access_count=5, memory_type="semantic")
        proc = calculate_stability(importance=0.8, access_count=5, memory_type="procedural")
        epi = calculate_stability(importance=0.8, access_count=5, memory_type="episodic")
        assert sem > proc > epi


class TestCalculateRetention:
    """retention = (1 + t / (c * S)) ** -decay_exponent."""

    def test_just_accessed_is_full_retention(self) -> None:
        fact = Fact(content="fresh")
        now = fact.last_accessed
        retention = calculate_retention(fact, now=now)
        assert retention == pytest.approx(1.0, abs=1e-9)

    def test_retention_decreases_over_time(self) -> None:
        base_time = datetime(2026, 1, 1, tzinfo=UTC)
        fact = Fact(content="test", last_accessed=base_time, importance=0.8)

        r_1h = calculate_retention(fact, now=base_time + timedelta(hours=1))
        r_24h = calculate_retention(fact, now=base_time + timedelta(hours=24))
        r_7d = calculate_retention(fact, now=base_time + timedelta(days=7))

        assert 0.0 < r_7d < r_24h < r_1h <= 1.0

    def test_zero_stability_returns_zero(self) -> None:
        fact = Fact(content="test", importance=0.0)
        now = fact.last_accessed + timedelta(hours=1)
        retention = calculate_retention(fact, now=now)
        assert retention == 0.0

    def test_high_importance_retains_longer(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        one_week_later = base + timedelta(days=7)

        low = Fact(content="low", importance=0.2, last_accessed=base)
        high = Fact(content="high", importance=0.95, last_accessed=base)

        assert calculate_retention(low, now=one_week_later) < calculate_retention(
            high, now=one_week_later
        )

    def test_frequently_accessed_retains_longer(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        one_week_later = base + timedelta(days=7)

        rarely = Fact(content="rare", access_count=0, last_accessed=base)
        often = Fact(content="often", access_count=50, last_accessed=base)

        assert calculate_retention(rarely, now=one_week_later) < calculate_retention(
            often, now=one_week_later
        )

    def test_episodic_decays_faster_than_semantic(self) -> None:
        """Episodic memory type decays faster than semantic."""
        base = datetime(2026, 1, 1, tzinfo=UTC)
        one_week = base + timedelta(days=7)

        episodic = Fact(
            content="event happened", type=MemoryType.EPISODIC, importance=0.8, last_accessed=base
        )
        semantic = Fact(
            content="core fact", type=MemoryType.SEMANTIC, importance=0.8, last_accessed=base
        )

        assert calculate_retention(episodic, now=one_week) < calculate_retention(
            semantic, now=one_week
        )

    def test_spaced_accesses_improve_retention(self) -> None:
        """Fact with spaced access intervals retains better."""
        base = datetime(2026, 1, 1, tzinfo=UTC)
        two_weeks = base + timedelta(days=14)

        cramped = Fact(
            content="cramped",
            importance=0.8,
            access_count=5,
            access_intervals=[0.1, 0.1, 0.1, 0.1, 0.1],
            last_accessed=base,
        )
        spaced = Fact(
            content="spaced",
            importance=0.8,
            access_count=5,
            access_intervals=[48.0, 48.0, 48.0, 48.0, 48.0],
            last_accessed=base,
        )

        assert calculate_retention(cramped, now=two_weeks) < calculate_retention(
            spaced, now=two_weeks
        )


class TestApplyDecay:
    """apply_decay updates retention_score on a list of facts."""

    def test_updates_retention_scores(self, sample_facts: list[Fact]) -> None:
        # Set all facts to have old last_accessed
        past = datetime(2025, 6, 1, tzinfo=UTC)
        for fact in sample_facts:
            fact.last_accessed = past
            fact.retention_score = 1.0

        now = datetime(2026, 3, 28, tzinfo=UTC)
        updated = apply_decay(sample_facts, now=now)

        for fact in updated:
            assert 0.0 <= fact.retention_score <= 1.0
            # PROCEDURAL facts are decay-immune (ConflictPolicy), stay at 1.0.
            if fact.type == MemoryType.PROCEDURAL:
                assert fact.retention_score == 1.0
            else:
                # After ~10 months, retention should be well below 1.0
                assert fact.retention_score < 1.0

    def test_empty_list(self) -> None:
        result = apply_decay([])
        assert result == []

    def test_does_not_modify_fresh_facts(self) -> None:
        fact = Fact(content="fresh", importance=0.9)
        result = apply_decay([fact])
        assert result[0].retention_score == pytest.approx(1.0, abs=0.01)
