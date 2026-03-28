"""Tests for agentmemo.forgetting — Ebbinghaus curve core math."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from agentmemo.forgetting import apply_decay, calculate_retention, calculate_stability
from agentmemo.types import Fact


class TestCalculateStability:
    """stability = base_hours * importance * (1 + log(1 + access_count))."""

    def test_default_importance_no_access(self) -> None:
        # base=168h, importance=0.8, access=0 → 168 * 0.8 * (1 + log(1)) = 134.4
        stability = calculate_stability(importance=0.8, access_count=0)
        assert stability == pytest.approx(168.0 * 0.8 * 1.0, rel=1e-6)

    def test_high_importance_many_accesses(self) -> None:
        stability = calculate_stability(importance=1.0, access_count=100)
        # 168 * 1.0 * (1 + log(101)) ≈ 168 * 5.615 ≈ 943.4
        expected = 168.0 * 1.0 * (1.0 + math.log(101))
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


class TestCalculateRetention:
    """retention = exp(-time_hours / stability)."""

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
            # After ~10 months, retention should be well below 1.0
            assert fact.retention_score < 1.0

    def test_empty_list(self) -> None:
        result = apply_decay([])
        assert result == []

    def test_does_not_modify_fresh_facts(self) -> None:
        fact = Fact(content="fresh", importance=0.9)
        result = apply_decay([fact])
        assert result[0].retention_score == pytest.approx(1.0, abs=0.01)


