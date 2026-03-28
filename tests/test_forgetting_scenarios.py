"""Tests for realistic forgetting scenarios over various time spans."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agentmemo.forgetting import calculate_retention
from agentmemo.types import Fact

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


class TestShortTermDecay:
    """Facts within hours — should retain well."""

    def test_1_hour_high_importance(self) -> None:
        fact = Fact(content="test", importance=0.9, last_accessed=BASE_TIME)
        r = calculate_retention(fact, now=BASE_TIME + timedelta(hours=1))
        assert r > 0.99

    def test_6_hours_medium_importance(self) -> None:
        fact = Fact(content="test", importance=0.5, last_accessed=BASE_TIME)
        r = calculate_retention(fact, now=BASE_TIME + timedelta(hours=6))
        assert r > 0.90


class TestMediumTermDecay:
    """Facts within days — retention depends on importance and access."""

    def test_1_day_never_accessed(self) -> None:
        fact = Fact(
            content="test",
            importance=0.5,
            access_count=0,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=1))
        assert 0.5 < r < 1.0

    def test_3_days_frequently_accessed(self) -> None:
        fact = Fact(
            content="test",
            importance=0.8,
            access_count=20,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=3))
        assert r > 0.90

    def test_7_days_low_importance_fades(self) -> None:
        fact = Fact(
            content="test",
            importance=0.2,
            access_count=0,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=7))
        assert r < 0.2


class TestLongTermDecay:
    """Facts over weeks/months — only important, accessed facts survive."""

    def test_30_days_high_importance_high_access(self) -> None:
        fact = Fact(
            content="core preference",
            importance=0.95,
            access_count=50,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=30))
        assert r > 0.50

    def test_30_days_low_importance_forgotten(self) -> None:
        fact = Fact(
            content="mentioned once",
            importance=0.2,
            access_count=0,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=30))
        assert r < 0.05

    def test_90_days_critical_fact_still_alive(self) -> None:
        fact = Fact(
            content="user name",
            importance=1.0,
            access_count=100,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=90))
        assert r > 0.20

    def test_365_days_everything_fades_eventually(self) -> None:
        fact = Fact(
            content="old info",
            importance=0.5,
            access_count=5,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=365))
        assert r < 0.01


class TestReinforcementEffect:
    """Accessing a fact should slow its decay."""

    def test_access_reinforces_retention(self) -> None:
        """Same fact: 0 accesses vs 50 accesses after 2 weeks."""
        two_weeks = BASE_TIME + timedelta(days=14)

        unused = Fact(
            content="test",
            importance=0.7,
            access_count=0,
            last_accessed=BASE_TIME,
        )
        used = Fact(
            content="test",
            importance=0.7,
            access_count=50,
            last_accessed=BASE_TIME,
        )

        r_unused = calculate_retention(unused, now=two_weeks)
        r_used = calculate_retention(used, now=two_weeks)

        assert r_used > r_unused
        assert r_used - r_unused > 0.1  # meaningful difference
