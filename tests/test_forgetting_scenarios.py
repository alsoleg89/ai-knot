"""Tests for realistic forgetting scenarios over various time spans."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_knot.forgetting import calculate_retention
from ai_knot.types import Fact, MemoryType

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
        assert r < 0.90  # Fades but semantic type retains longer

    def test_7_days_episodic_low_importance_fades_fast(self) -> None:
        """Episodic low-importance fact decays faster than semantic."""
        fact = Fact(
            content="test",
            type=MemoryType.EPISODIC,
            importance=0.2,
            access_count=0,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=7))
        assert r < 0.70


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
        assert r < 0.65  # Semantic type retains longer (decay exponent 0.8)

    def test_30_days_episodic_low_importance_forgotten(self) -> None:
        """Episodic low-importance fact is well forgotten after 30 days."""
        fact = Fact(
            content="mentioned once",
            type=MemoryType.EPISODIC,
            importance=0.2,
            access_count=0,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=30))
        assert r < 0.35

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
        assert r < 0.50


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
        assert r_used - r_unused > 0.05  # meaningful difference


class TestTypeAwareDecay:
    """Different memory types decay at different rates (Tulving, 1972)."""

    def test_semantic_retains_longest(self) -> None:
        """Semantic facts persist longer than procedural and episodic."""
        one_month = BASE_TIME + timedelta(days=30)

        semantic = Fact(
            content="core fact",
            type=MemoryType.SEMANTIC,
            importance=0.7,
            last_accessed=BASE_TIME,
        )
        procedural = Fact(
            content="preference",
            type=MemoryType.PROCEDURAL,
            importance=0.7,
            last_accessed=BASE_TIME,
        )
        episodic = Fact(
            content="event",
            type=MemoryType.EPISODIC,
            importance=0.7,
            last_accessed=BASE_TIME,
        )

        r_sem = calculate_retention(semantic, now=one_month)
        r_proc = calculate_retention(procedural, now=one_month)
        r_epi = calculate_retention(episodic, now=one_month)

        assert r_sem > r_proc > r_epi

    def test_episodic_fades_within_weeks(self) -> None:
        """Low-importance episodic fact should lose most retention in 2 weeks."""
        fact = Fact(
            content="meeting happened",
            type=MemoryType.EPISODIC,
            importance=0.3,
            last_accessed=BASE_TIME,
        )
        r = calculate_retention(fact, now=BASE_TIME + timedelta(days=14))
        assert r < 0.60


class TestSpacingEffect:
    """FSRS-inspired spacing effect — when accesses happen matters."""

    def test_spaced_beats_cramped_retention(self) -> None:
        """Same access count, but spaced intervals give better retention."""
        two_weeks = BASE_TIME + timedelta(days=14)

        cramped = Fact(
            content="crammed",
            importance=0.8,
            access_count=5,
            access_intervals=[0.1, 0.1, 0.1, 0.1, 0.1],
            last_accessed=BASE_TIME,
        )
        spaced = Fact(
            content="spaced",
            importance=0.8,
            access_count=5,
            access_intervals=[48.0, 48.0, 48.0, 48.0, 48.0],
            last_accessed=BASE_TIME,
        )

        assert calculate_retention(spaced, now=two_weeks) > calculate_retention(
            cramped, now=two_weeks
        )

    def test_no_intervals_is_neutral(self) -> None:
        """Without interval data, spacing factor is 1.0 (neutral)."""
        two_weeks = BASE_TIME + timedelta(days=14)

        with_intervals = Fact(
            content="test",
            importance=0.8,
            access_count=5,
            access_intervals=[24.0, 24.0, 24.0, 24.0, 24.0],
            last_accessed=BASE_TIME,
        )
        without_intervals = Fact(
            content="test",
            importance=0.8,
            access_count=5,
            access_intervals=[],
            last_accessed=BASE_TIME,
        )

        r_with = calculate_retention(with_intervals, now=two_weeks)
        r_without = calculate_retention(without_intervals, now=two_weeks)

        # 24h mean interval gives spacing_factor ~1.0, so these should be close
        assert abs(r_with - r_without) < 0.05
