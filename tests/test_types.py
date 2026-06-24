"""Tests for ai_knot.types — Fact, ConversationTurn, MemoryType."""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot.types import ConversationTurn, Fact, MemoryType


class TestMemoryType:
    """MemoryType enum correctness."""

    def test_values(self) -> None:
        assert MemoryType.SEMANTIC == "semantic"
        assert MemoryType.PROCEDURAL == "procedural"
        assert MemoryType.EPISODIC == "episodic"

    def test_from_string(self) -> None:
        assert MemoryType("semantic") is MemoryType.SEMANTIC
        assert MemoryType("procedural") is MemoryType.PROCEDURAL
        assert MemoryType("episodic") is MemoryType.EPISODIC

    def test_members_count(self) -> None:
        assert len(MemoryType) == 3


class TestFact:
    """Fact dataclass behavior."""

    def test_defaults(self) -> None:
        fact = Fact(content="test")
        assert fact.content == "test"
        assert fact.type is MemoryType.SEMANTIC
        assert fact.importance == 0.8
        assert fact.retention_score == 1.0
        assert fact.access_count == 0
        assert fact.tags == []
        assert len(fact.id) == 8
        assert isinstance(fact.created_at, datetime)
        assert fact.created_at.tzinfo is not None

    def test_custom_values(self) -> None:
        fact = Fact(
            content="Custom fact",
            type=MemoryType.PROCEDURAL,
            importance=0.5,
            tags=["test"],
        )
        assert fact.type is MemoryType.PROCEDURAL
        assert fact.importance == 0.5
        assert fact.tags == ["test"]

    def test_unique_ids(self) -> None:
        facts = [Fact(content=f"fact {i}") for i in range(100)]
        ids = {f.id for f in facts}
        assert len(ids) == 100

    def test_timestamps_are_utc(self) -> None:
        fact = Fact(content="utc test")
        assert fact.created_at.tzinfo == UTC
        assert fact.last_accessed.tzinfo == UTC

    def test_created_at_and_last_accessed_close(self) -> None:
        fact = Fact(content="timing test")
        delta = abs((fact.last_accessed - fact.created_at).total_seconds())
        assert delta < 1.0


class TestIsActive:
    """Fact.is_active() temporal validity checks."""

    def test_active_no_bounds(self) -> None:
        from datetime import UTC, datetime

        fact = Fact(content="no bounds")
        assert fact.is_active(datetime.now(UTC)) is True

    def test_active_past_valid_from(self) -> None:
        from datetime import UTC, datetime, timedelta

        fact = Fact(content="started in the past")
        fact.valid_from = datetime.now(UTC) - timedelta(hours=1)
        assert fact.is_active() is True

    def test_inactive_future_valid_from(self) -> None:
        from datetime import UTC, datetime, timedelta

        fact = Fact(content="starts in the future")
        fact.valid_from = datetime.now(UTC) + timedelta(hours=1)
        assert fact.is_active() is False

    def test_inactive_past_valid_until(self) -> None:
        from datetime import UTC, datetime, timedelta

        fact = Fact(content="already expired")
        fact.valid_until = datetime.now(UTC) - timedelta(hours=1)
        assert fact.is_active() is False

    def test_active_within_window(self) -> None:
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        fact = Fact(content="within window")
        fact.valid_from = now - timedelta(hours=2)
        fact.valid_until = now + timedelta(hours=2)
        assert fact.is_active(now) is True

    def test_inactive_before_window(self) -> None:
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        fact = Fact(content="window starts in the future")
        fact.valid_from = now + timedelta(hours=1)
        fact.valid_until = now + timedelta(hours=3)
        assert fact.is_active(now) is False


class TestConversationTurn:
    """ConversationTurn dataclass behavior."""

    def test_creation(self) -> None:
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_assistant_role(self) -> None:
        turn = ConversationTurn(role="assistant", content="Hi there")
        assert turn.role == "assistant"
