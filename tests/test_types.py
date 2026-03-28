"""Tests for agentmemo.types — Fact, ConversationTurn, MemoryType."""

from __future__ import annotations

from datetime import UTC, datetime

from agentmemo.types import ConversationTurn, Fact, MemoryType


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


class TestConversationTurn:
    """ConversationTurn dataclass behavior."""

    def test_creation(self) -> None:
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_assistant_role(self) -> None:
        turn = ConversationTurn(role="assistant", content="Hi there")
        assert turn.role == "assistant"
