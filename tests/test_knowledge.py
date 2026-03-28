"""Tests for agentmemo.knowledge — KnowledgeBase main class."""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from agentmemo.knowledge import KnowledgeBase
from agentmemo.storage.yaml_storage import YAMLStorage
from agentmemo.types import ConversationTurn, Fact, MemoryType


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="test_agent", storage=storage)


class TestAdd:
    """Adding facts to the knowledge base."""

    def test_add_returns_fact(self, kb: KnowledgeBase) -> None:
        fact = kb.add("User prefers Python", importance=0.9)
        assert isinstance(fact, Fact)
        assert fact.content == "User prefers Python"
        assert fact.importance == 0.9

    def test_add_with_type(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Always use type hints", type=MemoryType.PROCEDURAL)
        assert fact.type == MemoryType.PROCEDURAL

    def test_add_with_tags(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Works at Sber", tags=["profile", "work"])
        assert fact.tags == ["profile", "work"]

    def test_add_persists(self, kb: KnowledgeBase) -> None:
        kb.add("Persistent fact")
        # Reload from storage
        facts = kb._storage.load(kb._agent_id)
        assert len(facts) == 1
        assert facts[0].content == "Persistent fact"

    def test_add_multiple(self, kb: KnowledgeBase) -> None:
        kb.add("Fact one")
        kb.add("Fact two")
        kb.add("Fact three")
        facts = kb._storage.load(kb._agent_id)
        assert len(facts) == 3


class TestRecall:
    """Querying the knowledge base."""

    def test_recall_returns_string(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")
        result = kb.recall("what language?")
        assert isinstance(result, str)
        assert "Python" in result

    def test_recall_empty_kb(self, kb: KnowledgeBase) -> None:
        result = kb.recall("anything")
        assert result == ""

    def test_recall_top_k(self, kb: KnowledgeBase) -> None:
        for i in range(10):
            kb.add(f"Fact number {i}")
        result = kb.recall("fact", top_k=3)
        # Should contain at most 3 facts in the result
        assert result.count("[") <= 3

    def test_recall_increments_access_count(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python", importance=0.9)
        kb.recall("Python")
        facts = kb._storage.load(kb._agent_id)
        # At least one fact should have access_count > 0
        accessed = [f for f in facts if f.access_count > 0]
        assert len(accessed) >= 1


class TestForget:
    """Removing facts from the knowledge base."""

    def test_forget_by_id(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Forgettable fact")
        kb.forget(fact.id)
        facts = kb._storage.load(kb._agent_id)
        assert len(facts) == 0

    def test_forget_nonexistent(self, kb: KnowledgeBase) -> None:
        kb.add("Keep this")
        kb.forget("nonexistent_id")  # should not raise
        facts = kb._storage.load(kb._agent_id)
        assert len(facts) == 1


class TestDecay:
    """Applying forgetting curve to all facts."""

    def test_decay_updates_retention(self, kb: KnowledgeBase) -> None:
        from datetime import datetime, timedelta, timezone

        fact = kb.add("Old fact")
        # Manually set old last_accessed
        facts = kb._storage.load(kb._agent_id)
        facts[0].last_accessed = datetime(2025, 1, 1, tzinfo=timezone.utc)
        kb._storage.save(kb._agent_id, facts)

        kb.decay()

        updated = kb._storage.load(kb._agent_id)
        assert updated[0].retention_score < 1.0


class TestStats:
    """Knowledge base statistics."""

    def test_stats_on_empty_kb(self, kb: KnowledgeBase) -> None:
        stats = kb.stats()
        assert stats["total_facts"] == 0

    def test_stats_with_facts(self, kb: KnowledgeBase) -> None:
        kb.add("Fact 1", type=MemoryType.SEMANTIC)
        kb.add("Fact 2", type=MemoryType.PROCEDURAL)
        kb.add("Fact 3", type=MemoryType.EPISODIC)

        stats = kb.stats()
        assert stats["total_facts"] == 3
        assert stats["by_type"]["semantic"] == 1
        assert stats["by_type"]["procedural"] == 1
        assert stats["by_type"]["episodic"] == 1
