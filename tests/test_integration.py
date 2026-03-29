"""Integration tests — full lifecycle: add → recall → decay → recall → forget."""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import MemoryType


@pytest.fixture(params=["yaml", "sqlite"])
def kb(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> KnowledgeBase:
    """Parametrized KB — runs each test with both backends."""
    if request.param == "yaml":
        storage = YAMLStorage(base_dir=str(tmp_path / "yaml"))
    else:
        storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    return KnowledgeBase(agent_id="integration_test", storage=storage)


class TestFullLifecycle:
    """End-to-end agent memory lifecycle."""

    def test_add_recall_forget(self, kb: KnowledgeBase) -> None:
        # Add facts
        kb.add("User prefers Python", type=MemoryType.PROCEDURAL, importance=0.9)
        kb.add("User works at Sber", type=MemoryType.SEMANTIC, importance=0.95)
        f3 = kb.add("Deploy failed on Tuesday", type=MemoryType.EPISODIC, importance=0.4)

        # Recall should find relevant facts
        result = kb.recall("what language does the user prefer?")
        assert "Python" in result

        result = kb.recall("where does the user work?")
        assert "Sber" in result

        # Forget episodic fact
        kb.forget(f3.id)
        facts = kb._storage.load(kb._agent_id)
        assert len(facts) == 2
        assert all(f.id != f3.id for f in facts)

    def test_decay_reduces_old_facts(self, kb: KnowledgeBase) -> None:
        # Add a fact and manually age it
        kb.add("Old information", importance=0.3)

        facts = kb._storage.load(kb._agent_id)
        facts[0].last_accessed = datetime(2025, 1, 1, tzinfo=UTC)
        facts[0].retention_score = 1.0
        kb._storage.save(kb._agent_id, facts)

        # Apply decay
        kb.decay()

        updated = kb._storage.load(kb._agent_id)
        assert updated[0].retention_score < 0.5

    def test_access_reinforces_retention(self, kb: KnowledgeBase) -> None:
        kb.add("Important fact about Python", importance=0.9)

        # Access it multiple times
        for _ in range(5):
            kb.recall("Python")

        facts = kb._storage.load(kb._agent_id)
        assert facts[0].access_count >= 5

    def test_stats_reflect_state(self, kb: KnowledgeBase) -> None:
        kb.add("Semantic", type=MemoryType.SEMANTIC)
        kb.add("Procedural", type=MemoryType.PROCEDURAL)
        kb.add("Episodic", type=MemoryType.EPISODIC)

        stats = kb.stats()
        assert stats["total_facts"] == 3
        assert stats["by_type"]["semantic"] == 1
        assert stats["by_type"]["procedural"] == 1
        assert stats["by_type"]["episodic"] == 1


class TestMultiAgentIsolation:
    """Multiple agents sharing the same storage don't leak data."""

    def test_agents_isolated(self, tmp_path: pathlib.Path) -> None:
        storage = YAMLStorage(base_dir=str(tmp_path))

        alice = KnowledgeBase(agent_id="alice", storage=storage)
        bob = KnowledgeBase(agent_id="bob", storage=storage)

        alice.add("Alice's secret: prefers Rust")
        bob.add("Bob's secret: prefers Java")

        alice_result = alice.recall("preference")
        bob_result = bob.recall("preference")

        assert "Rust" in alice_result
        assert "Java" not in alice_result
        assert "Java" in bob_result
        assert "Rust" not in bob_result


class TestRecallFormatting:
    """Recall output format should be injectable into prompts."""

    def test_recall_format(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python", type=MemoryType.PROCEDURAL)
        kb.add("User works at Sber", type=MemoryType.SEMANTIC)

        result = kb.recall("user info")
        # Should contain type annotations
        assert "[" in result  # e.g., [procedural] or [semantic]
        assert "User" in result

    def test_recall_multiline(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A")
        kb.add("Fact B")
        kb.add("Fact C")

        result = kb.recall("facts", top_k=3)
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) >= 2
