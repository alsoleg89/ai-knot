"""Tests for ai_knot.knowledge — KnowledgeBase main class."""

from __future__ import annotations

import os
import pathlib
from unittest.mock import patch

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import ConversationTurn, Fact, MemoryType


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
        from datetime import UTC, datetime

        kb.add("Old fact")
        # Manually set old last_accessed
        facts = kb._storage.load(kb._agent_id)
        facts[0].last_accessed = datetime(2025, 1, 1, tzinfo=UTC)
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


class TestRecallFacts:
    """Structured retrieval returning Fact objects."""

    def test_recall_facts_returns_fact_objects(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")
        results = kb.recall_facts("what language?")
        assert isinstance(results, list)
        assert all(isinstance(f, Fact) for f in results)

    def test_recall_facts_empty_kb(self, kb: KnowledgeBase) -> None:
        assert kb.recall_facts("anything") == []

    def test_recall_facts_respects_top_k(self, kb: KnowledgeBase) -> None:
        for i in range(10):
            kb.add(f"Fact number {i} about deployment")
        results = kb.recall_facts("deployment", top_k=2)
        assert len(results) <= 2

    def test_recall_facts_content_matches_query(self, kb: KnowledgeBase) -> None:
        kb.add("User deploys on Fridays")
        kb.add("User prefers tea over coffee")
        results = kb.recall_facts("deployment day")
        assert len(results) >= 1
        assert any("Friday" in f.content for f in results)

    def test_recall_facts_updates_access_count(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python", importance=0.9)
        kb.recall_facts("Python")
        facts = kb._storage.load(kb._agent_id)
        accessed = [f for f in facts if f.access_count > 0]
        assert len(accessed) >= 1


class TestRecallByTag:
    """Tag-based filtering of stored facts."""

    def test_recall_by_tag_finds_tagged(self, kb: KnowledgeBase) -> None:
        kb.add("User works at Sber", tags=["profile"])
        results = kb.recall_by_tag("profile")
        assert len(results) == 1
        assert results[0].content == "User works at Sber"

    def test_recall_by_tag_empty_when_no_match(self, kb: KnowledgeBase) -> None:
        kb.add("Some fact", tags=["other"])
        assert kb.recall_by_tag("nonexistent") == []

    def test_recall_by_tag_ignores_untagged(self, kb: KnowledgeBase) -> None:
        kb.add("Tagged fact", tags=["work"])
        kb.add("Untagged fact")
        results = kb.recall_by_tag("work")
        assert len(results) == 1
        assert results[0].content == "Tagged fact"

    def test_recall_by_tag_multiple_facts(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A", tags=["important"])
        kb.add("Fact B", tags=["important"])
        kb.add("Fact C", tags=["other"])
        results = kb.recall_by_tag("important")
        assert len(results) == 2


class TestLearnApiKey:
    """learn() must raise ValueError when no API key is available."""

    def test_learn_raises_without_api_key(self, kb: KnowledgeBase) -> None:
        turns = [ConversationTurn(role="user", content="Deploy on Fridays")]
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
            pytest.raises(ValueError, match="No API key"),
        ):
            kb.learn(turns, provider="openai", api_key=None)

    def test_learn_empty_turns_returns_empty_without_key(self, kb: KnowledgeBase) -> None:
        # empty turns → early return before key check
        result = kb.learn([], provider="openai", api_key=None)
        assert result == []


class TestRecallFactsWithScores:
    """recall_facts_with_scores() returns (Fact, float) pairs."""

    def test_returns_pairs(self, kb: KnowledgeBase) -> None:
        kb.add("User deploys on Fridays")
        pairs = kb.recall_facts_with_scores("deployment day")
        assert len(pairs) >= 1
        fact, score = pairs[0]
        assert isinstance(fact, Fact)
        assert isinstance(score, float)

    def test_empty_kb_returns_empty(self, kb: KnowledgeBase) -> None:
        assert kb.recall_facts_with_scores("anything") == []

    def test_scores_are_non_negative(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")
        pairs = kb.recall_facts_with_scores("language")
        assert all(score >= 0.0 for _, score in pairs)

    def test_top_k_limits_results(self, kb: KnowledgeBase) -> None:
        for i in range(10):
            kb.add(f"Deployment fact {i}")
        pairs = kb.recall_facts_with_scores("deployment", top_k=3)
        assert len(pairs) <= 3
