"""Edge-case tests for KnowledgeBase — empty KB, duplicates, unicode, scale."""

from __future__ import annotations

import pathlib

import pytest

from agentmemo.knowledge import KnowledgeBase
from agentmemo.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="edge", storage=YAMLStorage(base_dir=str(tmp_path)))


class TestEmptyKB:
    """Operations on a knowledge base with no facts."""

    def test_recall_empty(self, kb: KnowledgeBase) -> None:
        assert kb.recall("anything") == ""

    def test_forget_on_empty(self, kb: KnowledgeBase) -> None:
        kb.forget("nonexistent")  # should not raise

    def test_decay_on_empty(self, kb: KnowledgeBase) -> None:
        kb.decay()  # should not raise

    def test_stats_on_empty(self, kb: KnowledgeBase) -> None:
        stats = kb.stats()
        assert stats["total_facts"] == 0


class TestUnicode:
    """Non-ASCII content handling."""

    def test_russian_content(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Пользователь предпочитает Python")
        assert fact.content == "Пользователь предпочитает Python"

    def test_chinese_content(self, kb: KnowledgeBase) -> None:
        fact = kb.add("用户喜欢使用Docker部署")
        assert fact.content == "用户喜欢使用Docker部署"

    def test_emoji_content(self, kb: KnowledgeBase) -> None:
        fact = kb.add("User loves 🐍 Python")
        assert "🐍" in fact.content

    def test_recall_unicode(self, kb: KnowledgeBase) -> None:
        kb.add("Пользователь работает в Сбере")
        result = kb.recall("Сбер")
        assert "Сбер" in result


class TestDuplicates:
    """Adding similar or identical facts."""

    def test_exact_duplicate_allowed(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")
        kb.add("User prefers Python")
        facts = kb._storage.load(kb._agent_id)
        # Both stored — dedup is extractor's job, not add's
        assert len(facts) == 2

    def test_different_ids_for_same_content(self, kb: KnowledgeBase) -> None:
        f1 = kb.add("Same content")
        f2 = kb.add("Same content")
        assert f1.id != f2.id


class TestScale:
    """Performance with many facts."""

    def test_hundred_facts(self, kb: KnowledgeBase) -> None:
        for i in range(100):
            kb.add(f"Fact number {i} about topic {i % 10}")

        facts = kb._storage.load(kb._agent_id)
        assert len(facts) == 100

        result = kb.recall("topic 5", top_k=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_recall_with_many_facts(self, kb: KnowledgeBase) -> None:
        for i in range(50):
            kb.add(f"Python is great for task {i}", importance=0.5)
        kb.add("User absolutely loves Rust", importance=0.99)

        result = kb.recall("Rust")
        assert "Rust" in result


class TestLongContent:
    """Very long fact content."""

    def test_long_content_stored(self, kb: KnowledgeBase) -> None:
        long_text = "A" * 10_000
        fact = kb.add(long_text)
        assert len(fact.content) == 10_000

    def test_long_content_recalled(self, kb: KnowledgeBase) -> None:
        kb.add("Short fact about Python")
        kb.add("X" * 5000 + " Python " + "Y" * 5000)
        result = kb.recall("Python")
        assert isinstance(result, str)
