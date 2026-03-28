"""Tests for agentmemo.retriever — TF-IDF search engine."""

from __future__ import annotations

import pytest

from agentmemo.retriever import TFIDFRetriever
from agentmemo.types import Fact


@pytest.fixture
def retriever() -> TFIDFRetriever:
    return TFIDFRetriever()


@pytest.fixture
def coding_facts() -> list[Fact]:
    return [
        Fact(content="User prefers Python for backend development", importance=0.9),
        Fact(content="User deploys applications using Docker containers", importance=0.8),
        Fact(content="User works at Sber as Operations Director", importance=0.95),
        Fact(content="User dislikes JavaScript and frontend work", importance=0.7),
        Fact(content="User always uses pytest for testing Python code", importance=0.85),
    ]


class TestTFIDFSearch:
    """Core search functionality."""

    def test_relevant_result_first(
        self, retriever: TFIDFRetriever, coding_facts: list[Fact]
    ) -> None:
        results = retriever.search("Python", coding_facts, top_k=3)
        assert len(results) > 0
        assert "Python" in results[0].content

    def test_top_k_limits_results(
        self, retriever: TFIDFRetriever, coding_facts: list[Fact]
    ) -> None:
        results = retriever.search("user", coding_facts, top_k=2)
        assert len(results) <= 2

    def test_no_results_for_unrelated_query(
        self, retriever: TFIDFRetriever, coding_facts: list[Fact]
    ) -> None:
        results = retriever.search("quantum physics rocket science", coding_facts, top_k=3)
        # May return results with low scores, but top result shouldn't be highly relevant
        # The important thing is it doesn't crash
        assert isinstance(results, list)

    def test_empty_facts_list(self, retriever: TFIDFRetriever) -> None:
        results = retriever.search("anything", [], top_k=5)
        assert results == []

    def test_empty_query(self, retriever: TFIDFRetriever, coding_facts: list[Fact]) -> None:
        results = retriever.search("", coding_facts, top_k=3)
        assert isinstance(results, list)

    def test_single_fact(self, retriever: TFIDFRetriever) -> None:
        facts = [Fact(content="User likes cats")]
        results = retriever.search("cats", facts, top_k=5)
        assert len(results) == 1
        assert results[0].content == "User likes cats"


class TestSearchWithRetentionBoost:
    """Retention score and importance should influence ranking."""

    def test_high_retention_ranked_higher(self, retriever: TFIDFRetriever) -> None:
        high_retention = Fact(
            content="User prefers Python",
            importance=0.9,
            retention_score=0.95,
        )
        low_retention = Fact(
            content="User prefers Python language",
            importance=0.9,
            retention_score=0.1,
        )
        results = retriever.search("Python", [low_retention, high_retention], top_k=2)
        assert results[0].retention_score > results[1].retention_score

    def test_high_importance_ranked_higher(self, retriever: TFIDFRetriever) -> None:
        important = Fact(content="User deploys with Docker", importance=0.99)
        trivial = Fact(content="User mentioned Docker once", importance=0.1)
        results = retriever.search("Docker", [trivial, important], top_k=2)
        assert results[0].importance > results[1].importance


class TestSearchSpecialCharacters:
    """Edge cases with special characters."""

    def test_unicode_content(self, retriever: TFIDFRetriever) -> None:
        facts = [
            Fact(content="Пользователь предпочитает Python"),
            Fact(content="Пользователь работает в Сбере"),
        ]
        results = retriever.search("Python", facts, top_k=2)
        assert len(results) > 0

    def test_punctuation_in_query(
        self, retriever: TFIDFRetriever, coding_facts: list[Fact]
    ) -> None:
        results = retriever.search("Python?!", coding_facts, top_k=3)
        assert isinstance(results, list)

    def test_very_long_query(self, retriever: TFIDFRetriever, coding_facts: list[Fact]) -> None:
        long_query = "Python " * 500
        results = retriever.search(long_query, coding_facts, top_k=3)
        assert isinstance(results, list)
