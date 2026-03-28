"""Tests for retriever quality — real-world relevance scenarios."""

from __future__ import annotations

import pytest

from agentmemo.retriever import TFIDFRetriever
from agentmemo.types import Fact


@pytest.fixture
def retriever() -> TFIDFRetriever:
    return TFIDFRetriever()


@pytest.fixture
def diverse_kb() -> list[Fact]:
    """A realistic knowledge base with diverse facts."""
    return [
        Fact(content="User works at Sber as Operations Director", importance=0.95),
        Fact(content="User prefers Python, dislikes async code", importance=0.85),
        Fact(content="User deploys on Kubernetes with Docker", importance=0.80),
        Fact(content="User's team has 12 backend engineers", importance=0.70),
        Fact(content="User prefers concise responses without emoji", importance=0.75),
        Fact(content="Project uses PostgreSQL 16 with RLS", importance=0.90),
        Fact(content="User wants all code tested with pytest", importance=0.85),
        Fact(content="User's preferred editor is VS Code", importance=0.60),
        Fact(content="User manages a content creation agency", importance=0.80),
        Fact(content="User wants FastAPI for all new APIs", importance=0.85),
    ]


class TestRelevanceScenarios:
    """Does the retriever return the RIGHT facts for real queries?"""

    def test_deployment_query(self, retriever: TFIDFRetriever, diverse_kb: list[Fact]) -> None:
        results = retriever.search("how should I deploy this service?", diverse_kb, top_k=3)
        contents = [r.content for r in results]
        # Should find Docker/Kubernetes fact
        assert any("Docker" in c or "Kubernetes" in c or "deploy" in c for c in contents)

    def test_coding_preferences_query(
        self, retriever: TFIDFRetriever, diverse_kb: list[Fact]
    ) -> None:
        results = retriever.search("what programming language to use?", diverse_kb, top_k=3)
        contents = [r.content for r in results]
        assert any("Python" in c for c in contents)

    def test_database_query(self, retriever: TFIDFRetriever, diverse_kb: list[Fact]) -> None:
        results = retriever.search("which database for this project?", diverse_kb, top_k=3)
        contents = [r.content for r in results]
        assert any("PostgreSQL" in c for c in contents)

    def test_testing_query(self, retriever: TFIDFRetriever, diverse_kb: list[Fact]) -> None:
        results = retriever.search("how to test the code?", diverse_kb, top_k=3)
        contents = [r.content for r in results]
        assert any("pytest" in c or "test" in c.lower() for c in contents)

    def test_api_framework_query(self, retriever: TFIDFRetriever, diverse_kb: list[Fact]) -> None:
        results = retriever.search("which API framework?", diverse_kb, top_k=3)
        contents = [r.content for r in results]
        assert any("FastAPI" in c or "API" in c for c in contents)

    def test_who_is_user_query(self, retriever: TFIDFRetriever, diverse_kb: list[Fact]) -> None:
        results = retriever.search("who is this user?", diverse_kb, top_k=3)
        # Should return profile-type facts
        assert len(results) > 0
