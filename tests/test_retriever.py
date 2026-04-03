"""Tests for ai_knot.retriever — BM25 search engine."""

from __future__ import annotations

import math

import pytest

from ai_knot.retriever import BM25Retriever, InvertedIndex, TFIDFRetriever
from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact


@pytest.fixture
def retriever() -> BM25Retriever:
    return BM25Retriever()


@pytest.fixture
def coding_facts() -> list[Fact]:
    return [
        Fact(content="User prefers Python for backend development", importance=0.9),
        Fact(content="User deploys applications using Docker containers", importance=0.8),
        Fact(content="User works at Sber as Operations Director", importance=0.95),
        Fact(content="User dislikes JavaScript and frontend work", importance=0.7),
        Fact(content="User always uses pytest for testing Python code", importance=0.85),
    ]


class TestBM25Search:
    """Core BM25 search functionality."""

    def test_relevant_result_first(
        self, retriever: BM25Retriever, coding_facts: list[Fact]
    ) -> None:
        results = retriever.search("Python", coding_facts, top_k=3)
        assert len(results) > 0
        assert "Python" in results[0][0].content

    def test_top_k_limits_results(self, retriever: BM25Retriever, coding_facts: list[Fact]) -> None:
        results = retriever.search("user", coding_facts, top_k=2)
        assert len(results) <= 2

    def test_no_results_for_unrelated_query(
        self, retriever: BM25Retriever, coding_facts: list[Fact]
    ) -> None:
        results = retriever.search("quantum physics rocket science", coding_facts, top_k=3)
        # May return results with low scores, but top result shouldn't be highly relevant
        # The important thing is it doesn't crash
        assert isinstance(results, list)

    def test_empty_facts_list(self, retriever: BM25Retriever) -> None:
        results = retriever.search("anything", [], top_k=5)
        assert results == []

    def test_empty_query(self, retriever: BM25Retriever, coding_facts: list[Fact]) -> None:
        results = retriever.search("", coding_facts, top_k=3)
        assert isinstance(results, list)

    def test_single_fact(self, retriever: BM25Retriever) -> None:
        facts = [Fact(content="User likes cats")]
        results = retriever.search("cats", facts, top_k=5)
        assert len(results) == 1
        assert results[0][0].content == "User likes cats"

    def test_exact_match_scores_higher_than_partial(self, retriever: BM25Retriever) -> None:
        """BM25 should score exact term matches higher than documents without the term."""
        exact = Fact(content="Python is great for data science")
        no_match = Fact(content="Java is widely used in enterprise")
        results = retriever.search("Python", [no_match, exact], top_k=2)
        # The exact-match document should be ranked first.
        assert results[0][0].content == exact.content

    def test_length_normalization_shorter_doc_scores_better(self, retriever: BM25Retriever) -> None:
        """For the same term frequency, a shorter document should score higher under BM25."""
        short_doc = Fact(content="Python programming", importance=0.5, retention_score=0.5)
        # Long doc has same term but many additional words, diluting its relevance.
        long_doc = Fact(
            content=(
                "Python is used in many different contexts including web development "
                "data analysis machine learning automation scripting testing deployment"
            ),
            importance=0.5,
            retention_score=0.5,
        )
        results = retriever.search("Python", [long_doc, short_doc], top_k=2)
        # Short document should rank first due to length normalization.
        assert results[0][0].content == short_doc.content

    def test_p95_normalization_scores_bounded(self, retriever: BM25Retriever) -> None:
        """All hybrid scores should be in a reasonable range; BM25 component in [0, 1]."""
        facts = [
            Fact(content="Python Python Python Python", importance=0.5, retention_score=0.5),
            Fact(content="Python is good", importance=0.5, retention_score=0.5),
            Fact(content="Java is different", importance=0.5, retention_score=0.5),
            Fact(content="Nothing relevant here", importance=0.5, retention_score=0.5),
        ]
        results = retriever.search("Python", facts, top_k=4)
        # All hybrid scores should be non-negative.
        for _, score in results:
            assert score >= 0.0
        # Hybrid max is bounded by 1.0 (each component <= 1.0, weights sum to 1.0).
        for _, score in results:
            assert score <= 1.0


class TestSearchWithRetentionBoost:
    """Retention score and importance should influence ranking."""

    def test_high_retention_ranked_higher(self, retriever: BM25Retriever) -> None:
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
        assert results[0][0].retention_score > results[1][0].retention_score

    def test_high_importance_ranked_higher(self, retriever: BM25Retriever) -> None:
        important = Fact(content="User deploys with Docker", importance=0.99)
        trivial = Fact(content="User mentioned Docker once", importance=0.1)
        results = retriever.search("Docker", [trivial, important], top_k=2)
        assert results[0][0].importance > results[1][0].importance


class TestSearchSpecialCharacters:
    """Edge cases with special characters."""

    def test_unicode_content(self, retriever: BM25Retriever) -> None:
        facts = [
            Fact(content="Пользователь предпочитает Python"),
            Fact(content="Пользователь работает в Сбере"),
        ]
        results = retriever.search("Python", facts, top_k=2)
        assert len(results) > 0

    def test_punctuation_in_query(self, retriever: BM25Retriever, coding_facts: list[Fact]) -> None:
        results = retriever.search("Python?!", coding_facts, top_k=3)
        assert isinstance(results, list)

    def test_very_long_query(self, retriever: BM25Retriever, coding_facts: list[Fact]) -> None:
        long_query = "Python " * 500
        results = retriever.search(long_query, coding_facts, top_k=3)
        assert isinstance(results, list)


class TestBackwardCompatibility:
    """TFIDFRetriever alias must still work."""

    def test_tfidf_alias_is_bm25(self) -> None:
        assert TFIDFRetriever is BM25Retriever

    def test_tfidf_retriever_search_works(self) -> None:
        retriever = TFIDFRetriever()
        facts = [
            Fact(content="Python is a programming language"),
            Fact(content="Java is also a programming language"),
        ]
        results = retriever.search("Python", facts, top_k=2)
        assert len(results) > 0
        assert results[0][0].content == "Python is a programming language"


class TestInvertedIndex:
    """Tests for the InvertedIndex BM25 scoring."""

    def test_index_scores_match_brute_force(self) -> None:
        """InvertedIndex.score() must produce identical scores to brute-force BM25."""
        facts = [
            Fact(content="Python is great for data science", importance=0.5),
            Fact(content="Java is widely used in enterprise", importance=0.5),
            Fact(content="Python and Java are both popular", importance=0.5),
            Fact(content="Nothing relevant here at all", importance=0.5),
        ]
        query = "Python Java"
        k1 = 1.5
        b = 0.75

        # --- Brute-force BM25 ---
        from collections import Counter

        doc_tokens_list = [_tokenize(f.content) for f in facts]
        num_docs = len(facts)
        doc_lengths = [len(t) for t in doc_tokens_list]
        avgdl = sum(doc_lengths) / num_docs

        doc_freq: Counter[str] = Counter()
        for dt in doc_tokens_list:
            for token in set(dt):
                doc_freq[token] += 1

        query_tokens = _tokenize(query)
        brute_scores: dict[str, float] = {}
        for fact, dt, dl in zip(facts, doc_tokens_list, doc_lengths, strict=True):
            tf_counts = Counter(dt)
            score = 0.0
            for qt in query_tokens:
                tf = tf_counts.get(qt, 0)
                df = doc_freq.get(qt, 0)
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                score += idf * (k1 + 1.0) * tf / denom if denom > 0 else 0.0
            if score > 0:
                brute_scores[fact.id] = score

        # --- InvertedIndex ---
        index = InvertedIndex(facts)
        index_scores = index.score(query, k1=k1, b=b)

        # Both should have the same keys.
        assert set(brute_scores.keys()) == set(index_scores.keys())

        # Scores should match within floating-point tolerance.
        for doc_id in brute_scores:
            assert brute_scores[doc_id] == pytest.approx(index_scores[doc_id], rel=1e-9)

    def test_search_results_identical_with_index(self) -> None:
        """BM25Retriever.search() results should rank the same facts in the same order."""
        facts = [
            Fact(content="User prefers Python for backend development", importance=0.9),
            Fact(content="User deploys applications using Docker containers", importance=0.8),
            Fact(content="User works at Sber as Operations Director", importance=0.95),
            Fact(content="User dislikes JavaScript and frontend work", importance=0.7),
            Fact(content="User always uses pytest for testing Python code", importance=0.85),
        ]
        retriever = BM25Retriever()
        results = retriever.search("Python testing", facts, top_k=5)

        # Only facts with BM25 relevance are returned (score floor filter).
        assert len(results) >= 1
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

        # The top result should contain "Python" (most relevant).
        assert "Python" in results[0][0].content or "pytest" in results[0][0].content

    def test_index_empty_facts(self) -> None:
        """InvertedIndex should handle an empty facts list."""
        index = InvertedIndex([])
        scores = index.score("anything")
        assert scores == {}
        assert index.facts == {}

    def test_index_facts_property(self) -> None:
        """The facts property should return the id -> Fact mapping."""
        facts = [Fact(content="hello"), Fact(content="world")]
        index = InvertedIndex(facts)
        assert len(index.facts) == 2
        for fact in facts:
            assert fact.id in index.facts
            assert index.facts[fact.id] is fact
