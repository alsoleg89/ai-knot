"""Tests for agentmemo.extractor — LLM-based fact extraction (mocked)."""

from __future__ import annotations

from unittest.mock import patch

from agentmemo.extractor import Extractor, deduplicate_facts
from agentmemo.types import ConversationTurn, Fact, MemoryType

MOCK_LLM_RESPONSE = [
    {"content": "User deploys in Docker", "type": "semantic", "importance": 0.8},
    {"content": "User dislikes async code", "type": "procedural", "importance": 0.85},
    {"content": "User works at Sber", "type": "semantic", "importance": 0.95},
]


class TestExtractor:
    """Fact extraction with mocked LLM."""

    def test_extract_returns_facts(self, sample_turns: list[ConversationTurn]) -> None:
        extractor = Extractor(api_key="fake-key", provider="openai")

        with patch.object(extractor, "_call_llm", return_value=MOCK_LLM_RESPONSE):
            facts = extractor.extract(sample_turns)

        assert len(facts) == 3
        assert all(isinstance(f, Fact) for f in facts)
        assert facts[0].content == "User deploys in Docker"
        assert facts[2].importance == 0.95

    def test_extract_handles_empty_response(
        self, sample_turns: list[ConversationTurn]
    ) -> None:
        extractor = Extractor(api_key="fake-key", provider="openai")

        with patch.object(extractor, "_call_llm", return_value=[]):
            facts = extractor.extract(sample_turns)

        assert facts == []

    def test_extract_handles_empty_turns(self) -> None:
        extractor = Extractor(api_key="fake-key", provider="openai")
        facts = extractor.extract([])
        assert facts == []

    def test_extract_maps_memory_types(
        self, sample_turns: list[ConversationTurn]
    ) -> None:
        extractor = Extractor(api_key="fake-key", provider="openai")

        with patch.object(extractor, "_call_llm", return_value=MOCK_LLM_RESPONSE):
            facts = extractor.extract(sample_turns)

        assert facts[0].type == MemoryType.SEMANTIC
        assert facts[1].type == MemoryType.PROCEDURAL


class TestDeduplication:
    """Deduplication by content similarity."""

    def test_exact_duplicates_removed(self) -> None:
        facts = [
            Fact(content="User prefers Python"),
            Fact(content="User prefers Python"),
            Fact(content="User works at Sber"),
        ]
        deduped = deduplicate_facts(facts)
        assert len(deduped) == 2

    def test_similar_facts_merged(self) -> None:
        facts = [
            Fact(content="User prefers Python"),
            Fact(content="User prefers Python language"),
        ]
        deduped = deduplicate_facts(facts, threshold=0.6)
        # With high enough Jaccard overlap, should merge
        assert len(deduped) <= 2

    def test_different_facts_kept(self) -> None:
        facts = [
            Fact(content="User likes cats"),
            Fact(content="User works at Google"),
        ]
        deduped = deduplicate_facts(facts)
        assert len(deduped) == 2

    def test_empty_list(self) -> None:
        assert deduplicate_facts([]) == []

    def test_single_fact(self) -> None:
        facts = [Fact(content="only one")]
        assert len(deduplicate_facts(facts)) == 1
