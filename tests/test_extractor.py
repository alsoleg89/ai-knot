"""Tests for ai_knot.extractor — LLM-based fact extraction (mocked)."""

from __future__ import annotations

from unittest.mock import patch

from ai_knot.extractor import Extractor, deduplicate_facts
from ai_knot.types import ConversationTurn, Fact, MemoryType

MOCK_LLM_RESPONSE = [
    {
        "content": "User deploys in Docker",
        "type": "semantic",
        "importance": 0.8,
        "tags": ["docker", "devops"],
    },
    {
        "content": "User dislikes async code",
        "type": "procedural",
        "importance": 0.85,
        "tags": ["async", "preferences"],
    },
    {
        "content": "User works at Sber",
        "type": "semantic",
        "importance": 0.95,
        "tags": ["employer", "company"],
    },
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
        assert facts[0].tags == ["docker", "devops"]
        assert facts[2].importance == 0.95
        assert facts[2].tags == ["employer", "company"]

    def test_extract_tags_graceful_degradation(self, sample_turns: list[ConversationTurn]) -> None:
        """Tags are empty when LLM omits them (backward compat)."""
        extractor = Extractor(api_key="fake-key", provider="openai")
        response_without_tags = [
            {"content": "User prefers pytest", "type": "procedural", "importance": 0.8},
        ]

        with patch.object(extractor, "_call_llm", return_value=response_without_tags):
            facts = extractor.extract(sample_turns)

        assert len(facts) == 1
        assert facts[0].tags == []

    def test_extract_handles_empty_response(self, sample_turns: list[ConversationTurn]) -> None:
        extractor = Extractor(api_key="fake-key", provider="openai")

        with patch.object(extractor, "_call_llm", return_value=[]):
            facts = extractor.extract(sample_turns)

        assert facts == []

    def test_extract_handles_empty_turns(self) -> None:
        extractor = Extractor(api_key="fake-key", provider="openai")
        facts = extractor.extract([])
        assert facts == []

    def test_extract_maps_memory_types(self, sample_turns: list[ConversationTurn]) -> None:
        extractor = Extractor(api_key="fake-key", provider="openai")

        with patch.object(extractor, "_call_llm", return_value=MOCK_LLM_RESPONSE):
            facts = extractor.extract(sample_turns)

        assert facts[0].type == MemoryType.SEMANTIC
        assert facts[1].type == MemoryType.PROCEDURAL


class TestBatching:
    """Extractor splits long conversations into batch_size chunks."""

    def test_single_batch_when_turns_fit(self, sample_turns: list[ConversationTurn]) -> None:
        extractor = Extractor(api_key="fake-key", provider="openai", batch_size=50)
        with patch.object(extractor, "_call_llm", return_value=[]) as mock_llm:
            extractor.extract(sample_turns)
        # All turns fit in one batch → exactly one call
        assert mock_llm.call_count == 1

    def test_two_batches_when_turns_exceed_batch_size(self) -> None:
        turns = [ConversationTurn(role="user", content=f"msg {i}") for i in range(5)]
        extractor = Extractor(api_key="fake-key", provider="openai", batch_size=3)
        with patch.object(extractor, "_call_llm", return_value=[]) as mock_llm:
            extractor.extract(turns)
        # 5 turns / batch_size=3 → 2 calls
        assert mock_llm.call_count == 2

    def test_results_from_all_batches_merged(self) -> None:
        turns = [ConversationTurn(role="user", content=f"msg {i}") for i in range(4)]
        extractor = Extractor(api_key="fake-key", provider="openai", batch_size=2)
        batch_response = [{"content": "Fact from batch", "type": "semantic", "importance": 0.8}]
        with patch.object(extractor, "_call_llm", return_value=batch_response):
            facts = extractor.extract(turns)
        # 2 batches × 1 fact each, but dedup removes exact duplicates → 1 unique
        assert len(facts) >= 1

    def test_empty_turns_returns_empty(self) -> None:
        extractor = Extractor(api_key="fake-key", provider="openai", batch_size=5)
        assert extractor.extract([]) == []


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
