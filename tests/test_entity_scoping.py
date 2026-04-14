"""Tests for entity-scoped retrieval (Phase 1-4).

Covers: entity dictionary, entity-mention inverted index, scoped recall,
graceful fallback, sandwich reorder, multi-hop round 2, AGGREGATION intent.
"""

from __future__ import annotations

import pathlib

import pytest

from ai_knot._query_intent import _classify_pool_query, _PoolQueryIntent
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="test_agent", storage=storage)


# ---------------------------------------------------------------------------
# Phase 1a: _build_entity_dictionary
# ---------------------------------------------------------------------------


class TestBuildEntityDictionary:
    def test_collects_entity_names(self, kb: KnowledgeBase) -> None:
        facts = [
            Fact(content="Tim reads a lot", entity="Tim", attribute="hobby"),
            Fact(content="Maria works at Google", entity="Maria", attribute="employer"),
        ]
        result = kb._build_entity_dictionary(facts)
        assert "tim" in result
        assert "maria" in result

    def test_collects_value_text(self, kb: KnowledgeBase) -> None:
        facts = [
            Fact(
                content="Tim's wife is Maria",
                entity="Tim",
                attribute="wife",
                value_text="Maria",
            ),
        ]
        result = kb._build_entity_dictionary(facts)
        assert "tim" in result
        assert "maria" in result

    def test_filters_short_entities(self, kb: KnowledgeBase) -> None:
        facts = [Fact(content="X is short", entity="X")]
        result = kb._build_entity_dictionary(facts)
        assert "x" not in result

    def test_filters_numeric_value_text(self, kb: KnowledgeBase) -> None:
        facts = [
            Fact(content="salary is 95000", entity="Tim", value_text="95000"),
            Fact(content="score is 3.14", entity="Tim", value_text="3.14"),
        ]
        result = kb._build_entity_dictionary(facts)
        assert "95000" not in result
        assert "3.14" not in result
        assert "tim" in result


# ---------------------------------------------------------------------------
# Phase 1b: _build_entity_mention_index
# ---------------------------------------------------------------------------


class TestBuildEntityMentionIndex:
    def test_indexes_content_mentions(self) -> None:
        f1 = Fact(content="Tim read a book about Python")
        f2 = Fact(content="Maria and Tim went hiking")
        f3 = Fact(content="Unrelated fact about weather")
        entity_dict = {"tim", "maria"}

        index = KnowledgeBase._build_entity_mention_index([f1, f2, f3], entity_dict)

        assert f1.id in index["tim"]
        assert f2.id in index["tim"]
        assert f2.id in index["maria"]
        assert f3.id not in index.get("tim", set())
        assert f3.id not in index.get("maria", set())

    def test_case_insensitive(self) -> None:
        f1 = Fact(content="TIM likes coffee")
        index = KnowledgeBase._build_entity_mention_index([f1], {"tim"})
        assert f1.id in index["tim"]

    def test_empty_entity_dict(self) -> None:
        f1 = Fact(content="Some content")
        index = KnowledgeBase._build_entity_mention_index([f1], set())
        assert len(index) == 0


# ---------------------------------------------------------------------------
# Phase 1c-d: Entity-scoped recall
# ---------------------------------------------------------------------------


class TestEntityScopedRecall:
    def test_scoped_recall_finds_all_entity_mentions(self, kb: KnowledgeBase) -> None:
        """Entity-scoped retrieval should find facts that mention the entity
        in content even if entity field is empty."""
        # 5 facts about Tim (mention in content), 20 noise facts
        tim_facts = [
            Fact(content="Tim read The Great Gatsby", entity="Tim", attribute="book"),
            Fact(content="Tim enjoys hiking in the mountains"),
            Fact(content="Tim's favorite color is blue"),
            Fact(content="Tim went to Stanford University"),
            Fact(content="Tim adopted a dog named Rex"),
        ]
        noise = [Fact(content=f"Completely unrelated topic number {i}") for i in range(20)]
        kb._storage.save(kb._agent_id, tim_facts + noise)

        results = kb.recall_facts("What do we know about Tim?", top_k=10)
        result_ids = {f.id for f in results}

        # All Tim facts should be in results
        for tf in tim_facts:
            assert tf.id in result_ids, f"Missing Tim fact: {tf.content}"

    def test_scoped_fallback_when_too_few(self, kb: KnowledgeBase) -> None:
        """When scoped set < top_k, fall back to full corpus search."""
        # Only 2 facts mention Tim, but top_k=5 → should fall back
        facts = [
            Fact(content="Tim likes coffee", entity="Tim", attribute="drink"),
            Fact(content="Tim reads books"),
            Fact(content="Alice works at Google"),
            Fact(content="Bob is a developer"),
            Fact(content="Carol studies physics"),
        ]
        kb._storage.save(kb._agent_id, facts)

        # top_k=5, only 2 Tim facts → scoped set too small → full corpus
        results = kb.recall_facts("Tell me about Tim", top_k=5)
        assert len(results) == 5  # all 5 facts returned (full corpus)

    def test_no_entity_falls_through(self, kb: KnowledgeBase) -> None:
        """Queries without known entities use normal full-corpus search."""
        facts = [
            Fact(content="Python is a programming language"),
            Fact(content="Java is also popular"),
            Fact(content="Rust is gaining traction"),
        ]
        kb._storage.save(kb._agent_id, facts)

        results = kb.recall_facts("What programming languages exist?", top_k=3)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Phase 2: Multi-hop round 2
# ---------------------------------------------------------------------------


class TestMultiHopEntityScoping:
    def test_multi_hop_discovers_linked_entities(self, kb: KnowledgeBase) -> None:
        """Round 2 should discover entities from round 1 results and find
        related facts."""
        # Tim → Maria (via value_text), Maria → Google (via entity)
        facts = [
            Fact(
                content="Tim's wife is Maria",
                entity="Tim",
                attribute="wife",
                value_text="Maria",
                slot_key="Tim::wife",
            ),
            Fact(
                content="Maria works at Google as an engineer",
                entity="Maria",
                attribute="employer",
                value_text="Google",
                slot_key="Maria::employer",
            ),
            Fact(
                content="Maria loves painting in her free time",
                entity="Maria",
                attribute="hobby",
                value_text="painting",
            ),
        ]
        noise = [Fact(content=f"Irrelevant filler fact {i} about random things") for i in range(20)]
        kb._storage.save(kb._agent_id, facts + noise)

        result = kb.recall("Where does Tim's wife work?", top_k=10)
        # Should find Maria and Google through multi-hop
        assert "Maria" in result
        assert "Google" in result


# ---------------------------------------------------------------------------
# Phase 3: Sandwich reorder
# ---------------------------------------------------------------------------


class TestSandwichReorder:
    def test_preserves_short_lists(self) -> None:
        pairs = [(Fact(content=f"fact {i}"), float(10 - i)) for i in range(8)]
        result = KnowledgeBase._sandwich_reorder(pairs)
        assert result == pairs

    def test_reorders_long_lists(self) -> None:
        pairs = [(Fact(content=f"fact {i}"), float(20 - i)) for i in range(15)]
        result = KnowledgeBase._sandwich_reorder(pairs)

        # Position 0 should still be the highest-scoring fact
        assert result[0] == pairs[0]

        # Positions -4 to -1 should be pairs[1:5] (high-scoring at the end)
        assert result[-4:] == pairs[1:5]

        # Middle should be the remaining facts
        assert result[1:-4] == pairs[5:]

    def test_exactly_10_items_unchanged(self) -> None:
        pairs = [(Fact(content=f"fact {i}"), float(10 - i)) for i in range(10)]
        result = KnowledgeBase._sandwich_reorder(pairs)
        assert result == pairs

    def test_11_items_reordered(self) -> None:
        pairs = [(Fact(content=f"fact {i}"), float(20 - i)) for i in range(11)]
        result = KnowledgeBase._sandwich_reorder(pairs)
        # Top-1 at start, items 1-4 at end
        assert result[0] == pairs[0]
        assert result[-4:] == pairs[1:5]


# ---------------------------------------------------------------------------
# Phase 4: AGGREGATION intent detection
# ---------------------------------------------------------------------------


class TestAggregationIntent:
    @pytest.fixture
    def entity_facts(self) -> list[Fact]:
        return [
            Fact(content="Tim reads books", entity="Tim", attribute="hobby"),
            Fact(content="Tim likes coffee", entity="Tim", attribute="drink"),
        ]

    def test_aggregation_with_list(self, entity_facts: list[Fact]) -> None:
        intent = _classify_pool_query("List all hobbies Tim has", entity_facts)
        assert intent == _PoolQueryIntent.AGGREGATION

    def test_aggregation_with_what_are(self, entity_facts: list[Fact]) -> None:
        intent = _classify_pool_query("What are Tim's favorite activities?", entity_facts)
        assert intent == _PoolQueryIntent.AGGREGATION

    def test_aggregation_with_how_many(self, entity_facts: list[Fact]) -> None:
        intent = _classify_pool_query("How many books has Tim read?", entity_facts)
        assert intent == _PoolQueryIntent.AGGREGATION

    def test_aggregation_with_tell_me_about(self, entity_facts: list[Fact]) -> None:
        intent = _classify_pool_query("Tell me about Tim's reading habits", entity_facts)
        assert intent == _PoolQueryIntent.AGGREGATION

    def test_entity_lookup_without_aggregation_vocab(self, entity_facts: list[Fact]) -> None:
        intent = _classify_pool_query("Tim's salary", entity_facts)
        assert intent == _PoolQueryIntent.ENTITY_LOOKUP

    def test_general_without_entity(self) -> None:
        facts = [Fact(content="The weather is nice")]
        intent = _classify_pool_query("How is the weather?", facts)
        assert intent == _PoolQueryIntent.GENERAL
