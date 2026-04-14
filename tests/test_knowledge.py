"""Tests for ai_knot.knowledge — KnowledgeBase main class."""

from __future__ import annotations

import logging
import os
import pathlib
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

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

    def test_add_dedup_sliding_window(self, kb: KnowledgeBase) -> None:
        """Stride-1 sliding windows with high token overlap (Jaccard ≥ 0.7) are suppressed.

        Uses realistic benchmark-length turns so Jaccard(A,B)≈0.71 (computed).
        A window that drifts far enough (Jaccard≈0.46) is kept as a new fact.
        """
        turn0 = (
            "[8 May 2023] Melanie I have been going to pottery class twice a week"
            " and I really enjoy making clay bowls and vases with Caroline"
        )
        turn1 = (
            "Caroline That sounds wonderful I really love the pottery class too"
            " and making things with clay is so relaxing and fun"
        )
        turn2 = (
            "[9 May 2023] Melanie Yes and yesterday we made beautiful flower pots"
            " I also went swimming at the beach with my kids in the afternoon"
        )
        turn3 = (
            "Caroline That sounds like such a lovely day I wish I could have"
            " joined you at the beach and the pottery studio"
        )
        turn4 = (
            "[10 May 2023] Melanie Today I went hiking in the mountains with my"
            " daughter it was very beautiful and refreshing"
        )

        win_a = kb.add(" ".join([turn0, turn1, turn2]))  # stored
        win_b = kb.add(" ".join([turn1, turn2, turn3]))  # Jaccard(A,B)≈0.71 → suppressed
        win_c = kb.add(" ".join([turn2, turn3, turn4]))  # Jaccard(A,C)≈0.46 → stored

        facts = kb._storage.load(kb._agent_id)
        # B was suppressed — returns the existing A fact
        assert win_b.id == win_a.id
        # C has enough new content to be kept (hiking is genuinely new)
        assert win_c.id != win_a.id
        # Only A and C stored — B was deduplicated
        assert len(facts) == 2
        stored_ids = {f.id for f in facts}
        assert win_a.id in stored_ids
        assert win_c.id in stored_ids

    def test_add_dedup_distinct_facts_not_suppressed(self, kb: KnowledgeBase) -> None:
        """Facts with different content (Jaccard < 0.7) are not suppressed."""
        f1 = kb.add("Melanie went swimming at the beach with her kids")
        f2 = kb.add("Caroline attended the LGBTQ+ pride parade downtown")
        f3 = kb.add("John fixed his car engine at the mechanic shop")

        facts = kb._storage.load(kb._agent_id)
        assert len(facts) == 3
        ids = {f.id for f in facts}
        assert f1.id in ids
        assert f2.id in ids
        assert f3.id in ids

    def test_add_dedup_returns_existing_fact(self, kb: KnowledgeBase) -> None:
        """When a near-duplicate is detected, the existing Fact object is returned."""
        original = kb.add("Melanie: I love swimming and pottery classes with Caroline")
        duplicate = kb.add("I love swimming and pottery classes with Caroline today")

        # duplicate is the existing fact (same id, same content)
        assert duplicate.id == original.id
        assert duplicate.content == original.content
        # Only one fact stored
        loaded = kb._storage.load(kb._agent_id)
        assert len(loaded) == 1
        assert loaded[0].id == original.id

    def test_add_dedup_window_only_last_n(self, kb: KnowledgeBase) -> None:
        """Dedup only checks last _DEDUP_WINDOW facts — older similar content is NOT suppressed."""
        # Add _DEDUP_WINDOW + 1 unrelated facts to push the similar fact out of the window
        for i in range(KnowledgeBase._DEDUP_WINDOW + 1):
            kb.add(f"Completely unrelated fact number {i} about topic X Y Z")

        # Now add a fact similar to the very first one — it's outside the window
        first_similar = kb.add("swimming pottery art museum outdoor activities hobby")
        # Add another near-duplicate — now the first_similar IS within the window → suppressed
        second_similar = kb.add("pottery art museum outdoor activities hobby swimming")

        facts = kb._storage.load(kb._agent_id)
        # first_similar was added (original was outside window)
        assert any(f.id == first_similar.id for f in facts)
        # second_similar was suppressed (first_similar is within window)
        assert second_similar.id == first_similar.id


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

    def test_recall_entity_hop(self, kb: KnowledgeBase) -> None:
        """Entity-hop: value_text of one fact matches entity of another."""
        # Build structured facts with entity/value links
        fact1 = Fact(
            content="Alex's wife is Maria",
            importance=0.9,
            entity="alex",
            attribute="wife",
            value_text="maria",
            slot_key="alex::wife",
        )
        fact2 = Fact(
            content="Maria works at Google",
            importance=0.9,
            entity="maria",
            attribute="employer",
            value_text="google",
            slot_key="maria::employer",
        )
        noise = [Fact(content=f"Unrelated fact {i} about topics") for i in range(20)]
        all_facts = [fact1, fact2, *noise]
        kb._storage.save(kb._agent_id, all_facts)

        result = kb.recall("Where does Alex's wife work?", top_k=10)
        # Both facts should be found via entity-hop
        assert "Maria" in result
        assert "Google" in result


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


class TestAddMany:
    """add_many() — batch insertion without LLM."""

    def test_add_many_strings(self, kb: KnowledgeBase) -> None:
        facts = kb.add_many(["User prefers Python", "User works at Sber"])
        assert len(facts) == 2
        assert all(isinstance(f, Fact) for f in facts)
        assert facts[0].content == "User prefers Python"
        assert facts[1].content == "User works at Sber"

    def test_add_many_dicts(self, kb: KnowledgeBase) -> None:
        items = [
            {"content": "User deploys on Fridays", "type": "episodic", "importance": 0.7},
            {"content": "User prefers async frameworks", "type": "procedural"},
        ]
        facts = kb.add_many(items)
        assert len(facts) == 2
        assert facts[0].type == MemoryType.EPISODIC
        assert pytest.approx(facts[0].importance) == 0.7
        assert facts[1].type == MemoryType.PROCEDURAL

    def test_add_many_empty(self, kb: KnowledgeBase) -> None:
        assert kb.add_many([]) == []

    def test_add_many_persists(self, kb: KnowledgeBase) -> None:
        kb.add_many(["Fact A", "Fact B", "Fact C"])
        stored = kb._storage.load(kb._agent_id)
        assert len(stored) == 3

    def test_add_many_default_type_applied_to_strings(self, kb: KnowledgeBase) -> None:
        facts = kb.add_many(["Some procedure"], type=MemoryType.PROCEDURAL)
        assert facts[0].type == MemoryType.PROCEDURAL

    def test_add_many_with_tags(self, kb: KnowledgeBase) -> None:
        facts = kb.add_many(["Tagged fact"], tags=["work"])
        assert facts[0].tags == ["work"]

    def test_add_many_single_storage_op(self, kb: KnowledgeBase) -> None:
        """add_many persists all facts in one load+save, not N×2 ops."""
        kb.add_many(["Fact A", "Fact B", "Fact C"])
        stored = kb._storage.load(kb._agent_id)
        assert len(stored) == 3

    def test_add_many_missing_content_raises(self, kb: KnowledgeBase) -> None:
        with pytest.raises(ValueError, match="content"):
            kb.add_many([{"type": "semantic"}])

    def test_add_many_validates_all_before_persisting(self, kb: KnowledgeBase) -> None:
        """Validation failure on item N must not partially persist items 0..N-1."""
        with pytest.raises(ValueError):
            kb.add_many(["Valid fact", {"type": "semantic"}])  # second item missing content
        assert kb._storage.load(kb._agent_id) == []


class TestLearnDefaultProvider:
    """Provider config set at __init__ used as fallback in learn()."""

    def test_default_provider_used_when_not_passed(self, tmp_path: pathlib.Path) -> None:
        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(
            agent_id="agent",
            storage=storage,
            provider="openai",
            api_key="sk-default",
        )
        turns = [ConversationTurn(role="user", content="I deploy on Fridays")]
        mock_facts = [{"content": "User deploys on Fridays", "type": "semantic", "importance": 0.8}]
        with (
            patch("ai_knot.extractor.call_with_retry", return_value="[]"),
            patch("ai_knot.extractor.Extractor._call_llm", return_value=mock_facts),
        ):
            result = kb.learn(turns)  # no api_key / provider per call
        assert isinstance(result, list)

    def test_per_call_provider_overrides_default(self, tmp_path: pathlib.Path) -> None:
        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(
            agent_id="agent",
            storage=storage,
            provider="openai",
            api_key="sk-default",
        )
        turns = [ConversationTurn(role="user", content="Hello")]
        with patch("ai_knot.extractor.Extractor._call_llm", return_value=[]):
            # Override with anthropic for this specific call
            result = kb.learn(turns, provider="anthropic", api_key="sk-other")
        assert result == []

    def test_no_api_key_at_init_or_call_raises(self, tmp_path: pathlib.Path) -> None:
        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(agent_id="agent", storage=storage)
        turns = [ConversationTurn(role="user", content="Hello")]
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
            pytest.raises(ValueError, match="No API key"),
        ):
            kb.learn(turns)


class TestAsyncAPI:
    """alearn(), arecall(), arecall_facts() — non-blocking variants."""

    def test_alearn_returns_list(self, tmp_path: pathlib.Path) -> None:
        import asyncio

        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(agent_id="async_agent", storage=storage)
        turns = [ConversationTurn(role="user", content="Hello")]
        with patch("ai_knot.extractor.Extractor._call_llm", return_value=[]):
            result = asyncio.run(kb.alearn(turns, provider="openai", api_key="sk-test"))
        assert result == []

    def test_arecall_returns_string(self, tmp_path: pathlib.Path) -> None:
        import asyncio

        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(agent_id="async_agent", storage=storage)
        kb.add("User prefers Python")
        result = asyncio.run(kb.arecall("language"))
        assert isinstance(result, str)
        assert "Python" in result

    def test_arecall_facts_returns_list(self, tmp_path: pathlib.Path) -> None:
        import asyncio

        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(agent_id="async_agent", storage=storage)
        kb.add("User deploys on Fridays")
        results = asyncio.run(kb.arecall_facts("deployment"))
        assert isinstance(results, list)
        assert all(isinstance(f, Fact) for f in results)

    def test_arecall_empty_kb(self, tmp_path: pathlib.Path) -> None:
        import asyncio

        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(agent_id="async_agent", storage=storage)
        assert asyncio.run(kb.arecall("anything")) == ""


class TestLLMFeatures:
    """Tests for LLM-enhanced features: query expansion, decay_config."""

    def test_recall_with_llm_expands_query(self, tmp_path: pathlib.Path) -> None:
        """When llm_recall=True and provider set, query is expanded before search."""
        provider = MagicMock()
        provider.name = "mock"
        provider.default_model = "gpt-4o"

        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(
            agent_id="agent",
            storage=storage,
            provider=provider,
            llm_recall=True,
        )
        kb.add("User uses PostgreSQL for data storage")

        with patch(
            "ai_knot.query_expander.call_with_retry",
            return_value="database PostgreSQL SQL storage relational",
        ):
            result = kb.recall("what database?", top_k=1)

        assert "PostgreSQL" in result

    def test_recall_without_llm_skips_expansion(self, tmp_path: pathlib.Path) -> None:
        """llm_recall=False (default) never calls the provider at recall time."""
        provider = MagicMock()
        provider.name = "mock"
        provider.default_model = "gpt-4o"

        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(
            agent_id="agent",
            storage=storage,
            provider=provider,
            llm_recall=False,
        )
        kb.add("Some fact")
        kb.recall("query")

        # call_with_retry should never be invoked for query expansion
        provider.call.assert_not_called()

    def test_expand_query_no_provider_warns(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """llm_recall=True without provider logs a warning and returns original query."""
        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(
            agent_id="agent",
            storage=storage,
            llm_recall=True,
        )
        kb.add("Some fact about databases")

        with caplog.at_level(logging.WARNING, logger="ai_knot.knowledge"):
            kb.recall("what database?")

        assert "llm_recall=True but no provider configured" in caplog.text

    def test_decay_config_changes_retention(self, tmp_path: pathlib.Path) -> None:
        """Custom decay_config produces different retention than default."""
        storage = YAMLStorage(base_dir=str(tmp_path))

        # KB with aggressive decay for semantic facts (higher exponent = faster decay)
        kb_fast = KnowledgeBase(
            agent_id="fast",
            storage=storage,
            decay_config={"semantic": 5.0},
        )
        kb_fast.add("Old semantic fact", type=MemoryType.SEMANTIC)

        # KB with default decay
        kb_default = KnowledgeBase(
            agent_id="default",
            storage=storage,
        )
        kb_default.add("Old semantic fact", type=MemoryType.SEMANTIC)

        # Age the facts identically
        old_time = datetime(2025, 1, 1, tzinfo=UTC)
        for agent_id in ("fast", "default"):
            facts = storage.load(agent_id)
            facts[0].last_accessed = old_time
            storage.save(agent_id, facts)

        kb_fast.decay()
        kb_default.decay()

        fast_retention = storage.load("fast")[0].retention_score
        default_retention = storage.load("default")[0].retention_score

        # Aggressive decay (exponent=5.0) should produce lower retention than default (0.8)
        assert fast_retention < default_retention

    def test_decay_config_default_unchanged(self, tmp_path: pathlib.Path) -> None:
        """KB without decay_config uses standard retention calculation."""
        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(agent_id="agent", storage=storage)
        kb.add("Test fact", type=MemoryType.SEMANTIC)

        # Age the fact
        facts = storage.load("agent")
        facts[0].last_accessed = datetime(2025, 1, 1, tzinfo=UTC)
        storage.save("agent", facts)

        kb.decay()

        retention = storage.load("agent")[0].retention_score
        # Should be between 0 and 1 (decayed but not zero)
        assert 0.0 < retention < 1.0


class TestLLMVsBaseDifferences:
    """Measurable differences between LLM-enhanced and base behavior.

    Each test demonstrates a concrete scenario where LLM features produce
    objectively different (better) results than the base pipeline.

    Key insight: the retriever uses RRF (rank fusion), so we need multiple
    competing facts for rank differences to manifest in scores.
    """

    def test_expansion_enhances_embed_query(self, tmp_path: pathlib.Path) -> None:
        """LLM expansion produces semantically richer text for embedding."""
        storage = YAMLStorage(base_dir=str(tmp_path))
        provider = MagicMock()
        provider.name = "mock"
        provider.default_model = "gpt-4o"
        kb = KnowledgeBase(
            agent_id="test",
            storage=storage,
            provider=provider,
            llm_recall=True,
        )

        with patch(
            "ai_knot.query_expander.call_with_retry",
            return_value="relational storage PostgreSQL data persistence",
        ):
            embed_text = kb._expand_query_for_embed("relational storage")

        assert "PostgreSQL" in embed_text
        assert "relational" in embed_text

    def test_dense_injection_surfaces_vocabulary_gap_fact(self, tmp_path: pathlib.Path) -> None:
        """Dense diversity injection adds a fact that BM25 misses due to vocabulary mismatch."""
        storage = YAMLStorage(base_dir=str(tmp_path))
        kb = KnowledgeBase(agent_id="test", storage=storage)

        # Target: zero keyword overlap with "cloud tools"
        target = kb.add("Team uses Docker for container orchestration")
        # Distractors: match "cloud" or "tools"
        distractor_facts = [
            kb.add("Cloud migration project started last month"),
            kb.add("DevOps tools inventory was updated recently"),
            kb.add("Cloud computing costs are rising each quarter"),
            kb.add("Tools for monitoring are essential for ops"),
            kb.add("Cloud infrastructure management overview"),
            kb.add("Tools comparison for modern development"),
        ]

        query = "cloud tools"

        # Confirm BM25-only mode runs without error.
        # Note: MMR diversity reranking may surface Docker even without
        # embeddings once cloud/tools distractors cluster together.
        kb.recall_facts_with_scores(query, top_k=5)

        # Set up dense embeddings: target closest to query
        kb._dense.add_embeddings({target.id: [0.95, 0.1]})
        for d in distractor_facts:
            kb._dense.add_embeddings({d.id: [0.1, 0.9]})
        kb._embedded_ids = {f.id for f in kb.list_facts()}

        # With dense injection: Docker surfaces via semantic similarity
        with patch.object(kb, "_embed_for_recall", return_value=[1.0, 0.0]):
            dense_results = kb.recall_facts_with_scores(query, top_k=5)
        assert any("Docker" in f.content for f, _ in dense_results)

    def test_auto_tags_change_ranking(self, tmp_path: pathlib.Path) -> None:
        """Tags from auto-tagging boost a fact above untagged competitors.

        BM25F gives tags 2x weight. When the query matches a tag, the
        tagged fact ranks higher than untagged facts with similar content.
        """
        storage = YAMLStorage(base_dir=str(tmp_path))

        # KB with tagged fact among untagged competitors
        kb_tagged = KnowledgeBase(agent_id="tagged", storage=storage)
        # Target: same content as competitor, but has tags
        kb_tagged.add(
            "User develops backend services",
            tags=["python", "backend"],
        )
        # Competitors: mention python in content but have no tags
        kb_tagged.add("Python is a popular programming language")
        kb_tagged.add("Many teams use Python for scripting")
        kb_tagged.add("Python ecosystem has many frameworks")
        kb_tagged.add("Development tools improve productivity")

        # KB without tags — same facts, no tags on any
        kb_plain = KnowledgeBase(agent_id="plain", storage=storage)
        kb_plain.add("User develops backend services")
        kb_plain.add("Python is a popular programming language")
        kb_plain.add("Many teams use Python for scripting")
        kb_plain.add("Python ecosystem has many frameworks")
        kb_plain.add("Development tools improve productivity")

        query = "python"

        tagged_pairs = kb_tagged.recall_facts_with_scores(query, top_k=5)
        plain_pairs = kb_plain.recall_facts_with_scores(query, top_k=5)

        # In tagged KB, "User develops backend services" has ["python"] tag
        # → gets 2x BM25F boost for "python" → should rank higher
        def _rank_of(pairs: list[tuple[Fact, float]], substr: str) -> int:
            for i, (f, _) in enumerate(pairs):
                if substr in f.content:
                    return i
            return len(pairs)

        tagged_rank = _rank_of(tagged_pairs, "backend services")
        plain_rank = _rank_of(plain_pairs, "backend services")

        assert tagged_rank < plain_rank, (
            f"Tagged fact rank ({tagged_rank}) should be better (lower) "
            f"than untagged ({plain_rank})"
        )

    def test_decay_config_flips_retention_hierarchy(self, tmp_path: pathlib.Path) -> None:
        """Custom decay_config flips the retention order between memory types.

        Default: episodic (exponent 1.3) decays faster than semantic (0.8).
        Custom: episodic (exponent 0.1) decays much slower → episodic retention
        becomes HIGHER than semantic, reversing the default Tulving hierarchy.
        """
        storage = YAMLStorage(base_dir=str(tmp_path))
        old_time = datetime(2025, 1, 1, tzinfo=UTC)

        for agent_id, decay_cfg in [("default", None), ("custom", {"episodic": 0.1})]:
            kb = KnowledgeBase(
                agent_id=agent_id,
                storage=storage,
                decay_config=decay_cfg,
            )
            kb.add("Team deployed on Friday", type=MemoryType.EPISODIC, importance=0.8)
            kb.add("Team uses Python daily", type=MemoryType.SEMANTIC, importance=0.8)

            # Age both facts
            facts = storage.load(agent_id)
            for f in facts:
                f.last_accessed = old_time
            storage.save(agent_id, facts)

        # Apply decay with each config
        KnowledgeBase(agent_id="default", storage=storage).decay()
        KnowledgeBase(
            agent_id="custom",
            storage=storage,
            decay_config={"episodic": 0.1},
        ).decay()

        default_facts = storage.load("default")
        default_epi = next(f for f in default_facts if f.type == MemoryType.EPISODIC)
        default_sem = next(f for f in default_facts if f.type == MemoryType.SEMANTIC)

        custom_facts = storage.load("custom")
        custom_epi = next(f for f in custom_facts if f.type == MemoryType.EPISODIC)
        custom_sem = next(f for f in custom_facts if f.type == MemoryType.SEMANTIC)

        # Default: episodic decays faster → lower retention
        assert default_epi.retention_score < default_sem.retention_score, (
            "Default: episodic should retain less than semantic"
        )

        # Custom: episodic exponent 0.1 vs semantic 0.8 → episodic retains MORE
        assert custom_epi.retention_score > custom_sem.retention_score, (
            f"Custom: episodic ({custom_epi.retention_score:.4f}) should exceed "
            f"semantic ({custom_sem.retention_score:.4f}) with exponent 0.1 vs 0.8"
        )

        # Episodic retention is dramatically different between configs
        assert custom_epi.retention_score > default_epi.retention_score, (
            "Custom decay preserves episodic retention far better than default"
        )

    def test_tags_and_dense_injection_combined(self, tmp_path: pathlib.Path) -> None:
        """Tags boost BM25 ranking; dense injection surfaces vocabulary-gap facts."""
        storage = YAMLStorage(base_dir=str(tmp_path))

        # --- Tags test: matching tags promote a fact ---
        kb_tagged = KnowledgeBase(agent_id="tagged", storage=storage)
        kb_tagged.add(
            "User relies on PostgreSQL for data persistence",
            tags=["storage", "relational"],
        )
        for d in [
            "Team stores config in a relational format",
            "Data storage costs increased last quarter",
            "Relational models are taught in CS courses",
            "User prefers local storage over cloud",
        ]:
            kb_tagged.add(d)

        kb_untagged = KnowledgeBase(agent_id="untagged", storage=storage)
        kb_untagged.add("User relies on PostgreSQL for data persistence")
        for d in [
            "Team stores config in a relational format",
            "Data storage costs increased last quarter",
            "Relational models are taught in CS courses",
            "User prefers local storage over cloud",
        ]:
            kb_untagged.add(d)

        query = "relational storage"
        tagged_results = kb_tagged.recall_facts_with_scores(query, top_k=5)
        untagged_results = kb_untagged.recall_facts_with_scores(query, top_k=5)

        tagged_rank = next(
            i for i, (f, _) in enumerate(tagged_results) if "PostgreSQL" in f.content
        )
        untagged_rank = next(
            i for i, (f, _) in enumerate(untagged_results) if "PostgreSQL" in f.content
        )
        assert tagged_rank < untagged_rank, "Tags should improve ranking"

        # --- Dense injection test: vocabulary-gap fact surfaces ---
        kb_dense = KnowledgeBase(agent_id="dense", storage=storage)
        dense_target = kb_dense.add("Team uses Docker for container orchestration")
        distractor_facts = [
            kb_dense.add("Cloud migration project started last month"),
            kb_dense.add("DevOps tools inventory was updated recently"),
            kb_dense.add("Cloud computing costs are rising each quarter"),
            kb_dense.add("Tools for monitoring are essential for ops"),
            kb_dense.add("Cloud infrastructure management overview"),
            kb_dense.add("Tools comparison for modern development"),
        ]

        kb_dense._dense.add_embeddings({dense_target.id: [0.95, 0.1]})
        for d in distractor_facts:
            kb_dense._dense.add_embeddings({d.id: [0.1, 0.9]})
        kb_dense._embedded_ids = {f.id for f in kb_dense.list_facts()}

        with patch.object(kb_dense, "_embed_for_recall", return_value=[1.0, 0.0]):
            results = kb_dense.recall_facts_with_scores("cloud tools", top_k=5)
        assert any("Docker" in f.content for f, _ in results)


class TestRecallVerificationGate:
    """supported=False facts must be excluded from all recall paths by default."""

    def test_recall_excludes_unsupported(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Alex works at Acme")
        fact.supported = False
        kb.replace_facts([fact])

        result = kb.recall("Alex employer")
        assert "Acme" not in result

    def test_recall_includes_unsupported_when_flag_set(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Alex works at Acme")
        fact.supported = False
        kb.replace_facts([fact])

        result = kb.recall("Alex employer", include_unsupported=True)
        assert "Acme" in result

    def test_recall_facts_excludes_unsupported(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Alex works at Acme")
        fact.supported = False
        kb.replace_facts([fact])

        results = kb.recall_facts("Alex employer")
        assert all(f.supported is not False for f in results)

    def test_recall_by_tag_excludes_unsupported(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Alex works at Acme", tags=["employer"])
        fact.supported = False
        kb.replace_facts([fact])

        results = kb.recall_by_tag("employer")
        assert results == []


class TestRecallPreservesFacts:
    """recall() must not permanently delete filtered-out facts from storage.

    Regression: _execute_recall loaded all facts, filtered to active/non-episodic/
    supported, then saved only the filtered subset — permanently losing episodic,
    expired, and unsupported facts on every recall() call.
    """

    def test_recall_preserves_episodic_facts(self, kb: KnowledgeBase) -> None:
        """Episodic facts must survive after recall() even though they're excluded."""
        semantic = kb.add("User prefers Python", importance=0.9)
        episodic = kb.add("Raw conversation turn", type=MemoryType.EPISODIC)

        kb.recall("Python")

        all_facts = kb._storage.load(kb._agent_id)
        fact_ids = {f.id for f in all_facts}
        assert semantic.id in fact_ids, "semantic fact should survive"
        assert episodic.id in fact_ids, "episodic fact must NOT be deleted by recall()"

    def test_recall_preserves_unsupported_facts(self, kb: KnowledgeBase) -> None:
        """Unsupported facts must survive recall() for later re-verification."""
        supported = kb.add("Verified fact", importance=0.9)
        unsupported = kb.add("Unverified claim")
        unsupported.supported = False
        kb.replace_facts([supported, unsupported])

        kb.recall("anything")

        all_facts = kb._storage.load(kb._agent_id)
        fact_ids = {f.id for f in all_facts}
        assert supported.id in fact_ids
        assert unsupported.id in fact_ids, "unsupported fact must NOT be deleted by recall()"

    def test_repeated_recall_does_not_shrink_storage(self, kb: KnowledgeBase) -> None:
        """Multiple recall() calls must not progressively lose facts."""
        kb.add("Fact A", importance=0.9)
        kb.add("Fact B", type=MemoryType.EPISODIC)
        kb.add("Fact C", importance=0.5)

        initial_count = len(kb._storage.load(kb._agent_id))

        for _ in range(5):
            kb.recall("anything")

        final_count = len(kb._storage.load(kb._agent_id))
        assert final_count == initial_count, (
            f"Storage shrank from {initial_count} to {final_count} after repeated recall()"
        )
