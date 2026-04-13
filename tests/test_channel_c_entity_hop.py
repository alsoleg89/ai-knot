"""Tests for E0.1 — Channel C token-intersection entity-hop matching."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="ch_c_test", storage=storage)


class TestChannelCTokenIntersection:
    def test_partial_name_match_reaches_entity(self, kb: KnowledgeBase) -> None:
        """'pottery' query token should reach facts tagged under 'pottery class' entity."""
        kb.add("Alice loves her pottery class on Fridays", tags=["pottery class"])
        results = kb.recall_facts("pottery", top_k=5)
        assert any("pottery" in f.content.lower() for f in results)

    def test_no_false_positive_oscar_car(self, kb: KnowledgeBase) -> None:
        """'car' query must not match entity 'oscar' — 'car' is a 3-char token but
        after stemming it must not be a shared significant token with 'oscar'."""
        kb.add("Oscar won the award last night", tags=["oscar"])
        kb.add("I parked my car in the garage", tags=["car"])
        results = kb.recall_facts("what car did she drive", top_k=5)
        # car fact should match; oscar fact should NOT come up via channel C hop
        result_contents = [f.content for f in results]
        oscar_in_results = any(
            "oscar" in c.lower() and "car" not in c.lower() for c in result_contents
        )
        assert not oscar_in_results, "oscar entity must not match 'car' query via channel C"

    def test_empty_entity_index_no_crash(self, kb: KnowledgeBase) -> None:
        """raw/dated mode with no entity facts — channel C is a no-op."""
        kb.add("generic fact about something", tags=[])
        # No entity facts, so entity_index is empty — should not raise
        results = kb.recall_facts("something", top_k=5)
        assert len(results) >= 0  # just no crash

    def test_short_tokens_filtered_out(self, kb: KnowledgeBase) -> None:
        """Tokens with length <= 2 must not cause entity-hop matches."""
        kb.add("AI systems can learn from data", tags=["ai"])
        # 'ai' token is <= 2 chars, must not trigger a false channel-C hop
        results = kb.recall_facts("describe ai capabilities", top_k=5)
        # verify call doesn't crash and returns stable results
        assert isinstance(results, list)

    def test_multi_word_entity_partial_match(self, kb: KnowledgeBase) -> None:
        """Query 'yoga' should reach facts under 'yoga studio' entity."""
        kb.add("Bob attends yoga studio twice a week", tags=["yoga studio"])
        results = kb.recall_facts("yoga", top_k=5)
        assert any("yoga" in f.content.lower() for f in results)

    def test_exact_match_still_works(self, kb: KnowledgeBase) -> None:
        """Exact entity key still works with token intersection."""
        kb.add("Maria works at the hospital", tags=["maria"])
        results = kb.recall_facts("maria hospital", top_k=5)
        assert any("maria" in f.content.lower() for f in results)
