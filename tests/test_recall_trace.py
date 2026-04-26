"""Tests for the per-stage trace API in _execute_recall / recall_facts_with_trace."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="trace_test", storage=storage)


class TestTraceKeys:
    def test_all_stage_keys_present(self, kb: KnowledgeBase) -> None:
        """recall_facts_with_trace returns a dict with all expected stage keys."""
        kb.add("Melanie plays violin on Tuesdays", tags=["music"])
        kb.add("John likes programming in Python", tags=["tech"])
        _, trace = kb.recall_facts_with_trace("violin music", top_k=5)

        assert "stage1_candidates" in trace
        assert "stage3_rrf" in trace
        assert "stage3b_dense_guarantee" in trace
        assert "stage4a_ddsa" in trace
        assert "stage4b_mmr" in trace

    def test_stage1_channel_keys(self, kb: KnowledgeBase) -> None:
        """stage1_candidates contains channel breakdown keys."""
        kb.add("Alice likes hiking", tags=["outdoor"])
        _, trace = kb.recall_facts_with_trace("hiking", top_k=5)

        s1 = trace["stage1_candidates"]
        assert "from_bm25" in s1
        assert "from_rare_tokens" in s1
        assert "from_entity_hop" in s1
        assert "from_dense" in s1
        assert "total" in s1
        assert "dense_scores" in s1
        assert isinstance(s1["total"], int)

    def test_stage3_selected_ids_are_subset_of_pool(self, kb: KnowledgeBase) -> None:
        """All selected_ids in stage3 must be candidates from stage1."""
        for i in range(10):
            kb.add(f"Fact number {i} about cooking", tags=["food"])
        _, trace = kb.recall_facts_with_trace("cooking", top_k=5)

        all_candidates = (
            set(trace["stage1_candidates"]["from_bm25"])
            | set(trace["stage1_candidates"]["from_rare_tokens"])
            | set(trace["stage1_candidates"]["from_entity_hop"])
            | set(trace["stage1_candidates"]["from_dense"])
        )
        for fid in trace["stage3_rrf"]["selected_ids"]:
            assert fid in all_candidates, f"selected id {fid!r} not in any channel"

    def test_mmr_dropped_ids_disjoint_from_output(self, kb: KnowledgeBase) -> None:
        """Facts in stage4b_mmr.dropped_ids must not appear in final results."""
        for i in range(20):
            kb.add(f"Music fact number {i}", tags=["music"])
        pairs, trace = kb.recall_facts_with_trace("music fact", top_k=5)

        result_ids = {f.id for f, _ in pairs}
        for fid in trace["stage4b_mmr"]["dropped_ids"]:
            assert fid not in result_ids, f"dropped fact {fid!r} still in output"

    def test_trace_none_default_no_overhead(self, kb: KnowledgeBase) -> None:
        """recall_facts (no trace) still returns correct results — no regression."""
        kb.add("Alice loves hiking", tags=["outdoor"])
        results = kb.recall_facts("hiking", top_k=5)
        assert len(results) == 1
        assert "hiking" in results[0].content

    def test_single_fact_trace(self, kb: KnowledgeBase) -> None:
        """Trace works correctly with a single fact in the KB."""
        kb.add("Only fact", tags=["solo"])
        pairs, trace = kb.recall_facts_with_trace("only fact", top_k=5)
        assert len(pairs) == 1
        assert trace["stage3b_dense_guarantee"]["applied"] is False
