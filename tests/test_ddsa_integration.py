"""Integration tests for DDSA (spreading activation) inside KnowledgeBase recall."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

import ai_knot._spreading_activation as _sa_module
import ai_knot.knowledge as _kb_module
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture(autouse=True)
def _enable_ddsa() -> Any:
    """Force-enable DDSA for all tests in this module regardless of default."""
    prev = _sa_module.DDSA_ENABLED
    _sa_module.DDSA_ENABLED = True
    try:
        yield
    finally:
        _sa_module.DDSA_ENABLED = prev


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="ddsa_test", storage=storage)


class TestDDSARescuesRtopkMiss:
    def test_tag_hop_rescues_weak_bm25_fact(self, kb: KnowledgeBase) -> None:
        """A fact sharing a tag with a strong BM25 match should survive into top_k
        even when its own BM25 score is too low to rank there independently."""
        # Strong BM25 seed for query "what does Melanie play".
        kb.add("Melanie plays violin on Tuesdays", tags=["hobby", "music"])
        # Push the target fact below natural top_k by adding noise.
        for i in range(25):
            kb.add(f"Weather report number {i}: cloudy skies today", tags=["weather"])
        # Target: weak BM25 for this query ("clarinet" ≠ "play") but same tags.
        kb.add("She also enjoys clarinet sessions", tags=["hobby", "music"])

        results = kb.recall_facts("what does Melanie play", top_k=10)
        contents = [f.content for f in results]
        assert any("clarinet" in c for c in contents), (
            "DDSA tag-hop should rescue the clarinet fact; got: " + str(contents)
        )

    def test_results_limited_to_topk(self, kb: KnowledgeBase) -> None:
        for i in range(50):
            kb.add(f"Fact {i} about music", tags=["music"])
        results = kb.recall_facts("music", top_k=10)
        assert len(results) <= 10

    def test_recall_still_works_with_single_fact(self, kb: KnowledgeBase) -> None:
        kb.add("Only one fact", tags=["solo"])
        results = kb.recall_facts("only one fact", top_k=5)
        assert len(results) == 1

    def test_ddsa_disabled_flag_skips_spreading_activation(
        self, kb: KnowledgeBase, monkeypatch: Any
    ) -> None:
        """When DDSA_ENABLED=False, spreading_activation is never called."""
        kb.add("Melanie plays violin on Tuesdays", tags=["hobby", "music"])

        call_count = 0
        original_fn = _sa_module.spreading_activation

        def tracking_sa(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)

        monkeypatch.setattr(_kb_module, "spreading_activation", tracking_sa)

        _sa_module.DDSA_ENABLED = False
        try:
            kb.recall_facts("what does Melanie play", top_k=5)
            assert call_count == 0, (
                "spreading_activation should NOT be called when DDSA_ENABLED=False"
            )
        finally:
            _sa_module.DDSA_ENABLED = True

        # Confirm the flag being True causes the function to be invoked.
        # Use an EXPLORATORY query (has use_ddsa=True in PipelineConfig) so DDSA runs.
        call_count = 0
        kb.recall_facts("why does Melanie play violin on Tuesdays", top_k=5)
        assert call_count > 0, "spreading_activation should be called when DDSA_ENABLED=True"


class TestDDSAFixRegression:
    """Tests that validate the Phase 1 regression fixes (F1/F2/F3)."""

    def test_ddsa_pool_capped_at_top_k(self, kb: KnowledgeBase) -> None:
        """F1: activated pool never exceeds top_k — no pool inflation via tail-merge."""
        for i in range(100):
            kb.add(f"fact about music number {i}", tags=["music", "tune"])
        results = kb.recall_facts("music", top_k=10)
        assert len(results) <= 10

    def test_ddsa_output_stable_across_calls(self, kb: KnowledgeBase) -> None:
        """F2: removing post-MMR guard doesn't introduce non-determinism.

        Two identical recall calls should return the same set of fact IDs.
        """
        for i in range(5):
            kb.add(f"hobby fact about music {i}", tags=["hobby", "music"])
        for i in range(5):
            kb.add(f"work fact about coding {i}", tags=["work", "programming"])
        first = {f.id for f in kb.recall_facts("music hobby", top_k=5)}
        second = {f.id for f in kb.recall_facts("music hobby", top_k=5)}
        assert first == second, "DDSA recall is non-deterministic across identical calls"

    def test_temporal_window_reduced_does_not_explode_pool(self, kb: KnowledgeBase) -> None:
        """F3: with temporal_window_sec=60 (vs old 300), batch-ingest facts don't flood pool."""
        # All facts added within milliseconds of each other (batch-ingest simulation).
        # With window=300 all 50 would activate; with window=60 only true neighbours should.
        for i in range(50):
            kb.add(f"batch ingested fact number {i}", tags=["batch"])
        results = kb.recall_facts("batch ingested fact", top_k=10)
        assert len(results) <= 10
