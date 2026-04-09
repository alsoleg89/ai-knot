"""Tests for multi_agent.assembly — coverage-aware fact selection."""

from __future__ import annotations

from ai_knot.multi_agent.assembly import CoverageAwareAssembler
from ai_knot.multi_agent.models import CandidateFact
from ai_knot.multi_agent.scoring import DiversityPolicy
from ai_knot.types import Fact


def _candidate(
    content: str,
    *,
    facet_id: str,
    score: float,
    agent: str = "agent_a",
    specificity: float = 0.7,
    near_miss: float = 0.0,
) -> CandidateFact:
    f = Fact(content=content)
    f.origin_agent_id = agent
    return CandidateFact(
        fact=f,
        base_score=score,
        facet_scores={facet_id: score},
        specificity_score=specificity,
        near_miss_penalty=near_miss,
    )


class TestCoverageAwareAssembler:
    def setup_method(self) -> None:
        self.assembler = CoverageAwareAssembler(diversity=DiversityPolicy())

    def test_covers_all_facets(self) -> None:
        candidates = {
            "f0": [_candidate("crypto shard", facet_id="f0", score=0.8, agent="a1")],
            "f1": [_candidate("graph shard", facet_id="f1", score=0.7, agent="a2")],
            "f2": [_candidate("satellite shard", facet_id="f2", score=0.6, agent="a3")],
        }
        result = self.assembler.assemble(candidates_by_facet=candidates, top_k=10)
        assert result.coverage_score == 1.0
        assert len(result.selected) == 3
        assert result.uncovered_facets == set()

    def test_diversity_cap_prevents_monopoly(self) -> None:
        candidates = {
            "f0": [
                _candidate("fact1", facet_id="f0", score=0.9, agent="a1"),
                _candidate("fact2", facet_id="f0", score=0.8, agent="a1"),
                _candidate("fact3", facet_id="f0", score=0.7, agent="a1"),
                _candidate("fact4", facet_id="f0", score=0.6, agent="a1"),
                _candidate("fact5", facet_id="f0", score=0.5, agent="a1"),
            ],
            "f1": [_candidate("other", facet_id="f1", score=0.3, agent="a2")],
        }
        result = self.assembler.assemble(candidates_by_facet=candidates, top_k=5, n_publishers=3)
        # Agent a1 shouldn't fill all 5 slots.
        a1_count = sum(1 for c in result.selected if c.fact.origin_agent_id == "a1")
        assert a1_count < 5

    def test_near_miss_deprioritised(self) -> None:
        candidates = {
            "f0": [
                _candidate("overview", facet_id="f0", score=0.9, agent="a1", near_miss=0.6),
                _candidate("specific", facet_id="f0", score=0.7, agent="a2", near_miss=0.0),
            ],
        }
        result = self.assembler.assemble(candidates_by_facet=candidates, top_k=2)
        # The specific fact should be ranked higher despite lower base score.
        assert result.selected[0].fact.content == "specific"

    def test_empty_candidates(self) -> None:
        result = self.assembler.assemble(candidates_by_facet={}, top_k=5)
        assert result.selected == []
        assert result.coverage_score == 0.0

    def test_backfill_after_coverage(self) -> None:
        candidates = {
            "f0": [
                _candidate("shard1", facet_id="f0", score=0.9, agent="a1"),
                _candidate("shard1b", facet_id="f0", score=0.5, agent="a3"),
            ],
            "f1": [
                _candidate("shard2", facet_id="f1", score=0.8, agent="a2"),
            ],
        }
        result = self.assembler.assemble(candidates_by_facet=candidates, top_k=5)
        assert result.coverage_score == 1.0
        # Should have at least 2 (coverage) + backfill.
        assert len(result.selected) >= 2

    def test_constrained_facet_first(self) -> None:
        """Facet with fewer candidates should be served first."""
        candidates = {
            "f0": [
                _candidate("easy1", facet_id="f0", score=0.9, agent="a1"),
                _candidate("easy2", facet_id="f0", score=0.8, agent="a2"),
            ],
            "f1": [
                _candidate("rare", facet_id="f1", score=0.5, agent="a3"),
            ],
        }
        result = self.assembler.assemble(candidates_by_facet=candidates, top_k=3)
        assert result.coverage_score == 1.0
        # The rare fact must be selected.
        selected_contents = [c.fact.content for c in result.selected]
        assert "rare" in selected_contents
