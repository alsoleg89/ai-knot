"""Tests for TeamInsightStore — in-memory team-level knowledge cache."""

from __future__ import annotations

from ai_knot.multi_agent.insights import TeamInsight, TeamInsightStore
from ai_knot.multi_agent.models import AssemblyResult, CandidateFact
from ai_knot.tokenizer import tokenize
from ai_knot.types import Fact


def _make_candidate(content: str, agent_id: str, fact_id: str = "") -> CandidateFact:
    """Helper to build a CandidateFact with minimal fields."""
    f = Fact(content=content, origin_agent_id=agent_id)
    if fact_id:
        f.id = fact_id
    return CandidateFact(fact=f, base_score=1.0)


def _make_insight(
    summary: str,
    *,
    insight_id: str = "ins1",
    reuse_count: int = 0,
) -> TeamInsight:
    tokens = tuple(tokenize(summary))
    return TeamInsight(
        insight_id=insight_id,
        summary=summary,
        tokens=tokens,
        supporting_fact_ids=("f1",),
        supporting_agents=("a1", "a2"),
        reuse_count=reuse_count,
    )


class TestRememberAndRetrieve:
    def test_remember_and_retrieve(self) -> None:
        store = TeamInsightStore()
        insight = _make_insight("kubernetes pod deployment scaling")
        store.remember(insight)
        assert store.count == 1

        results = store.retrieve("kubernetes deployment")
        assert len(results) == 1
        assert results[0].insight_id == "ins1"


class TestRetrieveRanking:
    def test_retrieve_ranking(self) -> None:
        store = TeamInsightStore()
        # Insight about k8s
        k8s = _make_insight("kubernetes pod deployment scaling", insight_id="k8s")
        # Insight about database
        db = _make_insight("postgres database replication sharding", insight_id="db")
        store.remember(k8s)
        store.remember(db)

        results = store.retrieve("kubernetes deployment pods")
        assert len(results) >= 1
        assert results[0].insight_id == "k8s"


class TestPromoteFromAssembly:
    def test_promote_from_assembly(self) -> None:
        store = TeamInsightStore(min_coverage_to_promote=0.8)
        result = AssemblyResult(
            selected=[
                _make_candidate("fact about caching", "agent-1", "f1"),
                _make_candidate("fact about redis", "agent-2", "f2"),
            ],
            covered_facets={"f0", "f1"},
            uncovered_facets=set(),
            coverage_score=1.0,
        )
        insight = store.promote_from_assembly(result, query="caching redis")
        assert insight is not None
        assert store.count == 1
        assert "agent-1" in insight.supporting_agents
        assert "agent-2" in insight.supporting_agents
        assert len(insight.supporting_fact_ids) == 2

    def test_promote_rejects_low_coverage(self) -> None:
        store = TeamInsightStore(min_coverage_to_promote=0.8)
        result = AssemblyResult(
            selected=[
                _make_candidate("fact 1", "agent-1", "f1"),
                _make_candidate("fact 2", "agent-2", "f2"),
            ],
            covered_facets={"f0"},
            uncovered_facets={"f1", "f2"},
            coverage_score=0.3,
        )
        assert store.promote_from_assembly(result) is None
        assert store.count == 0

    def test_promote_rejects_single_agent(self) -> None:
        store = TeamInsightStore(min_coverage_to_promote=0.5)
        result = AssemblyResult(
            selected=[
                _make_candidate("fact 1", "agent-1", "f1"),
                _make_candidate("fact 2", "agent-1", "f2"),
            ],
            covered_facets={"f0", "f1"},
            uncovered_facets=set(),
            coverage_score=1.0,
        )
        assert store.promote_from_assembly(result) is None
        assert store.count == 0

    def test_promote_rejects_single_fact(self) -> None:
        store = TeamInsightStore(min_coverage_to_promote=0.5)
        result = AssemblyResult(
            selected=[
                _make_candidate("only fact", "agent-1", "f1"),
            ],
            covered_facets={"f0"},
            uncovered_facets=set(),
            coverage_score=1.0,
        )
        assert store.promote_from_assembly(result) is None
        assert store.count == 0


class TestReuseCountBoost:
    def test_reuse_count_boost(self) -> None:
        store = TeamInsightStore()
        # Two insights with similar tokens but different reuse counts.
        low = _make_insight(
            "kubernetes deployment scaling",
            insight_id="low",
            reuse_count=0,
        )
        high = _make_insight(
            "kubernetes deployment scaling",
            insight_id="high",
            reuse_count=100,
        )
        store.remember(low)
        store.remember(high)

        results = store.retrieve("kubernetes deployment scaling")
        assert len(results) == 2
        assert results[0].insight_id == "high"


class TestEmptyStore:
    def test_empty_store(self) -> None:
        store = TeamInsightStore()
        assert store.count == 0
        assert store.retrieve("anything") == []
