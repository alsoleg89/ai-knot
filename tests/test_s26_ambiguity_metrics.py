"""Tests for the S26 ambiguity-aware metrics (Option C).

These lock the semantics that separate the *unsolvable* conceptual exact-agent
recall from the *solvable* identifiable variant and the *always-achievable*
domain (equivalence) recall.
"""

from __future__ import annotations

import importlib

import pytest

from ai_knot.multi_agent.facets import ConjunctiveFacetPlanner
from ai_knot.multi_agent.router import QueryShapeRouter
from ai_knot.types import Fact
from tests.eval.benchmark.base import BenchmarkMetrics, RetrievalResult, ScenarioResult
from tests.eval.benchmark.ma_gate import GATE, evaluate_gate, gate_passed

# scenarios/__init__ rebinds submodule names to their run() functions, so import
# the module object directly.
s26 = importlib.import_module("tests.eval.benchmark.scenarios.s26_sparse_assembly")


def test_agent_domain_parses_specialist_id() -> None:
    assert s26._agent_domain("specialist_0000") == s26._domain(0)[0]
    # Domains repeat every 20 → agents 0 and 20 share a domain.
    assert s26._agent_domain("specialist_0000") == s26._agent_domain("specialist_0020")
    assert s26._agent_domain(None) is None
    assert s26._agent_domain("querier") is None


def test_domain_class_size_grows_with_pool() -> None:
    # Each target is one of N/20 same-content peers — the ambiguity that caps
    # conceptual exact-agent recall.
    assert s26._domain_class_size(0, 10) == 1
    assert s26._domain_class_size(0, 100) == 5
    assert s26._domain_class_size(0, 1000) == 50


def test_conceptual_query_cannot_name_target() -> None:
    targets = s26._select_targets(1000, 0)
    markers = [s26._rare_marker(t) for t in targets]
    q = s26._build_query(targets)
    assert not any(m in q for m in markers)


def test_marker_query_decomposes_one_marker_per_facet() -> None:
    targets = s26._select_targets(1000, 0)
    markers = [s26._rare_marker(t) for t in targets]
    q = s26._build_query_with_markers(targets)
    assert all(m in q for m in markers)

    routed = QueryShapeRouter().route(
        q,
        requesting_agent_id="q",
        active_facts=[],
        requesting_agent_fact_count=0,
        topic_channel="",
    )
    facets = ConjunctiveFacetPlanner().decompose(routed)
    # The identifiable variant must route through the facet pipeline (>=2 facets)
    # with each marker surviving into its own facet, so literal rescue can act.
    assert len(facets) == 3
    surviving = {m for f in facets for m in markers if m.lower() in [t.lower() for t in f.tokens]}
    assert surviving == set(markers)


class ReturnedSources:
    def __init__(self, texts, *, publishers=None):
        self.result = RetrievalResult(
            texts,
            list(range(len(texts))),
            0,
            facts=[]
            if publishers is None
            else [
                Fact(text, origin_agent_id=publisher)
                for text, publisher in zip(texts, publishers, strict=True)
            ],
        )

    async def pool_retrieve(self, *args, **kwargs):
        return self.result


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["generic", "near_miss"])
async def test_domain_authorship_alone_cannot_pass_shard_coverage(kind):
    targets = s26._select_targets(1000, 0)
    texts = [
        s26._distractor_fact(i)
        if kind == "generic"
        else s26._near_miss_fact(i, [s26._domain(t)[2] for t in targets])
        for i in targets
    ]
    result = await s26._measure_at_checkpoint(
        ReturnedSources(texts, publishers=[s26._agent_id(i) for i in targets]), 1000, 1
    )
    assert result["equivalence_recall"] == 0
    assert result["target_shard_recall"] == 0
    metrics = BenchmarkMetrics("synthetic", selected_scenarios=[s26.SCENARIO_ID])
    metrics.scenario_results = [
        ScenarioResult(
            s26.SCENARIO_ID,
            "synthetic",
            {"equivalence_recall_at_1000": [result["equivalence_recall"]]},
            None,
            None,
        )
    ]
    threshold = tuple(t for t in GATE if t.metric == "equivalence_recall_at_1000")
    assert not gate_passed(evaluate_gate(metrics, threshold))


@pytest.mark.anyio
async def test_equivalent_peer_shards_cover_domains_without_becoming_exact_targets():
    peers = [t + 20 for t in s26._select_targets(1000, 0)]
    result = await s26._measure_at_checkpoint(
        ReturnedSources(
            [s26._shard_fact(i) for i in peers], publishers=[s26._agent_id(i) for i in peers]
        ),
        1000,
        1,
    )
    assert result["equivalence_recall"] == 1
    assert result["target_shard_recall"] == result["marker_in_query_recall"] == 0
    assert result["distractor_rate"] == 1  # Existing exact-target definition, not usefulness.


@pytest.mark.anyio
@pytest.mark.parametrize("with_facts", [False, True])
async def test_duplicate_rows_cannot_multiply_one_target_identity(with_facts):
    first = s26._select_targets(1000, 0)[0]
    result = await s26._measure_at_checkpoint(
        ReturnedSources(
            [s26._shard_fact(first)] * 5,
            publishers=[s26._agent_id(first)] * 5 if with_facts else None,
        ),
        1000,
        1,
    )
    assert result["target_shard_recall"] == result["marker_in_query_recall"] == pytest.approx(1 / 3)
    assert result["equivalence_recall"] == pytest.approx(1 / 3)
    assert result["agent_diversity"] == pytest.approx(1 / 3)
    assert result["all_shards_covered"] == 0
    assert result["last_target_rank"] == 1


@pytest.mark.anyio
async def test_marker_must_belong_to_the_fact_publisher():
    first, second, _ = s26._select_targets(1000, 0)
    result = await s26._measure_at_checkpoint(
        ReturnedSources([s26._shard_fact(second)], publishers=[s26._agent_id(first)]), 1000, 1
    )
    assert result["target_shard_recall"] == result["marker_in_query_recall"] == 0
    assert result["equivalence_recall"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("with_facts", [False, True])
async def test_marker_prefix_is_not_a_known_source(with_facts):
    first = s26._select_targets(1000, 0)[0]
    wrong = s26._shard_fact(first).replace(s26._rare_marker(first), s26._rare_marker(first) + "0")
    result = await s26._measure_at_checkpoint(
        ReturnedSources([wrong], publishers=[s26._agent_id(first)] if with_facts else None), 1000, 1
    )
    assert result["target_shard_recall"] == result["marker_in_query_recall"] == 0
    assert result["equivalence_recall"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("with_facts", [False, True])
async def test_actual_target_sources_keep_full_coverage(with_facts):
    targets = s26._select_targets(1000, 0)
    result = await s26._measure_at_checkpoint(
        ReturnedSources(
            [s26._shard_fact(t) for t in targets],
            publishers=[s26._agent_id(t) for t in targets] if with_facts else None,
        ),
        1000,
        1,
    )
    assert (
        result["target_shard_recall"]
        == result["marker_in_query_recall"]
        == result["equivalence_recall"]
        == 1
    )
    assert result["all_shards_covered"] == 1


@pytest.mark.anyio
async def test_repeated_domain_facets_count_as_covered_without_exact_peer_identity():
    targets = s26._select_targets(60, 0)
    assert targets == [0, 20, 40]
    result = await s26._measure_at_checkpoint(
        ReturnedSources([s26._shard_fact(0)], publishers=[s26._agent_id(0)]), 60, 1
    )
    assert result["equivalence_recall"] == 1
    assert result["target_shard_recall"] == pytest.approx(1 / 3)
    assert result["mean_equivalence_class_size"] == 3
