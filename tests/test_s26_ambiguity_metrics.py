"""Tests for the S26 ambiguity-aware metrics (Option C).

These lock the semantics that separate the *unsolvable* conceptual exact-agent
recall from the *solvable* identifiable variant and the *always-achievable*
domain (equivalence) recall.
"""

from __future__ import annotations

import importlib

from ai_knot.multi_agent.facets import ConjunctiveFacetPlanner
from ai_knot.multi_agent.router import QueryShapeRouter

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
