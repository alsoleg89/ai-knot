"""Tests for multi_agent.facets — conjunctive decomposition."""

from __future__ import annotations

from ai_knot.multi_agent.facets import ConjunctiveFacetPlanner
from ai_knot.multi_agent.models import RoutedPoolQuery


def _routed(query: str) -> RoutedPoolQuery:
    return RoutedPoolQuery(raw_query=query, intent="multi_source")


class TestConjunctiveFacetPlanner:
    def setup_method(self) -> None:
        self.planner = ConjunctiveFacetPlanner()

    def test_comma_and_split(self) -> None:
        routed = _routed(
            "How can a system integrate encrypted computation, "
            "partitioning data structures, and orbital data alignment?"
        )
        facets = self.planner.decompose(routed)
        assert len(facets) == 3
        assert all(f.facet_type == "domain" for f in facets)

    def test_comma_split_two_facets(self) -> None:
        routed = _routed("How to combine fluid dynamics simulation and compiler optimization?")
        facets = self.planner.decompose(routed)
        assert len(facets) >= 2

    def test_single_facet_fallback(self) -> None:
        routed = _routed("what is the database config?")
        facets = self.planner.decompose(routed)
        assert len(facets) == 1
        assert facets[0].facet_id == "f0"
        assert facets[0].text == "what is the database config?"

    def test_meta_explanatory_rejected(self) -> None:
        routed = _routed("what is X and how does it work?")
        facets = self.planner.decompose(routed)
        # "how does it work" is meta-explanatory, should not produce 2 facets.
        assert len(facets) == 1

    def test_short_clauses_rejected(self) -> None:
        routed = _routed("combine A and B")
        facets = self.planner.decompose(routed)
        # "A" and "B" have < 2 content tokens each.
        assert len(facets) == 1

    def test_facet_ids_sequential(self) -> None:
        routed = _routed(
            "integrate cryptographic protocols, graph databases, and satellite telemetry"
        )
        facets = self.planner.decompose(routed)
        assert len(facets) >= 2
        for i, f in enumerate(facets):
            assert f.facet_id == f"f{i}"

    def test_max_five_facets(self) -> None:
        routed = _routed(
            "combine quantum computing, bioinformatics, supply chain, "
            "embedded firmware, acoustic modelling, CFD simulation, "
            "and edge computing approaches"
        )
        facets = self.planner.decompose(routed)
        assert len(facets) <= 5

    def test_s26_style_query(self) -> None:
        """Test with actual S26 benchmark query pattern."""
        routed = _routed(
            "How can a system integrate encrypted computation without decryption, "
            "partitioning interconnected data structures, "
            "and orbital data stream alignment into a unified pipeline?"
        )
        facets = self.planner.decompose(routed)
        assert len(facets) == 3
        # Each facet should have content tokens.
        for f in facets:
            assert len(f.tokens) >= 2
