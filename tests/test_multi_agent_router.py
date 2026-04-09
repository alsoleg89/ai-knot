"""Tests for multi_agent.router — query intent classification."""

from __future__ import annotations

from ai_knot.multi_agent.router import (
    BROAD_DISCOVERY,
    ENTITY_LOOKUP,
    GENERAL,
    INCIDENT,
    MULTI_SOURCE,
    QueryShapeRouter,
    classify_pool_query,
)
from ai_knot.types import Fact


def _fact(*, entity: str = "", origin: str = "agent_a") -> Fact:
    f = Fact(content="test")
    f.entity = entity
    f.origin_agent_id = origin
    return f


class TestClassifyPoolQuery:
    def test_incident_time_pattern(self) -> None:
        assert classify_pool_query("what happened at 10:30?", []) == INCIDENT

    def test_incident_keyword(self) -> None:
        assert classify_pool_query("describe the outage", []) == INCIDENT

    def test_broad_discovery(self) -> None:
        facts = [
            _fact(origin="a1"),
            _fact(origin="a2"),
            _fact(origin="a3"),
        ]
        assert (
            classify_pool_query(
                "what does the team know?",
                facts,
                requesting_agent_fact_count=0,
            )
            == BROAD_DISCOVERY
        )

    def test_broad_discovery_needs_3_publishers(self) -> None:
        facts = [_fact(origin="a1"), _fact(origin="a2")]
        assert (
            classify_pool_query(
                "what does the team know?",
                facts,
                requesting_agent_fact_count=0,
            )
            != BROAD_DISCOVERY
        )

    def test_entity_lookup(self) -> None:
        facts = [_fact(entity="PostgreSQL")]
        assert classify_pool_query("what about PostgreSQL?", facts) == ENTITY_LOOKUP

    def test_entity_lookup_short_entity_ignored(self) -> None:
        facts = [_fact(entity="DB")]
        assert classify_pool_query("what about DB?", facts) != ENTITY_LOOKUP

    def test_multi_source_keyword(self) -> None:
        assert classify_pool_query("integrate logging across services", []) == MULTI_SOURCE

    def test_multi_source_long_and(self) -> None:
        assert (
            classify_pool_query(
                "How can we combine encryption and monitoring and logging in one system?",
                [],
            )
            == MULTI_SOURCE
        )

    def test_multi_source_beats_broad_discovery_for_conjunctive(self) -> None:
        """Comma-separated conjunctive query with aggregation stem should be
        MULTI_SOURCE even when the agent has zero private facts."""
        facts = [_fact(origin=f"a{i}") for i in range(10)]
        assert (
            classify_pool_query(
                "How can a system integrate encrypted computation, "
                "atmospheric modeling, and task scheduling into a pipeline?",
                facts,
                requesting_agent_fact_count=0,
            )
            == MULTI_SOURCE
        )

    def test_general_fallback(self) -> None:
        assert classify_pool_query("tell me about the config", []) == GENERAL


class TestQueryShapeRouter:
    def test_route_returns_routed_query(self) -> None:
        router = QueryShapeRouter()
        routed = router.route(
            "integrate X and Y across systems",
            requesting_agent_id="test",
            active_facts=[],
            requesting_agent_fact_count=5,
        )
        assert routed.intent == MULTI_SOURCE
        assert routed.raw_query == "integrate X and Y across systems"
        assert routed.use_expertise_routing is True

    def test_route_general_no_expertise(self) -> None:
        router = QueryShapeRouter()
        routed = router.route(
            "what is the config?",
            requesting_agent_id="test",
            active_facts=[],
            requesting_agent_fact_count=5,
        )
        assert routed.intent == GENERAL
        assert routed.use_expertise_routing is False
