"""Tests for AgentExpertiseIndex route-before-retrieve."""

from __future__ import annotations

from ai_knot.multi_agent.expertise import AgentExpertiseIndex
from ai_knot.types import Fact


def _make_fact(
    content: str,
    agent_id: str,
    *,
    entity: str = "",
    tags: list[str] | None = None,
    canonical_surface: str = "",
    version: int = 0,
) -> Fact:
    return Fact(
        content=content,
        origin_agent_id=agent_id,
        entity=entity,
        tags=tags or [],
        canonical_surface=canonical_surface,
        version=version,
    )


def _default_trust(agent_id: str) -> float:
    return 0.8


def test_build_creates_profiles() -> None:
    """Build from 3 agents with different content, verify profiles created."""
    facts = [
        _make_fact("Python type hints are important", "agent-py"),
        _make_fact("Rust ownership model prevents bugs", "agent-rust"),
        _make_fact("Go concurrency uses goroutines", "agent-go"),
    ]
    idx = AgentExpertiseIndex()
    idx.build(facts, _default_trust)

    assert idx.built
    assert len(idx.profiles) == 3
    assert "agent-py" in idx.profiles
    assert "agent-rust" in idx.profiles
    assert "agent-go" in idx.profiles

    py_profile = idx.profiles["agent-py"]
    assert py_profile.published_facts == 1
    assert py_profile.trust_score == 0.8
    assert len(py_profile.content_terms) > 0


def test_stale_detection() -> None:
    """Build, check not stale, add a fact, check stale."""
    facts = [
        _make_fact("Kubernetes orchestration basics", "agent-devops"),
    ]
    idx = AgentExpertiseIndex()
    idx.build(facts, _default_trust)

    assert not idx.is_stale(facts)

    # Adding a new fact changes the version fingerprint.
    facts_updated = facts + [
        _make_fact("Docker container networking", "agent-devops"),
    ]
    assert idx.is_stale(facts_updated)


def test_top_agents_for_facet() -> None:
    """Build from agents with known content, query specific terms."""
    facts = [
        _make_fact("Python async await concurrency patterns", "agent-py"),
        _make_fact("Python decorators and metaclasses", "agent-py"),
        _make_fact("Kubernetes pod scheduling and scaling", "agent-devops"),
        _make_fact("Database indexing and query optimization", "agent-db"),
    ]
    idx = AgentExpertiseIndex()
    idx.build(facts, _default_trust)

    hits = idx.top_agents_for_facet(("python",))
    assert len(hits) >= 1
    assert hits[0].agent_id == "agent-py"
    assert "python" in hits[0].matched_terms


def test_top_agents_for_query() -> None:
    """Convenience method tokenizes and delegates correctly."""
    facts = [
        _make_fact("Machine learning gradient descent", "agent-ml"),
        _make_fact("Web server HTTP routing", "agent-web"),
    ]
    idx = AgentExpertiseIndex()
    idx.build(facts, _default_trust)

    hits = idx.top_agents_for_query("machine learning")
    assert len(hits) >= 1
    assert hits[0].agent_id == "agent-ml"


def test_trust_weighting() -> None:
    """Agent with higher trust scores higher for same content overlap."""
    facts = [
        _make_fact("Deployment pipeline CI CD", "trusted-agent"),
        _make_fact("Deployment pipeline CI CD", "untrusted-agent"),
    ]

    def _varied_trust(agent_id: str) -> float:
        return 0.95 if agent_id == "trusted-agent" else 0.3

    idx = AgentExpertiseIndex()
    idx.build(facts, _varied_trust)

    hits = idx.top_agents_for_facet(("deploy",))
    assert len(hits) == 2
    assert hits[0].agent_id == "trusted-agent"
    assert hits[0].score > hits[1].score


def test_empty_index() -> None:
    """top_agents returns empty list before build."""
    idx = AgentExpertiseIndex()
    assert not idx.built
    assert idx.top_agents_for_facet(("anything",)) == []
    assert idx.top_agents_for_query("anything") == []


def test_entity_contributes_to_domains() -> None:
    """Entity names are indexed as domain terms."""
    facts = [
        _make_fact(
            "Salary is 95000",
            "agent-hr",
            entity="Alex Chen",
        ),
    ]
    idx = AgentExpertiseIndex()
    idx.build(facts, _default_trust)

    profile = idx.profiles["agent-hr"]
    assert profile.domains.total() > 0

    hits = idx.top_agents_for_query("Alex Chen")
    assert len(hits) >= 1
    assert hits[0].agent_id == "agent-hr"


def test_tags_indexed() -> None:
    """Tags are accumulated in the profile."""
    facts = [
        _make_fact("Some fact", "agent-a", tags=["devops", "cloud"]),
        _make_fact("Another fact", "agent-a", tags=["devops"]),
    ]
    idx = AgentExpertiseIndex()
    idx.build(facts, _default_trust)

    profile = idx.profiles["agent-a"]
    assert profile.tags["devops"] == 2
    assert profile.tags["cloud"] == 1


def test_empty_facet_tokens() -> None:
    """Empty facet tokens return no hits."""
    facts = [_make_fact("Something", "agent-a")]
    idx = AgentExpertiseIndex()
    idx.build(facts, _default_trust)

    assert idx.top_agents_for_facet(()) == []
