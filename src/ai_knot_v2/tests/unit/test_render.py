"""Unit tests for bench/ccb/render.py — ESWP render operator."""

from __future__ import annotations

from ai_knot_v2.bench.ccb.render import render_pack_eswp
from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom


def _atom(
    *,
    predicate: str = "prefers",
    subject: str = "Alice",
    object_value: str | None = "hiking",
    polarity: str = "pos",
    risk_class: str = "preference",
    regret_charge: float = 0.5,
    credence: float = 1.0,
    valid_from: int | None = None,
    valid_until: int | None = None,
    entity_orbit_id: str = "entity:alice",
    action_affect_mask: int = 0,
) -> MemoryAtom:
    uid = new_ulid()
    return MemoryAtom(
        atom_id=uid,
        agent_id="agent-1",
        user_id="user-1",
        variables=("alice",),
        causal_graph=(),
        kernel_kind="point",
        kernel_payload={},
        intervention_domain=("alice",),
        predicate=predicate,
        subject=subject,
        object_value=object_value,
        polarity=polarity,  # type: ignore[arg-type]
        valid_from=valid_from,
        valid_until=valid_until,
        observation_time=1_700_000_000,
        belief_time=1_700_000_000,
        granularity="instant",  # type: ignore[arg-type]
        entity_orbit_id=entity_orbit_id,
        transport_provenance=("session-1",),
        depends_on=(),
        depended_by=(),
        risk_class=risk_class,  # type: ignore[arg-type]
        risk_severity=0.3,
        regret_charge=regret_charge,
        irreducibility_score=1.0,
        protection_energy=0.4,
        action_affect_mask=action_affect_mask,
        credence=credence,
        evidence_episodes=(uid,),
        synthesis_method="regex",
        validation_tests=(),
        contradiction_events=(),
    )


class TestRenderPackEswp:
    def test_factual_pred_format(self) -> None:
        atom = _atom(
            predicate="is",
            object_value="doctor",
            risk_class="identity",
            regret_charge=0.9,
            credence=1.0,
        )
        rendered = render_pack_eswp([atom], "what does Alice do")
        assert "[fact]" in rendered
        assert "doctor" in rendered

    def test_event_pred_format(self) -> None:
        atom = _atom(predicate="visited", object_value="Paris", regret_charge=0.5)
        rendered = render_pack_eswp([atom], "where did Alice go")
        assert "[event]" in rendered
        assert "Paris" in rendered

    def test_event_with_temporal_range(self) -> None:
        atom = _atom(
            predicate="visited",
            object_value="Paris",
            valid_from=1_700_000_000,
            valid_until=1_700_100_000,
        )
        rendered = render_pack_eswp([atom], "when did Alice visit Paris")
        assert "valid" in rendered
        assert "–" in rendered

    def test_high_charge_first(self) -> None:
        low = _atom(
            predicate="prefers",
            subject="Low",
            object_value="l",
            regret_charge=0.1,
            credence=1.0,
            entity_orbit_id="entity:low",
        )
        high = _atom(
            predicate="prefers",
            subject="High",
            object_value="h",
            regret_charge=0.9,
            credence=1.0,
            entity_orbit_id="entity:high",
        )
        rendered = render_pack_eswp([low, high], "query")
        assert rendered.index("High") < rendered.index("Low")

    def test_credence_multiplies_charge(self) -> None:
        # charge=0.9, credence=0.1 → product=0.09 < charge=0.5, credence=0.9 → 0.45
        low_product = _atom(
            subject="LowProd", regret_charge=0.9, credence=0.1, entity_orbit_id="entity:lp"
        )
        high_product = _atom(
            subject="HighProd", regret_charge=0.5, credence=0.9, entity_orbit_id="entity:hp"
        )
        rendered = render_pack_eswp([low_product, high_product], "query")
        assert rendered.index("HighProd") < rendered.index("LowProd")

    def test_neg_polarity_prefix(self) -> None:
        atom = _atom(predicate="is", object_value="doctor", polarity="neg")
        rendered = render_pack_eswp([atom], "is Alice a doctor")
        assert "[neg-fact]" in rendered

    def test_neg_event_polarity(self) -> None:
        atom = _atom(predicate="visited", object_value="Paris", polarity="neg")
        rendered = render_pack_eswp([atom], "did Alice visit Paris")
        assert "[neg-event]" in rendered

    def test_empty_atoms_returns_empty(self) -> None:
        assert render_pack_eswp([], "any query") == ""

    def test_no_llm_import(self) -> None:
        import ast
        import pathlib

        src = pathlib.Path("src/ai_knot_v2/bench/ccb/render.py").read_text()
        tree = ast.parse(src)
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        llm_keywords = {"openai", "anthropic", "gpt", "claude", "litellm", "langchain"}
        assert not any(any(kw in n for kw in llm_keywords) for n in names)

    def test_predicate_underscores_replaced(self) -> None:
        atom = _atom(predicate="works_at", object_value="TechCorp")
        rendered = render_pack_eswp([atom], "where does Alice work")
        assert "works at" in rendered
        assert "works_at" not in rendered
