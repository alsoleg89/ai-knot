"""Unit tests for bench/cwp/lineage_render.py and bench/cwp/persistence.py."""

from __future__ import annotations

from ai_knot_v2.bench.cwp.lineage_render import render_pack_cwp
from ai_knot_v2.bench.cwp.persistence import (
    PCTSignature,
    compute_pct_signatures,
    cwp_priority,
)
from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.episode import RawEpisode


def _atom(
    *,
    predicate: str = "is",
    subject: str = "Alice",
    object_value: str = "doctor",
    polarity: str = "pos",
    risk_class: str = "identity",
    regret_charge: float = 0.7,
    credence: float = 1.0,
    evidence_episodes: tuple[str, ...] = (),
    valid_from: int | None = None,
    valid_until: int | None = None,
) -> MemoryAtom:
    uid = new_ulid()
    return MemoryAtom(
        atom_id=uid,
        agent_id="a",
        user_id="u",
        variables=(),
        causal_graph=(),
        kernel_kind="point",
        kernel_payload={},
        intervention_domain=(),
        predicate=predicate,
        subject=subject,
        object_value=object_value,
        polarity=polarity,  # type: ignore[arg-type]
        valid_from=valid_from,
        valid_until=valid_until,
        observation_time=0,
        belief_time=0,
        granularity="instant",  # type: ignore[arg-type]
        entity_orbit_id="entity:alice",
        transport_provenance=(),
        depends_on=(),
        depended_by=(),
        risk_class=risk_class,  # type: ignore[arg-type]
        risk_severity=0.5,
        regret_charge=regret_charge,
        irreducibility_score=1.0,
        protection_energy=0.5,
        action_affect_mask=0,
        credence=credence,
        evidence_episodes=evidence_episodes,
        synthesis_method="regex",
        validation_tests=(),
        contradiction_events=(),
    )


def _episode(ep_id: str, text: str, ts: int = 1_700_000_000, speaker: str = "Alice") -> RawEpisode:
    return RawEpisode(
        episode_id=ep_id,
        agent_id="a",
        user_id=speaker,
        session_id="s1",
        turn_index=0,
        speaker="user",
        text=text,
        timestamp=ts,
    )


class TestComputePctSignatures:
    def test_isolated_atom_has_zero_betweenness(self) -> None:
        a = _atom(evidence_episodes=("ep-1",))
        sigs = compute_pct_signatures([a])
        assert sigs[a.atom_id].betweenness == 0.0

    def test_co_cited_atoms_share_betweenness(self) -> None:
        a = _atom(evidence_episodes=("ep-1",))
        b = _atom(evidence_episodes=("ep-1",))
        sigs = compute_pct_signatures([a, b])
        assert sigs[a.atom_id].betweenness >= 1.0
        assert sigs[b.atom_id].betweenness >= 1.0

    def test_persistence_increases_with_co_atoms(self) -> None:
        a = _atom(evidence_episodes=("ep-1",))
        peers = [_atom(evidence_episodes=("ep-1",)) for _ in range(5)]
        sigs = compute_pct_signatures([a, *peers])
        assert sigs[a.atom_id].persistence_0 > 0.5

    def test_contradiction_marks_cycle(self) -> None:
        a = _atom(predicate="is", polarity="pos", evidence_episodes=("ep-1",))
        b = _atom(predicate="is", polarity="neg", evidence_episodes=("ep-1",))
        sigs = compute_pct_signatures([a, b])
        assert sigs[a.atom_id].cycle_membership == 1.0

    def test_no_contradiction_zero_cycle(self) -> None:
        a = _atom(predicate="is", polarity="pos", evidence_episodes=("ep-1",))
        b = _atom(predicate="has", polarity="pos", evidence_episodes=("ep-1",))
        sigs = compute_pct_signatures([a, b])
        assert sigs[a.atom_id].cycle_membership == 0.0


class TestCwpPriority:
    def test_higher_persistence_higher_priority(self) -> None:
        a = _atom(credence=1.0)
        sig_low = PCTSignature(a.atom_id, persistence_0=0.1, betweenness=0.0, cycle_membership=0.0)
        sig_high = PCTSignature(a.atom_id, persistence_0=0.9, betweenness=0.0, cycle_membership=0.0)
        assert cwp_priority(a, sig_high) > cwp_priority(a, sig_low)

    def test_contradiction_lowers_priority(self) -> None:
        a = _atom(credence=1.0)
        sig_clean = PCTSignature(
            a.atom_id, persistence_0=0.5, betweenness=1.0, cycle_membership=0.0
        )
        sig_loop = PCTSignature(a.atom_id, persistence_0=0.5, betweenness=1.0, cycle_membership=1.0)
        assert cwp_priority(a, sig_clean) > cwp_priority(a, sig_loop)


class TestRenderPackCwp:
    def test_empty_atoms_returns_empty(self) -> None:
        assert render_pack_cwp([], {}, "any query") == ""

    def test_factual_claim_marked_fact(self) -> None:
        a = _atom(predicate="is", subject="Alice", object_value="doctor")
        rendered = render_pack_cwp([a], {}, "what is Alice's job")
        assert "[fact]" in rendered
        assert "Alice is doctor" in rendered

    def test_event_claim_marked_event(self) -> None:
        a = _atom(predicate="visited", subject="Alice", object_value="Paris")
        rendered = render_pack_cwp([a], {}, "where")
        assert "[event]" in rendered

    def test_negative_polarity_marked(self) -> None:
        a = _atom(predicate="is", object_value="doctor", polarity="neg")
        rendered = render_pack_cwp([a], {}, "is alice doctor")
        assert "NOT" in rendered

    def test_supporting_observations_rendered(self) -> None:
        ep = _episode("ep-1", "Caroline: I'm allergic to penicillin")
        a = _atom(
            predicate="allergic_to",
            subject="Caroline",
            object_value="penicillin",
            evidence_episodes=("ep-1",),
        )
        rendered = render_pack_cwp([a], {"ep-1": ep}, "is caroline allergic")
        assert "←" in rendered
        assert "penicillin" in rendered.lower()
        assert "Caroline" in rendered

    def test_pct_signatures_change_ordering(self) -> None:
        a1 = _atom(subject="LowPriority", regret_charge=0.5, credence=1.0)
        a2 = _atom(subject="HighPriority", regret_charge=0.5, credence=1.0)
        sigs = {
            a1.atom_id: PCTSignature(a1.atom_id, 0.1, 0.0, 0.0),
            a2.atom_id: PCTSignature(a2.atom_id, 0.9, 5.0, 0.0),
        }
        rendered = render_pack_cwp([a1, a2], {}, "q", pct_signatures=sigs)
        assert rendered.index("HighPriority") < rendered.index("LowPriority")

    def test_max_atoms_limit_respected(self) -> None:
        atoms = [_atom(subject=f"Person{i}") for i in range(10)]
        rendered = render_pack_cwp(atoms, {}, "q", max_atoms=3)
        # Count claim lines (start with "[N] [fact]" or "[N] [event]")
        claim_lines = [line for line in rendered.split("\n") if line.startswith("[")]
        # Each claim is one line; supporting obs are separate lines starting with "    ←"
        top_level = [line for line in claim_lines if not line.startswith("    ")]
        assert len(top_level) == 3

    def test_no_llm_import(self) -> None:
        import ast
        import pathlib

        for fn in ("lineage_render.py", "persistence.py"):
            p = pathlib.Path(f"src/ai_knot_v2/bench/cwp/{fn}")
            tree = ast.parse(p.read_text())
            names: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
            llm_keywords = {"openai", "anthropic", "gpt", "claude", "litellm", "langchain"}
            assert not any(any(kw in n for kw in llm_keywords) for n in names), f"{fn}: {names}"
