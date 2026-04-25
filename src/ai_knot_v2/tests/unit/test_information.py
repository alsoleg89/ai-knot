"""Unit tests for core/information.py — ESWP Level-3 charge formulas."""

from __future__ import annotations

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.information import compute_irreducibility, compute_regret_charge_v2


def _atom(
    *,
    risk_severity: float = 0.5,
    action_affect_mask: int = 0,
    predicate: str = "prefers",
    valid_from: int | None = None,
    valid_until: int | None = None,
    risk_class: str = "preference",
) -> MemoryAtom:
    uid = new_ulid()
    return MemoryAtom(
        atom_id=uid,
        agent_id="agent-1",
        user_id="user-1",
        variables=("user_1",),
        causal_graph=(),
        kernel_kind="point",
        kernel_payload={},
        intervention_domain=("user_1",),
        predicate=predicate,
        subject="user-1",
        object_value="hiking",
        polarity="pos",
        valid_from=valid_from,
        valid_until=valid_until,
        observation_time=1_700_000_000,
        belief_time=1_700_000_000,
        granularity="instant",  # type: ignore[arg-type]
        entity_orbit_id="entity:user_1",
        transport_provenance=("session-1",),
        depends_on=(),
        depended_by=(),
        risk_class=risk_class,  # type: ignore[arg-type]
        risk_severity=risk_severity,
        regret_charge=0.0,
        irreducibility_score=1.0,
        protection_energy=0.4,
        action_affect_mask=action_affect_mask,
        credence=0.9,
        evidence_episodes=(uid,),
        synthesis_method="regex",
        validation_tests=(),
        contradiction_events=(),
    )


class TestComputeRegretChargeV2:
    def test_zero_risk_zero_charge(self) -> None:
        a = _atom(risk_severity=0.0, action_affect_mask=0)
        assert compute_regret_charge_v2(a) == 0.0

    def test_high_action_bits_increases_charge(self) -> None:
        low = _atom(risk_severity=0.5, action_affect_mask=0b00001)
        high = _atom(risk_severity=0.5, action_affect_mask=0b11110)
        assert compute_regret_charge_v2(high) > compute_regret_charge_v2(low)

    def test_curvature_increases_charge(self) -> None:
        a = _atom(risk_severity=0.5, action_affect_mask=0)
        assert compute_regret_charge_v2(a, curvature=1.0) > compute_regret_charge_v2(
            a, curvature=0.0
        )

    def test_charge_bounded_at_one(self) -> None:
        a = _atom(risk_severity=1.0, action_affect_mask=0xFF)
        assert compute_regret_charge_v2(a, curvature=1.0) <= 1.0

    def test_high_risk_higher_than_low(self) -> None:
        lo = _atom(risk_severity=0.1)
        hi = _atom(risk_severity=0.9)
        assert compute_regret_charge_v2(hi) > compute_regret_charge_v2(lo)


class TestComputeIrreducibility:
    def test_no_peers_returns_one(self) -> None:
        a = _atom()
        assert compute_irreducibility(a, []) == 1.0

    def test_identical_peer_reduces_score(self) -> None:
        a = _atom(predicate="prefers", valid_from=0, valid_until=1000)
        peer = _atom(predicate="prefers", valid_from=0, valid_until=1000)
        assert compute_irreducibility(a, [peer]) < 0.9

    def test_different_predicate_peer_no_effect(self) -> None:
        a = _atom(predicate="prefers", valid_from=0, valid_until=1000)
        peer = _atom(predicate="visited", valid_from=0, valid_until=1000)
        # no overlap on same predicate → still close to 1.0
        score = compute_irreducibility(a, [peer])
        assert score >= 0.9

    def test_non_overlapping_intervals_no_effect(self) -> None:
        a = _atom(predicate="prefers", valid_from=0, valid_until=100)
        peer = _atom(predicate="prefers", valid_from=500, valid_until=600)
        score = compute_irreducibility(a, [peer])
        assert score >= 0.9

    def test_floor_at_point_one(self) -> None:
        a = _atom(predicate="prefers")
        peers = [_atom(predicate="prefers") for _ in range(100)]
        assert compute_irreducibility(a, peers) >= 0.1
