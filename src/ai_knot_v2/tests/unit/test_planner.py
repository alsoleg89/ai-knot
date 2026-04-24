"""Sprint 5 / Sprint 11 — Evidence Planner unit tests.

Tests: reader_cost, reduction_score, utility, contradiction detection,
plan_evidence_pack, handle_contradictions, and tri-temporal Allen-relation scoring.
No LOCOMO data. All synthetic.
"""

from __future__ import annotations

import dataclasses
import time

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.library import AtomLibrary
from ai_knot_v2.core.types import ReaderBudget
from ai_knot_v2.ops.planner import (
    detect_contradictions,
    handle_contradictions,
    plan_evidence_pack,
    reader_cost,
    reduction_score,
    temporal_allen_bonus,
    utility,
)


def _atom(
    risk_class: str = "preference",
    predicate: str = "prefers",
    subject: str = "user-1",
    object_value: str | None = "hiking",
    polarity: str = "pos",
    risk_severity: float = 0.2,
    credence: float = 0.9,
    regret_charge: float = 0.2,
    entity_orbit_id: str = "entity:user_x",
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
        subject=subject,
        object_value=object_value,
        polarity=polarity,  # type: ignore[arg-type]
        valid_from=None,
        valid_until=None,
        observation_time=1_700_000_000,
        belief_time=1_700_000_000,
        granularity="instant",  # type: ignore[arg-type]
        entity_orbit_id=entity_orbit_id,
        transport_provenance=("session-1",),
        depends_on=(),
        depended_by=(),
        risk_class=risk_class,  # type: ignore[arg-type]
        risk_severity=risk_severity,
        regret_charge=regret_charge,
        irreducibility_score=1.0,
        protection_energy=0.4,
        action_affect_mask=0,
        credence=credence,
        evidence_episodes=(uid,),
        synthesis_method="regex",
        validation_tests=(),
        contradiction_events=(),
    )


_BUDGET = ReaderBudget(max_atoms=10, max_tokens=500, require_dependency_closure=True)


class TestReaderCost:
    def test_returns_positive_int(self) -> None:
        atom = _atom()
        assert reader_cost(atom) >= 1

    def test_longer_object_costs_more(self) -> None:
        short = _atom(object_value="ok")
        long = _atom(object_value="a very long description of what this atom says about hiking")
        assert reader_cost(long) > reader_cost(short)

    def test_none_object_has_base_cost(self) -> None:
        atom = _atom(object_value=None)
        assert reader_cost(atom) >= 1

    def test_result_is_int(self) -> None:
        assert isinstance(reader_cost(_atom()), int)


class TestReductionScore:
    def test_high_risk_scores_higher(self) -> None:
        high = _atom(risk_severity=0.9, risk_class="safety")
        low = _atom(risk_severity=0.1, risk_class="preference")
        assert reduction_score(high, "query", []) > reduction_score(low, "query", [])

    def test_text_overlap_increases_score(self) -> None:
        on_topic = _atom(object_value="hiking activities enjoyment")
        off_topic = _atom(object_value="quantum physics theory")
        score_on = reduction_score(on_topic, "what hiking activities does user enjoy", [])
        score_off = reduction_score(off_topic, "what hiking activities does user enjoy", [])
        assert score_on > score_off

    def test_neg_polarity_lower_than_pos(self) -> None:
        pos = _atom(polarity="pos", risk_severity=0.5)
        neg = dataclasses.replace(pos, atom_id=new_ulid(), polarity="neg")
        assert reduction_score(pos, "query", []) > reduction_score(neg, "query", [])

    def test_score_non_negative(self) -> None:
        atom = _atom()
        assert reduction_score(atom, "any query", []) >= 0.0

    def test_credence_scales_score(self) -> None:
        high_cred = _atom(credence=1.0)
        low_cred = _atom(credence=0.1)
        assert reduction_score(high_cred, "query", []) > reduction_score(low_cred, "query", [])


class TestUtility:
    def test_utility_is_positive(self) -> None:
        atom = _atom(risk_severity=0.5)
        assert utility(atom, "query", []) > 0.0

    def test_utility_is_float(self) -> None:
        assert isinstance(utility(_atom(), "query", []), float)

    def test_utility_is_reduction_over_cost(self) -> None:
        atom = _atom()
        r = reduction_score(atom, "query", [])
        c = reader_cost(atom)
        assert abs(utility(atom, "query", []) - r / c) < 1e-9


class TestContradictionDetection:
    def test_same_entity_same_predicate_different_polarity(self) -> None:
        a = _atom(predicate="is", subject="alice", object_value="doctor", polarity="pos")
        b = dataclasses.replace(a, atom_id=new_ulid(), polarity="neg")
        pairs = detect_contradictions([a, b])
        assert len(pairs) == 1

    def test_different_entity_no_contradiction(self) -> None:
        a = _atom(predicate="is", subject="alice", object_value="doctor", entity_orbit_id="e:alice")
        b = _atom(predicate="is", subject="bob", object_value="nurse", entity_orbit_id="e:bob")
        assert detect_contradictions([a, b]) == []

    def test_same_entity_different_predicate_no_contradiction(self) -> None:
        a = _atom(predicate="is", subject="alice", object_value="doctor")
        b = dataclasses.replace(a, atom_id=new_ulid(), predicate="prefers")
        assert detect_contradictions([a, b]) == []

    def test_identity_different_object_contradiction(self) -> None:
        a = _atom(predicate="is", subject="alice", object_value="teacher", polarity="pos")
        b = dataclasses.replace(a, atom_id=new_ulid(), object_value="doctor")
        pairs = detect_contradictions([a, b])
        assert len(pairs) == 1

    def test_no_contradictions_empty(self) -> None:
        assert detect_contradictions([]) == []

    def test_no_contradictions_one_atom(self) -> None:
        assert detect_contradictions([_atom()]) == []


class TestHandleContradictions:
    def test_higher_credence_wins(self) -> None:
        a = _atom(predicate="is", subject="alice", object_value="doctor", credence=0.9)
        b = dataclasses.replace(a, atom_id=new_ulid(), object_value="nurse", credence=0.5)
        resolved, abstains = handle_contradictions([a, b])
        assert a in resolved
        assert b not in resolved
        assert abstains == []

    def test_equal_credence_both_removed(self) -> None:
        a = _atom(predicate="is", subject="alice", object_value="doctor", credence=0.7)
        b = dataclasses.replace(a, atom_id=new_ulid(), object_value="nurse", credence=0.7)
        resolved, abstains = handle_contradictions([a, b])
        assert a not in resolved
        assert b not in resolved
        assert len(abstains) == 2

    def test_no_contradictions_returns_unchanged(self) -> None:
        atoms = [_atom(), _atom(risk_class="medical")]
        resolved, abstains = handle_contradictions(atoms)
        assert len(resolved) == 2
        assert abstains == []


class TestPlanEvidencePack:
    def test_empty_atoms_returns_empty_pack(self) -> None:
        pack = plan_evidence_pack([], "query", _BUDGET)
        assert pack.atoms == ()

    def test_pack_respects_max_atoms(self) -> None:
        atoms = [_atom() for _ in range(20)]
        budget = ReaderBudget(max_atoms=5, max_tokens=10000, require_dependency_closure=False)
        pack = plan_evidence_pack(atoms, "query", budget)
        assert len(pack.atoms) <= 5

    def test_pack_respects_token_budget(self) -> None:
        atoms = [_atom(object_value="x" * 100) for _ in range(10)]
        budget = ReaderBudget(max_atoms=20, max_tokens=50, require_dependency_closure=False)
        pack = plan_evidence_pack(atoms, "query", budget)
        # Each atom with 100-char object costs ~37+ tokens; few should fit in 50
        assert len(pack.atoms) <= 3

    def test_pack_has_utility_scores(self) -> None:
        atoms = [_atom()]
        pack = plan_evidence_pack(atoms, "query", _BUDGET)
        assert "atom_utilities" in pack.utility_scores

    def test_high_utility_atom_selected_first(self) -> None:
        low_risk = _atom(risk_severity=0.1, object_value="general stuff")
        high_risk = _atom(risk_severity=0.9, risk_class="medical", object_value="heart disease")
        budget = ReaderBudget(max_atoms=1, max_tokens=1000, require_dependency_closure=False)
        pack = plan_evidence_pack([low_risk, high_risk], "health conditions", budget)
        assert high_risk.atom_id in pack.atoms

    def test_contradictions_resolved_in_pack(self) -> None:
        a = _atom(predicate="is", subject="alice", object_value="doctor", credence=0.9)
        b = dataclasses.replace(a, atom_id=new_ulid(), object_value="nurse", credence=0.5)
        pack = plan_evidence_pack([a, b], "alice profession", _BUDGET)
        # Only the higher-credence atom should be in the pack
        assert a.atom_id in pack.atoms
        assert b.atom_id not in pack.atoms

    def test_pack_id_is_ulid(self) -> None:
        pack = plan_evidence_pack([_atom()], "query", _BUDGET)
        assert len(pack.pack_id) == 26

    def test_library_dependency_closure(self) -> None:
        lib = AtomLibrary()
        dep = _atom(risk_class="identity", object_value="alice id info")
        lib.add(dep)
        main = dataclasses.replace(
            _atom(),
            atom_id=new_ulid(),
            depends_on=(dep.atom_id,),
        )
        lib.add(main)
        pack = plan_evidence_pack([main], "query", _BUDGET, library=lib)
        # Both main and dep should be in pack after closure
        assert dep.atom_id in pack.atoms


class TestTemporalAllenBonus:
    """Tests for tri-temporal Allen-relation bonus (Sprint 11)."""

    def _timed_atom(self, vf: int, vu: int, obs_time: int | None = None) -> MemoryAtom:
        """Create atom with explicit valid interval."""
        a = _atom()
        obs = obs_time if obs_time is not None else int(time.time())
        return dataclasses.replace(
            a, atom_id=new_ulid(), valid_from=vf, valid_until=vu, observation_time=obs
        )

    def test_overlapping_interval_gives_bonus(self) -> None:
        """Atom interval overlapping query window → Allen bonus > 0."""
        now = int(time.time())
        # Query window: [now, now+86400] (today)
        # Atom interval: [now-1000, now+1000] → overlaps
        atom = self._timed_atom(now - 1000, now + 1000, obs_time=now)
        bonus = temporal_allen_bonus(atom, now, now + 86400)
        assert bonus > 0.0

    def test_disjoint_interval_no_allen_bonus(self) -> None:
        """Atom interval entirely before query window → no Allen bonus, only recency."""
        now = int(time.time())
        # Atom: 10 years ago interval; query: today
        old = now - 10 * 365 * 86400
        atom = self._timed_atom(old, old + 86400, obs_time=old)
        bonus_with_allen = temporal_allen_bonus(atom, now, now + 86400)
        bonus_no_interval = temporal_allen_bonus(
            dataclasses.replace(atom, atom_id=new_ulid(), valid_from=None, valid_until=None),
            now,
            now + 86400,
        )
        # Allen adds 0.0 for PRECEDES; the difference is purely Allen
        assert bonus_with_allen <= bonus_no_interval + 0.01  # no Allen bonus added

    def test_no_query_window_no_allen_bonus(self) -> None:
        """When query has no temporal window, Allen axis is skipped."""
        now = int(time.time())
        atom = self._timed_atom(now, now + 86400, obs_time=now)
        bonus = temporal_allen_bonus(atom, None, None)
        # Only recency bonus applies; should be > 0 (recent atom) but < 0.3 max
        assert 0.0 <= bonus <= 0.3 + 0.01

    def test_recent_atom_higher_recency(self) -> None:
        """Recently observed atom gets higher recency bonus than old atom."""
        now = int(time.time())
        recent = _atom()
        recent = dataclasses.replace(recent, atom_id=new_ulid(), observation_time=now)
        old = _atom()
        old = dataclasses.replace(old, atom_id=new_ulid(), observation_time=now - 5 * 365 * 86400)
        bonus_recent = temporal_allen_bonus(recent, None, None)
        bonus_old = temporal_allen_bonus(old, None, None)
        assert bonus_recent > bonus_old

    def test_reduction_score_uses_temporal_bonus(self) -> None:
        """reduction_score with temporal args gives higher score for timed atoms matching query."""
        now = int(time.time())
        # Two atoms: one timed to match query window, one with no interval
        timed = _atom()
        timed = dataclasses.replace(
            timed,
            atom_id=new_ulid(),
            valid_from=now,
            valid_until=now + 86400,
            observation_time=now,
        )
        untimed = _atom()
        untimed = dataclasses.replace(untimed, atom_id=new_ulid(), observation_time=now)

        score_timed = reduction_score(timed, "query today", [], query_vf=now, query_vu=now + 86400)
        score_untimed = reduction_score(
            untimed, "query today", [], query_vf=now, query_vu=now + 86400
        )
        # Timed atom gets Allen bonus; untimed doesn't
        assert score_timed > score_untimed
