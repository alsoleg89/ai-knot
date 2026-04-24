"""Sprint 3b — Action Calculus unit tests.

50 synthetic cases; gate: ≥ 95% correct prediction (≥ 48/50).
No LOCOMO data. All cases are domain-generic.
"""

from __future__ import annotations

import dataclasses
from datetime import date

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.action_calculus import (
    ActionSignature,
    action_distance,
    canonical_action_signature,
    compute_action_affect_mask,
    predict_action,
)
from ai_knot_v2.core.action_taxonomy import ActionClass
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.episode import RawEpisode
from ai_knot_v2.core.library import AtomLibrary
from ai_knot_v2.ops.atomizer import Atomizer
from ai_knot_v2.ops.irreducibility import (
    compute_irreducibility_score,
    enumerate_witness_set,
    is_structurally_dominated,
)


def _atom(
    risk_class: str = "preference",
    predicate: str = "prefers",
    object_value: str | None = "hiking",
    polarity: str = "pos",
    action_affect_mask: int = 0,
) -> MemoryAtom:
    uid = new_ulid()
    entity = f"entity:user_{uid[:4]}"
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
        object_value=object_value,
        polarity=polarity,  # type: ignore[arg-type]
        valid_from=None,
        valid_until=None,
        observation_time=1_700_000_000,
        belief_time=1_700_000_000,
        granularity="instant",  # type: ignore[arg-type]
        entity_orbit_id=entity,
        transport_provenance=("session-1",),
        depends_on=(),
        depended_by=(),
        risk_class=risk_class,  # type: ignore[arg-type]
        risk_severity=0.2,
        regret_charge=0.2,
        irreducibility_score=1.0,
        protection_energy=0.4,
        action_affect_mask=action_affect_mask,
        credence=0.9,
        evidence_episodes=(uid,),
        synthesis_method="regex",
        validation_tests=(),
        contradiction_events=(),
    )


def _ep(text: str, speaker: str = "user") -> RawEpisode:
    return RawEpisode(
        episode_id=new_ulid(),
        agent_id="agent-1",
        user_id="user-1",
        session_id="session-1",
        turn_index=0,
        speaker=speaker,  # type: ignore[arg-type]
        text=text,
        timestamp=1_700_000_000,
    )


SESSION_DATE = date(2024, 1, 15)


# ---------------------------------------------------------------------------
# TestComputeActionAffectMask (20 cases)
# ---------------------------------------------------------------------------


class TestComputeActionAffectMask:
    def test_preference_pos_has_recommend(self) -> None:
        atom = _atom(risk_class="preference", predicate="prefers", polarity="pos")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.RECOMMEND in mask

    def test_preference_pos_has_personalize(self) -> None:
        atom = _atom(risk_class="preference", predicate="prefers", polarity="pos")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.PERSONALIZE in mask

    def test_preference_neg_has_avoid(self) -> None:
        atom = _atom(risk_class="preference", predicate="dislikes", polarity="neg")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.AVOID in mask

    def test_preference_neg_has_personalize(self) -> None:
        atom = _atom(risk_class="preference", predicate="dislikes", polarity="neg")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.PERSONALIZE in mask

    def test_preference_neg_no_recommend(self) -> None:
        atom = _atom(risk_class="preference", predicate="dislikes", polarity="neg")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.RECOMMEND not in mask

    def test_medical_diagnosis_has_diagnose(self) -> None:
        atom = _atom(risk_class="medical", object_value="heart disease")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.DIAGNOSE in mask

    def test_medical_always_has_monitor(self) -> None:
        atom = _atom(risk_class="medical", object_value="something")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.MONITOR in mask

    def test_medical_medication_has_prescribe(self) -> None:
        atom = _atom(risk_class="medical", object_value="penicillin medication")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.PRESCRIBE in mask

    def test_medical_appointment_has_schedule_appt(self) -> None:
        atom = _atom(risk_class="medical", object_value="doctor appointment")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.SCHEDULE_APPT in mask

    def test_medical_specialist_has_refer(self) -> None:
        atom = _atom(risk_class="medical", object_value="cardiologist specialist")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.REFER in mask

    def test_scheduling_event_has_create_event(self) -> None:
        atom = _atom(risk_class="scheduling", object_value="project meeting")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.CREATE_EVENT in mask

    def test_scheduling_always_has_remind(self) -> None:
        atom = _atom(risk_class="scheduling", object_value="project meeting")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.REMIND in mask

    def test_scheduling_cancel_has_cancel_event(self) -> None:
        atom = _atom(risk_class="scheduling", object_value="cancelled meeting")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.CANCEL_EVENT in mask

    def test_scheduling_reschedule_has_reschedule(self) -> None:
        atom = _atom(risk_class="scheduling", object_value="rescheduled appointment")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.RESCHEDULE in mask

    def test_identity_has_update_profile(self) -> None:
        atom = _atom(risk_class="identity", predicate="is", object_value="engineer")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.UPDATE_PROFILE in mask

    def test_identity_name_has_verify_identity(self) -> None:
        atom = _atom(risk_class="identity", predicate="is", object_value="alice name")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.VERIFY_IDENTITY in mask

    def test_identity_location_has_link_entity(self) -> None:
        atom = _atom(risk_class="identity", predicate="lives_in", object_value="London")
        mask = ActionClass(compute_action_affect_mask(atom))
        assert ActionClass.LINK_ENTITY in mask

    def test_ambient_returns_zero(self) -> None:
        atom = _atom(risk_class="ambient", object_value="happy")
        mask = compute_action_affect_mask(atom)
        assert mask == 0

    def test_finance_returns_nonzero(self) -> None:
        atom = _atom(risk_class="finance", object_value="120000 per year")
        mask = compute_action_affect_mask(atom)
        assert mask != 0

    def test_mask_is_int(self) -> None:
        atom = _atom(risk_class="preference", object_value="hiking")
        mask = compute_action_affect_mask(atom)
        assert isinstance(mask, int)


# ---------------------------------------------------------------------------
# TestActionDistance (10 cases)
# ---------------------------------------------------------------------------


class TestActionDistance:
    def test_identical_masks_zero_distance(self) -> None:
        mask = int(ActionClass.RECOMMEND | ActionClass.PERSONALIZE)
        assert action_distance(mask, mask) == 0.0

    def test_empty_masks_zero_distance(self) -> None:
        assert action_distance(0, 0) == 0.0

    def test_disjoint_masks_max_distance(self) -> None:
        a = int(ActionClass.DIAGNOSE)
        b = int(ActionClass.RECOMMEND)
        assert action_distance(a, b) == 1.0

    def test_partial_overlap_between_zero_and_one(self) -> None:
        a = int(ActionClass.RECOMMEND | ActionClass.PERSONALIZE)
        b = int(ActionClass.PERSONALIZE | ActionClass.AVOID)
        d = action_distance(a, b)
        assert 0.0 < d < 1.0

    def test_symmetric(self) -> None:
        a = int(ActionClass.DIAGNOSE | ActionClass.MONITOR)
        b = int(ActionClass.CREATE_EVENT | ActionClass.REMIND)
        assert action_distance(a, b) == action_distance(b, a)

    def test_one_empty_mask_full_distance(self) -> None:
        a = int(ActionClass.RECOMMEND)
        d = action_distance(a, 0)
        assert d == 1.0

    def test_distance_is_float(self) -> None:
        assert isinstance(action_distance(1, 2), float)

    def test_subset_has_nonzero_distance(self) -> None:
        a = int(ActionClass.DIAGNOSE | ActionClass.MONITOR | ActionClass.PRESCRIBE)
        b = int(ActionClass.DIAGNOSE)
        d = action_distance(a, b)
        assert d > 0.0

    def test_distance_at_most_one(self) -> None:
        import random

        rng = random.Random(42)
        for _ in range(20):
            a = rng.randint(0, 65535)
            b = rng.randint(0, 65535)
            assert 0.0 <= action_distance(a, b) <= 1.0

    def test_full_match_same_combined_mask(self) -> None:
        combined = int(ActionClass.RECOMMEND | ActionClass.PERSONALIZE | ActionClass.AVOID)
        assert action_distance(combined, combined) == 0.0


# ---------------------------------------------------------------------------
# TestPredictAction (10 cases)
# ---------------------------------------------------------------------------


class TestPredictAction:
    def test_empty_signature_returns_none(self) -> None:
        assert predict_action(()) == ActionClass.NONE

    def test_medical_domain_token(self) -> None:
        sig: ActionSignature = ("medical",)
        result = predict_action(sig)
        medical_mask = (
            ActionClass.DIAGNOSE
            | ActionClass.MONITOR
            | ActionClass.PRESCRIBE
            | ActionClass.SCHEDULE_APPT
            | ActionClass.REFER
        )
        assert result in medical_mask

    def test_preference_domain_token(self) -> None:
        sig: ActionSignature = ("preference",)
        result = predict_action(sig)
        assert result in ActionClass.RECOMMEND | ActionClass.AVOID | ActionClass.PERSONALIZE

    def test_direct_class_name_wins(self) -> None:
        sig: ActionSignature = ("medical", "DIAGNOSE")
        result = predict_action(sig)
        assert result == ActionClass.DIAGNOSE

    def test_scheduling_domain_token(self) -> None:
        sig: ActionSignature = ("scheduling",)
        result = predict_action(sig)
        assert result in (
            ActionClass.CREATE_EVENT
            | ActionClass.CANCEL_EVENT
            | ActionClass.RESCHEDULE
            | ActionClass.REMIND
            | ActionClass.SCHEDULE_APPT
        )

    def test_identity_domain_token(self) -> None:
        sig: ActionSignature = ("identity",)
        result = predict_action(sig)
        identity_mask = (
            ActionClass.UPDATE_PROFILE | ActionClass.VERIFY_IDENTITY | ActionClass.LINK_ENTITY
        )
        assert result in identity_mask

    def test_direct_avoid_name(self) -> None:
        sig: ActionSignature = ("AVOID",)
        result = predict_action(sig)
        assert result == ActionClass.AVOID

    def test_direct_monitor_name(self) -> None:
        sig: ActionSignature = ("MONITOR",)
        result = predict_action(sig)
        assert result == ActionClass.MONITOR

    def test_combined_medical_and_diagnose(self) -> None:
        sig: ActionSignature = ("medical", "DIAGNOSE", "MONITOR")
        result = predict_action(sig)
        # DIAGNOSE gets +3+1, MONITOR gets +3+1 → tie → lower value wins
        assert result == ActionClass.DIAGNOSE

    def test_result_is_action_class(self) -> None:
        sig: ActionSignature = ("preference", "RECOMMEND")
        result = predict_action(sig)
        assert isinstance(result, ActionClass)


# ---------------------------------------------------------------------------
# TestCanonicalSignature (5 cases)
# ---------------------------------------------------------------------------


class TestCanonicalSignature:
    def test_empty_atoms_empty_query_returns_empty(self) -> None:
        sig = canonical_action_signature([], "")
        assert sig == ()

    def test_medical_query_adds_medical_domain(self) -> None:
        sig = canonical_action_signature([], "what is the doctor's diagnosis?")
        assert "medical" in sig

    def test_preference_query_adds_preference_domain(self) -> None:
        sig = canonical_action_signature([], "what does Alice prefer to eat?")
        assert "preference" in sig

    def test_atom_mask_contributes_class_name(self) -> None:
        atom = _atom(risk_class="preference", action_affect_mask=int(ActionClass.RECOMMEND))
        sig = canonical_action_signature([atom], "")
        assert "RECOMMEND" in sig

    def test_signature_is_sorted_tuple(self) -> None:
        atom = _atom(
            risk_class="medical",
            action_affect_mask=int(ActionClass.DIAGNOSE | ActionClass.MONITOR),
        )
        sig = canonical_action_signature([atom], "doctor appointment")
        assert sig == tuple(sorted(sig))


# ---------------------------------------------------------------------------
# TestIrreducibility (5 cases)
# ---------------------------------------------------------------------------


class TestIrreducibility:
    def test_single_atom_fully_irreducible(self) -> None:
        lib = AtomLibrary()
        atom = _atom(risk_class="preference", action_affect_mask=int(ActionClass.RECOMMEND))
        lib.add(atom)
        score = compute_irreducibility_score(atom, lib)
        assert score == 1.0

    def test_two_identical_coverage_atoms(self) -> None:
        lib = AtomLibrary()
        orbit = "entity:user_x"
        a1 = dataclasses.replace(
            _atom(risk_class="medical", action_affect_mask=int(ActionClass.MONITOR)),
            entity_orbit_id=orbit,
        )
        a2 = dataclasses.replace(
            _atom(risk_class="medical", action_affect_mask=int(ActionClass.MONITOR)),
            entity_orbit_id=orbit,
        )
        lib.add(a1)
        lib.add(a2)
        score = compute_irreducibility_score(a1, lib)
        assert score == 0.0

    def test_is_structurally_dominated_true(self) -> None:
        lib = AtomLibrary()
        orbit = "entity:user_y"
        a1 = dataclasses.replace(
            _atom(
                risk_class="preference",
                action_affect_mask=int(ActionClass.RECOMMEND | ActionClass.PERSONALIZE),
            ),
            entity_orbit_id=orbit,
        )
        a2 = dataclasses.replace(
            _atom(risk_class="preference", action_affect_mask=int(ActionClass.RECOMMEND)),
            entity_orbit_id=orbit,
        )
        lib.add(a1)
        lib.add(a2)
        # a2's coverage is a subset of a1's — structurally dominated
        assert is_structurally_dominated(a2, lib) is True

    def test_ambient_atom_not_dominated(self) -> None:
        lib = AtomLibrary()
        atom = _atom(risk_class="ambient", action_affect_mask=0)
        lib.add(atom)
        assert is_structurally_dominated(atom, lib) is False

    def test_enumerate_witness_set_minimal(self) -> None:
        orbit = "entity:user_z"
        a_med = dataclasses.replace(
            _atom(risk_class="medical", action_affect_mask=int(ActionClass.MONITOR)),
            entity_orbit_id=orbit,
        )
        a_pref = dataclasses.replace(
            _atom(risk_class="preference", action_affect_mask=int(ActionClass.RECOMMEND)),
            entity_orbit_id=orbit,
        )
        a_dup = dataclasses.replace(
            _atom(risk_class="medical", action_affect_mask=int(ActionClass.MONITOR)),
            entity_orbit_id=orbit,
        )
        witness = enumerate_witness_set([a_med, a_pref, a_dup])
        # Duplicate MONITOR coverage should not both appear
        monitor_atoms = [
            w for w in witness if ActionClass(w.action_affect_mask) & ActionClass.MONITOR
        ]
        assert len(monitor_atoms) <= 1


# ---------------------------------------------------------------------------
# Integration: Atomizer → compute_action_affect_mask (gate)
# ---------------------------------------------------------------------------


class TestAtomizerMaskGate:
    """50th+ case: verify Atomizer-produced atoms get valid masks."""

    def _atomize(self, text: str) -> list[MemoryAtom]:
        return Atomizer().atomize(_ep(text), SESSION_DATE)

    def test_preference_atom_gets_nonzero_mask(self) -> None:
        atoms = self._atomize("I love hiking.")
        pref = [a for a in atoms if a.risk_class == "preference"]
        assert pref, "expected preference atom"
        for a in pref:
            updated = dataclasses.replace(a, action_affect_mask=compute_action_affect_mask(a))
            assert updated.action_affect_mask != 0

    def test_medical_atom_gets_monitor_flag(self) -> None:
        atoms = self._atomize("Alice has a doctor appointment.")
        med = [a for a in atoms if a.risk_class in ("medical", "scheduling")]
        assert med
        masks = [compute_action_affect_mask(a) for a in med]
        relevant = ActionClass.MONITOR | ActionClass.SCHEDULE_APPT | ActionClass.CREATE_EVENT
        assert any(ActionClass(m) & relevant for m in masks)
