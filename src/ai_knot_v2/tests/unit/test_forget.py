"""Sprint 8 — forget operation unit tests."""

from __future__ import annotations

import math

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.library import AtomLibrary
from ai_knot_v2.ops.forget import (
    BASE_DECAY_RATE,
    FORGET_THRESHOLD,
    decay_protection_energy,
    run_forget_pass,
    should_forget,
)
from ai_knot_v2.store.sqlite import SqliteStore


def _atom(
    risk_severity: float = 0.5,
    protection_energy: float = 1.0,
    entity_orbit_id: str = "orbit-1",
    risk_class: str = "preference",
) -> MemoryAtom:
    return MemoryAtom(
        atom_id=new_ulid(),
        agent_id="agent-1",
        user_id=None,
        variables=(),
        causal_graph=(),
        kernel_kind="point",
        kernel_payload={},
        intervention_domain=(),
        predicate="prefers",
        subject="Alice",
        object_value="hiking",
        polarity="pos",
        valid_from=None,
        valid_until=None,
        observation_time=0,
        belief_time=0,
        granularity="instant",
        entity_orbit_id=entity_orbit_id,
        transport_provenance=(),
        depends_on=(),
        depended_by=(),
        risk_class=risk_class,  # type: ignore[arg-type]
        risk_severity=risk_severity,
        regret_charge=0.0,
        irreducibility_score=0.0,
        protection_energy=protection_energy,
        action_affect_mask=0,
        credence=1.0,
        evidence_episodes=(),
        synthesis_method="regex",
        validation_tests=(),
        contradiction_events=(),
    )


class TestDecayProtectionEnergy:
    def test_decay_reduces_energy(self) -> None:
        atom = _atom(risk_severity=0.0, protection_energy=1.0)
        decayed = decay_protection_energy(atom, elapsed_days=30)
        assert decayed.protection_energy < 1.0

    def test_zero_days_no_change(self) -> None:
        atom = _atom(risk_severity=0.5, protection_energy=0.8)
        decayed = decay_protection_energy(atom, elapsed_days=0)
        assert abs(decayed.protection_energy - 0.8) < 1e-9

    def test_high_risk_decays_slower(self) -> None:
        low_risk = _atom(risk_severity=0.0, protection_energy=1.0)
        high_risk = _atom(risk_severity=1.0, protection_energy=1.0)
        elapsed = 100
        low_decayed = decay_protection_energy(low_risk, elapsed)
        high_decayed = decay_protection_energy(high_risk, elapsed)
        assert high_decayed.protection_energy > low_decayed.protection_energy

    def test_ode_formula_matches(self) -> None:
        atom = _atom(risk_severity=0.0, protection_energy=1.0)
        elapsed = 50.0
        k = BASE_DECAY_RATE / (1.0 + 0.0 * 5.0)
        expected = math.exp(-k * elapsed)
        decayed = decay_protection_energy(atom, elapsed)
        assert abs(decayed.protection_energy - expected) < 1e-9

    def test_energy_never_negative(self) -> None:
        atom = _atom(risk_severity=0.0, protection_energy=0.001)
        decayed = decay_protection_energy(atom, elapsed_days=10000)
        assert decayed.protection_energy >= 0.0

    def test_other_fields_unchanged(self) -> None:
        atom = _atom(risk_severity=0.5, protection_energy=1.0)
        decayed = decay_protection_energy(atom, elapsed_days=30)
        assert decayed.atom_id == atom.atom_id
        assert decayed.predicate == atom.predicate
        assert decayed.subject == atom.subject


class TestShouldForget:
    def test_forget_at_threshold(self) -> None:
        atom = _atom(protection_energy=FORGET_THRESHOLD)
        assert should_forget(atom)

    def test_forget_below_threshold(self) -> None:
        atom = _atom(protection_energy=0.0)
        assert should_forget(atom)

    def test_no_forget_above_threshold(self) -> None:
        atom = _atom(protection_energy=FORGET_THRESHOLD + 0.01)
        assert not should_forget(atom)

    def test_fresh_atom_not_forgotten(self) -> None:
        atom = _atom(protection_energy=1.0)
        assert not should_forget(atom)


class TestRunForgetPass:
    def _store_and_library(self, atoms: list[MemoryAtom]) -> tuple[SqliteStore, AtomLibrary]:
        store = SqliteStore(":memory:")
        library = AtomLibrary()
        for atom in atoms:
            store.save_atom(atom)
            library.add(atom)
        return store, library

    def test_removes_low_risk_after_many_days(self) -> None:
        # risk_severity=0.0: k=0.02, after 200 days E = exp(-4) ≈ 0.018 < 0.05
        atom = _atom(risk_severity=0.0, protection_energy=1.0)
        store, library = self._store_and_library([atom])
        forgotten, retained = run_forget_pass(library, store, elapsed_days=200)
        assert forgotten == 1
        assert retained == 0
        assert library.size() == 0

    def test_retains_high_risk_after_same_period(self) -> None:
        # risk_severity=1.0: k=0.02/6≈0.00333, after 200 days E = exp(-0.667) ≈ 0.51 > 0.05
        atom = _atom(risk_severity=1.0, protection_energy=1.0)
        store, library = self._store_and_library([atom])
        forgotten, retained = run_forget_pass(library, store, elapsed_days=200)
        assert forgotten == 0
        assert retained == 1
        assert library.size() == 1

    def test_retained_atom_has_decayed_energy(self) -> None:
        atom = _atom(risk_severity=1.0, protection_energy=1.0)
        store, library = self._store_and_library([atom])
        run_forget_pass(library, store, elapsed_days=100)
        updated = library.get(atom.atom_id)
        assert updated is not None
        assert updated.protection_energy < 1.0

    def test_mixed_atoms(self) -> None:
        low = _atom(risk_severity=0.0, protection_energy=1.0, entity_orbit_id="orbit-low")
        high = _atom(risk_severity=1.0, protection_energy=1.0, entity_orbit_id="orbit-high")
        store, library = self._store_and_library([low, high])
        forgotten, retained = run_forget_pass(library, store, elapsed_days=200)
        assert forgotten == 1
        assert retained == 1

    def test_empty_library_noop(self) -> None:
        store = SqliteStore(":memory:")
        library = AtomLibrary()
        forgotten, retained = run_forget_pass(library, store, elapsed_days=100)
        assert forgotten == 0
        assert retained == 0
