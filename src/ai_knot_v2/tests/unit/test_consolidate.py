"""Sprint 8 — consolidate operation unit tests."""

from __future__ import annotations

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.library import AtomLibrary
from ai_knot_v2.ops.consolidate import (
    ADJACENT_SECONDS,
    ConsolidateResult,
    consolidate_library,
    merge_intervals,
)
from ai_knot_v2.store.sqlite import SqliteStore

_DAY = 86400


def _atom(
    predicate: str = "visited",
    subject: str = "Alice",
    object_value: str = "hospital",
    polarity: str = "pos",
    entity_orbit_id: str = "orbit-1",
    valid_from: int | None = None,
    valid_until: int | None = None,
    credence: float = 0.8,
    evidence: tuple[str, ...] = (),
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
        predicate=predicate,
        subject=subject,
        object_value=object_value,
        polarity=polarity,  # type: ignore[arg-type]
        valid_from=valid_from,
        valid_until=valid_until,
        observation_time=0,
        belief_time=0,
        granularity="interval" if valid_from is not None else "instant",
        entity_orbit_id=entity_orbit_id,
        transport_provenance=(),
        depends_on=(),
        depended_by=(),
        risk_class="medical",
        risk_severity=0.5,
        regret_charge=0.0,
        irreducibility_score=0.0,
        protection_energy=1.0,
        action_affect_mask=0,
        credence=credence,
        evidence_episodes=evidence,
        synthesis_method="regex",
        validation_tests=(),
        contradiction_events=(),
    )


class TestMergeIntervals:
    def test_no_merge_single_atom(self) -> None:
        atom = _atom(valid_from=0, valid_until=_DAY * 10)
        result, removed = merge_intervals([atom])
        assert len(result) == 1
        assert removed == []

    def test_no_merge_disjoint_intervals(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 5)
        b = _atom(valid_from=_DAY * 10, valid_until=_DAY * 15)
        result, removed = merge_intervals([a, b])
        assert len(result) == 2
        assert removed == []

    def test_merge_overlapping_intervals(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 10, evidence=("ep-1",))
        b = _atom(valid_from=_DAY * 8, valid_until=_DAY * 20, evidence=("ep-2",))
        result, removed = merge_intervals([a, b])
        assert len(result) == 1
        assert len(removed) == 2
        merged = result[0]
        assert merged.valid_from == 0
        assert merged.valid_until == _DAY * 20

    def test_merge_adjacent_intervals(self) -> None:
        # gap = ADJACENT_SECONDS exactly → should merge
        a = _atom(valid_from=0, valid_until=_DAY * 5)
        b = _atom(valid_from=_DAY * 5 + ADJACENT_SECONDS, valid_until=_DAY * 10)
        result, removed = merge_intervals([a, b])
        assert len(result) == 1
        assert len(removed) == 2

    def test_no_merge_gap_beyond_adjacent(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 5)
        b = _atom(valid_from=_DAY * 5 + ADJACENT_SECONDS + 1, valid_until=_DAY * 10)
        result, removed = merge_intervals([a, b])
        assert len(result) == 2
        assert removed == []

    def test_merged_evidence_is_union(self) -> None:
        ep1, ep2 = new_ulid(), new_ulid()
        a = _atom(valid_from=0, valid_until=_DAY * 10, evidence=(ep1,))
        b = _atom(valid_from=_DAY * 8, valid_until=_DAY * 20, evidence=(ep2,))
        result, _ = merge_intervals([a, b])
        assert set(result[0].evidence_episodes) == {ep1, ep2}

    def test_merged_credence_is_max(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 10, credence=0.6)
        b = _atom(valid_from=_DAY * 8, valid_until=_DAY * 20, credence=0.9)
        result, _ = merge_intervals([a, b])
        assert result[0].credence == 0.9

    def test_merged_synthesis_method_is_fusion(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 10)
        b = _atom(valid_from=_DAY * 5, valid_until=_DAY * 15)
        result, _ = merge_intervals([a, b])
        assert result[0].synthesis_method == "fusion"

    def test_no_merge_different_predicates(self) -> None:
        a = _atom(predicate="visited", valid_from=0, valid_until=_DAY * 10)
        b = _atom(predicate="works_at", valid_from=_DAY * 5, valid_until=_DAY * 20)
        result, removed = merge_intervals([a, b])
        assert len(result) == 2
        assert removed == []

    def test_no_merge_different_objects(self) -> None:
        a = _atom(object_value="hospital", valid_from=0, valid_until=_DAY * 10)
        b = _atom(object_value="clinic", valid_from=_DAY * 5, valid_until=_DAY * 20)
        result, removed = merge_intervals([a, b])
        assert len(result) == 2
        assert removed == []

    def test_untimed_atoms_unchanged(self) -> None:
        a = _atom(valid_from=None, valid_until=None)
        b = _atom(valid_from=None, valid_until=None)
        result, removed = merge_intervals([a, b])
        assert len(result) == 2
        assert removed == []

    def test_three_way_merge(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 10, evidence=("ep1",))
        b = _atom(valid_from=_DAY * 8, valid_until=_DAY * 20, evidence=("ep2",))
        c = _atom(valid_from=_DAY * 18, valid_until=_DAY * 30, evidence=("ep3",))
        result, removed = merge_intervals([a, b, c])
        assert len(result) == 1
        assert len(removed) == 3
        merged = result[0]
        assert merged.valid_from == 0
        assert merged.valid_until == _DAY * 30
        assert set(merged.evidence_episodes) == {"ep1", "ep2", "ep3"}

    def test_mixed_timed_and_untimed(self) -> None:
        timed_a = _atom(valid_from=0, valid_until=_DAY * 10)
        timed_b = _atom(valid_from=_DAY * 5, valid_until=_DAY * 20)
        untimed = _atom(valid_from=None, valid_until=None)
        result, removed = merge_intervals([timed_a, timed_b, untimed])
        # timed pair merges; untimed stays
        assert len(result) == 2
        assert len(removed) == 2
        assert any(a.valid_from is None for a in result)


class TestConsolidateLibrary:
    def _setup(self, atoms: list[MemoryAtom]) -> tuple[AtomLibrary, SqliteStore]:
        store = SqliteStore(":memory:")
        library = AtomLibrary()
        for atom in atoms:
            store.save_atom(atom)
            library.add(atom)
        return library, store

    def test_noop_on_disjoint(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 5)
        b = _atom(valid_from=_DAY * 10, valid_until=_DAY * 15)
        library, store = self._setup([a, b])
        result = consolidate_library(library, store)
        assert result == ConsolidateResult(merged_count=0, atoms_removed=0, atoms_added=0)
        assert library.size() == 2

    def test_merges_overlapping_in_library(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 10)
        b = _atom(valid_from=_DAY * 8, valid_until=_DAY * 20)
        library, store = self._setup([a, b])
        result = consolidate_library(library, store)
        assert result.atoms_removed == 2
        assert result.atoms_added == 1
        assert library.size() == 1

    def test_merged_atom_in_store(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 10)
        b = _atom(valid_from=_DAY * 8, valid_until=_DAY * 20)
        library, store = self._setup([a, b])
        consolidate_library(library, store)
        merged = library.all_atoms()[0]
        assert store.get_atom(merged.atom_id) is not None

    def test_originals_removed_from_store(self) -> None:
        a = _atom(valid_from=0, valid_until=_DAY * 10)
        b = _atom(valid_from=_DAY * 8, valid_until=_DAY * 20)
        library, store = self._setup([a, b])
        consolidate_library(library, store)
        assert store.get_atom(a.atom_id) is None
        assert store.get_atom(b.atom_id) is None

    def test_noop_on_empty_library(self) -> None:
        library = AtomLibrary()
        store = SqliteStore(":memory:")
        result = consolidate_library(library, store)
        assert result == ConsolidateResult(merged_count=0, atoms_removed=0, atoms_added=0)

    def test_noop_on_untimed_atoms(self) -> None:
        a = _atom(valid_from=None, valid_until=None)
        b = _atom(valid_from=None, valid_until=None)
        library, store = self._setup([a, b])
        result = consolidate_library(library, store)
        assert result == ConsolidateResult(merged_count=0, atoms_removed=0, atoms_added=0)
        assert library.size() == 2
