"""AtomLibrary — in-memory typed index over MemoryAtom."""

from __future__ import annotations

import contextlib
from collections import defaultdict

from ai_knot_v2.core.atom import MemoryAtom


class AtomLibrary:
    def __init__(self) -> None:
        self._atoms: dict[str, MemoryAtom] = {}
        self._by_entity: defaultdict[str, list[str]] = defaultdict(list)
        self._by_predicate: defaultdict[str, list[str]] = defaultdict(list)
        self._by_risk_class: defaultdict[str, list[str]] = defaultdict(list)

    def add(self, atom: MemoryAtom) -> None:
        if atom.atom_id in self._atoms:
            return
        self._atoms[atom.atom_id] = atom
        self._by_entity[atom.entity_orbit_id].append(atom.atom_id)
        self._by_predicate[atom.predicate].append(atom.atom_id)
        self._by_risk_class[atom.risk_class].append(atom.atom_id)

    def get(self, atom_id: str) -> MemoryAtom | None:
        return self._atoms.get(atom_id)

    def query_by_entity(self, entity_orbit_id: str) -> list[MemoryAtom]:
        return [self._atoms[aid] for aid in self._by_entity.get(entity_orbit_id, [])]

    def query_by_predicate(self, predicate: str) -> list[MemoryAtom]:
        return [self._atoms[aid] for aid in self._by_predicate.get(predicate, [])]

    def query_by_risk_class(self, risk_class: str) -> list[MemoryAtom]:
        return [self._atoms[aid] for aid in self._by_risk_class.get(risk_class, [])]

    def dependency_closure(self, atom_ids: set[str]) -> set[str]:
        """Sprint 1 stub: returns input unchanged. Full transitive closure in Sprint 2."""
        return atom_ids

    def remove(self, atom_id: str) -> None:
        atom = self._atoms.pop(atom_id, None)
        if atom is None:
            return
        with contextlib.suppress(ValueError):
            self._by_entity[atom.entity_orbit_id].remove(atom_id)
        with contextlib.suppress(ValueError):
            self._by_predicate[atom.predicate].remove(atom_id)
        with contextlib.suppress(ValueError):
            self._by_risk_class[atom.risk_class].remove(atom_id)

    def all_atoms(self) -> list[MemoryAtom]:
        return list(self._atoms.values())

    def size(self) -> int:
        return len(self._atoms)
