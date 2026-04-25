"""CONSOLIDATE operation: interval-merge for temporal atoms.

Groups atoms by (entity_orbit_id, predicate, subject, object_value, polarity).
Within each group, merges atoms whose valid intervals overlap or are adjacent
(within ADJACENT_SECONDS of each other).

Merged atoms have:
- valid_from = min of constituent valid_from values
- valid_until = max of constituent valid_until values
- evidence_episodes = union of all constituent evidence sets
- credence = max credence among constituents
- synthesis_method = "fusion"
- a fresh atom_id (ULID)
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.library import AtomLibrary
from ai_knot_v2.core.provenance import AuditEvent
from ai_knot_v2.store.sqlite import SqliteStore

ADJACENT_SECONDS: int = 86400  # 1-day gap counts as adjacent (high-risk default)
_LOW_RISK_ADJACENT_SECONDS: int = 7 * 86400  # 7-day adjacency for low-risk atoms

_HIGH_RISK_CLASSES: frozenset[str] = frozenset({"safety", "medical", "legal", "identity"})


@dataclasses.dataclass(frozen=True, slots=True)
class ConsolidateResult:
    merged_count: int
    atoms_removed: int
    atoms_added: int


def _overlaps_or_adjacent(
    a: MemoryAtom, b: MemoryAtom, adjacent_seconds: int = ADJACENT_SECONDS
) -> bool:
    if a.valid_from is None or a.valid_until is None:
        return False
    if b.valid_from is None or b.valid_until is None:
        return False
    return (
        a.valid_from <= b.valid_until + adjacent_seconds
        and b.valid_from <= a.valid_until + adjacent_seconds
    )


def _merge_two(a: MemoryAtom, b: MemoryAtom) -> MemoryAtom:
    vf: int | None = a.valid_from
    vu: int | None = a.valid_until
    if b.valid_from is not None:
        vf = min(vf, b.valid_from) if vf is not None else b.valid_from
    if b.valid_until is not None:
        vu = max(vu, b.valid_until) if vu is not None else b.valid_until
    merged_evidence = tuple(sorted(set(a.evidence_episodes) | set(b.evidence_episodes)))
    return dataclasses.replace(
        a,
        atom_id=new_ulid(),
        valid_from=vf,
        valid_until=vu,
        evidence_episodes=merged_evidence,
        credence=max(a.credence, b.credence),
        synthesis_method="fusion",
    )


def _merge_group(
    atoms: list[MemoryAtom],
    adjacent_seconds: int = ADJACENT_SECONDS,
) -> list[MemoryAtom]:
    """Greedy interval merging within atoms sharing the same triple."""
    timed = [a for a in atoms if a.valid_from is not None and a.valid_until is not None]
    untimed = [a for a in atoms if a.valid_from is None or a.valid_until is None]

    if len(timed) <= 1:
        return atoms

    timed.sort(key=lambda a: a.valid_from if a.valid_from is not None else 0)

    merged: list[MemoryAtom] = [timed[0]]
    for atom in timed[1:]:
        if _overlaps_or_adjacent(merged[-1], atom, adjacent_seconds=adjacent_seconds):
            merged[-1] = _merge_two(merged[-1], atom)
        else:
            merged.append(atom)

    return merged + untimed


def merge_intervals(
    atoms: list[MemoryAtom],
    adjacent_seconds: int = ADJACENT_SECONDS,
) -> tuple[list[MemoryAtom], list[MemoryAtom]]:
    """Merge temporally overlapping atoms sharing the same constraint triple.

    Returns (result_atoms, removed_originals).
    result_atoms contains both unchanged atoms and new merged atoms.
    removed_originals contains every atom that was replaced by a merge.
    """
    GroupKey = tuple[str, str, str, str | None, str]
    groups: defaultdict[GroupKey, list[MemoryAtom]] = defaultdict(list)
    for atom in atoms:
        key: GroupKey = (
            atom.entity_orbit_id,
            atom.predicate,
            atom.subject,
            atom.object_value,
            atom.polarity,
        )
        groups[key].append(atom)

    result_atoms: list[MemoryAtom] = []
    removed_originals: list[MemoryAtom] = []

    for group_atoms in groups.values():
        merged = _merge_group(group_atoms, adjacent_seconds=adjacent_seconds)
        merged_ids = {a.atom_id for a in merged}
        truly_removed = [a for a in group_atoms if a.atom_id not in merged_ids]
        if truly_removed:
            removed_originals.extend(truly_removed)
        result_atoms.extend(merged)

    return result_atoms, removed_originals


def merge_intervals_rg(
    atoms: list[MemoryAtom],
) -> tuple[list[MemoryAtom], list[MemoryAtom]]:
    """RG-flow consolidation: risk-stratified interval merge.

    High-risk atoms (safety/medical/legal/identity): 1-day adjacency — conservative.
    Low-risk atoms: 7-day adjacency — more aggressive temporal fusion.
    """
    high_risk = [a for a in atoms if a.risk_class in _HIGH_RISK_CLASSES]
    low_risk = [a for a in atoms if a.risk_class not in _HIGH_RISK_CLASSES]

    high_result, high_removed = merge_intervals(high_risk, adjacent_seconds=ADJACENT_SECONDS)
    low_result, low_removed = merge_intervals(low_risk, adjacent_seconds=_LOW_RISK_ADJACENT_SECONDS)

    return high_result + low_result, high_removed + low_removed


def consolidate_library(
    library: AtomLibrary,
    store: SqliteStore,
) -> ConsolidateResult:
    """Run a full interval-merge consolidation pass on the library."""
    atoms = library.all_atoms()
    original_ids = {a.atom_id for a in atoms}
    result_atoms, removed_originals = merge_intervals_rg(atoms)

    if not removed_originals:
        return ConsolidateResult(merged_count=0, atoms_removed=0, atoms_added=0)

    new_atoms = [a for a in result_atoms if a.atom_id not in original_ids]
    now = int(time.time())

    for atom in removed_originals:
        library.remove(atom.atom_id)
        store.delete_atom(atom.atom_id)
        store.append_audit_event(
            AuditEvent(
                event_id=new_ulid(),
                operation="consolidate",
                atom_id=atom.atom_id,
                agent_id=atom.agent_id,
                timestamp=now,
                details={"action": "remove", "reason": "interval_merge"},
            )
        )

    for atom in new_atoms:
        library.add(atom)
        store.save_atom(atom)
        store.append_audit_event(
            AuditEvent(
                event_id=new_ulid(),
                operation="consolidate",
                atom_id=atom.atom_id,
                agent_id=atom.agent_id,
                timestamp=now,
                details={"action": "add", "synthesis_method": "fusion"},
            )
        )

    return ConsolidateResult(
        merged_count=len(new_atoms),
        atoms_removed=len(removed_originals),
        atoms_added=len(new_atoms),
    )
