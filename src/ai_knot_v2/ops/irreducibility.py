"""Structural irreducibility witness builder.

Determines whether a MemoryAtom is irreducible relative to its entity orbit —
i.e., it covers action-class ground that no other atom in the orbit covers.

No LLM. All operations are deterministic rule-based enumeration.
"""

from __future__ import annotations

from ai_knot_v2.core.action_calculus import compute_action_affect_mask
from ai_knot_v2.core.action_taxonomy import ActionClass
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.library import AtomLibrary


def compute_irreducibility_score(atom: MemoryAtom, library: AtomLibrary) -> float:
    """Return irreducibility score ∈ [0.0, 1.0].

    1.0 = fully irreducible (unique action coverage in entity orbit).
    0.0 = fully redundant (all action classes covered by peers).
    0.5 = neutral (atom has no action coverage, e.g., ambient risk class).
    """
    atom_mask = compute_action_affect_mask(atom)
    if atom_mask == 0:
        return 0.5

    peer_atoms = [
        a for a in library.query_by_entity(atom.entity_orbit_id) if a.atom_id != atom.atom_id
    ]

    covered_by_peers = 0
    for peer in peer_atoms:
        covered_by_peers |= compute_action_affect_mask(peer)

    # Bits in atom_mask not covered by any peer
    unique_bits = atom_mask & ~covered_by_peers
    atom_bit_count = bin(atom_mask).count("1")
    unique_bit_count = bin(unique_bits).count("1")
    return unique_bit_count / atom_bit_count


def is_structurally_dominated(atom: MemoryAtom, library: AtomLibrary) -> bool:
    """Return True if atom's action coverage is fully subsumed by peer atoms.

    An atom is dominated when:
    1. It has non-zero action coverage, AND
    2. All action bits are already covered by existing orbit peers.
    """
    atom_mask = compute_action_affect_mask(atom)
    if atom_mask == 0:
        return False

    peer_atoms = [
        a for a in library.query_by_entity(atom.entity_orbit_id) if a.atom_id != atom.atom_id
    ]

    covered_by_peers = 0
    for peer in peer_atoms:
        covered_by_peers |= compute_action_affect_mask(peer)

    return (atom_mask & ~covered_by_peers) == 0


def enumerate_witness_set(atoms: list[MemoryAtom]) -> list[MemoryAtom]:
    """Return a minimal irreducible subset of atoms covering all action classes.

    Greedy enumeration: pick atoms in descending risk_severity order,
    add to witness set only if they introduce new action-class bits.
    """
    sorted_atoms = sorted(atoms, key=lambda a: a.risk_severity, reverse=True)
    covered = 0
    witness: list[MemoryAtom] = []

    for atom in sorted_atoms:
        mask = compute_action_affect_mask(atom)
        new_bits = mask & ~covered
        if new_bits != 0 or mask == 0:
            witness.append(atom)
            covered |= mask

    return witness


def action_coverage_gap(atoms: list[MemoryAtom], target_mask: int) -> ActionClass:
    """Return the action class bits in target_mask not covered by any atom in atoms."""
    covered = 0
    for atom in atoms:
        covered |= compute_action_affect_mask(atom)
    gap_bits = target_mask & ~covered
    return ActionClass(gap_bits)
