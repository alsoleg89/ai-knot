"""Information-theoretic helpers — no LLM, no external dependencies.

Provides Level-3 ESWP formulas for regret_charge and irreducibility_score.
"""

from __future__ import annotations

from ai_knot_v2.core.atom import MemoryAtom


def compute_regret_charge_v2(atom: MemoryAtom, curvature: float = 0.0) -> float:
    """Compute regret charge as proxy for expected marginal free energy.

    Formula:
      action_bits  = popcount(action_affect_mask)
      delta_q      = risk_severity × (1 + 0.3 × action_bits)
      regret_charge = min(1.0, delta_q × (1 + 0.5×curvature) × (1 + 0.2×risk_severity))

    curvature = 1.0 when the atom has active contradiction events, else 0.0.
    """
    action_bits = bin(atom.action_affect_mask).count("1")
    delta_q = atom.risk_severity * (1.0 + 0.3 * action_bits)
    curvature_term = 1.0 + 0.5 * curvature
    danger_term = 1.0 + 0.2 * atom.risk_severity
    return min(1.0, delta_q * curvature_term * danger_term)


def compute_irreducibility(
    atom: MemoryAtom,
    peer_atoms: list[MemoryAtom],
) -> float:
    """Score how irreplaceable this atom is within its orbit.

    1.0 = no peers with same predicate → fully irreducible.
    Decreases as the number of temporally overlapping same-predicate peers grows.
    """
    if not peer_atoms:
        return 1.0

    def _overlaps(a: MemoryAtom, b: MemoryAtom) -> bool:
        if a.valid_from is None or b.valid_from is None:
            return True
        if a.valid_until is None or b.valid_until is None:
            return True
        return a.valid_from <= b.valid_until and b.valid_from <= a.valid_until

    overlapping = sum(
        1
        for p in peer_atoms
        if p.predicate == atom.predicate and p.atom_id != atom.atom_id and _overlaps(atom, p)
    )
    redundancy = overlapping / (len(peer_atoms) + 1)
    return max(0.1, 1.0 - redundancy)
