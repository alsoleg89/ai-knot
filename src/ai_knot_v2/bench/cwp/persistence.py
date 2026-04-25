"""Persistence proxy — atom betweenness centrality from evidence_episodes graph.

Approximation of Persistent Causal Topology (PCT) signature from CWP framework.
Real persistent homology is O(n^3); we use betweenness centrality on the
bipartite (atom × episode) graph as a tractable Level-2 approximation.

High betweenness ⇒ atom sits on many derivation paths ⇒ "backbone" candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from ai_knot_v2.core.atom import MemoryAtom


@dataclass(frozen=True)
class PCTSignature:
    """Persistent Causal Topology signature for one atom.

    Components (Level-2 approximation):
      persistence_0    — fraction of all atoms that share at least one evidence episode
                         with this atom (proxy for topological neighbourhood density).
      betweenness      — number of distinct (other-atom, shared-episode) pairs;
                         high ⇒ atom on many alternative derivation paths.
      cycle_membership — 1.0 if atom shares evidence with another atom that has
                         the same predicate but different polarity (contradiction loop),
                         0.0 otherwise.
    """

    atom_id: str
    persistence_0: float
    betweenness: float
    cycle_membership: float


def compute_pct_signatures(atoms: list[MemoryAtom]) -> dict[str, PCTSignature]:
    """Compute PCT signature for each atom in the corpus.

    Pure function over the static atom set — O(|atoms| * avg_evidence_episodes^2).
    """
    n = len(atoms) or 1

    # Inverted index: episode_id → set of atom_ids that cite it
    episode_atoms: dict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        for ep_id in atom.evidence_episodes:
            episode_atoms[ep_id].add(atom.atom_id)

    out: dict[str, PCTSignature] = {}
    by_id: dict[str, MemoryAtom] = {a.atom_id: a for a in atoms}

    for atom in atoms:
        co_atoms: Counter[str] = Counter()
        for ep_id in atom.evidence_episodes:
            for other_id in episode_atoms.get(ep_id, set()):
                if other_id != atom.atom_id:
                    co_atoms[other_id] += 1

        persistence_0 = len(co_atoms) / n if n else 0.0
        betweenness = float(sum(co_atoms.values()))

        cycle = 0.0
        for other_id in co_atoms:
            other = by_id.get(other_id)
            if other is None:
                continue
            if other.predicate == atom.predicate and other.polarity != atom.polarity:
                cycle = 1.0
                break

        out[atom.atom_id] = PCTSignature(
            atom_id=atom.atom_id,
            persistence_0=persistence_0,
            betweenness=betweenness,
            cycle_membership=cycle,
        )
    return out


def cwp_priority(
    atom: MemoryAtom,
    sig: PCTSignature,
    *,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.5,
) -> float:
    """CWP-derived ranking priority replacing raw regret_charge × credence.

    priority = α·persistence_0 + β·betweenness_norm + (1−γ·cycle_membership) · credence

    Higher = more important to keep in evidence pack.
    """
    betweenness_norm = sig.betweenness / (1.0 + sig.betweenness)
    contradiction_discount = 1.0 - gamma * sig.cycle_membership
    return (
        alpha * sig.persistence_0 + beta * betweenness_norm + contradiction_discount * atom.credence
    )
