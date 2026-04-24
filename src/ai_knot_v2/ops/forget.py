"""FORGET operation: protection-energy ODE decay + expiry removal.

First-order linear ODE: dE/dt = -k(severity) * E
Solution: E(t) = E0 * exp(-k * t)

Higher risk_severity → smaller k → slower decay.
Atoms with protection_energy ≤ FORGET_THRESHOLD are removed from library and store.
"""

from __future__ import annotations

import dataclasses
import math
import time

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.library import AtomLibrary
from ai_knot_v2.core.provenance import AuditEvent
from ai_knot_v2.store.sqlite import SqliteStore

BASE_DECAY_RATE: float = 0.02
FORGET_THRESHOLD: float = 0.05


def decay_protection_energy(atom: MemoryAtom, elapsed_days: float) -> MemoryAtom:
    """Return atom with protection_energy decayed by ODE solution over elapsed_days."""
    k = BASE_DECAY_RATE / (1.0 + atom.risk_severity * 5.0)
    new_energy = max(0.0, atom.protection_energy * math.exp(-k * elapsed_days))
    return dataclasses.replace(atom, protection_energy=new_energy)


def should_forget(atom: MemoryAtom) -> bool:
    return atom.protection_energy <= FORGET_THRESHOLD


def run_forget_pass(
    library: AtomLibrary,
    store: SqliteStore,
    elapsed_days: float,
) -> tuple[int, int]:
    """Decay all atoms; remove forgotten ones.

    Returns (forgotten_count, retained_count).
    """
    atoms = library.all_atoms()
    forgotten = 0
    retained = 0
    now = int(time.time())

    for atom in atoms:
        decayed = decay_protection_energy(atom, elapsed_days)
        if should_forget(decayed):
            library.remove(atom.atom_id)
            store.delete_atom(atom.atom_id)
            store.append_audit_event(
                AuditEvent(
                    event_id=new_ulid(),
                    operation="forget",
                    atom_id=atom.atom_id,
                    agent_id=atom.agent_id,
                    timestamp=now,
                    details={
                        "elapsed_days": elapsed_days,
                        "final_energy": decayed.protection_energy,
                        "risk_severity": atom.risk_severity,
                    },
                )
            )
            forgotten += 1
        else:
            library.remove(atom.atom_id)
            library.add(decayed)
            store.save_atom(decayed)
            retained += 1

    return forgotten, retained
