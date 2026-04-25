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

LANDAUER_FLOOR_SCALE: float = 0.02  # high-risk atoms floor at 0.02 × risk_severity
ACCESS_RATE_HALFLIFE_DAYS: float = 7.0  # access boost halflife in days


def decay_protection_energy(
    atom: MemoryAtom,
    elapsed_days: float,
    access_count_recent: int = 0,
    contradiction_count: int = 0,
) -> MemoryAtom:
    """Return atom with protection_energy decayed by Landauer-ODE over elapsed_days.

    Extends the base ODE with:
    - access_boost: recent accesses slow decay (halflife = ACCESS_RATE_HALFLIFE_DAYS)
    - curvature_boost: atoms in active contradictions decay slower
    - Landauer floor: high-risk atoms cannot decay below kB·T·ln2 proxy
    """
    k = BASE_DECAY_RATE / (1.0 + atom.risk_severity * 5.0)
    decayed = atom.protection_energy * math.exp(-k * elapsed_days)

    access_boost = 0.05 * access_count_recent * math.exp(-elapsed_days / ACCESS_RATE_HALFLIFE_DAYS)
    curvature_boost = 0.03 * min(contradiction_count, 3)

    new_energy = decayed + access_boost + curvature_boost
    landauer_floor = LANDAUER_FLOOR_SCALE * atom.risk_severity
    new_energy = max(landauer_floor, min(1.0, new_energy))

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
        contradiction_count = len(atom.contradiction_events)
        decayed = decay_protection_energy(
            atom, elapsed_days, contradiction_count=contradiction_count
        )
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
