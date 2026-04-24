"""SQL DDL for ai-knot v2 SQLite store.

All JSON columns use Python's json module for serialization.
Migrations are handled by sequential ADD COLUMN statements below.
"""

from __future__ import annotations

import sqlite3

_EPISODES = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id       TEXT PRIMARY KEY,
    agent_id         TEXT NOT NULL,
    user_id          TEXT,
    session_id       TEXT NOT NULL,
    turn_index       INTEGER NOT NULL,
    speaker          TEXT NOT NULL,
    text             TEXT NOT NULL,
    timestamp        INTEGER NOT NULL,
    metadata         TEXT NOT NULL
)
"""

_ATOMS = """
CREATE TABLE IF NOT EXISTS atoms (
    atom_id              TEXT PRIMARY KEY,
    agent_id             TEXT NOT NULL,
    user_id              TEXT,
    variables            TEXT NOT NULL,
    causal_graph         TEXT NOT NULL,
    kernel_kind          TEXT NOT NULL,
    kernel_payload       TEXT NOT NULL,
    intervention_domain  TEXT NOT NULL,
    predicate            TEXT NOT NULL,
    subject              TEXT NOT NULL,
    object_value         TEXT,
    polarity             TEXT NOT NULL,
    valid_from           INTEGER,
    valid_until          INTEGER,
    observation_time     INTEGER NOT NULL,
    belief_time          INTEGER NOT NULL,
    granularity          TEXT NOT NULL,
    entity_orbit_id      TEXT NOT NULL,
    transport_provenance TEXT NOT NULL,
    depends_on           TEXT NOT NULL,
    depended_by          TEXT NOT NULL,
    risk_class           TEXT NOT NULL,
    risk_severity        REAL NOT NULL,
    regret_charge        REAL NOT NULL,
    irreducibility_score REAL NOT NULL,
    protection_energy    REAL NOT NULL,
    action_affect_mask   INTEGER NOT NULL,
    credence             REAL NOT NULL,
    evidence_episodes    TEXT NOT NULL,
    synthesis_method     TEXT NOT NULL,
    validation_tests     TEXT NOT NULL,
    contradiction_events TEXT NOT NULL
)
"""

_EVIDENCE_PACKS = """
CREATE TABLE IF NOT EXISTS evidence_packs (
    pack_id         TEXT PRIMARY KEY,
    atoms           TEXT NOT NULL,
    utility_scores  TEXT NOT NULL
)
"""

_EVIDENCE_SPANS = """
CREATE TABLE IF NOT EXISTS evidence_spans (
    span_id         TEXT PRIMARY KEY,
    pack_id         TEXT NOT NULL,
    episode_id      TEXT NOT NULL,
    start_char      INTEGER NOT NULL,
    end_char        INTEGER NOT NULL,
    text            TEXT NOT NULL,
    relevance_score REAL NOT NULL,
    FOREIGN KEY (pack_id) REFERENCES evidence_packs(pack_id)
)
"""

_ATOM_DEPENDENCIES = """
CREATE TABLE IF NOT EXISTS atom_dependencies (
    from_atom_id TEXT NOT NULL,
    to_atom_id   TEXT NOT NULL,
    PRIMARY KEY (from_atom_id, to_atom_id)
)
"""

_AUDIT_TRAIL = """
CREATE TABLE IF NOT EXISTS audit_trail (
    event_id  TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    atom_id   TEXT,
    agent_id  TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    details   TEXT NOT NULL
)
"""

_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_atoms_entity ON atoms(entity_orbit_id)",
    "CREATE INDEX IF NOT EXISTS idx_atoms_predicate ON atoms(predicate)",
    "CREATE INDEX IF NOT EXISTS idx_atoms_risk ON atoms(risk_class)",
    "CREATE INDEX IF NOT EXISTS idx_atoms_agent ON atoms(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_atom ON audit_trail(atom_id)",
]

ALL_DDL = [
    _EPISODES,
    _ATOMS,
    _EVIDENCE_PACKS,
    _EVIDENCE_SPANS,
    _ATOM_DEPENDENCIES,
    _AUDIT_TRAIL,
    *_INDICES,
]


def create_all(conn: sqlite3.Connection) -> None:
    """Create all tables and indices. Idempotent (IF NOT EXISTS)."""
    with conn:
        for ddl in ALL_DDL:
            conn.execute(ddl)
