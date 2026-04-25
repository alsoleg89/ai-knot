"""PostgreSQL store for ai-knot v2 (Sprint 21 — production backend).

Implements the same interface as SqliteStore but uses psycopg3.
Connection string: postgresql://user:pass@host:port/dbname

Usage:
    from ai_knot_v2.store.postgres import PostgresStore
    store = PostgresStore("postgresql://user:pass@localhost/aiknot")

Requires: pip install psycopg[binary]
"""

from __future__ import annotations

import json
from typing import Any

from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.episode import RawEpisode
from ai_knot_v2.core.provenance import AuditEvent, AuditTrail
from ai_knot_v2.core.types import EvidencePack


def _j(obj: object) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# PostgreSQL DDL (same schema as SQLite, Postgres dialect)
# ---------------------------------------------------------------------------

_PG_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id  TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL,
    user_id     TEXT,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    speaker     TEXT NOT NULL,
    text        TEXT NOT NULL,
    timestamp   BIGINT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS atoms (
    atom_id              TEXT PRIMARY KEY,
    agent_id             TEXT NOT NULL,
    user_id              TEXT,
    variables            JSONB NOT NULL,
    causal_graph         JSONB NOT NULL,
    kernel_kind          TEXT NOT NULL,
    kernel_payload       JSONB NOT NULL,
    intervention_domain  JSONB NOT NULL,
    predicate            TEXT NOT NULL,
    subject              TEXT NOT NULL,
    object_value         TEXT,
    polarity             TEXT NOT NULL,
    valid_from           BIGINT,
    valid_until          BIGINT,
    observation_time     BIGINT NOT NULL,
    belief_time          BIGINT NOT NULL,
    granularity          TEXT NOT NULL,
    entity_orbit_id      TEXT NOT NULL,
    transport_provenance JSONB NOT NULL,
    depends_on           JSONB NOT NULL,
    depended_by          JSONB NOT NULL,
    risk_class           TEXT NOT NULL,
    risk_severity        REAL NOT NULL,
    regret_charge        REAL NOT NULL,
    irreducibility_score REAL NOT NULL,
    protection_energy    REAL NOT NULL,
    action_affect_mask   INTEGER NOT NULL,
    credence             REAL NOT NULL,
    evidence_episodes    JSONB NOT NULL,
    synthesis_method     TEXT NOT NULL,
    validation_tests     JSONB NOT NULL,
    contradiction_events JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_packs (
    pack_id        TEXT PRIMARY KEY,
    atoms          JSONB NOT NULL,
    utility_scores JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_trail (
    event_id  TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    atom_id   TEXT,
    agent_id  TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    details   JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_atoms_entity ON atoms(entity_orbit_id);
CREATE INDEX IF NOT EXISTS idx_atoms_predicate ON atoms(predicate);
CREATE INDEX IF NOT EXISTS idx_atoms_risk ON atoms(risk_class);
CREATE INDEX IF NOT EXISTS idx_atoms_agent ON atoms(agent_id);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_atom ON audit_trail(atom_id);
"""


# ---------------------------------------------------------------------------
# Row ↔ domain object converters (shared logic with SqliteStore)
# ---------------------------------------------------------------------------


def _atom_to_row(atom: MemoryAtom) -> dict[str, Any]:
    return {
        "atom_id": atom.atom_id,
        "agent_id": atom.agent_id,
        "user_id": atom.user_id,
        "variables": _j(list(atom.variables)),
        "causal_graph": _j([list(e) for e in atom.causal_graph]),
        "kernel_kind": atom.kernel_kind,
        "kernel_payload": _j(atom.kernel_payload),
        "intervention_domain": _j(list(atom.intervention_domain)),
        "predicate": atom.predicate,
        "subject": atom.subject or "",
        "object_value": atom.object_value,
        "polarity": atom.polarity,
        "valid_from": atom.valid_from,
        "valid_until": atom.valid_until,
        "observation_time": atom.observation_time,
        "belief_time": atom.belief_time,
        "granularity": atom.granularity,
        "entity_orbit_id": atom.entity_orbit_id,
        "transport_provenance": _j(list(atom.transport_provenance)),
        "depends_on": _j(list(atom.depends_on)),
        "depended_by": _j(list(atom.depended_by)),
        "risk_class": atom.risk_class,
        "risk_severity": atom.risk_severity,
        "regret_charge": atom.regret_charge,
        "irreducibility_score": atom.irreducibility_score,
        "protection_energy": atom.protection_energy,
        "action_affect_mask": atom.action_affect_mask,
        "credence": atom.credence,
        "evidence_episodes": _j(list(atom.evidence_episodes)),
        "synthesis_method": atom.synthesis_method,
        "validation_tests": _j(list(atom.validation_tests)),
        "contradiction_events": _j(list(atom.contradiction_events)),
    }


def _row_to_atom(row: dict[str, Any]) -> MemoryAtom:
    def _l(key: str) -> list[Any]:
        v = row[key]
        return json.loads(v) if isinstance(v, str) else (v or [])

    return MemoryAtom(
        atom_id=row["atom_id"],
        agent_id=row["agent_id"],
        user_id=row.get("user_id"),
        variables=tuple(_l("variables")),
        causal_graph=tuple(tuple(e) for e in _l("causal_graph")),
        kernel_kind=row["kernel_kind"],
        kernel_payload=json.loads(row["kernel_payload"])
        if isinstance(row["kernel_payload"], str)
        else row["kernel_payload"],
        intervention_domain=tuple(_l("intervention_domain")),
        predicate=row["predicate"],
        subject=row["subject"],
        object_value=row.get("object_value"),
        polarity=row["polarity"],
        valid_from=row.get("valid_from"),
        valid_until=row.get("valid_until"),
        observation_time=row["observation_time"],
        belief_time=row["belief_time"],
        granularity=row["granularity"],
        entity_orbit_id=row["entity_orbit_id"],
        transport_provenance=tuple(_l("transport_provenance")),
        depends_on=tuple(_l("depends_on")),
        depended_by=tuple(_l("depended_by")),
        risk_class=row["risk_class"],
        risk_severity=float(row["risk_severity"]),
        regret_charge=float(row["regret_charge"]),
        irreducibility_score=float(row["irreducibility_score"]),
        protection_energy=float(row["protection_energy"]),
        action_affect_mask=int(row["action_affect_mask"]),
        credence=float(row["credence"]),
        evidence_episodes=tuple(_l("evidence_episodes")),
        synthesis_method=row["synthesis_method"],
        validation_tests=tuple(_l("validation_tests")),
        contradiction_events=tuple(_l("contradiction_events")),
    )


# ---------------------------------------------------------------------------
# PostgresStore
# ---------------------------------------------------------------------------


class PostgresStore:
    """Postgres-backed store for ai-knot v2 atoms, episodes, and audit trail.

    Requires psycopg3: pip install psycopg[binary]
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError(
                "psycopg3 is required for PostgresStore: pip install psycopg[binary]"
            ) from exc

        self._conn = psycopg.connect(dsn, autocommit=False)
        self._create_tables()

    def _create_tables(self) -> None:
        with self._conn.cursor() as cur:
            for stmt in _PG_DDL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def save_episode(self, ep: RawEpisode) -> None:
        sql = """
            INSERT INTO episodes
              (episode_id, agent_id, user_id, session_id,
               turn_index, speaker, text, timestamp, metadata)
            VALUES (%(episode_id)s, %(agent_id)s, %(user_id)s, %(session_id)s,
                    %(turn_index)s, %(speaker)s, %(text)s, %(timestamp)s, %(metadata)s)
            ON CONFLICT (episode_id) DO NOTHING
        """
        with self._conn.cursor() as cur:
            cur.execute(
                sql,
                {
                    "episode_id": ep.episode_id,
                    "agent_id": ep.agent_id,
                    "user_id": ep.user_id,
                    "session_id": ep.session_id,
                    "turn_index": ep.turn_index,
                    "speaker": ep.speaker,
                    "text": ep.text,
                    "timestamp": ep.timestamp,
                    "metadata": _j({}),
                },
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Atoms
    # ------------------------------------------------------------------

    def save_atom(self, atom: MemoryAtom) -> None:
        row = _atom_to_row(atom)
        cols = ", ".join(row.keys())
        params = ", ".join(f"%({k})s" for k in row)
        sql = f"INSERT INTO atoms ({cols}) VALUES ({params}) ON CONFLICT (atom_id) DO NOTHING"
        with self._conn.cursor() as cur:
            cur.execute(sql, row)
        self._conn.commit()

    def get_atom(self, atom_id: str) -> MemoryAtom | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM atoms WHERE atom_id = %s", (atom_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return _row_to_atom(dict(zip(cols, row, strict=False)))

    def get_atoms_by_entity(self, entity_orbit_id: str) -> list[MemoryAtom]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM atoms WHERE entity_orbit_id = %s", (entity_orbit_id,))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return [_row_to_atom(dict(zip(cols, r, strict=False))) for r in rows]

    def get_atoms_by_agent(self, agent_id: str) -> list[MemoryAtom]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM atoms WHERE agent_id = %s", (agent_id,))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return [_row_to_atom(dict(zip(cols, r, strict=False))) for r in rows]

    def delete_atom(self, atom_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM atoms WHERE atom_id = %s", (atom_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Evidence packs
    # ------------------------------------------------------------------

    def save_evidence_pack(self, pack: EvidencePack) -> None:
        sql = """
            INSERT INTO evidence_packs (pack_id, atoms, utility_scores)
            VALUES (%s, %s, %s)
            ON CONFLICT (pack_id) DO NOTHING
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (pack.pack_id, _j(list(pack.atoms)), _j(pack.utility_scores)))
        self._conn.commit()

    def get_evidence_pack(self, pack_id: str) -> EvidencePack | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT atoms, utility_scores FROM evidence_packs WHERE pack_id = %s", (pack_id,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            atoms_raw, scores_raw = row
            atoms = json.loads(atoms_raw) if isinstance(atoms_raw, str) else (atoms_raw or [])
            scores = json.loads(scores_raw) if isinstance(scores_raw, str) else (scores_raw or {})
            return EvidencePack(
                pack_id=pack_id,
                atoms=tuple(atoms),
                spans=(),
                utility_scores=scores,
            )

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def append_audit_event(self, event: AuditEvent) -> None:
        sql = """
            INSERT INTO audit_trail (event_id, operation, atom_id, agent_id, timestamp, details)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
        """
        with self._conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    event.event_id,
                    event.operation,
                    event.atom_id,
                    event.agent_id,
                    event.timestamp,
                    _j(event.details),
                ),
            )
        self._conn.commit()

    def trace(self, atom_id: str) -> AuditTrail:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM audit_trail WHERE atom_id = %s ORDER BY timestamp",
                (atom_id,),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            events = tuple(
                AuditEvent(
                    event_id=r["event_id"],
                    operation=r["operation"],
                    atom_id=r["atom_id"],
                    agent_id=r["agent_id"],
                    timestamp=r["timestamp"],
                    details=json.loads(r["details"])
                    if isinstance(r["details"], str)
                    else r["details"],
                )
                for r in (dict(zip(cols, row, strict=False)) for row in rows)
            )
        return AuditTrail(atom_id=atom_id, events=events)
