"""SQLite store for ai-knot v2 — stdlib sqlite3 only, no ORM."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.episode import RawEpisode
from ai_knot_v2.core.provenance import AuditEvent, AuditTrail
from ai_knot_v2.core.types import EvidencePack, EvidenceSpan
from ai_knot_v2.store.schema import create_all


def _j(obj: object) -> str:
    return json.dumps(obj)


def _atom_to_row(atom: MemoryAtom) -> dict[str, object]:
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
        "subject": atom.subject,
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


def _row_to_atom(row: sqlite3.Row) -> MemoryAtom:
    r = dict(row)
    return MemoryAtom(
        atom_id=r["atom_id"],
        agent_id=r["agent_id"],
        user_id=r["user_id"],
        variables=tuple(json.loads(r["variables"])),
        causal_graph=tuple(tuple(e) for e in json.loads(r["causal_graph"])),
        kernel_kind=r["kernel_kind"],
        kernel_payload=json.loads(r["kernel_payload"]),
        intervention_domain=tuple(json.loads(r["intervention_domain"])),
        predicate=r["predicate"],
        subject=r["subject"],
        object_value=r["object_value"],
        polarity=r["polarity"],
        valid_from=r["valid_from"],
        valid_until=r["valid_until"],
        observation_time=r["observation_time"],
        belief_time=r["belief_time"],
        granularity=r["granularity"],
        entity_orbit_id=r["entity_orbit_id"],
        transport_provenance=tuple(json.loads(r["transport_provenance"])),
        depends_on=tuple(json.loads(r["depends_on"])),
        depended_by=tuple(json.loads(r["depended_by"])),
        risk_class=r["risk_class"],
        risk_severity=r["risk_severity"],
        regret_charge=r["regret_charge"],
        irreducibility_score=r["irreducibility_score"],
        protection_energy=r["protection_energy"],
        action_affect_mask=r["action_affect_mask"],
        credence=r["credence"],
        evidence_episodes=tuple(json.loads(r["evidence_episodes"])),
        synthesis_method=r["synthesis_method"],
        validation_tests=tuple(json.loads(r["validation_tests"])),
        contradiction_events=tuple(json.loads(r["contradiction_events"])),
    )


def _episode_to_row(ep: RawEpisode) -> dict[str, object]:
    return {
        "episode_id": ep.episode_id,
        "agent_id": ep.agent_id,
        "user_id": ep.user_id,
        "session_id": ep.session_id,
        "turn_index": ep.turn_index,
        "speaker": ep.speaker,
        "text": ep.text,
        "timestamp": ep.timestamp,
        "metadata": _j(ep.metadata),
    }


def _row_to_episode(row: sqlite3.Row) -> RawEpisode:
    r = dict(row)
    return RawEpisode(
        episode_id=r["episode_id"],
        agent_id=r["agent_id"],
        user_id=r["user_id"],
        session_id=r["session_id"],
        turn_index=r["turn_index"],
        speaker=r["speaker"],
        text=r["text"],
        timestamp=r["timestamp"],
        metadata=json.loads(r["metadata"]),
    )


class SqliteStore:
    """Thin SQLite wrapper for v2 memory store."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        create_all(self._conn)

    def close(self) -> None:
        self._conn.close()

    # --- Episodes ---

    def save_episode(self, ep: RawEpisode) -> None:
        row = _episode_to_row(ep)
        cols = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO episodes ({cols}) VALUES ({placeholders})", row
            )

    def get_episode(self, episode_id: str) -> RawEpisode | None:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        return _row_to_episode(row) if row else None

    # --- Atoms ---

    def save_atom(self, atom: MemoryAtom) -> None:
        row = _atom_to_row(atom)
        cols = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO atoms ({cols}) VALUES ({placeholders})", row
            )

    def get_atom(self, atom_id: str) -> MemoryAtom | None:
        row = self._conn.execute("SELECT * FROM atoms WHERE atom_id = ?", (atom_id,)).fetchone()
        return _row_to_atom(row) if row else None

    def get_atoms_by_entity(self, entity_orbit_id: str) -> list[MemoryAtom]:
        rows = self._conn.execute(
            "SELECT * FROM atoms WHERE entity_orbit_id = ?", (entity_orbit_id,)
        ).fetchall()
        return [_row_to_atom(r) for r in rows]

    def get_atoms_by_agent(self, agent_id: str) -> list[MemoryAtom]:
        rows = self._conn.execute("SELECT * FROM atoms WHERE agent_id = ?", (agent_id,)).fetchall()
        return [_row_to_atom(r) for r in rows]

    # --- Evidence packs ---

    def save_evidence_pack(self, pack: EvidencePack) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO evidence_packs (pack_id, atoms, utility_scores) "
                "VALUES (?, ?, ?)",
                (pack.pack_id, _j(list(pack.atoms)), _j(pack.utility_scores)),
            )
            for span in pack.spans:
                self._conn.execute(
                    "INSERT OR REPLACE INTO evidence_spans "
                    "(span_id, pack_id, episode_id, start_char, end_char, text, relevance_score) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        new_ulid(),
                        pack.pack_id,
                        span.episode_id,
                        span.start_char,
                        span.end_char,
                        span.text,
                        span.relevance_score,
                    ),
                )

    def get_evidence_pack(self, pack_id: str) -> EvidencePack | None:
        row = self._conn.execute(
            "SELECT * FROM evidence_packs WHERE pack_id = ?", (pack_id,)
        ).fetchone()
        if not row:
            return None
        span_rows = self._conn.execute(
            "SELECT * FROM evidence_spans WHERE pack_id = ?", (pack_id,)
        ).fetchall()
        spans = tuple(
            EvidenceSpan(
                episode_id=r["episode_id"],
                start_char=r["start_char"],
                end_char=r["end_char"],
                text=r["text"],
                relevance_score=r["relevance_score"],
            )
            for r in span_rows
        )
        return EvidencePack(
            pack_id=row["pack_id"],
            atoms=tuple(json.loads(row["atoms"])),
            spans=spans,
            utility_scores=json.loads(row["utility_scores"]),
        )

    # --- Audit trail ---

    def append_audit_event(self, event: AuditEvent) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO audit_trail "
                "(event_id, operation, atom_id, agent_id, timestamp, details) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.operation,
                    event.atom_id,
                    event.agent_id,
                    event.timestamp,
                    _j(event.details),
                ),
            )

    def trace(self, atom_id: str) -> AuditTrail:
        rows = self._conn.execute(
            "SELECT * FROM audit_trail WHERE atom_id = ? ORDER BY timestamp",
            (atom_id,),
        ).fetchall()
        events = tuple(
            AuditEvent(
                event_id=r["event_id"],
                operation=r["operation"],
                atom_id=r["atom_id"],
                agent_id=r["agent_id"],
                timestamp=r["timestamp"],
                details=json.loads(r["details"]),
            )
            for r in rows
        )
        return AuditTrail(atom_id=atom_id, events=events)
