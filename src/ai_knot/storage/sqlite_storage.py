"""SQLite-based storage backend — zero-server production storage."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_knot.types import Fact, MemoryType

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS facts (
    id          TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    content     TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'semantic',
    importance  REAL NOT NULL DEFAULT 0.8,
    retention   REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    tags        TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    source_snippets TEXT NOT NULL DEFAULT '[]',
    source_spans    TEXT NOT NULL DEFAULT '[]',
    supported       INTEGER NOT NULL DEFAULT 1,
    support_confidence REAL NOT NULL DEFAULT 1.0,
    verification_source TEXT NOT NULL DEFAULT 'manual',
    access_intervals TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (agent_id, id)
)
"""

_CREATE_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS snapshots (
    agent_id   TEXT NOT NULL,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    facts_json TEXT NOT NULL,
    PRIMARY KEY (agent_id, name)
)
"""


class SQLiteStorage:
    """Stores facts in a local SQLite database.

    Good for single-server production deployments where you want
    more robustness than YAML files but don't need a separate DB server.
    """

    def __init__(self, db_path: str = ".ai_knot/ai_knot.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_SNAPSHOTS_TABLE)
        self._migrate_db()

    def _migrate_db(self) -> None:
        """Add new columns to existing databases (backward compat)."""
        new_columns = {
            "source_snippets": "TEXT NOT NULL DEFAULT '[]'",
            "source_spans": "TEXT NOT NULL DEFAULT '[]'",
            "supported": "INTEGER NOT NULL DEFAULT 1",
            "support_confidence": "REAL NOT NULL DEFAULT 1.0",
            "verification_source": "TEXT NOT NULL DEFAULT 'manual'",
            "access_intervals": "TEXT NOT NULL DEFAULT '[]'",
        }
        with self._get_conn() as conn:
            cur = conn.execute("PRAGMA table_info(facts)")
            existing_cols = {row[1] for row in cur.fetchall()}
            for col, definition in new_columns.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {definition}")

    def save(self, agent_id: str, facts: list[Fact]) -> None:
        """Replace all facts for an agent."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM facts WHERE agent_id = ?", (agent_id,))
            for fact in facts:
                conn.execute(
                    """INSERT INTO facts
                       (id, agent_id, content, type, importance, retention,
                        access_count, tags, created_at, last_accessed,
                        source_snippets, source_spans, supported,
                        support_confidence, verification_source,
                        access_intervals)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        fact.id,
                        agent_id,
                        fact.content,
                        fact.type.value,
                        fact.importance,
                        fact.retention_score,
                        fact.access_count,
                        json.dumps(fact.tags),
                        fact.created_at.isoformat(),
                        fact.last_accessed.isoformat(),
                        json.dumps(fact.source_snippets),
                        json.dumps(fact.source_spans),
                        1 if fact.supported else 0,
                        fact.support_confidence,
                        fact.verification_source,
                        json.dumps(fact.access_intervals),
                    ),
                )
        logger.debug("Saved %d facts for agent '%s'", len(facts), agent_id)

    def load(self, agent_id: str) -> list[Fact]:
        """Load all facts for an agent."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, content, type, importance, retention,
                          access_count, tags, created_at, last_accessed,
                          source_snippets, source_spans, supported,
                          support_confidence, verification_source,
                          access_intervals
                   FROM facts WHERE agent_id = ?
                   ORDER BY created_at""",
                (agent_id,),
            ).fetchall()

        return [
            Fact(
                id=row[0],
                content=row[1],
                type=MemoryType(row[2]),
                importance=row[3],
                retention_score=row[4],
                access_count=row[5],
                tags=json.loads(row[6]),
                created_at=_parse_datetime(row[7]),
                last_accessed=_parse_datetime(row[8]),
                source_snippets=json.loads(row[9]),
                source_spans=json.loads(row[10]),
                supported=bool(row[11]),
                support_confidence=float(row[12]),
                verification_source=str(row[13]),
                access_intervals=json.loads(row[14]),
            )
            for row in rows
        ]

    def delete(self, agent_id: str, fact_id: str) -> None:
        """Remove a single fact by id."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM facts WHERE agent_id = ? AND id = ?",
                (agent_id, fact_id),
            )

    def list_agents(self) -> list[str]:
        """Return all agent_ids that have stored facts."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT DISTINCT agent_id FROM facts").fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # SnapshotCapable implementation
    # ------------------------------------------------------------------

    def save_snapshot(self, agent_id: str, name: str, facts: list[Fact]) -> None:
        """Persist a named snapshot (overwrites if name already exists)."""
        facts_data = [
            {
                "id": f.id,
                "content": f.content,
                "type": f.type.value,
                "importance": f.importance,
                "retention_score": f.retention_score,
                "access_count": f.access_count,
                "tags": f.tags,
                "created_at": f.created_at.isoformat(),
                "last_accessed": f.last_accessed.isoformat(),
                "source_snippets": f.source_snippets,
                "source_spans": f.source_spans,
                "supported": f.supported,
                "support_confidence": f.support_confidence,
                "verification_source": f.verification_source,
                "access_intervals": f.access_intervals,
            }
            for f in facts
        ]
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO snapshots (agent_id, name, created_at, facts_json)
                   VALUES (?, ?, ?, ?)""",
                (agent_id, name, datetime.now(UTC).isoformat(), json.dumps(facts_data)),
            )
        logger.debug("Saved snapshot '%s' for agent '%s'", name, agent_id)

    def load_snapshot(self, agent_id: str, name: str) -> list[Fact]:
        """Load facts from a named snapshot.

        Raises:
            KeyError: If no snapshot with the given name exists.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT facts_json FROM snapshots WHERE agent_id = ? AND name = ?",
                (agent_id, name),
            ).fetchone()

        if row is None:
            raise KeyError(f"Snapshot {name!r} not found for agent {agent_id!r}")

        raw: list[dict[str, Any]] = json.loads(row[0])
        return [
            Fact(
                id=str(entry["id"]),
                content=str(entry["content"]),
                type=MemoryType(str(entry["type"])),
                importance=float(entry["importance"]),
                retention_score=float(entry["retention_score"]),
                access_count=int(entry["access_count"]),
                tags=list(entry["tags"]),
                created_at=_parse_datetime(str(entry["created_at"])),
                last_accessed=_parse_datetime(str(entry["last_accessed"])),
                source_snippets=list(entry.get("source_snippets", [])),
                source_spans=list(entry.get("source_spans", [])),
                supported=bool(entry.get("supported", True)),
                support_confidence=float(entry.get("support_confidence", 1.0)),
                verification_source=str(entry.get("verification_source", "legacy")),
                access_intervals=[float(x) for x in entry.get("access_intervals", [])],
            )
            for entry in raw
        ]

    def list_snapshots(self, agent_id: str) -> list[str]:
        """Return snapshot names sorted by creation time (oldest first)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT name FROM snapshots WHERE agent_id = ? ORDER BY created_at",
                (agent_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def delete_snapshot(self, agent_id: str, name: str) -> None:
        """Delete a named snapshot. No-op if it does not exist."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM snapshots WHERE agent_id = ? AND name = ?",
                (agent_id, name),
            )


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-format datetime string, ensuring UTC timezone."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
