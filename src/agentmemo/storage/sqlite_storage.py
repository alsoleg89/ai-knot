"""SQLite-based storage backend — zero-server production storage."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from agentmemo.types import Fact, MemoryType

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
    PRIMARY KEY (agent_id, id)
)
"""


class SQLiteStorage:
    """Stores facts in a local SQLite database.

    Good for single-server production deployments where you want
    more robustness than YAML files but don't need a separate DB server.
    """

    def __init__(self, db_path: str = ".agentmemo/agentmemo.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(_CREATE_TABLE)

    def save(self, agent_id: str, facts: list[Fact]) -> None:
        """Replace all facts for an agent."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM facts WHERE agent_id = ?", (agent_id,))
            for fact in facts:
                conn.execute(
                    """INSERT INTO facts
                       (id, agent_id, content, type, importance, retention,
                        access_count, tags, created_at, last_accessed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    ),
                )
        logger.debug("Saved %d facts for agent '%s'", len(facts), agent_id)

    def load(self, agent_id: str) -> list[Fact]:
        """Load all facts for an agent."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, content, type, importance, retention,
                          access_count, tags, created_at, last_accessed
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


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-format datetime string, ensuring UTC timezone."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
