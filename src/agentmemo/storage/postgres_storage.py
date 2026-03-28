"""PostgreSQL storage backend — remote/cloud-ready production storage.

Provide a DSN (connection string) and the table is auto-created.
Requires ``psycopg[binary]>=3.1``: ``pip install agentmemo[postgres]``

Example DSN::

    postgresql://user:password@host:5432/dbname
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from agentmemo.types import Fact, MemoryType

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS agentmemo_facts (
    id            TEXT NOT NULL,
    agent_id      TEXT NOT NULL,
    content       TEXT NOT NULL,
    type          TEXT NOT NULL DEFAULT 'semantic',
    importance    DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    retention     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    access_count  INTEGER NOT NULL DEFAULT 0,
    tags          TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    PRIMARY KEY (agent_id, id)
)
"""


class PostgresStorage:
    """Stores facts in a PostgreSQL database.

    Provide a DSN and the required table is created automatically.
    Thread-safe via PostgreSQL's native concurrency model.

    Args:
        dsn: PostgreSQL connection string
            (e.g. ``postgresql://user:pass@host:5432/db``).
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PostgreSQL backend requires psycopg. "
                "Install it with: pip install 'agentmemo[postgres]'"
            ) from exc
        self._dsn = dsn
        self._init_db()

    def _get_conn(self) -> Any:
        import psycopg  # noqa: F811

        return psycopg.connect(self._dsn, autocommit=False)

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()

    def save(self, agent_id: str, facts: list[Fact]) -> None:
        """Replace all facts for an agent."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM agentmemo_facts WHERE agent_id = %s", (agent_id,))
            for fact in facts:
                conn.execute(
                    """INSERT INTO agentmemo_facts
                       (id, agent_id, content, type, importance, retention,
                        access_count, tags, created_at, last_accessed)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
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
            conn.commit()
        logger.debug("Saved %d facts for agent '%s'", len(facts), agent_id)

    def load(self, agent_id: str) -> list[Fact]:
        """Load all facts for an agent."""
        with self._get_conn() as conn:
            cur = conn.execute(
                """SELECT id, content, type, importance, retention,
                          access_count, tags, created_at, last_accessed
                   FROM agentmemo_facts WHERE agent_id = %s
                   ORDER BY created_at""",
                (agent_id,),
            )
            rows = cur.fetchall()

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
                "DELETE FROM agentmemo_facts WHERE agent_id = %s AND id = %s",
                (agent_id, fact_id),
            )
            conn.commit()

    def list_agents(self) -> list[str]:
        """Return all agent_ids that have stored facts."""
        with self._get_conn() as conn:
            cur = conn.execute("SELECT DISTINCT agent_id FROM agentmemo_facts")
            rows = cur.fetchall()
        return [row[0] for row in rows]


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-format datetime string, ensuring UTC timezone."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
