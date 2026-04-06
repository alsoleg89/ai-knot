"""PostgreSQL storage backend — remote/cloud-ready production storage.

Provide a DSN (connection string) and the table is auto-created.
Requires ``psycopg[binary]>=3.1``: ``pip install ai-knot[postgres]``

Example DSN::

    postgresql://user:password@host:5432/dbname
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from ai_knot.storage.base import parse_datetime as _parse_datetime
from ai_knot.types import Fact, MemoryType, MESIState

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS "ai-knot_facts" (
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
    source_snippets    TEXT NOT NULL DEFAULT '[]',
    source_spans       TEXT NOT NULL DEFAULT '[]',
    supported          INTEGER NOT NULL DEFAULT 1,
    support_confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    verification_source TEXT NOT NULL DEFAULT 'manual',
    access_intervals   TEXT NOT NULL DEFAULT '[]',
    origin_agent_id    TEXT NOT NULL DEFAULT '',
    visibility         TEXT NOT NULL DEFAULT 'private',
    source_verbatim    TEXT NOT NULL DEFAULT '',
    valid_from         TEXT NOT NULL DEFAULT '',
    valid_until        TEXT,
    entity             TEXT NOT NULL DEFAULT '',
    attribute          TEXT NOT NULL DEFAULT '',
    version            INTEGER NOT NULL DEFAULT 0,
    mesi_state         TEXT NOT NULL DEFAULT 'E',
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
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PostgreSQL backend requires psycopg. "
                "Install it with: pip install 'ai-knot[postgres]'"
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
        self._migrate_db()

    def _migrate_db(self) -> None:
        """Add new columns to existing databases (backward compat)."""
        new_columns = {
            "source_snippets": "TEXT NOT NULL DEFAULT '[]'",
            "source_spans": "TEXT NOT NULL DEFAULT '[]'",
            "supported": "INTEGER NOT NULL DEFAULT 1",
            "support_confidence": "DOUBLE PRECISION NOT NULL DEFAULT 1.0",
            "verification_source": "TEXT NOT NULL DEFAULT 'manual'",
            "access_intervals": "TEXT NOT NULL DEFAULT '[]'",
            "origin_agent_id": "TEXT NOT NULL DEFAULT ''",
            "visibility": "TEXT NOT NULL DEFAULT 'private'",
            "source_verbatim": "TEXT NOT NULL DEFAULT ''",
            "valid_from": "TEXT NOT NULL DEFAULT ''",
            "valid_until": "TEXT",
            "entity": "TEXT NOT NULL DEFAULT ''",
            "attribute": "TEXT NOT NULL DEFAULT ''",
            "version": "INTEGER NOT NULL DEFAULT 0",
            "mesi_state": "TEXT NOT NULL DEFAULT 'E'",
        }
        with self._get_conn() as conn:
            cur = conn.execute(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name = 'ai-knot_facts'"""
            )
            existing_cols = {row[0] for row in cur.fetchall()}
            for col, definition in new_columns.items():
                if col not in existing_cols:
                    conn.execute(f'ALTER TABLE "ai-knot_facts" ADD COLUMN {col} {definition}')
            conn.commit()

    def save(self, agent_id: str, facts: list[Fact]) -> None:
        """Replace all facts for an agent."""
        rows = [
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
                fact.origin_agent_id,
                fact.visibility,
                fact.source_verbatim,
                fact.valid_from.isoformat(),
                fact.valid_until.isoformat() if fact.valid_until is not None else None,
                fact.entity,
                fact.attribute,
                fact.version,
                fact.mesi_state,
            )
            for fact in facts
        ]
        with self._get_conn() as conn:
            conn.execute('DELETE FROM "ai-knot_facts" WHERE agent_id = %s', (agent_id,))
            conn.executemany(
                """INSERT INTO "ai-knot_facts"
                   (id, agent_id, content, type, importance, retention,
                    access_count, tags, created_at, last_accessed,
                    source_snippets, source_spans, supported,
                    support_confidence, verification_source,
                    access_intervals, origin_agent_id, visibility,
                    source_verbatim, valid_from, valid_until,
                    entity, attribute, version, mesi_state)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s)""",
                rows,
            )
            conn.commit()
        logger.debug("Saved %d facts for agent '%s'", len(facts), agent_id)

    def load(self, agent_id: str) -> list[Fact]:
        """Load all facts for an agent."""
        with self._get_conn() as conn:
            cur = conn.execute(
                """SELECT id, content, type, importance, retention,
                          access_count, tags, created_at, last_accessed,
                          source_snippets, source_spans, supported,
                          support_confidence, verification_source,
                          access_intervals, origin_agent_id, visibility,
                          source_verbatim, valid_from, valid_until,
                          entity, attribute, version, mesi_state
                   FROM "ai-knot_facts" WHERE agent_id = %s
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
                source_snippets=json.loads(row[9]),
                source_spans=json.loads(row[10]),
                supported=bool(row[11]),
                support_confidence=float(row[12]),
                verification_source=str(row[13]),
                access_intervals=json.loads(row[14]),
                origin_agent_id=str(row[15]),
                visibility=str(row[16]),
                source_verbatim=str(row[17]),
                valid_from=_parse_datetime(row[18]) if row[18] else datetime.now(UTC),
                valid_until=_parse_datetime(row[19]) if row[19] else None,
                entity=str(row[20]) if row[20] else "",
                attribute=str(row[21]) if row[21] else "",
                version=int(row[22]) if row[22] is not None else 0,
                mesi_state=MESIState(str(row[23])) if row[23] else MESIState.EXCLUSIVE,
            )
            for row in rows
        ]

    def delete(self, agent_id: str, fact_id: str) -> None:
        """Remove a single fact by id."""
        with self._get_conn() as conn:
            conn.execute(
                'DELETE FROM "ai-knot_facts" WHERE agent_id = %s AND id = %s',
                (agent_id, fact_id),
            )
            conn.commit()

    def list_agents(self) -> list[str]:
        """Return all agent_ids that have stored facts."""
        with self._get_conn() as conn:
            cur = conn.execute('SELECT DISTINCT agent_id FROM "ai-knot_facts"')
            rows = cur.fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # TemporalStorageCapable implementation (index-accelerated queries)
    # ------------------------------------------------------------------

    _SELECT_COLS = """SELECT id, content, type, importance, retention,
                          access_count, tags, created_at, last_accessed,
                          source_snippets, source_spans, supported,
                          support_confidence, verification_source,
                          access_intervals, origin_agent_id, visibility,
                          source_verbatim, valid_from, valid_until,
                          entity, attribute, version, mesi_state"""

    def _fact_from_row(self, row: tuple[Any, ...]) -> Fact:
        return Fact(
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
            origin_agent_id=str(row[15]),
            visibility=str(row[16]),
            source_verbatim=str(row[17]),
            valid_from=_parse_datetime(row[18]) if row[18] else datetime.now(UTC),
            valid_until=_parse_datetime(row[19]) if row[19] else None,
            entity=str(row[20]) if row[20] else "",
            attribute=str(row[21]) if row[21] else "",
            version=int(row[22]) if row[22] is not None else 0,
            mesi_state=MESIState(str(row[23])) if row[23] else MESIState.EXCLUSIVE,
        )

    def load_active(self, agent_id: str) -> list[Fact]:
        """Load only facts where valid_until IS NULL (index-accelerated)."""
        with self._get_conn() as conn:
            cur = conn.execute(
                f'{self._SELECT_COLS} FROM "ai-knot_facts"'
                " WHERE agent_id = %s AND valid_until IS NULL ORDER BY created_at",
                (agent_id,),
            )
            rows = cur.fetchall()
        return [self._fact_from_row(row) for row in rows]

    def load_since_version(self, agent_id: str, since: int, exclude_agent: str) -> list[Fact]:
        """MESI dirty pull: facts with version > since from agents other than exclude_agent."""
        with self._get_conn() as conn:
            cur = conn.execute(
                f'{self._SELECT_COLS} FROM "ai-knot_facts"'
                " WHERE agent_id = %s AND version > %s AND origin_agent_id != %s"
                " ORDER BY version",
                (agent_id, since, exclude_agent),
            )
            rows = cur.fetchall()
        return [self._fact_from_row(row) for row in rows]
