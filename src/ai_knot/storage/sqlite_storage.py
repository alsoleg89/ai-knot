"""SQLite-based storage backend — zero-server production storage."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_knot.storage.base import parse_datetime as _parse_datetime
from ai_knot.types import Fact, MemoryType, MESIState, SlotDelta

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
    origin_agent_id TEXT NOT NULL DEFAULT '',
    visibility      TEXT NOT NULL DEFAULT 'private',
    source_verbatim TEXT NOT NULL DEFAULT '',
    valid_from  TEXT NOT NULL DEFAULT '',
    valid_until TEXT,
    entity      TEXT NOT NULL DEFAULT '',
    attribute   TEXT NOT NULL DEFAULT '',
    version     INTEGER NOT NULL DEFAULT 0,
    mesi_state  TEXT NOT NULL DEFAULT 'E',
    canonical_surface TEXT NOT NULL DEFAULT '',
    witness_surface   TEXT NOT NULL DEFAULT '',
    prompt_surface    TEXT NOT NULL DEFAULT '',
    slot_key          TEXT NOT NULL DEFAULT '',
    value_text        TEXT NOT NULL DEFAULT '',
    qualifiers        TEXT NOT NULL DEFAULT '{}',
    state_confidence  REAL NOT NULL DEFAULT 1.0,
    topic_channel     TEXT NOT NULL DEFAULT '',
    visibility_scope  TEXT NOT NULL DEFAULT 'global',
    claim_key         TEXT NOT NULL DEFAULT '',
    memory_tier       TEXT NOT NULL DEFAULT 'private',
    PRIMARY KEY (agent_id, id)
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(agent_id, valid_until)",
    "CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(agent_id, entity, attribute)",
    "CREATE INDEX IF NOT EXISTS idx_facts_version ON facts(agent_id, version)",
]

_CREATE_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS snapshots (
    agent_id   TEXT NOT NULL,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    facts_json TEXT NOT NULL,
    PRIMARY KEY (agent_id, name)
)
"""

_INSERT_FACTS_SQL = """INSERT INTO facts
   (id, agent_id, content, type, importance, retention,
    access_count, tags, created_at, last_accessed,
    source_snippets, source_spans, supported,
    support_confidence, verification_source,
    access_intervals, origin_agent_id, visibility,
    source_verbatim, valid_from, valid_until,
    entity, attribute, version, mesi_state,
    canonical_surface, witness_surface, prompt_surface,
    slot_key, value_text, qualifiers, state_confidence,
    topic_channel, visibility_scope, claim_key, memory_tier)
   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


class SQLiteStorage:
    """Stores facts in a local SQLite database.

    Good for single-server production deployments where you want
    more robustness than YAML files but don't need a separate DB server.
    """

    #: v2 query-plane capability flag — checked by KnowledgeBase.query() and
    #: ingest_episode() as an explicit guard instead of Protocol isinstance().
    supports_v2_query_planes: bool = True

    def __init__(self, db_path: str = ".ai_knot/ai_knot.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Context manager that opens, yields, and always closes a SQLite connection."""
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_SNAPSHOTS_TABLE)
            for stmt in _CREATE_INDEXES:
                conn.execute(stmt)
            # v2 query planes — idempotent
            from ai_knot.migrations.v2_query_planes import apply_v2_migration

            apply_v2_migration(conn)
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
            "origin_agent_id": "TEXT NOT NULL DEFAULT ''",
            "visibility": "TEXT NOT NULL DEFAULT 'private'",
            "source_verbatim": "TEXT NOT NULL DEFAULT ''",
            "valid_from": "TEXT NOT NULL DEFAULT ''",
            "valid_until": "TEXT",
            "entity": "TEXT NOT NULL DEFAULT ''",
            "attribute": "TEXT NOT NULL DEFAULT ''",
            "version": "INTEGER NOT NULL DEFAULT 0",
            "mesi_state": "TEXT NOT NULL DEFAULT 'E'",
            "canonical_surface": "TEXT NOT NULL DEFAULT ''",
            "witness_surface": "TEXT NOT NULL DEFAULT ''",
            "prompt_surface": "TEXT NOT NULL DEFAULT ''",
            "slot_key": "TEXT NOT NULL DEFAULT ''",
            "value_text": "TEXT NOT NULL DEFAULT ''",
            "qualifiers": "TEXT NOT NULL DEFAULT '{}'",
            "state_confidence": "REAL NOT NULL DEFAULT 1.0",
            "topic_channel": "TEXT NOT NULL DEFAULT ''",
            "visibility_scope": "TEXT NOT NULL DEFAULT 'global'",
            "claim_key": "TEXT NOT NULL DEFAULT ''",
            "memory_tier": "TEXT NOT NULL DEFAULT 'private'",
        }
        with self._conn() as conn:
            cur = conn.execute("PRAGMA table_info(facts)")
            existing_cols = {row[1] for row in cur.fetchall()}
            for col, definition in new_columns.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {definition}")

    def save(self, agent_id: str, facts: list[Fact]) -> None:
        """Replace all facts for an agent."""
        rows = self._build_rows(agent_id, facts)
        self._execute_save(agent_id, rows)
        logger.debug("Saved %d facts for agent '%s'", len(facts), agent_id)

    def atomic_update(
        self,
        agent_id: str,
        fn: Callable[[list[Fact]], list[Fact]],
    ) -> None:
        """Load, transform, and save facts in a single EXCLUSIVE SQLite transaction.

        Blocks all other writers (including cross-process) for the duration,
        preventing lost-update races in shared-namespace publish operations.
        """
        conn = sqlite3.connect(self._db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            conn.execute("BEGIN EXCLUSIVE")
            rows = conn.execute(
                f"{self._SELECT_COLS} FROM facts WHERE agent_id = ? ORDER BY created_at",
                (agent_id,),
            ).fetchall()
            current = [self._fact_from_row(row) for row in rows]
            new_facts = fn(current)
            conn.execute("DELETE FROM facts WHERE agent_id = ?", (agent_id,))
            conn.executemany(_INSERT_FACTS_SQL, self._build_rows(agent_id, new_facts))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def save_atomic(self, agent_id: str, facts: list[Fact]) -> None:
        """Atomically replace facts using BEGIN IMMEDIATE to block concurrent writers."""
        rows = self._build_rows(agent_id, facts)
        conn = sqlite3.connect(self._db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM facts WHERE agent_id = ?", (agent_id,))
            conn.executemany(_INSERT_FACTS_SQL, rows)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        logger.debug("Atomically saved %d facts for agent '%s'", len(facts), agent_id)

    def _build_rows(self, agent_id: str, facts: list[Fact]) -> list[tuple]:  # type: ignore[type-arg]
        return [
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
                fact.canonical_surface,
                fact.witness_surface,
                fact.prompt_surface,
                fact.slot_key,
                fact.value_text,
                json.dumps(fact.qualifiers),
                fact.state_confidence,
                fact.topic_channel,
                fact.visibility_scope,
                fact.claim_key,
                fact.memory_tier,
            )
            for fact in facts
        ]

    def _execute_save(self, agent_id: str, rows: list[tuple]) -> None:  # type: ignore[type-arg]
        with self._conn() as conn:
            conn.execute("DELETE FROM facts WHERE agent_id = ?", (agent_id,))
            conn.executemany(_INSERT_FACTS_SQL, rows)

    def load(self, agent_id: str) -> list[Fact]:
        """Load all facts for an agent."""
        with self._conn() as conn:
            rows = conn.execute(
                f"{self._SELECT_COLS} FROM facts WHERE agent_id = ? ORDER BY created_at",
                (agent_id,),
            ).fetchall()

        return [self._fact_from_row(row) for row in rows]

    @staticmethod
    def _fact_from_row(row: tuple[Any, ...]) -> Fact:
        """Construct a Fact from a SELECT row (columns 0-32, matching _SELECT_COLS order)."""
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
            canonical_surface=str(row[24]) if row[24] else "",
            witness_surface=str(row[25]) if row[25] else "",
            prompt_surface=str(row[26]) if row[26] else "",
            slot_key=str(row[27]) if row[27] else "",
            value_text=str(row[28]) if row[28] else "",
            qualifiers=json.loads(row[29]) if row[29] else {},
            state_confidence=float(row[30]) if row[30] is not None else 1.0,
            topic_channel=str(row[31]) if row[31] else "",
            visibility_scope=str(row[32]) if row[32] else "global",
            claim_key=str(row[33]) if len(row) > 33 and row[33] else "",
            memory_tier=str(row[34]) if len(row) > 34 and row[34] else "private",
        )

    def delete(self, agent_id: str, fact_id: str) -> None:
        """Remove a single fact by id."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM facts WHERE agent_id = ? AND id = ?",
                (agent_id, fact_id),
            )

    def list_agents(self) -> list[str]:
        """Return all agent_ids that have stored facts."""
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT agent_id FROM facts").fetchall()
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
                          entity, attribute, version, mesi_state,
                          canonical_surface, witness_surface, prompt_surface,
                          slot_key, value_text, qualifiers, state_confidence,
                          topic_channel, visibility_scope, claim_key,
                          memory_tier"""

    def load_active(self, agent_id: str) -> list[Fact]:
        """Load only currently-active facts (index-accelerated).

        A fact is active when ``valid_until IS NULL`` (no expiry) and
        ``valid_from <= now`` (not yet future-dated).
        """
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                f"{self._SELECT_COLS} FROM facts"
                " WHERE agent_id = ? AND valid_until IS NULL"
                "   AND (valid_from = '' OR valid_from <= ?)"
                " ORDER BY created_at",
                (agent_id, now),
            ).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def load_since_version(self, agent_id: str, since: int, exclude_agent: str) -> list[Fact]:
        """MESI dirty pull: facts with version > since from agents other than exclude_agent."""
        with self._conn() as conn:
            rows = conn.execute(
                f"{self._SELECT_COLS} FROM facts"
                " WHERE agent_id = ? AND version > ? AND origin_agent_id != ?"
                " ORDER BY version",
                (agent_id, since, exclude_agent),
            ).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def load_active_frontier(self, agent_id: str) -> list[Fact]:
        """Return the latest active fact per slot_key (active frontier).

        For slotted facts, returns the highest-version active fact per slot.
        For unslotted facts, returns all active facts (no slot to collapse).
        Only returns facts where ``valid_from <= now`` (not future-dated).
        """
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            # Latest active version per slot (for slotted facts).
            slotted = conn.execute(
                f"{self._SELECT_COLS} FROM facts f"
                " INNER JOIN ("
                "   SELECT slot_key AS sk, MAX(version) AS max_v"
                "   FROM facts"
                "   WHERE agent_id = ? AND valid_until IS NULL AND slot_key != ''"
                "     AND (valid_from = '' OR valid_from <= ?)"
                "   GROUP BY sk"
                " ) m ON f.slot_key = m.sk AND f.version = m.max_v"
                " AND f.agent_id = ? AND f.valid_until IS NULL",
                (agent_id, now, agent_id),
            ).fetchall()
            # All active unslotted facts.
            unslotted = conn.execute(
                f"{self._SELECT_COLS} FROM facts"
                " WHERE agent_id = ? AND valid_until IS NULL AND slot_key = ''"
                "   AND (valid_from = '' OR valid_from <= ?)",
                (agent_id, now),
            ).fetchall()
        return [self._fact_from_row(r) for r in slotted + unslotted]

    def load_slot_deltas_since(
        self, agent_id: str, since_version: int, exclude_agent: str
    ) -> list[SlotDelta]:
        """Lightweight delta pull: slot changes since *since_version*, excluding *exclude_agent*.

        Returns ``SlotDelta`` objects instead of full ``Fact`` objects for
        bandwidth-efficient cross-agent synchronisation.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT slot_key, version, id, prompt_surface, content, valid_until, mesi_state"
                " FROM facts"
                " WHERE agent_id = ? AND version > ? AND origin_agent_id != ?"
                " ORDER BY version",
                (agent_id, since_version, exclude_agent),
            ).fetchall()

        deltas: list[SlotDelta] = []
        for slot_key, version, fact_id, prompt_surface, content, valid_until, mesi_state in rows:
            if valid_until is not None:
                op = "invalidate"
            elif str(mesi_state) == MESIState.MODIFIED:
                op = "supersede"
            else:
                op = "new"
            deltas.append(
                SlotDelta(
                    slot_key=str(slot_key) if slot_key else "",
                    version=int(version),
                    op=op,
                    fact_id=str(fact_id),
                    content=str(content),
                    prompt_surface=str(prompt_surface) if prompt_surface else "",
                )
            )
        return deltas

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
                "origin_agent_id": f.origin_agent_id,
                "visibility": f.visibility,
                "source_verbatim": f.source_verbatim,
                "valid_from": f.valid_from.isoformat(),
                "valid_until": f.valid_until.isoformat() if f.valid_until is not None else None,
                "entity": f.entity,
                "attribute": f.attribute,
                "version": f.version,
                "mesi_state": f.mesi_state,
                "canonical_surface": f.canonical_surface,
                "witness_surface": f.witness_surface,
                "prompt_surface": f.prompt_surface,
                "slot_key": f.slot_key,
                "value_text": f.value_text,
                "qualifiers": f.qualifiers,
                "state_confidence": f.state_confidence,
                "topic_channel": f.topic_channel,
                "visibility_scope": f.visibility_scope,
            }
            for f in facts
        ]
        with self._conn() as conn:
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
        with self._conn() as conn:
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
                origin_agent_id=str(entry.get("origin_agent_id", "")),
                visibility=str(entry.get("visibility", "private")),
                source_verbatim=str(entry.get("source_verbatim", "")),
                valid_from=_parse_datetime(str(entry["valid_from"]))
                if entry.get("valid_from")
                else datetime.now(UTC),
                valid_until=_parse_datetime(str(entry["valid_until"]))
                if entry.get("valid_until")
                else None,
                entity=str(entry.get("entity", "")),
                attribute=str(entry.get("attribute", "")),
                version=int(entry.get("version", 0)),
                mesi_state=MESIState(str(entry.get("mesi_state", "E"))),
                canonical_surface=str(entry.get("canonical_surface", "")),
                witness_surface=str(entry.get("witness_surface", "")),
                prompt_surface=str(entry.get("prompt_surface", "")),
                slot_key=str(entry.get("slot_key", "")),
                value_text=str(entry.get("value_text", "")),
                qualifiers={str(k): str(v) for k, v in entry.get("qualifiers", {}).items()},
                state_confidence=float(entry.get("state_confidence", 1.0)),
                topic_channel=str(entry.get("topic_channel", "")),
                visibility_scope=str(entry.get("visibility_scope", "global")),
            )
            for entry in raw
        ]

    def list_snapshots(self, agent_id: str) -> list[str]:
        """Return snapshot names sorted by creation time (oldest first)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name FROM snapshots WHERE agent_id = ? ORDER BY created_at",
                (agent_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def delete_snapshot(self, agent_id: str, name: str) -> None:
        """Delete a named snapshot. No-op if it does not exist."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM snapshots WHERE agent_id = ? AND name = ?",
                (agent_id, name),
            )

    # ------------------------------------------------------------------ #
    # RawEpisodeStore (v2)
    # ------------------------------------------------------------------ #

    def save_episodes(self, agent_id: str, episodes: list[Any]) -> None:
        """Upsert raw episodes."""
        if not episodes:
            return
        rows = [
            (
                ep.id,
                agent_id,
                ep.session_id,
                ep.turn_id,
                ep.speaker,
                ep.observed_at.isoformat(),
                ep.session_date.isoformat() if ep.session_date else None,
                ep.raw_text,
                json.dumps(ep.source_meta, ensure_ascii=False),
                ep.parent_episode_id,
            )
            for ep in episodes
        ]
        with self._conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO raw_episodes
                   (id, agent_id, session_id, turn_id, speaker, observed_at,
                    session_date, raw_text, source_meta, parent_episode_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def load_episodes(self, agent_id: str, *, session_id: str | None = None) -> list[Any]:
        """Load all episodes for agent, optionally filtered by session."""
        with self._conn() as conn:
            if session_id is not None:
                rows = conn.execute(
                    "SELECT id, agent_id, session_id, turn_id, speaker, observed_at, "
                    "session_date, raw_text, source_meta, parent_episode_id "
                    "FROM raw_episodes WHERE agent_id=? AND session_id=? ORDER BY observed_at",
                    (agent_id, session_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, agent_id, session_id, turn_id, speaker, observed_at, "
                    "session_date, raw_text, source_meta, parent_episode_id "
                    "FROM raw_episodes WHERE agent_id=? ORDER BY observed_at",
                    (agent_id,),
                ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def get_episode(self, agent_id: str, episode_id: str) -> Any:
        """Retrieve a single episode by id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, agent_id, session_id, turn_id, speaker, observed_at, "
                "session_date, raw_text, source_meta, parent_episode_id "
                "FROM raw_episodes WHERE agent_id=? AND id=?",
                (agent_id, episode_id),
            ).fetchone()
        return _row_to_episode(row) if row else None

    # ------------------------------------------------------------------ #
    # ClaimStore (v2)
    # ------------------------------------------------------------------ #

    def save_claims(self, agent_id: str, claims: list[Any]) -> None:
        """Upsert atomic claims."""
        if not claims:
            return
        rows = [_claim_to_row(agent_id, c) for c in claims]
        with self._conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO atomic_claims
                   (id, agent_id, kind, subject, relation, value_text, value_tokens,
                    qualifiers, polarity, event_time, observed_at, valid_from, valid_until,
                    confidence, salience, source_episode_id, source_spans,
                    materialization_version, materialized_at, slot_key, version, origin_agent_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def load_claims(
        self,
        agent_id: str,
        *,
        ids: list[str] | None = None,
        subjects: list[str] | None = None,
        kinds: list[Any] | None = None,
        active_only: bool = True,
    ) -> list[Any]:
        """Load claims with optional filters."""
        conditions = ["agent_id = ?"]
        params: list[Any] = [agent_id]
        if active_only:
            conditions.append("valid_until IS NULL")
        if ids is not None:
            placeholders = ",".join("?" * len(ids))
            conditions.append(f"id IN ({placeholders})")
            params.extend(ids)
        if subjects is not None:
            placeholders = ",".join("?" * len(subjects))
            conditions.append(f"subject IN ({placeholders})")
            params.extend(subjects)
        if kinds is not None:
            kind_vals = [k.value if hasattr(k, "value") else str(k) for k in kinds]
            placeholders = ",".join("?" * len(kind_vals))
            conditions.append(f"kind IN ({placeholders})")
            params.extend(kind_vals)
        where = " AND ".join(conditions)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, agent_id, kind, subject, relation, value_text, value_tokens, "
                f"qualifiers, polarity, event_time, observed_at, valid_from, valid_until, "
                f"confidence, salience, source_episode_id, source_spans, "
                f"materialization_version, materialized_at, slot_key, version, origin_agent_id "
                f"FROM atomic_claims WHERE {where} ORDER BY observed_at",
                params,
            ).fetchall()
        return [_row_to_claim(r) for r in rows]

    def iter_value_text(self, agent_id: str) -> list[tuple[str, str]]:
        """Return [(claim_id, search_text)] for BM25 retrieval.

        search_text = "{subject} {relation} {value_text}" for richer matching
        on subject and relation terms, not just value text.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, subject, relation, value_text FROM atomic_claims "
                "WHERE agent_id=? AND valid_until IS NULL",
                (agent_id,),
            ).fetchall()
        return [(r[0], f"{r[1] or ''} {r[2] or ''} {r[3]}".strip()) for r in rows]

    def replace_claims_for_episodes(
        self, agent_id: str, episode_ids: list[str], new_claims: list[Any]
    ) -> None:
        """Delete claims from these episodes and insert new ones."""
        if not episode_ids:
            return
        placeholders = ",".join("?" * len(episode_ids))
        with self._conn() as conn:
            conn.execute(
                f"DELETE FROM atomic_claims WHERE agent_id=? AND source_episode_id IN ({placeholders})",
                [agent_id, *episode_ids],
            )
            if new_claims:
                rows = [_claim_to_row(agent_id, c) for c in new_claims]
                conn.executemany(
                    """INSERT OR REPLACE INTO atomic_claims
                       (id, agent_id, kind, subject, relation, value_text, value_tokens,
                        qualifiers, polarity, event_time, observed_at, valid_from, valid_until,
                        confidence, salience, source_episode_id, source_spans,
                        materialization_version, materialized_at, slot_key, version, origin_agent_id)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )

    def delete_all_claims(self, agent_id: str) -> None:
        """Remove all claims for an agent."""
        with self._conn() as conn:
            conn.execute("DELETE FROM atomic_claims WHERE agent_id=?", (agent_id,))

    # ------------------------------------------------------------------ #
    # BundleStore (v2)
    # ------------------------------------------------------------------ #

    def save_bundles(
        self, agent_id: str, bundles: list[Any], memberships: dict[str, list[str]]
    ) -> None:
        """Persist bundles and their member claim lists.

        Idempotent: any existing bundles for the same (kind, topic) pairs are
        deleted first, including legacy random-id rows from older versions and
        shrinking member sets.  This ensures exactly one bundle per
        (agent_id, kind, topic) after the call.
        """
        if not bundles:
            return
        # (kind_val, topic) pairs we are about to write.
        kt_pairs = {
            (b.kind.value if hasattr(b.kind, "value") else str(b.kind), b.topic) for b in bundles
        }
        bundle_rows = [
            (
                b.id,
                agent_id,
                b.kind.value if hasattr(b.kind, "value") else str(b.kind),
                b.topic,
                b.bundle_score,
                b.score_formula,
                b.built_from_materialization_version,
                b.built_at.isoformat(),
            )
            for b in bundles
        ]
        member_rows = [
            (agent_id, bundle_id, claim_id, rank)
            for bundle_id, claim_ids in memberships.items()
            for rank, claim_id in enumerate(claim_ids)
        ]
        with self._conn() as conn:
            # Purge stale bundles (and their memberships) for each (kind, topic)
            # we are about to overwrite.  Handles both legacy random-id rows and
            # the case where the new bundle has fewer members than the old one.
            for kind_val, topic in kt_pairs:
                conn.execute(
                    "DELETE FROM bundle_members WHERE agent_id=? AND bundle_id IN "
                    "(SELECT id FROM support_bundles WHERE agent_id=? AND kind=? AND topic=?)",
                    (agent_id, agent_id, kind_val, topic),
                )
                conn.execute(
                    "DELETE FROM support_bundles WHERE agent_id=? AND kind=? AND topic=?",
                    (agent_id, kind_val, topic),
                )
            conn.executemany(
                """INSERT OR REPLACE INTO support_bundles
                   (id, agent_id, kind, topic, bundle_score, score_formula,
                    built_from_materialization_version, built_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                bundle_rows,
            )
            if member_rows:
                conn.executemany(
                    """INSERT OR REPLACE INTO bundle_members
                       (agent_id, bundle_id, claim_id, member_rank)
                       VALUES (?,?,?,?)""",
                    member_rows,
                )

    def load_bundles_by_topic(
        self,
        agent_id: str,
        topics: list[str],
        kinds: list[Any] | None = None,
    ) -> list[Any]:
        """Load bundles matching any of the given topics."""
        if not topics:
            return []
        conditions = ["agent_id = ?"]
        params: list[Any] = [agent_id]
        placeholders = ",".join("?" * len(topics))
        conditions.append(f"topic IN ({placeholders})")
        params.extend(topics)
        if kinds is not None:
            kind_vals = [k.value if hasattr(k, "value") else str(k) for k in kinds]
            kp = ",".join("?" * len(kind_vals))
            conditions.append(f"kind IN ({kp})")
            params.extend(kind_vals)
        where = " AND ".join(conditions)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, agent_id, kind, topic, bundle_score, score_formula, "
                f"built_from_materialization_version, built_at "
                f"FROM support_bundles WHERE {where} "
                f"ORDER BY kind, topic, built_at DESC, id",
                params,
            ).fetchall()
        return [_row_to_bundle(r) for r in rows]

    def load_bundle_members(self, agent_id: str, bundle_ids: list[str]) -> dict[str, list[str]]:
        """Return {bundle_id: [claim_id, ...]} for each requested bundle."""
        if not bundle_ids:
            return {}
        placeholders = ",".join("?" * len(bundle_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT bundle_id, claim_id FROM bundle_members "
                f"WHERE agent_id=? AND bundle_id IN ({placeholders}) ORDER BY bundle_id, member_rank",
                [agent_id, *bundle_ids],
            ).fetchall()
        result: dict[str, list[str]] = {bid: [] for bid in bundle_ids}
        for bundle_id, claim_id in rows:
            result.setdefault(bundle_id, []).append(claim_id)
        return result

    def invalidate_by_keys(self, agent_id: str, keys: list[Any]) -> int:
        """Delete bundles matching any dirty key; return count removed."""
        from ai_knot.query_types import BundleKind

        if not keys:
            return 0
        bundle_ids_to_delete: set[str] = set()
        with self._conn() as conn:
            for key in keys:
                if key.subject and key.relation:
                    topic = f"{key.subject}::{key.relation}"
                    rows = conn.execute(
                        "SELECT id FROM support_bundles WHERE agent_id=? AND topic=? "
                        "AND kind IN (?,?)",
                        (
                            agent_id,
                            topic,
                            BundleKind.STATE_TIMELINE.value,
                            BundleKind.RELATION_SUPPORT.value,
                        ),
                    ).fetchall()
                    bundle_ids_to_delete.update(r[0] for r in rows)
                elif key.subject:
                    rows = conn.execute(
                        "SELECT id FROM support_bundles WHERE agent_id=? AND topic=? "
                        "AND kind IN (?,?)",
                        (
                            agent_id,
                            key.subject,
                            BundleKind.ENTITY_TOPIC.value,
                            BundleKind.EVENT_NEIGHBORHOOD.value,
                        ),
                    ).fetchall()
                    bundle_ids_to_delete.update(r[0] for r in rows)
                elif key.bundle_kind and key.topic:
                    kind_val = (
                        key.bundle_kind.value
                        if hasattr(key.bundle_kind, "value")
                        else str(key.bundle_kind)
                    )
                    rows = conn.execute(
                        "SELECT id FROM support_bundles WHERE agent_id=? AND kind=? AND topic=?",
                        (agent_id, kind_val, key.topic),
                    ).fetchall()
                    bundle_ids_to_delete.update(r[0] for r in rows)

            if bundle_ids_to_delete:
                bp = ",".join("?" * len(bundle_ids_to_delete))
                conn.execute(
                    f"DELETE FROM bundle_members WHERE agent_id=? AND bundle_id IN ({bp})",
                    [agent_id, *bundle_ids_to_delete],
                )
                conn.execute(
                    f"DELETE FROM support_bundles WHERE agent_id=? AND id IN ({bp})",
                    [agent_id, *bundle_ids_to_delete],
                )
        return len(bundle_ids_to_delete)

    def clear_all_bundles(self, agent_id: str) -> None:
        """Remove all bundles and memberships for an agent."""
        with self._conn() as conn:
            conn.execute("DELETE FROM bundle_members WHERE agent_id=?", (agent_id,))
            conn.execute("DELETE FROM support_bundles WHERE agent_id=?", (agent_id,))

    # ------------------------------------------------------------------ #
    # MaterializationMetaStore (v2)
    # ------------------------------------------------------------------ #

    def load_materialization_meta(self, agent_id: str) -> dict[str, Any]:
        """Return metadata dict; empty dict if no record exists."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT schema_version, materialization_version, last_rebuild_at, "
                "dirty_keys_json, rebuild_status FROM materialization_meta WHERE agent_id=?",
                (agent_id,),
            ).fetchone()
        if row is None:
            return {}
        return {
            "schema_version": row[0],
            "materialization_version": row[1],
            "last_rebuild_at": row[2],
            "dirty_keys_json": row[3],
            "rebuild_status": row[4],
        }

    def save_materialization_meta(
        self,
        agent_id: str,
        *,
        schema_version: int,
        materialization_version: int,
        last_rebuild_at: datetime | None = None,
        dirty_keys_json: str = "[]",
        rebuild_status: str = "ready",
    ) -> None:
        """Upsert materialization metadata for an agent."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO materialization_meta
                   (agent_id, schema_version, materialization_version,
                    last_rebuild_at, dirty_keys_json, rebuild_status)
                   VALUES (?,?,?,?,?,?)""",
                (
                    agent_id,
                    schema_version,
                    materialization_version,
                    last_rebuild_at.isoformat() if last_rebuild_at else None,
                    dirty_keys_json,
                    rebuild_status,
                ),
            )


# ---------------------------------------------------------------------------
# Module-level row conversion helpers (no self reference)
# ---------------------------------------------------------------------------


def _row_to_episode(row: tuple[Any, ...]) -> Any:
    from ai_knot.query_types import RawEpisode
    from ai_knot.storage.base import parse_datetime as _pd

    (
        ep_id,
        agent_id,
        session_id,
        turn_id,
        speaker,
        observed_at,
        session_date,
        raw_text,
        source_meta_json,
        parent_episode_id,
    ) = row
    return RawEpisode(
        id=ep_id,
        agent_id=agent_id,
        session_id=session_id,
        turn_id=turn_id,
        speaker=speaker,
        observed_at=_pd(observed_at),
        session_date=_pd(session_date) if session_date else None,
        raw_text=raw_text,
        source_meta=json.loads(source_meta_json),
        parent_episode_id=parent_episode_id,
    )


def _claim_to_row(agent_id: str, c: Any) -> tuple[Any, ...]:
    return (
        c.id,
        agent_id,
        c.kind.value if hasattr(c.kind, "value") else str(c.kind),
        c.subject,
        c.relation,
        c.value_text,
        json.dumps(list(c.value_tokens), ensure_ascii=False),
        json.dumps(c.qualifiers, ensure_ascii=False),
        c.polarity,
        c.event_time.isoformat() if c.event_time else None,
        c.observed_at.isoformat(),
        c.valid_from.isoformat(),
        c.valid_until.isoformat() if c.valid_until else None,
        c.confidence,
        c.salience,
        c.source_episode_id,
        json.dumps([list(s) for s in c.source_spans], ensure_ascii=False),
        c.materialization_version,
        c.materialized_at.isoformat(),
        c.slot_key,
        c.version,
        c.origin_agent_id,
    )


def _row_to_claim(row: tuple[Any, ...]) -> Any:
    from ai_knot.query_types import AtomicClaim, ClaimKind
    from ai_knot.storage.base import parse_datetime as _pd

    (
        claim_id,
        agent_id,
        kind_str,
        subject,
        relation,
        value_text,
        value_tokens_json,
        qualifiers_json,
        polarity,
        event_time_str,
        observed_at_str,
        valid_from_str,
        valid_until_str,
        confidence,
        salience,
        source_episode_id,
        source_spans_json,
        materialization_version,
        materialized_at_str,
        slot_key,
        version,
        origin_agent_id,
    ) = row
    return AtomicClaim(
        id=claim_id,
        agent_id=agent_id,
        kind=ClaimKind(kind_str),
        subject=subject,
        relation=relation,
        value_text=value_text,
        value_tokens=tuple(json.loads(value_tokens_json)),
        qualifiers=json.loads(qualifiers_json),
        polarity=polarity,
        event_time=_pd(event_time_str) if event_time_str else None,
        observed_at=_pd(observed_at_str),
        valid_from=_pd(valid_from_str),
        valid_until=_pd(valid_until_str) if valid_until_str else None,
        confidence=confidence,
        salience=salience,
        source_episode_id=source_episode_id,
        source_spans=tuple(tuple(s) for s in json.loads(source_spans_json)),
        materialization_version=materialization_version,
        materialized_at=_pd(materialized_at_str),
        slot_key=slot_key,
        version=version,
        origin_agent_id=origin_agent_id,
    )


def _row_to_bundle(row: tuple[Any, ...]) -> Any:
    from ai_knot.query_types import BundleKind, SupportBundle
    from ai_knot.storage.base import parse_datetime as _pd

    (
        bundle_id,
        agent_id,
        kind_str,
        topic,
        bundle_score,
        score_formula,
        built_from_version,
        built_at_str,
    ) = row
    return SupportBundle(
        id=bundle_id,
        agent_id=agent_id,
        kind=BundleKind(kind_str),
        topic=topic,
        member_claim_ids=(),  # loaded separately
        score_formula=score_formula,
        bundle_score=bundle_score,
        built_from_materialization_version=built_from_version,
        built_at=_pd(built_at_str),
    )
