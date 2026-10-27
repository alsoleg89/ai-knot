"""Transaction-local SQL seam shared by the two durable backends."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, TypeAlias, runtime_checkable

from ai_knot.always_on.changes import access_projection, evidence_revision
from ai_knot.types import Fact

SQLScalar: TypeAlias = str | int | float | bytes | None
SQLRow: TypeAlias = dict[str, SQLScalar]


@dataclass(frozen=True)
class SQLStatement:
    sqlite: str
    postgres: str


def sql(statement: str) -> SQLStatement:
    """Translate fixed internal AO SQL; never accepts user-controlled identifiers."""
    pg = re.sub(r"\bao_([a-z_]+)\b", r'"ai-knot_ao_\1"', statement).replace("?", "%s")
    pg = pg.replace("acl_grants", '"ai-knot_acl_grants"')
    return SQLStatement(statement, pg)


class AlwaysOnSession(Protocol):
    def load_facts(
        self, namespace: str, *, fact_id: str | None = None, slot_key: str | None = None
    ) -> list[Fact]: ...
    def replace_facts(self, namespace: str, facts: list[Fact]) -> None: ...
    def fetch_one(
        self, statement: SQLStatement, params: Sequence[SQLScalar] = ()
    ) -> SQLRow | None: ...
    def fetch_all(
        self, statement: SQLStatement, params: Sequence[SQLScalar] = ()
    ) -> list[SQLRow]: ...
    def execute(self, statement: SQLStatement, params: Sequence[SQLScalar] = ()) -> int: ...


@runtime_checkable
class AlwaysOnStorageCapable(Protocol):
    def always_on_transaction(
        self, namespaces: tuple[str, ...], *, write: bool
    ) -> AbstractContextManager[AlwaysOnSession]: ...


def namespace_lock_key(namespace: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(namespace.encode(), digest_size=8, person=b"ai-knot-ao-v1").digest(),
        "big",
        signed=True,
    )


class SQLSession:
    def __init__(
        self,
        conn: Any,
        namespaces: tuple[str, ...],
        *,
        write: bool,
        postgres: bool,
        loader: Callable[..., list[Fact]],
        saver: Callable[[Any, str, list[Fact]], None],
    ) -> None:
        self._connection = conn
        self._namespaces = namespaces
        self._write = write
        self._postgres = postgres
        self._loader = loader
        self._saver = saver

    def _check(self, namespace: str) -> None:
        if namespace not in self._namespaces:
            raise ValueError("namespace not declared in transaction")

    def load_facts(
        self, namespace: str, *, fact_id: str | None = None, slot_key: str | None = None
    ) -> list[Fact]:
        self._check(namespace)
        if fact_id is None and slot_key is None:
            return self._loader(self._connection, namespace)
        return self._loader(self._connection, namespace, fact_id=fact_id, slot_key=slot_key)

    def replace_facts(self, namespace: str, facts: list[Fact]) -> None:
        self._check(namespace)
        if not self._write:
            raise ValueError("read-only transaction")
        self._saver(self._connection, namespace, facts)

    def _cursor(self, statement: SQLStatement, params: Sequence[SQLScalar]) -> Any:
        query = statement.postgres if self._postgres else statement.sqlite
        return self._connection.execute(query, tuple(params))

    def fetch_all(self, statement: SQLStatement, params: Sequence[SQLScalar] = ()) -> list[SQLRow]:
        cur = self._cursor(statement, params)
        names = [column[0] for column in cur.description]
        result: list[SQLRow] = []
        for row in cur.fetchall():
            normalized = [
                value.astimezone(UTC).isoformat(timespec="microseconds")
                if isinstance(value, datetime)
                else value
                for value in row
            ]
            result.append(dict(zip(names, normalized, strict=True)))
        return result

    def fetch_one(self, statement: SQLStatement, params: Sequence[SQLScalar] = ()) -> SQLRow | None:
        rows = self.fetch_all(statement, params)
        return rows[0] if rows else None

    def execute(self, statement: SQLStatement, params: Sequence[SQLScalar] = ()) -> int:
        if not self._write:
            raise ValueError("read-only transaction")
        return int(self._cursor(statement, params).rowcount)


DDL = (
    (
        "CREATE TABLE IF NOT EXISTS ao_schema (id INTEGER PRIMARY KEY CHECK(id="
        "1), version INTEGER NOT NULL)"
    ),
    (
        "CREATE TABLE IF NOT EXISTS ao_streams (namespace TEXT PRIMARY KEY, ena"
        "bled INTEGER NOT NULL CHECK(enabled IN(0,1)), head_seq BIGINT NOT NULL"
        " DEFAULT 0 CHECK(head_seq>=0), min_retained_seq BIGINT NOT NULL DEFAUL"
        "T 1 CHECK(min_retained_seq>=1 AND min_retained_seq<=head_seq+1))"
    ),
    (
        "CREATE TABLE IF NOT EXISTS ao_changes (namespace TEXT NOT NULL, seq BI"
        "GINT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('insert','update','de"
        "lete','resync_required','acl_changed')), fact_id TEXT NOT NULL, slot_k"
        "ey TEXT NOT NULL, before_slot_key TEXT NOT NULL DEFAULT '', before_rev"
        "ision TEXT, after_revision TEXT, old_access_json TEXT, new_access_json"
        " TEXT, acl_subject TEXT, PRIMARY KEY(namespace,seq))"
    ),
    "CREATE INDEX IF NOT EXISTS ao_changes_fact ON ao_changes(namespace,fact_id,seq)",
    (
        "CREATE TABLE IF NOT EXISTS ao_subscriptions (owner TEXT NOT NULL, id T"
        "EXT NOT NULL, incarnation TEXT NOT NULL, namespace TEXT NOT NULL, sele"
        "ctor_json TEXT NOT NULL, cursor BIGINT NOT NULL CHECK(cursor>=0), debo"
        "unce_seconds DOUBLE PRECISION NOT NULL CHECK(debounce_seconds BETWEEN "
        "0 AND 3600), pending_since TEXT, inflight_token TEXT, inflight_through"
        "_seq BIGINT, inflight_events_json TEXT, inflight_resync INTEGER, infli"
        "ght_acl_digest TEXT, PRIMARY KEY(owner,id), CHECK((inflight_token IS N"
        "ULL AND inflight_through_seq IS NULL AND inflight_events_json IS NULL "
        "AND inflight_resync IS NULL AND inflight_acl_digest IS NULL) OR (infli"
        "ght_token IS NOT NULL AND inflight_through_seq IS NOT NULL AND infligh"
        "t_events_json IS NOT NULL AND inflight_resync IS NOT NULL AND inflight"
        "_acl_digest IS NOT NULL)))"
    ),
    "CREATE INDEX IF NOT EXISTS ao_subscriptions_cursor ON ao_subscriptions(namespace,cursor)",
    (
        "CREATE TABLE IF NOT EXISTS ao_delivery_receipts (owner TEXT NOT NULL, "
        "subscription_id TEXT NOT NULL, incarnation TEXT NOT NULL, delivery_id "
        "TEXT NOT NULL, through_seq BIGINT NOT NULL, PRIMARY KEY(owner,subscrip"
        "tion_id,incarnation,delivery_id))"
    ),
)


def migrate_freshness(session: AlwaysOnSession) -> None:
    columns = session.fetch_all(
        SQLStatement(
            "SELECT name FROM pragma_table_info('ao_freshness')",
            "SELECT column_name AS name FROM information_schema.columns "
            "WHERE table_schema=current_schema() AND table_name='ai-knot_ao_freshness'",
        )
    )
    names = {row["name"] for row in columns}
    for name, declaration in (
        ("revision", "BIGINT NOT NULL DEFAULT 1"),
        ("dirty", "INTEGER NOT NULL DEFAULT 1"),
    ):
        if name not in names:
            session.execute(sql(f"ALTER TABLE ao_freshness ADD COLUMN {name} {declaration}"))


def migrate(session: AlwaysOnSession) -> None:
    for statement in DDL:
        session.execute(sql(statement))
    row = session.fetch_one(sql("SELECT version FROM ao_schema WHERE id=1"))
    if row is not None and int(row["version"] or 0) > 6:
        raise RuntimeError("unsupported always-on schema version")
    session.execute(sql("INSERT INTO ao_schema(id,version) VALUES(1,2) ON CONFLICT(id) DO NOTHING"))
    if row is not None and int(row["version"] or 0) >= 5:
        migrate_freshness(session)


def append_event(
    session: AlwaysOnSession,
    namespace: str,
    *,
    kind: str,
    fact_id: str = "",
    slot_key: str = "",
    before_slot_key: str = "",
    before_revision: str | None = None,
    after_revision: str | None = None,
    old_access_json: str | None = None,
    new_access_json: str | None = None,
    acl_subject: str | None = None,
) -> None:
    row = session.fetch_one(sql("SELECT enabled FROM ao_streams WHERE namespace=?"), (namespace,))
    if row is None or row["enabled"] != 1:
        return
    session.execute(
        sql("UPDATE ao_streams SET head_seq=head_seq+1 WHERE namespace=?"), (namespace,)
    )
    head = session.fetch_one(sql("SELECT head_seq FROM ao_streams WHERE namespace=?"), (namespace,))
    assert head is not None
    session.execute(
        sql(
            "INSERT INTO ao_changes(namespace,seq,kind,fact_id,slot_key,before_slot"
            "_key,before_revision,after_revision,old_access_json,new_access_json,ac"
            "l_subject) VALUES(?,?,?,?,?,?,?,?,?,?,?)"
        ),
        (
            namespace,
            head["head_seq"],
            kind,
            fact_id,
            slot_key,
            before_slot_key,
            before_revision,
            after_revision,
            old_access_json,
            new_access_json,
            acl_subject,
        ),
    )


def journal_diff(
    session: AlwaysOnSession, namespace: str, before: list[Fact], after: list[Fact]
) -> None:
    state = session.fetch_one(
        sql(
            "SELECT (SELECT enabled FROM ao_streams WHERE namespace=?) AS enabled, "
            "(SELECT version FROM ao_schema WHERE id=1) AS schema_version"
        ),
        (namespace,),
    )
    enabled = state is not None and state["enabled"] == 1
    policies = state is not None and int(state["schema_version"] or 0) >= 5
    if policies:
        policies = (
            session.fetch_one(
                sql("SELECT 1 AS present FROM ao_freshness WHERE namespace=? LIMIT 1"), (namespace,)
            )
            is not None
        )
    if not enabled and not policies:
        return
    old = {fact.id: fact for fact in before}
    new = {fact.id: fact for fact in after}
    for fact_id in sorted(old.keys() | new.keys()):
        prior, current = old.get(fact_id), new.get(fact_id)
        a = evidence_revision(prior) if prior else None
        b = evidence_revision(current) if current else None
        if a == b:
            continue
        if policies:
            session.execute(
                sql(
                    "UPDATE ao_freshness SET dirty=CASE WHEN bound_revision=? THEN 0 ELSE 1 END, "
                    "revision=revision+1 WHERE namespace=? "
                    "AND subject_kind='fact' AND subject_key=?"
                ),
                (b, namespace, fact_id),
            )
            for key in {
                fact.slot_key for fact in (prior, current) if fact is not None and fact.slot_key
            }:
                session.execute(
                    sql(
                        "UPDATE ao_freshness SET dirty=1, revision=revision+1 "
                        "WHERE namespace=? AND subject_kind='slot' AND subject_key=?"
                    ),
                    (namespace, key),
                )
        if enabled:
            append_event(
                session,
                namespace,
                kind="delete" if current is None else "insert" if prior is None else "update",
                fact_id=fact_id,
                slot_key=(current or prior).slot_key,  # type: ignore[union-attr]
                before_slot_key=prior.slot_key if prior else "",
                before_revision=a,
                after_revision=b,
                old_access_json=access_projection(prior) if prior else None,
                new_access_json=access_projection(current) if current else None,
            )


def enable_stream(session: AlwaysOnSession, namespace: str, now: datetime) -> None:
    if now.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    existing = session.fetch_one(
        sql("SELECT enabled FROM ao_streams WHERE namespace=?"), (namespace,)
    )
    if existing is not None:
        return
    session.execute(sql("INSERT INTO ao_streams(namespace,enabled) VALUES(?,1)"), (namespace,))
    if session.load_facts(namespace):
        append_event(session, namespace, kind="resync_required")
