"""Executable SQLite design check, NOT the AI Knot implementation.

Run: python3 docs/superpowers/reference/always_on_storage_reference.py
Stdlib only; temporary databases; no production imports, models, or external I/O.
Scope: private fact streams. ACL, PostgreSQL and production serializers need
their own integration tests; see 01-storage-code.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

MAX_BYTES = 65536
SCAN_LIMIT = 1000
EVIDENCE_FIELDS = (
    "content",
    "type",
    "slot_key",
    "entity",
    "attribute",
    "value_text",
    "canonical_surface",
    "witness_surface",
    "prompt_surface",
    "qualifiers",
    "source_snippets",
    "source_spans",
    "source_verbatim",
    "supported",
    "support_confidence",
    "verification_source",
    "state_confidence",
    "valid_from",
    "valid_until",
    "event_time",
    "origin_agent_id",
    "visibility",
    "visibility_scope",
    "claim_key",
    "topic_channel",
    "memory_tier",
)


def canonical(value: Any, *, bounded: bool = True) -> str:
    """Canonical UTF-8 JSON, rejecting invalid object keys and nonfinite floats."""

    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(k, str) or not k for k in item):
                raise ValueError("JSON keys must be nonempty strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")

    check(value)
    result = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    if bounded and len(result.encode("utf-8")) > MAX_BYTES:
        raise ValueError("JSON exceeds 65536 UTF-8 bytes")
    return result


def timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware datetime required")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Fact:
    """Reduced transport model; defaults are fixed, never wall-clock factories."""

    id: str
    content: str
    type: str = "semantic"
    slot_key: str = ""
    entity: str = ""
    attribute: str = ""
    value_text: str = ""
    canonical_surface: str = ""
    witness_surface: str = ""
    prompt_surface: str = ""
    qualifiers: dict[str, str] | None = None
    source_snippets: tuple[str, ...] = ()
    source_spans: tuple[str, ...] = ()
    source_verbatim: str = ""
    supported: bool = True
    support_confidence: float = 1.0
    verification_source: str = "manual"
    state_confidence: float = 1.0
    valid_from: str = "2026-10-26T00:00:00.000000+00:00"
    valid_until: str | None = None
    event_time: str | None = None
    origin_agent_id: str = ""
    visibility: str = "private"
    visibility_scope: str = "global"
    claim_key: str = ""
    topic_channel: str = ""
    memory_tier: str = "private"
    access_count: int = 0
    last_accessed: str = "2026-10-26T00:00:00.000000+00:00"
    access_intervals: tuple[float, ...] = ()
    retention_score: float = 1.0
    importance: float = 0.8
    tags: tuple[str, ...] = ()
    version: int = 0
    mesi_state: str = "E"

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["qualifiers"] = result["qualifiers"] or {}
        for name in ("valid_from", "valid_until", "event_time"):
            if result[name] is not None:
                result[name] = timestamp(datetime.fromisoformat(result[name]))
        if not self.id or self.type not in {"semantic", "procedural", "episodic"}:
            raise ValueError("invalid fact identity/type")
        canonical(result)
        return result


def evidence_revision(fact: Fact) -> str:
    payload = fact.payload()
    data = canonical({k: payload[k] for k in EVIDENCE_FIELDS})
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Change:
    seq: int
    kind: str
    fact_id: str
    slot_key: str
    before_slot_key: str
    before_revision: str | None
    after_revision: str | None


@dataclass(frozen=True)
class Batch:
    delivery_id: str | None = None
    events: tuple[Change, ...] = ()
    resync_required: bool = False


class InvalidDelivery(ValueError):
    pass


class CursorExpired(ValueError):
    pass


DDL = """
CREATE TABLE IF NOT EXISTS ao_schema(version INTEGER PRIMARY KEY CHECK(version=1));
INSERT OR IGNORE INTO ao_schema VALUES(1);
CREATE TABLE IF NOT EXISTS facts(
 namespace TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
 PRIMARY KEY(namespace,id));
CREATE TABLE IF NOT EXISTS ao_streams(
 namespace TEXT PRIMARY KEY, enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
 head_seq INTEGER NOT NULL DEFAULT 0, min_retained_seq INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS ao_changes(
 namespace TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
 fact_id TEXT NOT NULL, slot_key TEXT NOT NULL, before_slot_key TEXT NOT NULL,
 before_revision TEXT, after_revision TEXT, PRIMARY KEY(namespace,seq));
CREATE TABLE IF NOT EXISTS ao_subscriptions(
 owner TEXT NOT NULL, id TEXT NOT NULL, incarnation TEXT NOT NULL,
 namespace TEXT NOT NULL, selector TEXT NOT NULL, cursor INTEGER NOT NULL,
 debounce REAL NOT NULL, pending_since TEXT,
 inflight_token TEXT, inflight_through INTEGER, inflight_json TEXT,
 PRIMARY KEY(owner,id));
CREATE INDEX IF NOT EXISTS ao_subscriptions_retention
 ON ao_subscriptions(namespace,cursor);
CREATE TABLE IF NOT EXISTS ao_delivery_receipts(
 owner TEXT NOT NULL, subscription_id TEXT NOT NULL, incarnation TEXT NOT NULL,
 token TEXT NOT NULL, through_seq INTEGER NOT NULL,
 PRIMARY KEY(owner,subscription_id,incarnation,token));
"""


class Store:
    """Private stream protocol demonstrator, deliberately not a public facade."""

    def __init__(self, path: Path, clock: Callable[[], datetime]) -> None:
        self.path, self.clock = path, clock
        with self.transaction() as conn:
            # executescript would commit a preceding transaction: execute DDL singly.
            for statement in DDL.split(";"):
                if statement.strip():
                    conn.execute(statement)

    @contextmanager
    def transaction(
        self, *, write: bool = True, timeout: float = 0.05
    ) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=timeout)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @staticmethod
    def _load(conn: sqlite3.Connection, namespace: str) -> list[Fact]:
        return [
            Fact(**json.loads(row[0]))
            for row in conn.execute(
                "SELECT payload FROM facts WHERE namespace=? ORDER BY id", (namespace,)
            )
        ]

    def load(self, namespace: str) -> list[Fact]:
        with self.transaction(write=False) as conn:
            return self._load(conn, namespace)

    @staticmethod
    def _append(
        conn: sqlite3.Connection, namespace: str, kind: str, old: Fact | None, new: Fact | None
    ) -> None:
        conn.execute("UPDATE ao_streams SET head_seq=head_seq+1 WHERE namespace=?", (namespace,))
        seq = conn.execute(
            "SELECT head_seq FROM ao_streams WHERE namespace=?", (namespace,)
        ).fetchone()[0]
        fact = new or old
        conn.execute(
            "INSERT INTO ao_changes VALUES(?,?,?,?,?,?,?,?)",
            (
                namespace,
                seq,
                kind,
                fact.id if fact else "",
                fact.slot_key if fact else "",
                old.slot_key if old else "",
                evidence_revision(old) if old else None,
                evidence_revision(new) if new else None,
            ),
        )

    def enable(self, namespace: str) -> None:
        if not namespace:
            raise ValueError("empty namespace")
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT 1 FROM ao_streams WHERE namespace=?", (namespace,)
            ).fetchone()
            if existing:
                return
            conn.execute("INSERT INTO ao_streams(namespace,enabled) VALUES(?,1)", (namespace,))
            if self._load(conn, namespace):
                self._append(conn, namespace, "resync_required", None, None)

    def _replace(
        self, conn: sqlite3.Connection, namespace: str, facts: list[Fact], *, fail_after: str = ""
    ) -> None:
        before = {f.id: f for f in self._load(conn, namespace)}
        after = {f.id: f for f in facts}
        if len(after) != len(facts):
            raise ValueError("duplicate fact ID")
        rows = [(namespace, f.id, canonical(f.payload())) for f in facts]
        conn.execute("DELETE FROM facts WHERE namespace=?", (namespace,))
        conn.executemany("INSERT INTO facts VALUES(?,?,?)", rows)
        if fail_after == "facts":
            raise RuntimeError("injected after facts")
        if not conn.execute(
            "SELECT 1 FROM ao_streams WHERE namespace=? AND enabled=1", (namespace,)
        ).fetchone():
            return
        for fact_id in sorted(before.keys() | after.keys()):
            old, new = before.get(fact_id), after.get(fact_id)
            if old and new and evidence_revision(old) == evidence_revision(new):
                continue
            kind = "insert" if old is None else "delete" if new is None else "update"
            self._append(conn, namespace, kind, old, new)
            if fail_after == "event":
                raise RuntimeError("injected after event/head")

    def mutate(
        self, namespace: str, fn: Callable[[list[Fact]], list[Fact]], *, fail_after: str = ""
    ) -> None:
        with self.transaction() as conn:
            self._replace(conn, namespace, fn(self._load(conn, namespace)), fail_after=fail_after)

    def save(self, namespace: str, facts: list[Fact], *, fail_after: str = "") -> None:
        self.mutate(namespace, lambda _: facts, fail_after=fail_after)

    def head(self, namespace: str) -> int:
        with self.transaction(write=False) as conn:
            row = conn.execute(
                "SELECT head_seq FROM ao_streams WHERE namespace=?", (namespace,)
            ).fetchone()
            return row[0] if row else 0

    def subscribe(
        self,
        owner: str,
        subscription_id: str,
        namespace: str,
        *,
        start: str = "now",
        fact_ids: tuple[str, ...] = (),
        slot_keys: tuple[str, ...] = (),
        debounce: float = 0,
    ) -> None:
        if namespace != owner:
            raise PermissionError("reference supports private owner namespace only")
        if (
            not owner
            or not subscription_id
            or start not in {"now", "beginning"}
            or isinstance(debounce, bool)
            or not math.isfinite(debounce)
            or not 0 <= debounce <= 3600
        ):
            raise ValueError("invalid subscription")
        if any(not isinstance(v, str) or not v for v in (*fact_ids, *slot_keys)):
            raise ValueError("empty selector key")
        selector = canonical(
            {"fact_ids": sorted(set(fact_ids)), "slot_keys": sorted(set(slot_keys))}
        )
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM ao_subscriptions WHERE owner=? AND id=?", (owner, subscription_id)
            ).fetchone()
            if row:
                if (row["namespace"], row["selector"], row["debounce"]) != (
                    namespace,
                    selector,
                    debounce,
                ):
                    raise ValueError("RevisionConflict")
                return
            stream = conn.execute(
                "SELECT * FROM ao_streams WHERE namespace=? AND enabled=1", (namespace,)
            ).fetchone()
            if stream is None:
                raise ValueError("enable stream explicitly first")
            # beginning with pruned history demands resync, never a silent jump.
            cursor = stream["head_seq"] if start == "now" else 0
            if cursor < stream["min_retained_seq"] - 1:
                raise CursorExpired("beginning pruned; subscribe now, then capture current state")
            conn.execute(
                """INSERT INTO ao_subscriptions
                (owner,id,incarnation,namespace,selector,cursor,debounce)
                VALUES(?,?,?,?,?,?,?)""",
                (owner, subscription_id, uuid4().hex, namespace, selector, cursor, debounce),
            )

    @staticmethod
    def _subscription(conn: sqlite3.Connection, owner: str, sid: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM ao_subscriptions WHERE owner=? AND id=?", (owner, sid)
        ).fetchone()
        if row is None:
            raise KeyError(sid)
        return row

    @staticmethod
    def _batch(token: str, payload: str) -> Batch:
        data = json.loads(payload)
        return Batch(token, tuple(Change(**e) for e in data["events"]), data["resync"])

    def poll(self, owner: str, sid: str, *, limit: int = 100) -> Batch:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be 1..500")
        now = self.clock()
        timestamp(now)
        with self.transaction() as conn:
            sub = self._subscription(conn, owner, sid)
            stream = conn.execute(
                "SELECT * FROM ao_streams WHERE namespace=?", (sub["namespace"],)
            ).fetchone()
            if sub["cursor"] < stream["min_retained_seq"] - 1:
                raise CursorExpired("subscription behind retained history")
            if sub["inflight_token"]:
                return self._batch(sub["inflight_token"], sub["inflight_json"])
            rows = conn.execute(
                """SELECT * FROM ao_changes
                WHERE namespace=? AND seq>? ORDER BY seq LIMIT ?""",
                (sub["namespace"], sub["cursor"], SCAN_LIMIT),
            ).fetchall()
            if not rows:
                return Batch()
            selector = json.loads(sub["selector"])
            events: dict[str, Change] = {}
            through, resync = sub["cursor"], False
            for row in rows:
                matches = (
                    not selector["fact_ids"]
                    and not selector["slot_keys"]
                    or row["fact_id"] in selector["fact_ids"]
                    or row["slot_key"] in selector["slot_keys"]
                    or row["before_slot_key"] in selector["slot_keys"]
                )
                if row["kind"] == "resync_required":
                    resync = True
                elif matches:
                    if row["fact_id"] not in events and len(events) == limit:
                        break
                    event = Change(**{k: row[k] for k in Change.__dataclass_fields__})
                    previous = events.get(event.fact_id)
                    if previous:
                        event = replace(
                            event,
                            before_revision=previous.before_revision,
                            before_slot_key=previous.before_slot_key,
                        )
                    events[event.fact_id] = event
                through = row["seq"]
            if (events or resync) and sub["debounce"]:
                since = (
                    datetime.fromisoformat(sub["pending_since"]) if sub["pending_since"] else now
                )
                if sub["pending_since"] is None:
                    conn.execute(
                        """UPDATE ao_subscriptions SET pending_since=?
                        WHERE owner=? AND id=?""",
                        (timestamp(since), owner, sid),
                    )
                if now < since + timedelta(seconds=sub["debounce"]):
                    return Batch()
            token = uuid4().hex
            payload = canonical(
                {"events": [asdict(e) for e in events.values()], "resync": resync}, bounded=False
            )
            conn.execute(
                """UPDATE ao_subscriptions SET inflight_token=?,
                inflight_through=?, inflight_json=?, pending_since=NULL
                WHERE owner=? AND id=?""",
                (token, through, payload, owner, sid),
            )
            return self._batch(token, payload)

    def ack(self, owner: str, sid: str, token: str) -> None:
        with self.transaction() as conn:
            try:
                sub = self._subscription(conn, owner, sid)
            except KeyError as exc:
                raise InvalidDelivery("unknown owner/subscription") from exc
            if conn.execute(
                """SELECT 1 FROM ao_delivery_receipts WHERE owner=?
                AND subscription_id=? AND incarnation=? AND token=?""",
                (owner, sid, sub["incarnation"], token),
            ).fetchone():
                return
            if not token or token != sub["inflight_token"]:
                raise InvalidDelivery("token was not issued for this subscription")
            conn.execute(
                "INSERT INTO ao_delivery_receipts VALUES(?,?,?,?,?)",
                (owner, sid, sub["incarnation"], token, sub["inflight_through"]),
            )
            conn.execute(
                """UPDATE ao_subscriptions SET cursor=inflight_through,
                inflight_token=NULL,inflight_through=NULL,inflight_json=NULL
                WHERE owner=? AND id=?""",
                (owner, sid),
            )

    def unsubscribe(self, owner: str, sid: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM ao_subscriptions WHERE owner=? AND id=?", (owner, sid))
            conn.execute(
                "DELETE FROM ao_delivery_receipts WHERE owner=? AND subscription_id=?", (owner, sid)
            )

    def prune(self, namespace: str, through: int, *, force: bool = False) -> None:
        with self.transaction() as conn:
            stream = conn.execute(
                "SELECT * FROM ao_streams WHERE namespace=?", (namespace,)
            ).fetchone()
            minimum = conn.execute(
                "SELECT MIN(cursor) FROM ao_subscriptions WHERE namespace=?", (namespace,)
            ).fetchone()[0]
            if not 0 <= through <= stream["head_seq"]:
                raise ValueError("invalid prune boundary")
            if not force and minimum is not None and through > minimum:
                raise ValueError("prune exceeds acknowledged retention frontier")
            conn.execute(
                "DELETE FROM ao_changes WHERE namespace=? AND seq<=?", (namespace, through)
            )
            conn.execute(
                """UPDATE ao_streams SET min_retained_seq=MAX(min_retained_seq,?)
                WHERE namespace=?""",
                (through + 1, namespace),
            )


class ReferenceChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory.db"
        self.now = datetime(2026, 10, 26, tzinfo=UTC)
        self.store = Store(self.path, lambda: self.now)
        self.store.enable("alice")

    def restart(self) -> Store:
        return Store(self.path, lambda: self.now)

    def test_fact_event_head_rollback_and_callback_rollback(self) -> None:
        self.store.save("alice", [Fact("f", "old")])
        for failure in ("facts", "event"):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.store.save("alice", [Fact("f", "new")], fail_after=failure)
            self.assertEqual(self.restart().load("alice")[0].content, "old")
            self.assertEqual(self.store.head("alice"), 1)

        def broken(_: list[Fact]) -> list[Fact]:
            raise RuntimeError("callback")

        with self.assertRaisesRegex(RuntimeError, "callback"):
            self.store.mutate("alice", broken)
        self.assertEqual(self.store.head("alice"), 1)

    def test_telemetry_suppressed_semantic_and_acl_fields_change_revision(self) -> None:
        original = Fact("f", "old")
        self.store.save("alice", [original])
        telemetry = replace(
            original,
            access_count=7,
            retention_score=0.1,
            importance=0.2,
            tags=("rank",),
            version=99,
            mesi_state="S",
        )
        self.store.save("alice", [telemetry])
        self.assertEqual(self.store.head("alice"), 1)
        self.assertEqual(evidence_revision(original), evidence_revision(telemetry))
        for field, value in (
            ("content", "new"),
            ("visibility_scope", "team"),
            ("memory_tier", "org"),
            ("valid_until", self.now.isoformat()),
        ):
            self.assertNotEqual(
                evidence_revision(original), evidence_revision(replace(original, **{field: value}))
            )
        self.store.save("alice", [])
        with self.store.transaction(write=False) as conn:
            tombstone = conn.execute("SELECT * FROM ao_changes WHERE seq=2").fetchone()
        self.assertEqual(tombstone["fact_id"], "f")
        self.assertEqual(tombstone["before_revision"], evidence_revision(telemetry))
        self.assertIsNone(tombstone["after_revision"])

    def test_baseline_enable_is_persisted_without_historical_invention(self) -> None:
        self.store.save("bob", [Fact("old", "pre-enable")])
        self.assertEqual(self.store.head("bob"), 0)
        self.restart().enable("bob")
        self.restart().enable("bob")
        self.assertEqual(self.store.head("bob"), 1)
        self.store.subscribe("bob", "s", "bob", start="beginning")
        batch = self.store.poll("bob", "s")
        self.assertTrue(batch.resync_required)
        self.assertEqual(batch.events, ())

    def test_poll_replay_ack_receipts_owner_and_reused_id(self) -> None:
        self.store.subscribe("alice", "s", "alice")
        self.store.save("alice", [Fact("f", "one")])
        first = self.store.poll("alice", "s")
        self.assertEqual(self.restart().poll("alice", "s"), first)
        with self.assertRaises(InvalidDelivery):
            self.store.ack("bob", "s", first.delivery_id)
        with self.assertRaises(InvalidDelivery):
            self.store.ack("alice", "s", "future-token")
        self.restart().ack("alice", "s", first.delivery_id)
        self.store.save("alice", [Fact("f", "two")])
        second = self.store.poll("alice", "s")
        self.store.ack("alice", "s", second.delivery_id)
        self.restart().ack("alice", "s", first.delivery_id)
        self.assertEqual(self.store.poll("alice", "s"), Batch())
        self.store.unsubscribe("alice", "s")
        self.store.unsubscribe("alice", "s")
        self.store.subscribe("alice", "s", "alice")
        with self.assertRaises(InvalidDelivery):
            self.store.ack("alice", "s", second.delivery_id)

    def test_scan_bound_empty_delivery_coalescing_and_page_limit(self) -> None:
        self.store.subscribe("alice", "s", "alice", fact_ids=("z",))
        facts = [Fact(f"f{i:04}", "hidden by selector") for i in range(1000)]
        self.store.save("alice", facts + [Fact("z", "visible")])
        empty = self.store.poll("alice", "s")
        self.assertIsNotNone(empty.delivery_id)
        self.assertEqual(empty.events, ())
        self.store.ack("alice", "s", empty.delivery_id)
        self.assertEqual(self.store.poll("alice", "s").events[0].fact_id, "z")
        self.store.subscribe("alice", "all", "alice")
        self.store.save("alice", [Fact("x", "1"), Fact("y", "2")])
        batch = self.store.poll("alice", "all", limit=1)
        self.assertEqual(len(batch.events), 1)
        self.assertEqual(self.restart().poll("alice", "all", limit=500), batch)

    def test_slot_move_matches_old_slot_and_coalesces_first_revision(self) -> None:
        first = Fact("f", "one", slot_key="old")
        self.store.save("alice", [first])
        self.store.subscribe("alice", "s", "alice", slot_keys=("old",))
        self.store.save("alice", [replace(first, content="two")])
        self.store.save("alice", [replace(first, content="three", slot_key="new")])
        batch = self.store.poll("alice", "s")
        self.assertEqual(len(batch.events), 1)
        self.assertEqual(batch.events[0].before_revision, evidence_revision(first))
        self.assertEqual(batch.events[0].before_slot_key, "old")

    def test_debounce_persisted_and_forced_prune_expires_cursor(self) -> None:
        self.store.subscribe("alice", "s", "alice", debounce=30)
        self.store.save("alice", [Fact("f", "one")])
        self.assertEqual(self.store.poll("alice", "s"), Batch())
        self.now += timedelta(seconds=29)
        self.assertEqual(self.restart().poll("alice", "s"), Batch())
        self.now += timedelta(seconds=1)
        self.assertIsNotNone(self.restart().poll("alice", "s").delivery_id)
        with self.assertRaisesRegex(ValueError, "frontier"):
            self.store.prune("alice", 1)
        self.store.prune("alice", 1, force=True)
        with self.assertRaises(CursorExpired):
            self.restart().poll("alice", "s")

    def test_sqlite_uncommitted_head_invisible_lock_failure_consumes_no_seq(self) -> None:
        # WAL lets an independent reader observe the prior committed head.
        with sqlite3.connect(self.path) as setup:
            setup.execute("PRAGMA journal_mode=WAL")
        with self.store.transaction() as first:
            self.store._replace(first, "alice", [Fact("f", "first")])
            self.assertEqual(self.store.head("alice"), 0)
            with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                self.store.save("alice", [Fact("g", "second")])
        self.store.mutate("alice", lambda current: current + [Fact("g", "second")])
        self.assertEqual(self.store.head("alice"), 2)
        self.assertEqual({f.id for f in self.store.load("alice")}, {"f", "g"})

    def test_validation_failures_before_state_change(self) -> None:
        for bad in (0, 501, True, 1.5):
            with self.assertRaises(ValueError):
                self.store.poll("alice", "missing", limit=bad)
        for bad in (-1, 3601, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.store.subscribe("alice", "s", "alice", debounce=bad)
        for fact in (
            Fact("f", "x" * 65536),
            Fact("f", "x", state_confidence=float("nan")),
            Fact("f", "x", valid_from="2026-10-26T00:00:00"),
        ):
            with self.assertRaises(ValueError):
                self.store.save("alice", [fact])
        self.assertEqual(self.store.load("alice"), [])
        self.assertEqual(self.store.head("alice"), 0)
        with self.assertRaises(ValueError):
            canonical({"": "invalid key"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
