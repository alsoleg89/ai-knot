"""Durable ACL grants + append-only audit ledger: storage round-trip + parity.

``ACLStoreCapable`` persists per-agent read-scope grants (previously in-memory only
on ``SharedMemoryPool``) and ``EventLedgerCapable`` records the trust-change and
fact-usage event stream (previously only an aggregate snapshot existed). These
tests lock both protocols across SQLite and YAML, and parity between them. The
Postgres variants are skipped unless ``AI_KNOT_TEST_PG_DSN`` is set (CI has no
Postgres), matching the rest of the suite.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from ai_knot.storage.base import ACLStoreCapable, EventLedgerCapable
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture(params=["sqlite", "yaml"])
def store(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> object:
    if request.param == "sqlite":
        return SQLiteStorage(str(tmp_path / "t.db"))
    return YAMLStorage(base_dir=str(tmp_path / "yaml"))


# ---- protocol conformance ---------------------------------------------------


def test_backends_implement_protocols(store: object) -> None:
    assert isinstance(store, ACLStoreCapable)
    assert isinstance(store, EventLedgerCapable)


# ---- ACL grants -------------------------------------------------------------


def test_grant_round_trip(store: ACLStoreCapable) -> None:
    store.save_grant("alice", "team:infra", granted_at="2026-01-01T00:00:00", granted_by="admin")
    assert store.load_grants() == {"alice": {"team:infra"}}


def test_grant_multiple_scopes_per_agent(store: ACLStoreCapable) -> None:
    store.save_grant("alice", "team:infra", granted_at="t1")
    store.save_grant("alice", "team:sec", granted_at="t2")
    assert store.load_grants() == {"alice": {"team:infra", "team:sec"}}


def test_grant_upsert_does_not_duplicate(store: ACLStoreCapable) -> None:
    store.save_grant("alice", "team:infra", granted_at="t1", granted_by="a")
    store.save_grant("alice", "team:infra", granted_at="t2", granted_by="b")
    assert store.load_grants() == {"alice": {"team:infra"}}


def test_grant_isolation_between_agents(store: ACLStoreCapable) -> None:
    store.save_grant("alice", "team:infra", granted_at="t1")
    store.save_grant("bob", "team:sec", granted_at="t2")
    assert store.load_grants() == {"alice": {"team:infra"}, "bob": {"team:sec"}}


def test_revoke_grant(store: ACLStoreCapable) -> None:
    store.save_grant("alice", "team:infra", granted_at="t1")
    store.save_grant("alice", "team:sec", granted_at="t2")
    store.revoke_grant("alice", "team:infra")
    assert store.load_grants() == {"alice": {"team:sec"}}


def test_revoke_last_scope_drops_agent(store: ACLStoreCapable) -> None:
    store.save_grant("alice", "team:infra", granted_at="t1")
    store.revoke_grant("alice", "team:infra")
    assert store.load_grants() == {}


def test_revoke_missing_grant_is_noop(store: ACLStoreCapable) -> None:
    store.revoke_grant("nobody", "team:none")  # must not raise
    assert store.load_grants() == {}


def test_empty_grants(store: ACLStoreCapable) -> None:
    assert store.load_grants() == {}


# ---- audit ledger -----------------------------------------------------------


def test_trust_events_order_and_payload(store: EventLedgerCapable) -> None:
    store.append_trust_event(ts="t1", agent_id="alice", event_type="publish", delta=0.0)
    store.append_trust_event(
        ts="t2", agent_id="bob", event_type="quick_inv", delta=-0.5, reason="stale"
    )
    store.append_trust_event(ts="t3", agent_id="alice", event_type="use", delta=0.1)
    events = store.load_trust_events()
    assert [e["ts"] for e in events] == ["t1", "t2", "t3"]  # insertion order
    assert events[1]["agent_id"] == "bob"
    assert events[1]["event_type"] == "quick_inv"
    assert events[1]["delta"] == -0.5
    assert events[1]["reason"] == "stale"


def test_trust_events_filter_by_agent(store: EventLedgerCapable) -> None:
    store.append_trust_event(ts="t1", agent_id="alice", event_type="publish", delta=0.0)
    store.append_trust_event(ts="t2", agent_id="bob", event_type="publish", delta=0.0)
    store.append_trust_event(ts="t3", agent_id="alice", event_type="use", delta=0.1)
    alice = store.load_trust_events("alice")
    assert [e["ts"] for e in alice] == ["t1", "t3"]


def test_usage_events_filter_by_fact(store: EventLedgerCapable) -> None:
    store.append_usage_event(ts="t1", fact_id="f1", agent_id="alice", recall_session="s1")
    store.append_usage_event(ts="t2", fact_id="f2", agent_id="alice")
    store.append_usage_event(ts="t3", fact_id="f1", agent_id="bob")
    f1 = store.load_usage_events("f1")
    assert [e["ts"] for e in f1] == ["t1", "t3"]
    assert f1[0]["recall_session"] == "s1"


def test_empty_ledgers(store: EventLedgerCapable) -> None:
    assert store.load_trust_events() == []
    assert store.load_usage_events() == []


# ---- cross-backend parity ---------------------------------------------------


def test_sqlite_yaml_parity(tmp_path: pathlib.Path) -> None:
    sq = SQLiteStorage(str(tmp_path / "t.db"))
    ya = YAMLStorage(base_dir=str(tmp_path / "yaml"))
    for s in (sq, ya):
        s.save_grant("alice", "team:infra", granted_at="t1", granted_by="admin")
        s.save_grant("bob", "team:sec", granted_at="t2")
        s.append_trust_event(ts="t1", agent_id="alice", event_type="publish", delta=0.0)
        s.append_usage_event(ts="t2", fact_id="f1", agent_id="bob")

    assert sq.load_grants() == ya.load_grants()

    def _strip_seq(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{k: v for k, v in r.items() if k != "seq"} for r in rows]

    assert _strip_seq(sq.load_trust_events()) == _strip_seq(ya.load_trust_events())
    assert _strip_seq(sq.load_usage_events()) == _strip_seq(ya.load_usage_events())


# ---- Postgres (skipped without a live DB) -----------------------------------

_PG_DSN = os.environ.get("AI_KNOT_TEST_PG_DSN", "")


@pytest.mark.skipif(not _PG_DSN, reason="AI_KNOT_TEST_PG_DSN not set (no live Postgres)")
def test_postgres_acl_and_ledger() -> None:
    pytest.importorskip("psycopg")
    from ai_knot.storage.postgres_storage import PostgresStorage

    store = PostgresStorage(dsn=_PG_DSN)
    try:
        store.save_grant("pg_alice", "team:infra", granted_at="t1", granted_by="admin")
        assert store.load_grants().get("pg_alice") == {"team:infra"}
        store.append_trust_event(ts="t1", agent_id="pg_alice", event_type="publish", delta=0.2)
        events = store.load_trust_events("pg_alice")
        assert events and events[-1]["delta"] == 0.2
    finally:
        store.revoke_grant("pg_alice", "team:infra")
