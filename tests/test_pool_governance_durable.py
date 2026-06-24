"""SharedMemoryPool wiring of durable ACL grants + the trust-event audit ledger.

Verifies the pool persists read-scope grants and appends trust events when
``persist_stats=True`` and the backend supports the new protocols, and that the
default (off) path keeps the prior in-memory behaviour.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

from ai_knot.knowledge import KnowledgeBase
from ai_knot.pool import SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage

_FIXED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


# ---- durable ACL ------------------------------------------------------------


def test_grants_persist_across_restart(tmp_path: pathlib.Path) -> None:
    db = str(tmp_path / "pool.db")
    pool = SharedMemoryPool(SQLiteStorage(db), persist_stats=True)
    pool.grant_read("alice", "team:infra")
    pool.grant_read("alice", "team:sec")
    pool.grant_read("bob", "team:infra")

    # A fresh pool on the same store restores the grants.
    pool2 = SharedMemoryPool(SQLiteStorage(db), persist_stats=True)
    assert pool2.read_scopes == {"alice": {"team:infra", "team:sec"}, "bob": {"team:infra"}}


def test_grants_in_memory_when_persist_off(tmp_path: pathlib.Path) -> None:
    db = str(tmp_path / "pool.db")
    pool = SharedMemoryPool(SQLiteStorage(db), persist_stats=False)
    pool.grant_read("alice", "team:infra")
    # Without persistence, a fresh pool starts empty (prior behaviour preserved).
    pool2 = SharedMemoryPool(SQLiteStorage(db), persist_stats=False)
    assert pool2.read_scopes == {}


# ---- trust-event ledger -----------------------------------------------------


def test_publish_appends_trust_event(tmp_path: pathlib.Path) -> None:
    store = SQLiteStorage(str(tmp_path / "pool.db"))
    pool = SharedMemoryPool(store, persist_stats=True, clock=lambda: _FIXED)
    pool.register("agent_a")
    kb = KnowledgeBase("agent_a", storage=store)
    f1 = kb.add("Alex earns 95k")
    f2 = kb.add("Bob lives in Berlin")

    pool.publish("agent_a", [f1.id, f2.id], kb=kb)

    publishes = [e for e in store.load_trust_events("agent_a") if e["event_type"] == "publish"]
    assert publishes, "publish should append a trust event"
    assert publishes[-1]["delta"] == 2.0
    assert publishes[-1]["ts"] == "2026-01-02T03:04:05+00:00"  # injected clock


def test_no_ledger_when_persist_off(tmp_path: pathlib.Path) -> None:
    store = SQLiteStorage(str(tmp_path / "pool.db"))
    pool = SharedMemoryPool(store, persist_stats=False)
    pool.register("agent_a")
    kb = KnowledgeBase("agent_a", storage=store)
    f1 = kb.add("Alex earns 95k")

    pool.publish("agent_a", [f1.id], kb=kb)

    assert store.load_trust_events() == []
