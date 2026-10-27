from __future__ import annotations

import threading
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact


def test_transaction_rolls_back_facts_and_event_head(tmp_path):
    from ai_knot.always_on.storage import SQLStatement, enable_stream

    store = SQLiteStorage(str(tmp_path / "state.db"))
    with store.always_on_transaction(("owner",), write=True) as tx:
        enable_stream(tx, "owner", datetime.now(UTC))
    with pytest.raises(RuntimeError), store.always_on_transaction(("owner",), write=True) as tx:
        tx.replace_facts("owner", [Fact(id="f", content="old")])
        raise RuntimeError("fault")
    assert store.load("owner") == []
    with store.always_on_transaction(("owner",), write=False) as tx:
        assert (
            tx.fetch_one(
                SQLStatement(
                    "SELECT head_seq FROM ao_streams", 'SELECT head_seq FROM "ai-knot_ao_streams"'
                )
            )["head_seq"]
            == 0
        )


def test_atomic_updates_and_telemetry_digest(tmp_path):
    from ai_knot.always_on.changes import evidence_revision

    store = SQLiteStorage(str(tmp_path / "state.db"))
    barrier = threading.Barrier(2)

    def worker(i):
        barrier.wait()
        for j in range(10):
            store.atomic_update(
                "owner", lambda rows, j=j: rows + [Fact(id=f"{i}-{j}", content="hello")]
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(store.load("owner")) == 20
    f = store.load("owner")[0]
    assert evidence_revision(f) == evidence_revision(replace(f, access_count=20))
    assert evidence_revision(f) != evidence_revision(replace(f, content="changed"))


def test_sql_rollback_and_scope_guard(sql_storage, clock):
    from ai_knot.always_on.storage import enable_stream, sql

    with sql_storage.always_on_transaction(("owner",), write=True) as session:
        enable_stream(session, "owner", clock())
    with (
        pytest.raises(RuntimeError),
        sql_storage.always_on_transaction(("owner",), write=True) as session,
    ):
        session.replace_facts("owner", [Fact(id="committed", content="value")])
        assert (
            session.fetch_one(sql("SELECT head_seq FROM ao_streams WHERE namespace=?"), ("owner",))[
                "head_seq"
            ]
            == 1
        )
        raise RuntimeError("after head before commit")
    assert sql_storage.load("owner") == []
    with sql_storage.always_on_transaction(("owner",), write=False) as session:
        assert (
            session.fetch_one(sql("SELECT head_seq FROM ao_streams WHERE namespace=?"), ("owner",))[
                "head_seq"
            ]
            == 0
        )
        with pytest.raises(ValueError, match="namespace"):
            session.load_facts("peer")
        with pytest.raises(ValueError, match="read-only"):
            session.replace_facts("owner", [])


def test_commit_order_does_not_skip_delayed_writer(sql_storage, clock):
    from ai_knot.always_on.storage import enable_stream, sql

    with sql_storage.always_on_transaction(("owner",), write=True) as session:
        enable_stream(session, "owner", clock())
    entered = threading.Event()
    release = threading.Event()
    second_attempted = threading.Event()
    failures = []

    def first():
        try:
            with sql_storage.always_on_transaction(("owner",), write=True) as tx:
                tx.replace_facts("owner", [Fact(id="one", content="first")])
                entered.set()
                assert release.wait(10)
        except BaseException as exc:
            failures.append(exc)

    def second():
        try:
            assert entered.wait(10)
            second_attempted.set()
            sql_storage.atomic_update(
                "owner", lambda facts: facts + [Fact(id="two", content="second")]
            )
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    assert entered.wait(10) and second_attempted.wait(10)
    try:
        with sql_storage.always_on_transaction(("owner",), write=False) as session:
            assert (
                session.fetch_one(
                    sql("SELECT head_seq FROM ao_streams WHERE namespace=?"), ("owner",)
                )["head_seq"]
                == 0
            )
    finally:
        release.set()
    for thread in threads:
        thread.join(10)
    assert not any(thread.is_alive() for thread in threads)
    assert failures == []
    assert {fact.id for fact in sql_storage.load("owner")} == {"one", "two"}
    with sql_storage.always_on_transaction(("owner",), write=False) as session:
        assert [
            row["seq"]
            for row in session.fetch_all(
                sql("SELECT seq FROM ao_changes WHERE namespace=? ORDER BY seq"), ("owner",)
            )
        ] == [1, 2]


def test_digest_rejects_duplicate_ids_and_nonfinite(sql_storage, clock):
    from ai_knot.always_on.storage import enable_stream

    with sql_storage.always_on_transaction(("owner",), write=True) as session:
        enable_stream(session, "owner", clock())
    f = Fact(id="one", content="safe")
    sql_storage.save("owner", [f])
    with pytest.raises(ValueError, match="duplicate"):
        sql_storage.save("owner", [f, f])
    with pytest.raises(ValueError):
        sql_storage.save("owner", [replace(f, state_confidence=float("nan"))])
    assert sql_storage.load("owner")[0].content == "safe"


def test_postgres_concurrent_empty_schema_initialization(sql_storage):
    """Concurrent constructors must serialize CREATE TABLE and all migrations."""
    from ai_knot.storage.postgres_storage import PostgresStorage

    if not isinstance(sql_storage, PostgresStorage):
        pytest.skip("PostgreSQL schema initialization regression")
    import uuid

    import psycopg
    from psycopg.conninfo import make_conninfo
    from psycopg.sql import SQL, Identifier

    schema = "ao_init_" + uuid.uuid4().hex
    with psycopg.connect(sql_storage._dsn, autocommit=True) as conn:
        conn.execute(SQL("CREATE SCHEMA {}").format(Identifier(schema)))
    barrier = threading.Barrier(4)
    failures = []
    dsn = make_conninfo(sql_storage._dsn, options="-csearch_path=" + schema)

    def construct():
        try:
            barrier.wait()
            PostgresStorage(dsn)
        except BaseException as exc:
            failures.append(exc)

    try:
        threads = [threading.Thread(target=construct) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
        assert not any(thread.is_alive() for thread in threads)
        assert failures == []
        assert PostgresStorage(dsn).load("owner") == []
    finally:
        with psycopg.connect(sql_storage._dsn, autocommit=True) as conn:
            conn.execute(SQL("DROP SCHEMA {} CASCADE").format(Identifier(schema)))


def test_snapshot_roundtrip_preserves_ao_history(sql_storage, clock):
    from ai_knot.always_on import AlwaysOnMemory
    from ai_knot.always_on.subscriptions import WatchSelector
    from ai_knot.knowledge import KnowledgeBase

    f = Fact(
        id="one",
        content="before",
        claim_key="claim",
        memory_tier="org",
        valid_from=clock(),
        created_at=clock(),
        last_accessed=clock(),
    )
    sql_storage.save("owner", [f])
    kb = KnowledgeBase("owner", storage=sql_storage)
    ao = AlwaysOnMemory(kb, clock=clock)
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    intent = ao.create_intent(client_key="work", goal="durable")
    kb.snapshot("backup")
    sql_storage.save("owner", [replace(f, content="after")])
    kb.restore("backup")
    restored = sql_storage.load("owner")[0]
    assert (restored.content, restored.claim_key, restored.memory_tier) == (
        "before",
        "claim",
        "org",
    )
    assert ao.get_intent(intent.id).id == intent.id
    batch = ao.poll("watch")
    assert batch.events[0].kind == "update"
