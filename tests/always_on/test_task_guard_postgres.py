"""Guard validation must see a writer that committed while its lock was awaited."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager

import pytest

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory, IntentState, StaleEvidence
from ai_knot.storage.postgres_storage import PostgresStorage
from ai_knot.types import Fact


class ObservedConnection(AbstractContextManager):
    def __init__(self, connection, waiting, pids):
        self.connection = connection
        self.waiting = waiting
        self.pids = pids

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, *args):
        return self.connection.__exit__(*args)

    def execute(self, query, *args, **kwargs):
        if "pg_advisory_xact_lock" in query:
            self.pids.append(self.connection.info.backend_pid)
            self.waiting.set()
        return self.connection.execute(query, *args, **kwargs)


@pytest.mark.parametrize("namespace", ["owner", "__shared__"])
def test_default_repeatable_read_does_not_hide_a_preceding_locked_writer(
    sql_storage, clock, monkeypatch, namespace
):
    if not isinstance(sql_storage, PostgresStorage):
        pytest.skip("PostgreSQL advisory lock/isolation regression")
    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    options = conninfo_to_dict(sql_storage._dsn).get("options", "")
    dsn = make_conninfo(
        sql_storage._dsn, options=options + " -cdefault_transaction_isolation=repeatable\\ read"
    )
    storage = PostgresStorage(dsn)
    with storage._get_conn() as conn:
        assert conn.execute("SHOW transaction_isolation").fetchone()[0] == "repeatable read"
    kb = KnowledgeBase("owner", storage=storage, embed_url="")
    source = Fact("Existing selected observation", id="source", qualifiers={"task_key": "rollout"})
    kb.replace_facts([source])
    pool = SharedMemoryPool(storage, persist_stats=True, embed_url="", clock=clock)
    pool.grant_read("owner", "ops")
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    intent = memory.create_intent(client_key="one", goal="Plan")
    lease = memory.claim(intent.id, worker_id="host")
    guard = memory.task_guard(task_key="rollout")
    basis = memory.capture_evidence((("owner", source.id),))
    before = memory.get_intent(intent.id)
    waiting = threading.Event()
    pids = []
    original = storage._get_conn
    main = threading.get_ident()

    def connection():
        conn = original()
        return conn if threading.get_ident() == main else ObservedConnection(conn, waiting, pids)

    monkeypatch.setattr(storage, "_get_conn", connection)

    def checkpoint():
        return memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=before.revision,
            state=IntentState.READY,
            progress={},
            basis=basis,
            context_guard=guard,
        )

    with ThreadPoolExecutor(max_workers=1) as worker:
        with storage.always_on_transaction((namespace,), write=True) as held:
            new = Fact("A new separate condition now applies", id="new", visibility_scope="ops")
            held.replace_facts(namespace, [source, new] if namespace == "owner" else [new])
            pending = worker.submit(checkpoint)
            assert waiting.wait(timeout=5), "Guarded writer did not attempt a namespace lock"
            observed = False
            deadline = time.monotonic() + 5
            with psycopg.connect(sql_storage._dsn, autocommit=True) as observer:
                while time.monotonic() < deadline:
                    row = observer.execute(
                        "SELECT wait_event_type,wait_event FROM pg_stat_activity WHERE pid=%s",
                        (pids[0],),
                    ).fetchone()
                    if row == ("Lock", "advisory"):
                        observed = True
                        break
                    if pending.done():
                        break
                    time.sleep(0.01)
            assert observed, "Guarded writer did not wait on the held source namespace"
        with pytest.raises(StaleEvidence):
            pending.result(timeout=5)
    assert memory.get_intent(intent.id) == before
    assert memory.review_context(basis).complete


def test_repeatable_read_initializer_observes_committed_schema_version_after_lock(
    sql_storage, monkeypatch
):
    if not isinstance(sql_storage, PostgresStorage):
        pytest.skip("PostgreSQL schema lock/isolation regression")
    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    from ai_knot.always_on.storage import namespace_lock_key

    options = conninfo_to_dict(sql_storage._dsn).get("options", "")
    dsn = make_conninfo(
        sql_storage._dsn, options=options + " -cdefault_transaction_isolation=repeatable\\ read"
    )
    main = threading.get_ident()
    waiting = threading.Event()
    pids = []
    original = PostgresStorage._get_conn

    def connection(self):
        conn = original(self)
        return conn if threading.get_ident() == main else ObservedConnection(conn, waiting, pids)

    monkeypatch.setattr(PostgresStorage, "_get_conn", connection)

    with ThreadPoolExecutor(max_workers=1) as worker:
        with psycopg.connect(sql_storage._dsn) as held:
            schema = held.execute("SELECT current_schema()").fetchone()[0]
            held.execute(
                "SELECT pg_advisory_xact_lock(%s)", (namespace_lock_key("__schema__:" + schema),)
            )
            held.execute('UPDATE "ai-knot_ao_schema" SET version=9 WHERE id=1')
            pending = worker.submit(PostgresStorage, dsn)
            assert waiting.wait(timeout=5)
            observed = False
            deadline = time.monotonic() + 5
            with psycopg.connect(sql_storage._dsn, autocommit=True) as observer:
                while time.monotonic() < deadline:
                    row = observer.execute(
                        "SELECT wait_event_type,wait_event FROM pg_stat_activity WHERE pid=%s",
                        (pids[0],),
                    ).fetchone()
                    if row == ("Lock", "advisory"):
                        observed = True
                        break
                    if pending.done():
                        break
                    time.sleep(0.01)
            assert observed, "Initializer did not wait on the migration namespace"
        with pytest.raises(RuntimeError, match="unsupported always-on schema version"):
            pending.result(timeout=5)
