"""Backing-read budgets must preserve SQL snapshot and source admission behavior."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from tests.eval.context_read_profile import measured, seeded


@pytest.mark.parametrize("operation", ["snapshot", "lexical"])
def test_shared_root_checks_reuse_scanned_rows_within_one_read(sql_storage, operation):
    memory = seeded(sql_storage)

    def call():
        if operation == "snapshot":
            return memory.task_snapshot(task_key="deploy", max_facts=128, max_chars=65536)
        return memory.select_context(
            ("deployment recovery",), task_key="deploy", max_facts=128, max_chars=65536
        )

    result, counts = measured(sql_storage, call)
    facts = result if isinstance(result, tuple) else result.facts
    assert len(facts) == 64
    assert {f.ref.fact_id for f in facts} == {f"lesson-{i:03d}" for i in range(64)}
    assert counts["total_reads"] < 200


def test_review_reuses_shared_dependencies_without_a_whole_namespace_scan(sql_storage, monkeypatch):
    memory = seeded(sql_storage)
    refs = tuple(
        f.ref for f in memory.task_snapshot(task_key="deploy", max_facts=128, max_chars=65536)
    )
    loader = type(sql_storage)._load_on_connection

    def targeted(self, conn, namespace, *, fact_id=None, slot_key=None):
        assert fact_id is not None or slot_key is not None
        return loader(self, conn, namespace, fact_id=fact_id, slot_key=slot_key)

    monkeypatch.setattr(type(sql_storage), "_load_on_connection", targeted)
    result, counts = measured(sql_storage, lambda: memory.review_context(refs, max_chars=65536))
    assert result.complete and len(result.accepted) == 64
    assert counts["total_reads"] < 300


@pytest.mark.parametrize("change", ["source", "acl"])
def test_concurrent_commit_keeps_one_snapshot_then_next_read_rechecks(
    sql_storage, monkeypatch, change
):
    memory = seeded(sql_storage, lessons=2, roots=1)
    before = memory.task_snapshot(task_key="deploy")
    loader = type(sql_storage)._load_on_connection
    injected = False

    def write():
        if change == "acl":
            memory._pool.revoke_read("reader", "ops")
        else:
            sql_storage.atomic_update(
                "__shared__", lambda facts: [replace(f, content="Changed source") for f in facts]
            )

    def during(self, conn, namespace, *, fact_id=None, slot_key=None):
        nonlocal injected
        rows = loader(self, conn, namespace, fact_id=fact_id, slot_key=slot_key)
        if not injected and namespace == "reader" and fact_id is None and slot_key is None:
            injected = True
            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(write).result(timeout=5)
        return rows

    monkeypatch.setattr(type(sql_storage), "_load_on_connection", during)
    assert memory.task_snapshot(task_key="deploy") == before
    assert injected
    assert memory.task_snapshot(task_key="deploy") == ()
    assert not memory.review_context(tuple(f.ref for f in before)).complete


def test_reused_filters_keep_raw_sql_order_namespaces_and_read_only_rows(sql_storage, clock):
    from dataclasses import asdict

    from ai_knot.always_on.reads import ReadCache
    from ai_knot.always_on.storage import sql
    from ai_knot.types import Fact

    own = [
        Fact("first", id="same", slot_key="target", created_at=clock()),
        Fact("second", id="second", slot_key="target", created_at=clock()),
        Fact("unslotted", id="plain", created_at=clock()),
    ]
    sql_storage.save("reader", own)
    sql_storage.save("__shared__", [replace(own[0], content="other namespace")])
    sql_storage.save_grant("reader", "ops", granted_at=clock().isoformat())
    with sql_storage.always_on_transaction(("reader", "__shared__"), write=False) as session:
        for preload in (True, False):
            reused = ReadCache(session)
            if preload:
                reused.load_facts("reader")
            for namespace in ("reader", "__shared__"):
                for fact_id in ("missing", "same", "plain", "", None):
                    for slot_key in ("target", "missing", "", None):
                        fields = {"fact_id": fact_id, "slot_key": slot_key}
                        expected = [asdict(f) for f in session.load_facts(namespace, **fields)]
                        actual = reused.load_facts(namespace, **fields)
                        assert [asdict(f) for f in actual] == expected
                        actual.clear()
                        assert [
                            asdict(f) for f in reused.load_facts(namespace, **fields)
                        ] == expected
            query = sql("SELECT scope FROM acl_grants WHERE agent_id=?")
            row = reused.fetch_one(query, ("reader",))
            assert row == {"scope": "ops"}
            row["scope"] = "mutated local result"
            assert reused.fetch_all(query, ("reader",)) == [{"scope": "ops"}]
            with pytest.raises(ValueError, match="namespace"):
                reused.load_facts("not-declared")
            with pytest.raises(ValueError, match="read-only"):
                reused.replace_facts("reader", [])
            with pytest.raises(ValueError, match="read-only"):
                reused.execute(query, ("reader",))
