"""Pool telemetry must merge across independently constructed hosts."""

from concurrent.futures import ThreadPoolExecutor

from ai_knot.knowledge import KnowledgeBase
from ai_knot.pool import SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage


def independent(store):
    if isinstance(store, SQLiteStorage):
        return SQLiteStorage(str(store._db_path))
    return type(store)(store._dsn)


def test_stale_pools_publish_recall_restart(sql_storage):
    first = SharedMemoryPool(sql_storage, persist_stats=True)
    second = SharedMemoryPool(independent(sql_storage), persist_stats=True)
    first.register("writer")
    second.register("writer")
    kb = KnowledgeBase("writer", storage=sql_storage)
    one = kb.add("Python is the primary programming language")
    two = kb.add("Docker is the deployment platform")
    first.publish("writer", [one.id], kb=kb)
    second.publish("writer", [two.id], kb=kb)
    assert first.recall("Python primary programming language", "reader-a")
    assert second.recall("Docker deployment platform", "reader-b")
    first.flush_stats()
    second.flush_stats()
    before = sql_storage.load_pool_stats()
    first.flush_stats()
    second.flush_stats()
    assert sql_storage.load_pool_stats() == before
    restored = SharedMemoryPool(independent(sql_storage), persist_stats=True)
    assert restored._publish_count == {"writer": 2}
    assert restored._used_count["writer"] >= 2
    assert {"reader-a", "reader-b"} <= set().union(*restored._fact_consumers.values())
    first.recall("Python language", "reader-c")
    assert first._publish_count == restored._publish_count


def test_atomic_deltas_concurrent_and_legacy_snapshot(sql_storage):
    sql_storage.save_pool_stats({"publish_count": {"legacy": 9}, "fact_consumers": {"f": ["old"]}})
    stores = [independent(sql_storage) for _ in range(4)]

    def write(store):
        for _ in range(10):
            store.merge_pool_stats(
                {"publish_count": {"legacy": 1}, "fact_consumers": {"f": ["new"]}}
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, stores))
    result = sql_storage.load_pool_stats()
    assert result["publish_count"] == {"legacy": 49}
    assert result["fact_consumers"]["f"] == ["new", "old"]


def test_recall_refreshes_peer_trust_without_replaying_local_usage(sql_storage):
    first = SharedMemoryPool(sql_storage, persist_stats=True)
    second = SharedMemoryPool(independent(sql_storage), persist_stats=True)
    first.register("writer")
    kb = KnowledgeBase("writer", storage=sql_storage)
    fact = kb.add("Python is the primary language")
    first.publish("writer", [fact.id], kb=kb)
    for _ in range(3):
        assert first.recall("Python primary language", "reader", top_k=1)
    first.flush_stats()
    assert second.recall("Python primary language", "writer", top_k=1)
    assert second._used_count == {"writer": 3}
    assert second.get_trust("writer") == first.get_trust("writer")
    second.flush_stats()
    first.flush_stats()
    assert sql_storage.load_pool_stats()["used_count"] == {"writer": 3}


def test_publish_rolls_back_facts_stats_and_events(sql_storage, monkeypatch):
    import pytest

    import ai_knot.pool as pool_module

    pool = SharedMemoryPool(sql_storage, persist_stats=True)
    pool.register("writer")
    kb = KnowledgeBase("writer", storage=sql_storage)
    fact = kb.add("Python language")
    original = pool_module.merge_stats

    def fail(*args):
        raise RuntimeError("interrupted before stats commit")

    monkeypatch.setattr(pool_module, "merge_stats", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        pool.publish("writer", [fact.id], kb=kb)
    assert sql_storage.load("__shared__") == []
    assert sql_storage.load_pool_stats() == {}
    assert sql_storage.load_trust_events() == []
    assert pool._publish_count == {}
    monkeypatch.setattr(pool_module, "merge_stats", original)
    assert pool.publish("writer", [fact.id], kb=kb)
    assert sql_storage.load_pool_stats()["publish_count"] == {"writer": 1}
    assert len(sql_storage.load_trust_events()) == 1


def test_separate_processes_merge_legacy_stats(sql_storage):
    import os
    import subprocess
    import sys

    sqlite = isinstance(sql_storage, SQLiteStorage)
    location = str(sql_storage._db_path) if sqlite else sql_storage._dsn
    script = """
import sys
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.postgres_storage import PostgresStorage
store = (SQLiteStorage if sys.argv[1] == 'sqlite' else PostgresStorage)(sys.argv[2])
for _ in range(8):
    store.merge_pool_stats({'publish_count': {'writer': 1}, 'fact_consumers': {'f': [sys.argv[3]]}})
"""
    sql_storage.save_pool_stats({"publish_count": {"writer": 5}, "fact_consumers": {"f": ["old"]}})
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                "sqlite" if sqlite else "postgres",
                location,
                f"reader-{index}",
            ],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(3)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
    assert sql_storage.load_pool_stats()["publish_count"] == {"writer": 29}
    assert sql_storage.load_pool_stats()["fact_consumers"] == {
        "f": ["old", "reader-0", "reader-1", "reader-2"]
    }


def test_concurrent_pools_publish_preserves_all_evidence_and_stats(sql_storage):
    pools = [SharedMemoryPool(independent(sql_storage), persist_stats=True) for _ in range(4)]
    kb = KnowledgeBase("writer", storage=sql_storage)
    facts = [kb.add(f"Independent deployment detail number {index}") for index in range(4)]
    for pool in pools:
        pool.register("writer")

    def publish(item):
        pool, fact = item
        return pool.publish("writer", [fact.id], kb=kb)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(publish, zip(pools, facts, strict=True)))
    assert all(len(result) == 1 for result in results)
    assert {fact.id for fact in sql_storage.load("__shared__")} == {fact.id for fact in facts}
    assert sql_storage.load_pool_stats()["publish_count"] == {"writer": 4}
    assert len(sql_storage.load_trust_events("writer")) == 4
    for pool in pools:
        pool.flush_stats()
        pool.flush_stats()
    assert sql_storage.load_pool_stats()["publish_count"] == {"writer": 4}
