"""Tests for SQLite storage backend."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact, MemoryType


class TestSQLiteSaveLoad:
    """Basic save/load round-trip."""

    def test_save_and_load(self, sqlite_storage: SQLiteStorage, sample_facts: list[Fact]) -> None:
        sqlite_storage.save("agent1", sample_facts)
        loaded = sqlite_storage.load("agent1")
        assert len(loaded) == len(sample_facts)
        assert loaded[0].content == sample_facts[0].content

    def test_load_nonexistent_agent(self, sqlite_storage: SQLiteStorage) -> None:
        loaded = sqlite_storage.load("nonexistent")
        assert loaded == []

    def test_overwrite_replaces(self, sqlite_storage: SQLiteStorage) -> None:
        facts_v1 = [Fact(content="version 1")]
        facts_v2 = [Fact(content="version 2"), Fact(content="version 2b")]

        sqlite_storage.save("agent1", facts_v1)
        sqlite_storage.save("agent1", facts_v2)

        loaded = sqlite_storage.load("agent1")
        assert len(loaded) == 2

    def test_preserves_all_fields(self, sqlite_storage: SQLiteStorage) -> None:
        fact = Fact(
            content="Full field test",
            type=MemoryType.EPISODIC,
            importance=0.33,
            retention_score=0.66,
            access_count=7,
            tags=["x", "y"],
        )
        sqlite_storage.save("agent1", [fact])
        loaded = sqlite_storage.load("agent1")[0]

        assert loaded.content == fact.content
        assert loaded.type == fact.type
        assert loaded.importance == pytest.approx(fact.importance)
        assert loaded.retention_score == pytest.approx(fact.retention_score)
        assert loaded.access_count == fact.access_count
        assert loaded.tags == fact.tags
        assert loaded.id == fact.id


class TestSQLiteMultiAgent:
    """Multiple agents in the same database."""

    def test_agents_isolated(self, sqlite_storage: SQLiteStorage) -> None:
        sqlite_storage.save("alice", [Fact(content="Alice fact")])
        sqlite_storage.save("bob", [Fact(content="Bob fact")])

        alice_facts = sqlite_storage.load("alice")
        bob_facts = sqlite_storage.load("bob")

        assert len(alice_facts) == 1
        assert len(bob_facts) == 1
        assert alice_facts[0].content == "Alice fact"
        assert bob_facts[0].content == "Bob fact"

    def test_list_agents(self, sqlite_storage: SQLiteStorage) -> None:
        sqlite_storage.save("agent_a", [Fact(content="a")])
        sqlite_storage.save("agent_b", [Fact(content="b")])

        agents = sqlite_storage.list_agents()
        assert set(agents) == {"agent_a", "agent_b"}


class TestSQLiteDelete:
    """Deleting individual facts."""

    def test_delete_fact(self, sqlite_storage: SQLiteStorage) -> None:
        facts = [Fact(content="keep"), Fact(content="delete me")]
        sqlite_storage.save("agent1", facts)

        sqlite_storage.delete("agent1", facts[1].id)

        loaded = sqlite_storage.load("agent1")
        assert len(loaded) == 1
        assert loaded[0].content == "keep"

    def test_delete_nonexistent_fact(self, sqlite_storage: SQLiteStorage) -> None:
        sqlite_storage.save("agent1", [Fact(content="only")])
        sqlite_storage.delete("agent1", "nonexistent_id")
        loaded = sqlite_storage.load("agent1")
        assert len(loaded) == 1


class TestLoadActiveValidFrom:
    """load_active() must respect valid_from (P1 fix)."""

    def test_load_active_excludes_future_facts(self, sqlite_storage: SQLiteStorage) -> None:
        from datetime import UTC, datetime, timedelta

        future_fact = Fact(content="not yet active")
        future_fact.valid_from = datetime.now(UTC) + timedelta(hours=1)
        sqlite_storage.save("agent1", [future_fact])

        active = sqlite_storage.load_active("agent1")
        assert all(f.id != future_fact.id for f in active)

    def test_load_active_includes_past_valid_from(self, sqlite_storage: SQLiteStorage) -> None:
        from datetime import UTC, datetime, timedelta

        past_fact = Fact(content="already active")
        past_fact.valid_from = datetime.now(UTC) - timedelta(hours=1)
        sqlite_storage.save("agent1", [past_fact])

        active = sqlite_storage.load_active("agent1")
        assert any(f.id == past_fact.id for f in active)


class TestSaveAtomic:
    """save_atomic() correctness and concurrency."""

    def test_save_atomic_round_trip(self, sqlite_storage: SQLiteStorage) -> None:
        facts = [Fact(content="atomic fact A"), Fact(content="atomic fact B")]
        sqlite_storage.save_atomic("agent1", facts)
        loaded = sqlite_storage.load("agent1")
        contents = {f.content for f in loaded}
        assert contents == {"atomic fact A", "atomic fact B"}

    def test_save_atomic_replaces_all(self, sqlite_storage: SQLiteStorage) -> None:
        sqlite_storage.save_atomic("agent1", [Fact(content="v1")])
        sqlite_storage.save_atomic("agent1", [Fact(content="v2a"), Fact(content="v2b")])
        loaded = sqlite_storage.load("agent1")
        assert len(loaded) == 2
        assert all(f.content.startswith("v2") for f in loaded)

    def test_save_atomic_concurrent(self, tmp_dir: pathlib.Path) -> None:
        import threading

        db_path = str(tmp_dir / "concurrent.db")
        storage = SQLiteStorage(db_path=db_path)
        storage.save("agent1", [])
        storage.save("agent2", [])

        errors: list[Exception] = []

        def write(agent_id: str, content: str) -> None:
            try:
                storage.save_atomic(agent_id, [Fact(content=content)])
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=write, args=("agent1", "from thread 1"))
        t2 = threading.Thread(target=write, args=("agent2", "from thread 2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        assert len(storage.load("agent1")) == 1
        assert len(storage.load("agent2")) == 1


class TestConnectionManagement:
    """_conn() context manager closes connections and emits no ResourceWarning."""

    def test_no_resource_warnings_on_repeated_ops(self, tmp_dir: pathlib.Path) -> None:
        import warnings

        db_path = str(tmp_dir / "leaks.db")
        storage = SQLiteStorage(db_path=db_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i in range(50):
                fact = Fact(content=f"fact {i}")
                storage.save("agent1", [fact])
                storage.load("agent1")
                storage.delete("agent1", fact.id)

        resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert resource_warnings == []


class TestSQLitePersistence:
    """Data survives across SQLiteStorage instances."""

    def test_data_persists_after_reopen(self, tmp_dir: pathlib.Path) -> None:
        db_path = str(tmp_dir / "persist.db")

        # Write with one instance
        storage1 = SQLiteStorage(db_path=db_path)
        storage1.save("agent1", [Fact(content="persistent")])

        # Read with a fresh instance
        storage2 = SQLiteStorage(db_path=db_path)
        loaded = storage2.load("agent1")

        assert len(loaded) == 1
        assert loaded[0].content == "persistent"
