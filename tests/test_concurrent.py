"""Thread-safety tests for YAML and SQLite storage backends."""

from __future__ import annotations

import threading
from pathlib import Path

from agentmemo.storage.sqlite_storage import SQLiteStorage
from agentmemo.storage.yaml_storage import YAMLStorage


def _add_facts(storage: YAMLStorage | SQLiteStorage, agent_id: str, n: int) -> None:
    """Helper: load, append n facts with incrementing content, save."""
    from agentmemo.types import Fact, MemoryType

    for i in range(n):
        facts = storage.load(agent_id)
        facts.append(
            Fact(
                content=f"fact-{threading.get_ident()}-{i}",
                type=MemoryType.SEMANTIC,
                importance=0.5,
            )
        )
        storage.save(agent_id, facts)


class TestYAMLConcurrent:
    """YAML storage: concurrent reads and writes must not corrupt data."""

    def test_concurrent_writes_no_corruption(self, tmp_path: Path) -> None:
        storage = YAMLStorage(base_dir=str(tmp_path))
        threads = [
            threading.Thread(target=_add_facts, args=(storage, "agent1", 5)) for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        facts = storage.load("agent1")
        # Each thread did 5 saves of the whole list; final state must be valid YAML.
        assert isinstance(facts, list)
        for fact in facts:
            assert fact.content  # no empty / corrupted entries

    def test_concurrent_reads_never_raise(self, tmp_path: Path) -> None:
        storage = YAMLStorage(base_dir=str(tmp_path))
        from agentmemo.types import Fact, MemoryType

        storage.save(
            "agent2",
            [Fact(content="initial", type=MemoryType.SEMANTIC, importance=0.9)],
        )

        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(10):
                    storage.load("agent2")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Reader threads raised: {errors}"

    def test_write_read_consistency(self, tmp_path: Path) -> None:
        """A value saved by one thread must be readable by another."""
        storage = YAMLStorage(base_dir=str(tmp_path))
        from agentmemo.types import Fact, MemoryType

        sentinel = "unique-sentinel-value"
        written = threading.Event()

        def writer() -> None:
            storage.save(
                "agent3",
                [Fact(content=sentinel, type=MemoryType.EPISODIC, importance=1.0)],
            )
            written.set()

        w = threading.Thread(target=writer)
        w.start()
        written.wait(timeout=5.0)
        w.join()

        facts = storage.load("agent3")
        assert any(f.content == sentinel for f in facts)


class TestSQLiteConcurrent:
    """SQLite storage: WAL mode allows concurrent access without locking errors."""

    def test_concurrent_saves_no_exception(self, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")
        storage = SQLiteStorage(db_path=db)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                _add_facts(storage, "agent1", 5)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Worker threads raised: {errors}"
        facts = storage.load("agent1")
        assert isinstance(facts, list)

    def test_concurrent_reads_never_raise(self, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")
        storage = SQLiteStorage(db_path=db)
        from agentmemo.types import Fact, MemoryType

        storage.save(
            "agent2",
            [Fact(content="hello", type=MemoryType.PROCEDURAL, importance=0.7)],
        )

        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(10):
                    storage.load("agent2")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Reader threads raised: {errors}"
