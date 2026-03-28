"""Tests ensuring YAML and SQLite backends produce identical results."""

from __future__ import annotations

import pathlib

import pytest

from agentmemo.storage.sqlite_storage import SQLiteStorage
from agentmemo.storage.yaml_storage import YAMLStorage
from agentmemo.types import Fact


@pytest.fixture
def both_storages(tmp_path: pathlib.Path) -> tuple[YAMLStorage, SQLiteStorage]:
    yaml = YAMLStorage(base_dir=str(tmp_path / "yaml"))
    sqlite = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    return yaml, sqlite


class TestStorageCompatibility:
    """Both backends must behave identically."""

    def test_save_load_identical(
        self,
        both_storages: tuple[YAMLStorage, SQLiteStorage],
        sample_facts: list[Fact],
    ) -> None:
        yaml_store, sqlite_store = both_storages

        yaml_store.save("agent1", sample_facts)
        sqlite_store.save("agent1", sample_facts)

        yaml_facts = yaml_store.load("agent1")
        sqlite_facts = sqlite_store.load("agent1")

        assert len(yaml_facts) == len(sqlite_facts)
        for yf, sf in zip(yaml_facts, sqlite_facts, strict=True):
            assert yf.content == sf.content
            assert yf.type == sf.type
            assert yf.importance == pytest.approx(sf.importance)
            assert yf.id == sf.id

    def test_empty_load_identical(
        self, both_storages: tuple[YAMLStorage, SQLiteStorage]
    ) -> None:
        yaml_store, sqlite_store = both_storages
        assert yaml_store.load("ghost") == sqlite_store.load("ghost") == []

    def test_delete_identical(
        self, both_storages: tuple[YAMLStorage, SQLiteStorage]
    ) -> None:
        yaml_store, sqlite_store = both_storages
        facts = [Fact(content="keep"), Fact(content="remove")]

        for store in (yaml_store, sqlite_store):
            store.save("agent1", facts)
            store.delete("agent1", facts[1].id)

        yaml_result = yaml_store.load("agent1")
        sqlite_result = sqlite_store.load("agent1")

        assert len(yaml_result) == len(sqlite_result) == 1
        assert yaml_result[0].content == sqlite_result[0].content == "keep"

    def test_list_agents_identical(
        self, both_storages: tuple[YAMLStorage, SQLiteStorage]
    ) -> None:
        yaml_store, sqlite_store = both_storages

        for store in (yaml_store, sqlite_store):
            store.save("a", [Fact(content="a")])
            store.save("b", [Fact(content="b")])

        assert set(yaml_store.list_agents()) == set(sqlite_store.list_agents())
