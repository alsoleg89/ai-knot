"""Regression tests: ``topic_channel`` and ``visibility_scope`` survive a storage round-trip.

These two multi-agent routing / access-scope fields were persisted by the SQLite
and YAML backends but silently DROPPED by the Postgres backend (the columns were
missing) — a correctness bug for shared-pool deployments on Postgres. These tests
lock parity across all three backends. The Postgres test is skipped unless
``AI_KNOT_TEST_PG_DSN`` is set (CI has no Postgres), matching the rest of the suite.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact


def _fact() -> Fact:
    return Fact(content="deploy via FluxCD", topic_channel="devops", visibility_scope="team:infra")


def test_sqlite_round_trip(tmp_path: pathlib.Path) -> None:
    store = SQLiteStorage(str(tmp_path / "t.db"))
    store.save("a", [_fact()])
    loaded = store.load("a")[0]
    assert loaded.topic_channel == "devops"
    assert loaded.visibility_scope == "team:infra"


def test_yaml_round_trip(tmp_path: pathlib.Path) -> None:
    store = YAMLStorage(base_dir=str(tmp_path))
    store.save("a", [_fact()])
    loaded = store.load("a")[0]
    assert loaded.topic_channel == "devops"
    assert loaded.visibility_scope == "team:infra"


def test_sqlite_yaml_parity(tmp_path: pathlib.Path) -> None:
    sq = SQLiteStorage(str(tmp_path / "t.db"))
    ya = YAMLStorage(base_dir=str(tmp_path / "yaml"))
    fact = _fact()
    sq.save("a", [fact])
    ya.save("a", [fact])
    s_loaded = sq.load("a")[0]
    y_loaded = ya.load("a")[0]
    assert (s_loaded.topic_channel, s_loaded.visibility_scope) == (
        y_loaded.topic_channel,
        y_loaded.visibility_scope,
    )


_PG_DSN = os.environ.get("AI_KNOT_TEST_PG_DSN", "")


@pytest.mark.skipif(not _PG_DSN, reason="AI_KNOT_TEST_PG_DSN not set (no live Postgres)")
def test_postgres_round_trip() -> None:
    pytest.importorskip("psycopg")
    from ai_knot.storage.postgres_storage import PostgresStorage

    store = PostgresStorage(dsn=_PG_DSN)
    agent = "field_parity_test_agent"
    try:
        store.save(agent, [_fact()])
        loaded = store.load(agent)[0]
        assert loaded.topic_channel == "devops"
        assert loaded.visibility_scope == "team:infra"
    finally:
        store.save(agent, [])
