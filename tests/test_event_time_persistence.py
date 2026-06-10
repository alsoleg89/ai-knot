"""Regression tests: ``Fact.event_time`` survives storage round-trip.

``event_time`` is the STRUCTURED real-world anchor (when a memory was formed /
uttered) — the standard field exposed by production memory systems
(Mem0 ``timestamp``/``reference_date``, memvid ``ExplicitHeader``). It is the
prerequisite for temporal-reasoning and knowledge-update queries that anchor on
*when* a fact was true (e.g. LongMemEval). Historically the field was set in
memory but DROPPED by every storage backend on save/load (only
``qualifiers["event_date"]`` round-tripped). These tests lock the persistence in
all three backends, mirroring how ``valid_until`` (a ``datetime | None``) is
already serialized.

The Postgres test is skipped unless a live server DSN is provided via
``AI_KNOT_TEST_PG_DSN`` (CI has no Postgres), matching the rest of the suite.
"""

from __future__ import annotations

import os
import pathlib
from datetime import UTC, datetime

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact

# A fixed, tz-aware anchor (8 May 2023) — the kind of per-session haystack
# timestamp LongMemEval supplies. tz-aware so the ISO round-trip is exact.
ANCHOR = datetime(2023, 5, 8, 13, 56, tzinfo=UTC)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


class TestSQLiteEventTime:
    def test_event_time_round_trip(self, sqlite_storage: SQLiteStorage) -> None:
        sqlite_storage.save("agent1", [Fact(content="met Anna at the gala", event_time=ANCHOR)])
        loaded = sqlite_storage.load("agent1")[0]
        assert loaded.event_time == ANCHOR

    def test_event_time_none_round_trips_as_none(self, sqlite_storage: SQLiteStorage) -> None:
        sqlite_storage.save("agent1", [Fact(content="no anchor here")])
        loaded = sqlite_storage.load("agent1")[0]
        assert loaded.event_time is None

    def test_event_time_survives_snapshot(self, sqlite_storage: SQLiteStorage) -> None:
        sqlite_storage.save_snapshot(
            "agent1", "snap", [Fact(content="anchored", event_time=ANCHOR)]
        )
        loaded = sqlite_storage.load_snapshot("agent1", "snap")[0]
        assert loaded.event_time == ANCHOR

    def test_event_time_survives_load_active(self, sqlite_storage: SQLiteStorage) -> None:
        # load_active uses the same _SELECT_COLS / _fact_from_row path.
        sqlite_storage.save("agent1", [Fact(content="active anchored", event_time=ANCHOR)])
        active = sqlite_storage.load_active("agent1")
        assert len(active) == 1
        assert active[0].event_time == ANCHOR


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


class TestYAMLEventTime:
    def test_event_time_round_trip(self, yaml_storage: YAMLStorage) -> None:
        yaml_storage.save("agent1", [Fact(content="met Anna at the gala", event_time=ANCHOR)])
        loaded = yaml_storage.load("agent1")[0]
        assert loaded.event_time == ANCHOR

    def test_event_time_none_round_trips_as_none(self, yaml_storage: YAMLStorage) -> None:
        yaml_storage.save("agent1", [Fact(content="no anchor here")])
        loaded = yaml_storage.load("agent1")[0]
        assert loaded.event_time is None

    def test_event_time_omitted_from_yaml_when_none(self, yaml_storage: YAMLStorage) -> None:
        # Compact-YAML contract: default (None) values are not written to disk.
        yaml_storage.save("agent1", [Fact(content="no anchor")])
        raw = (pathlib.Path(yaml_storage._base_dir) / "agent1" / "knowledge.yaml").read_text()
        assert "event_time" not in raw

    def test_event_time_survives_snapshot(self, yaml_storage: YAMLStorage) -> None:
        yaml_storage.save_snapshot("agent1", "snap", [Fact(content="anchored", event_time=ANCHOR)])
        loaded = yaml_storage.load_snapshot("agent1", "snap")[0]
        assert loaded.event_time == ANCHOR


# ---------------------------------------------------------------------------
# Cross-backend parity (YAML == SQLite)
# ---------------------------------------------------------------------------


def test_yaml_sqlite_event_time_identical(tmp_path: pathlib.Path) -> None:
    yaml = YAMLStorage(base_dir=str(tmp_path / "yaml"))
    sqlite = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    facts = [
        Fact(content="anchored fact", event_time=ANCHOR),
        Fact(content="unanchored fact"),
    ]
    yaml.save("agent1", facts)
    sqlite.save("agent1", facts)
    yf = {f.content: f.event_time for f in yaml.load("agent1")}
    sf = {f.content: f.event_time for f in sqlite.load("agent1")}
    assert yf == sf
    assert yf["anchored fact"] == ANCHOR
    assert yf["unanchored fact"] is None


# ---------------------------------------------------------------------------
# add() → reload integration (the path the LongMemEval harness exercises)
# ---------------------------------------------------------------------------


def test_add_persists_event_time_sqlite(tmp_path: pathlib.Path) -> None:
    kb = KnowledgeBase(
        agent_id="conv-0",
        storage=SQLiteStorage(db_path=str(tmp_path / "kb.db")),
    )
    kb.add("I started my new job at Globex", event_time=ANCHOR)
    # Reload through a *fresh* KnowledgeBase to prove it came off disk, not memory.
    kb2 = KnowledgeBase(
        agent_id="conv-0",
        storage=SQLiteStorage(db_path=str(tmp_path / "kb.db")),
    )
    stored = kb2.list_facts()
    assert stored
    assert stored[0].event_time == ANCHOR
    # The anchor must NOT leak into the indexed content (no date text-prefix hack).
    assert "2023" not in stored[0].content


# ---------------------------------------------------------------------------
# Postgres (live-server only — skipped in CI)
# ---------------------------------------------------------------------------

_PG_DSN = os.environ.get("AI_KNOT_TEST_PG_DSN", "")


@pytest.mark.skipif(not _PG_DSN, reason="AI_KNOT_TEST_PG_DSN not set (no live Postgres)")
def test_event_time_round_trip_postgres() -> None:
    pytest.importorskip("psycopg")
    from ai_knot.storage.postgres_storage import PostgresStorage

    store = PostgresStorage(dsn=_PG_DSN)
    agent = "evt_time_test_agent"
    try:
        store.save(agent, [Fact(content="pg anchored", event_time=ANCHOR)])
        loaded = store.load(agent)[0]
        assert loaded.event_time == ANCHOR
        store.save(agent, [Fact(content="pg unanchored")])
        assert store.load(agent)[0].event_time is None
    finally:
        store.save(agent, [])
