"""Real SQL backends shared by the always-on integration tests."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.base import StorageBackend
from ai_knot.storage.sqlite_storage import SQLiteStorage

if TYPE_CHECKING:
    from ai_knot.always_on import AlwaysOnMemory


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 27, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float = 0, **kwargs: float) -> None:
        self.now += timedelta(seconds=seconds, **kwargs)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()

    class ClockDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return clock().astimezone(tz) if tz is not None else clock().replace(tzinfo=None)

    # Fact defaults and foreground writes must use the same time as the facade.
    # Otherwise tests start failing once real time passes the fixed scenario date.
    for module in ("types", "knowledge", "learning", "_resolve"):
        monkeypatch.setattr(f"ai_knot.{module}.datetime", ClockDatetime)
    return clock


@pytest.fixture(params=["sqlite", "postgres"])
def sql_storage(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StorageBackend]:
    if request.param == "sqlite":
        yield SQLiteStorage(str(tmp_path / "always-on.db"))
        return
    dsn = os.environ.get("AI_KNOT_TEST_PG_DSN")
    if not dsn:
        if os.environ.get("AI_KNOT_REQUIRE_POSTGRES") == "1":
            pytest.fail("AI_KNOT_TEST_PG_DSN is required for the PostgreSQL gate")
        pytest.skip("AI_KNOT_TEST_PG_DSN not set")
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from ai_knot.storage.postgres_storage import PostgresStorage

    schema = "ao_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield PostgresStorage(make_conninfo(dsn, options=f"-csearch_path={schema}"))
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def kb(sql_storage: StorageBackend) -> KnowledgeBase:
    return KnowledgeBase("owner", storage=sql_storage)


@pytest.fixture
def ao_factory(sql_storage: StorageBackend, clock: FakeClock) -> Callable[..., AlwaysOnMemory]:
    def factory(**kwargs: object) -> AlwaysOnMemory:
        from ai_knot.always_on import AlwaysOnMemory

        return AlwaysOnMemory(KnowledgeBase("owner", storage=sql_storage), clock=clock, **kwargs)

    return factory
