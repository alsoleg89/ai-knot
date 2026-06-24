"""Tests for create_storage() factory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ai_knot.storage import create_storage
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage

_HAS_PSYCOPG = importlib.util.find_spec("psycopg") is not None


class TestCreateStorage:
    """create_storage() returns correct backend types."""

    def test_yaml_backend(self, tmp_path: Path) -> None:
        storage = create_storage("yaml", base_dir=str(tmp_path))
        assert isinstance(storage, YAMLStorage)

    def test_sqlite_backend(self, tmp_path: Path) -> None:
        storage = create_storage("sqlite", base_dir=str(tmp_path))
        assert isinstance(storage, SQLiteStorage)

    def test_sqlite_honors_explicit_dsn_path(self, tmp_path: Path) -> None:
        """Regression: an explicit sqlite path (dsn / AI_KNOT_DB_PATH) must win.

        Previously create_storage ignored dsn for sqlite and always used
        <base_dir>/ai_knot.db, so every KB collided in one shared file
        regardless of the requested path (cross-run contamination).
        """
        explicit = str(tmp_path / "isolated.db")
        storage = create_storage("sqlite", base_dir=str(tmp_path), dsn=explicit)
        assert isinstance(storage, SQLiteStorage)
        assert storage._db_path == explicit

    def test_sqlite_defaults_when_no_dsn(self, tmp_path: Path) -> None:
        storage = create_storage("sqlite", base_dir=str(tmp_path))
        assert storage._db_path == str(tmp_path / "ai_knot.db")

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_storage("redis")

    def test_postgres_without_dsn_raises(self) -> None:
        with pytest.raises(ValueError, match="DSN"):
            create_storage("postgres")

    @pytest.mark.skipif(
        _HAS_PSYCOPG,
        reason="exercises the missing-psycopg ImportError path; with psycopg installed "
        "create_storage would instead try to connect to the fake DSN",
    )
    def test_postgres_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should read AI_KNOT_DSN from env when dsn= is not passed."""
        # We can't actually connect, but we can verify the env var path works
        # by checking that it tries to import psycopg (not ValueError for missing DSN)
        monkeypatch.setenv("AI_KNOT_DSN", "postgresql://fake:fake@localhost/fake")
        with pytest.raises(ImportError, match="psycopg"):
            create_storage("postgres")
