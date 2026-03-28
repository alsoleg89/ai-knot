"""Tests for create_storage() factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentmemo.storage import create_storage
from agentmemo.storage.sqlite_storage import SQLiteStorage
from agentmemo.storage.yaml_storage import YAMLStorage


class TestCreateStorage:
    """create_storage() returns correct backend types."""

    def test_yaml_backend(self, tmp_path: Path) -> None:
        storage = create_storage("yaml", base_dir=str(tmp_path))
        assert isinstance(storage, YAMLStorage)

    def test_sqlite_backend(self, tmp_path: Path) -> None:
        storage = create_storage("sqlite", base_dir=str(tmp_path))
        assert isinstance(storage, SQLiteStorage)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_storage("redis")

    def test_postgres_without_dsn_raises(self) -> None:
        with pytest.raises(ValueError, match="DSN"):
            create_storage("postgres")

    def test_postgres_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should read AGENTMEMO_DSN from env when dsn= is not passed."""
        # We can't actually connect, but we can verify the env var path works
        # by checking that it tries to import psycopg (not ValueError for missing DSN)
        monkeypatch.setenv("AGENTMEMO_DSN", "postgresql://fake:fake@localhost/fake")
        with pytest.raises(ImportError, match="psycopg"):
            create_storage("postgres")
