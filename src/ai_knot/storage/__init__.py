"""Storage backends for ai_knot."""

from __future__ import annotations

import os

from ai_knot.storage.base import SnapshotCapable, StorageBackend
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage

__all__ = ["SQLiteStorage", "SnapshotCapable", "StorageBackend", "YAMLStorage", "create_storage"]

_BACKENDS = {
    "yaml": "YAMLStorage",
    "sqlite": "SQLiteStorage",
    "postgres": "PostgresStorage",
}


def create_storage(
    backend: str, *, base_dir: str = ".ai_knot", dsn: str | None = None
) -> StorageBackend:  # noqa: E501
    """Create a storage backend by name.

    Args:
        backend: One of "yaml", "sqlite", "postgres".
        base_dir: Directory for file-based backends (yaml, sqlite).
        dsn: Connection string for remote backends (postgres).
            Also read from ``AI_KNOT_DSN`` env var if not provided.

    Returns:
        A storage backend instance.

    Raises:
        ValueError: If the backend name is unknown.
    """
    if backend == "yaml":
        return YAMLStorage(base_dir=base_dir)
    if backend == "sqlite":
        db_path = dsn if dsn else os.path.join(base_dir, "ai_knot.db")
        return SQLiteStorage(db_path=db_path)
    if backend == "postgres":
        resolved_dsn = dsn or os.environ.get("AI_KNOT_DSN")
        if not resolved_dsn:
            raise ValueError(
                "PostgreSQL backend requires a DSN. "
                "Pass dsn= or set the AI_KNOT_DSN environment variable."
            )
        from ai_knot.storage.postgres_storage import PostgresStorage

        return PostgresStorage(dsn=resolved_dsn)
    raise ValueError(f"Unknown storage backend {backend!r}. Choose from: {', '.join(_BACKENDS)}")
