"""Storage backends for agentmemo."""

from __future__ import annotations

import os

from agentmemo.storage.base import StorageBackend
from agentmemo.storage.sqlite_storage import SQLiteStorage
from agentmemo.storage.yaml_storage import YAMLStorage

__all__ = ["SQLiteStorage", "StorageBackend", "YAMLStorage", "create_storage"]

_BACKENDS = {
    "yaml": "YAMLStorage",
    "sqlite": "SQLiteStorage",
    "postgres": "PostgresStorage",
}


def create_storage(
    backend: str, *, base_dir: str = ".agentmemo", dsn: str | None = None
) -> StorageBackend:  # noqa: E501
    """Create a storage backend by name.

    Args:
        backend: One of "yaml", "sqlite", "postgres".
        base_dir: Directory for file-based backends (yaml, sqlite).
        dsn: Connection string for remote backends (postgres).
            Also read from ``AGENTMEMO_DSN`` env var if not provided.

    Returns:
        A storage backend instance.

    Raises:
        ValueError: If the backend name is unknown.
    """
    if backend == "yaml":
        return YAMLStorage(base_dir=base_dir)
    if backend == "sqlite":
        return SQLiteStorage(db_path=os.path.join(base_dir, "agentmemo.db"))
    if backend == "postgres":
        resolved_dsn = dsn or os.environ.get("AGENTMEMO_DSN")
        if not resolved_dsn:
            raise ValueError(
                "PostgreSQL backend requires a DSN. "
                "Pass dsn= or set the AGENTMEMO_DSN environment variable."
            )
        from agentmemo.storage.postgres_storage import PostgresStorage

        return PostgresStorage(dsn=resolved_dsn)
    raise ValueError(f"Unknown storage backend {backend!r}. Choose from: {', '.join(_BACKENDS)}")
