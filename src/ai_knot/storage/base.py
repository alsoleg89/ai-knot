"""Storage backend protocol definition."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from ai_knot.types import Fact


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-format datetime string, ensuring UTC timezone."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class StorageBackend(Protocol):
    """Interface that all storage backends must implement.

    Backends are responsible for persisting and retrieving Fact objects,
    keyed by agent_id. Each agent has its own isolated namespace.
    """

    def save(self, agent_id: str, facts: list[Fact]) -> None:
        """Persist the full list of facts for an agent (replaces existing)."""
        ...

    def load(self, agent_id: str) -> list[Fact]:
        """Load all facts for an agent. Returns empty list if none exist."""
        ...

    def delete(self, agent_id: str, fact_id: str) -> None:
        """Remove a single fact by id. No-op if fact doesn't exist."""
        ...

    def list_agents(self) -> list[str]:
        """Return all agent_ids that have stored facts."""
        ...


@runtime_checkable
class TemporalStorageCapable(Protocol):
    """Optional extension for backends with index-accelerated temporal queries.

    Not required by ``StorageBackend``.  ``SharedMemoryPool`` checks
    ``isinstance(storage, TemporalStorageCapable)`` at runtime and falls back
    to Python-level filtering on YAML backends.
    """

    def load_active(self, agent_id: str) -> list[Fact]:
        """Load only facts where ``valid_until IS NULL`` (index-accelerated)."""
        ...

    def load_since_version(self, agent_id: str, since: int, exclude_agent: str) -> list[Fact]:
        """MESI dirty pull: facts with version > since, from agents other than exclude_agent."""
        ...


@runtime_checkable
class SnapshotCapable(Protocol):
    """Optional extension protocol for backends that support named snapshots.

    Not required by ``StorageBackend``. ``KnowledgeBase`` checks
    ``isinstance(storage, SnapshotCapable)`` at runtime and raises
    ``NotImplementedError`` when the backend does not implement it.
    """

    def save_snapshot(self, agent_id: str, name: str, facts: list[Fact]) -> None:
        """Persist a named snapshot for an agent (overwrites if name exists)."""
        ...

    def load_snapshot(self, agent_id: str, name: str) -> list[Fact]:
        """Load facts from a named snapshot.

        Raises:
            KeyError: If no snapshot with the given name exists.
        """
        ...

    def list_snapshots(self, agent_id: str) -> list[str]:
        """Return snapshot names for an agent, sorted by creation time (oldest first)."""
        ...

    def delete_snapshot(self, agent_id: str, name: str) -> None:
        """Delete a named snapshot. No-op if the snapshot does not exist."""
        ...
