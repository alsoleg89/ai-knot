"""Storage backend protocol definition."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from ai_knot.types import Fact, SlotDelta


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

    def load_active_frontier(self, agent_id: str) -> list[Fact]:
        """Return the latest active fact per slot_key (active frontier).

        For slotted facts (``slot_key != ""``), returns the highest-version
        active fact per slot.  For unslotted facts, returns all active facts
        (each unslotted fact has a unique identity with no slot to collapse).
        """
        ...

    def load_slot_deltas_since(
        self, agent_id: str, since_version: int, exclude_agent: str
    ) -> list[SlotDelta]:
        """Lightweight delta pull: slot changes since *since_version*, excluding *exclude_agent*.

        Returns ``SlotDelta`` records instead of full ``Fact`` objects, making
        cross-agent sync roughly one order of magnitude cheaper in token cost.
        """
        ...

    def save_atomic(self, agent_id: str, facts: list[Fact]) -> None:
        """Atomically replace all facts for an agent using a database-level exclusive lock.

        For SQLite this uses ``BEGIN IMMEDIATE`` to prevent other writers from
        interleaving between the DELETE and INSERT operations.  YAML backends
        can fall back to the regular ``save()`` but should be documented as
        degraded (single-writer only).
        """
        ...


@runtime_checkable
class AtomicUpdateCapable(Protocol):
    """Optional extension for backends that support cross-process atomic load+save.

    Implementations must guarantee that the load, callback, and save execute
    as a single exclusive transaction, preventing lost updates when multiple
    processes share the same storage file.
    """

    def atomic_update(
        self,
        agent_id: str,
        fn: Callable[[list[Fact]], list[Fact]],
    ) -> None:
        """Load all facts for *agent_id*, apply *fn*, save the result atomically.

        The callback *fn* receives the current fact list and must return the
        updated list.  The entire load→transform→save cycle is protected by an
        exclusive database-level lock.
        """
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
