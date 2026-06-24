"""Storage backend protocol definition."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

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
class PoolStatsCapable(Protocol):
    """Optional extension for persisting shared-pool trust/usage telemetry.

    ``SharedMemoryPool`` keeps publish / use / quick-invalidation counters and
    per-fact consumer sets in memory; a backend implementing this protocol lets
    that social memory survive a process restart.  Opt-in via
    ``SharedMemoryPool(persist_stats=True)`` — pools default to in-memory only,
    so this adds no I/O to existing callers.
    """

    def save_pool_stats(self, stats: dict[str, Any]) -> None:
        """Persist the pool's trust/usage telemetry (overwrites any prior copy)."""
        ...

    def load_pool_stats(self) -> dict[str, Any]:
        """Load previously persisted telemetry, or an empty dict if none exists."""
        ...


@runtime_checkable
class ACLStoreCapable(Protocol):
    """Optional extension for durable per-agent read-scope grants (multi-agent ACL).

    ``SharedMemoryPool`` keeps read-scope grants (``grant_read``) in memory only,
    so a process restart forgets who may read which scope.  A backend implementing
    this protocol persists those grants; the pool reloads them on construction.
    """

    def save_grant(
        self, agent_id: str, scope: str, *, granted_at: str, granted_by: str = ""
    ) -> None:
        """Persist a read-scope grant (idempotent upsert on ``(agent_id, scope)``)."""
        ...

    def load_grants(self) -> dict[str, set[str]]:
        """Load all grants as ``{agent_id: {scope, ...}}`` (empty dict if none)."""
        ...

    def revoke_grant(self, agent_id: str, scope: str) -> None:
        """Remove a grant. No-op if the grant does not exist."""
        ...


@runtime_checkable
class EventLedgerCapable(Protocol):
    """Optional append-only audit ledger for trust changes and fact usage.

    ``PoolStatsCapable`` persists only an aggregate snapshot (current counters);
    this protocol records *when and why* trust changed and *which recall used a
    fact*, the event stream an audit needs.  Timestamps are caller-supplied ISO
    strings — storage never reads the clock, so runs stay deterministic.
    """

    def append_trust_event(
        self, *, ts: str, agent_id: str, event_type: str, delta: float, reason: str = ""
    ) -> None:
        """Append one trust-change event (publish / use / quick-invalidation / penalty)."""
        ...

    def append_usage_event(
        self, *, ts: str, fact_id: str, agent_id: str, recall_session: str = ""
    ) -> None:
        """Append one fact-usage event (a recall surfaced ``fact_id`` to ``agent_id``)."""
        ...

    def load_trust_events(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        """Return trust events in insertion order; filter by ``agent_id`` when given."""
        ...

    def load_usage_events(self, fact_id: str | None = None) -> list[dict[str, Any]]:
        """Return usage events in insertion order; filter by ``fact_id`` when given."""
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
