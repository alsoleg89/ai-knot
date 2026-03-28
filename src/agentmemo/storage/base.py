"""Storage backend protocol definition."""

from __future__ import annotations

from typing import Protocol

from agentmemo.types import Fact


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
