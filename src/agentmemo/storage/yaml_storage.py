"""YAML-based storage backend — human-readable, Git-friendly."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agentmemo.types import Fact, MemoryType

logger = logging.getLogger(__name__)


class YAMLStorage:
    """Stores facts as YAML files on disk.

    Directory layout:
        {base_dir}/{agent_id}/knowledge.yaml

    Each file is human-readable, editable, and Git-trackable.
    """

    def __init__(self, base_dir: str = ".agentmemo") -> None:
        self._base_dir = Path(base_dir)

    def save(self, agent_id: str, facts: list[Fact]) -> None:
        """Write all facts for an agent to a YAML file."""
        agent_dir = self._base_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        data: dict[str, dict[str, Any]] = {}
        for fact in facts:
            data[fact.id] = {
                "content": fact.content,
                "type": fact.type.value,
                "importance": fact.importance,
                "retention_score": fact.retention_score,
                "access_count": fact.access_count,
                "tags": fact.tags,
                "created_at": fact.created_at.isoformat(),
                "last_accessed": fact.last_accessed.isoformat(),
            }

        yaml_path = agent_dir / "knowledge.yaml"
        yaml_path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.debug("Saved %d facts for agent '%s' to %s", len(facts), agent_id, yaml_path)

    def load(self, agent_id: str) -> list[Fact]:
        """Read all facts for an agent from a YAML file."""
        yaml_path = self._base_dir / agent_id / "knowledge.yaml"
        if not yaml_path.exists():
            return []

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not raw:
            return []

        facts: list[Fact] = []
        for fact_id, entry in raw.items():
            facts.append(
                Fact(
                    id=str(fact_id),
                    content=entry["content"],
                    type=MemoryType(entry["type"]),
                    importance=float(entry["importance"]),
                    retention_score=float(entry["retention_score"]),
                    access_count=int(entry["access_count"]),
                    tags=list(entry.get("tags", [])),
                    created_at=_parse_datetime(entry["created_at"]),
                    last_accessed=_parse_datetime(entry["last_accessed"]),
                )
            )
        return facts

    def delete(self, agent_id: str, fact_id: str) -> None:
        """Remove a single fact by id."""
        facts = self.load(agent_id)
        filtered = [f for f in facts if f.id != fact_id]
        if len(filtered) < len(facts):
            self.save(agent_id, filtered)

    def list_agents(self) -> list[str]:
        """Return all agent_ids that have stored knowledge."""
        if not self._base_dir.exists():
            return []
        return [
            d.name
            for d in self._base_dir.iterdir()
            if d.is_dir() and (d / "knowledge.yaml").exists()
        ]


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-format datetime string, ensuring UTC timezone."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
