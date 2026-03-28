"""YAML-based storage backend — human-readable, Git-friendly."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agentmemo.types import Fact, MemoryType

logger = logging.getLogger(__name__)

# One lock per on-disk YAML file; keyed by resolved absolute path.
_file_locks: dict[str, threading.Lock] = {}
_file_locks_mutex = threading.Lock()


def _get_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _file_locks_mutex:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]


class YAMLStorage:
    """Stores facts as YAML files on disk.

    Directory layout:
        {base_dir}/{agent_id}/knowledge.yaml

    Each file is human-readable, editable, and Git-trackable.
    Concurrent writes from multiple threads are serialized per-file via a
    threading.Lock; the file is replaced atomically (write → fsync → rename).
    """

    def __init__(self, base_dir: str = ".agentmemo") -> None:
        self._base_dir = Path(base_dir)

    def save(self, agent_id: str, facts: list[Fact]) -> None:
        """Write all facts for an agent to a YAML file (atomic)."""
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
        yaml_text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

        lock = _get_lock(yaml_path)
        with lock:
            # Write to a temp file in the same directory, then rename atomically.
            fd, tmp_path = tempfile.mkstemp(dir=agent_dir, suffix=".yaml.tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(yaml_text)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, yaml_path)
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

        logger.debug("Saved %d facts for agent '%s' to %s", len(facts), agent_id, yaml_path)

    def load(self, agent_id: str) -> list[Fact]:
        """Read all facts for an agent from a YAML file."""
        yaml_path = self._base_dir / agent_id / "knowledge.yaml"
        if not yaml_path.exists():
            return []

        lock = _get_lock(yaml_path)
        with lock:
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
