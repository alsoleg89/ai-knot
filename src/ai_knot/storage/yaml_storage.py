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

from ai_knot.storage.base import parse_datetime as _parse_datetime
from ai_knot.types import Fact, MemoryType, MESIState

# Use C extension loader when available (10-20x faster than pure Python).
_YamlLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

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

    def __init__(self, base_dir: str = ".ai_knot") -> None:
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
            # Only write evidence fields when non-default to keep YAML compact.
            if fact.source_snippets:
                data[fact.id]["source_snippets"] = fact.source_snippets
            if fact.source_spans:
                data[fact.id]["source_spans"] = fact.source_spans
            if not fact.supported:
                data[fact.id]["supported"] = fact.supported
            if fact.support_confidence != 1.0:
                data[fact.id]["support_confidence"] = fact.support_confidence
            if fact.verification_source != "manual":
                data[fact.id]["verification_source"] = fact.verification_source
            if fact.access_intervals:
                data[fact.id]["access_intervals"] = fact.access_intervals
            if fact.origin_agent_id:
                data[fact.id]["origin_agent_id"] = fact.origin_agent_id
            if fact.visibility != "private":
                data[fact.id]["visibility"] = fact.visibility
            if fact.source_verbatim:
                data[fact.id]["source_verbatim"] = fact.source_verbatim
            data[fact.id]["valid_from"] = fact.valid_from.isoformat()
            if fact.valid_until is not None:
                data[fact.id]["valid_until"] = fact.valid_until.isoformat()
            if fact.entity:
                data[fact.id]["entity"] = fact.entity
            if fact.attribute:
                data[fact.id]["attribute"] = fact.attribute
            if fact.version:
                data[fact.id]["version"] = fact.version
            if fact.mesi_state != "E":
                data[fact.id]["mesi_state"] = fact.mesi_state

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
            raw = yaml.load(yaml_path.read_text(encoding="utf-8"), Loader=_YamlLoader)

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
                    source_snippets=list(entry.get("source_snippets", [])),
                    source_spans=list(entry.get("source_spans", [])),
                    supported=bool(entry.get("supported", True)),
                    support_confidence=float(entry.get("support_confidence", 1.0)),
                    verification_source=str(entry.get("verification_source", "legacy")),
                    access_intervals=[float(x) for x in entry.get("access_intervals", [])],
                    origin_agent_id=str(entry.get("origin_agent_id", "")),
                    visibility=str(entry.get("visibility", "private")),
                    source_verbatim=str(entry.get("source_verbatim", "")),
                    valid_from=_parse_datetime(entry["valid_from"])
                    if "valid_from" in entry
                    else datetime.now(UTC),
                    valid_until=_parse_datetime(entry["valid_until"])
                    if "valid_until" in entry
                    else None,
                    entity=str(entry.get("entity", "")),
                    attribute=str(entry.get("attribute", "")),
                    version=int(entry.get("version", 0)),
                    mesi_state=MESIState(str(entry.get("mesi_state", "E"))),
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

    # ------------------------------------------------------------------
    # SnapshotCapable implementation
    # ------------------------------------------------------------------

    def _snapshot_dir(self, agent_id: str) -> Path:
        return self._base_dir / agent_id / "snapshots"

    def save_snapshot(self, agent_id: str, name: str, facts: list[Fact]) -> None:
        """Persist a named snapshot (overwrites if name already exists)."""
        snap_dir = self._snapshot_dir(agent_id)
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"{name}.yaml"

        data: dict[str, Any] = {}
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
            if fact.source_snippets:
                data[fact.id]["source_snippets"] = fact.source_snippets
            if fact.source_spans:
                data[fact.id]["source_spans"] = fact.source_spans
            if not fact.supported:
                data[fact.id]["supported"] = fact.supported
            if fact.support_confidence != 1.0:
                data[fact.id]["support_confidence"] = fact.support_confidence
            if fact.verification_source != "manual":
                data[fact.id]["verification_source"] = fact.verification_source
            if fact.access_intervals:
                data[fact.id]["access_intervals"] = fact.access_intervals
            if fact.origin_agent_id:
                data[fact.id]["origin_agent_id"] = fact.origin_agent_id
            if fact.visibility != "private":
                data[fact.id]["visibility"] = fact.visibility
            if fact.source_verbatim:
                data[fact.id]["source_verbatim"] = fact.source_verbatim
            data[fact.id]["valid_from"] = fact.valid_from.isoformat()
            if fact.valid_until is not None:
                data[fact.id]["valid_until"] = fact.valid_until.isoformat()
            if fact.entity:
                data[fact.id]["entity"] = fact.entity
            if fact.attribute:
                data[fact.id]["attribute"] = fact.attribute
            if fact.version:
                data[fact.id]["version"] = fact.version
            if fact.mesi_state != "E":
                data[fact.id]["mesi_state"] = fact.mesi_state

        yaml_text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        lock = _get_lock(snap_path)
        with lock:
            fd, tmp_path = tempfile.mkstemp(dir=snap_dir, suffix=".yaml.tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(yaml_text)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, snap_path)
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

        logger.debug("Saved snapshot '%s' for agent '%s'", name, agent_id)

    def load_snapshot(self, agent_id: str, name: str) -> list[Fact]:
        """Load facts from a named snapshot.

        Raises:
            KeyError: If no snapshot with the given name exists.
        """
        snap_path = self._snapshot_dir(agent_id) / f"{name}.yaml"
        if not snap_path.exists():
            raise KeyError(f"Snapshot {name!r} not found for agent {agent_id!r}")

        lock = _get_lock(snap_path)
        with lock:
            raw = yaml.load(snap_path.read_text(encoding="utf-8"), Loader=_YamlLoader)

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
                    source_snippets=list(entry.get("source_snippets", [])),
                    source_spans=list(entry.get("source_spans", [])),
                    supported=bool(entry.get("supported", True)),
                    support_confidence=float(entry.get("support_confidence", 1.0)),
                    verification_source=str(entry.get("verification_source", "legacy")),
                    access_intervals=[float(x) for x in entry.get("access_intervals", [])],
                    origin_agent_id=str(entry.get("origin_agent_id", "")),
                    visibility=str(entry.get("visibility", "private")),
                    source_verbatim=str(entry.get("source_verbatim", "")),
                    valid_from=_parse_datetime(entry["valid_from"])
                    if "valid_from" in entry
                    else datetime.now(UTC),
                    valid_until=_parse_datetime(entry["valid_until"])
                    if "valid_until" in entry
                    else None,
                    entity=str(entry.get("entity", "")),
                    attribute=str(entry.get("attribute", "")),
                    version=int(entry.get("version", 0)),
                    mesi_state=MESIState(str(entry.get("mesi_state", "E"))),
                )
            )
        return facts

    def list_snapshots(self, agent_id: str) -> list[str]:
        """Return snapshot names sorted by modification time (oldest first)."""
        snap_dir = self._snapshot_dir(agent_id)
        if not snap_dir.exists():
            return []
        files = sorted(snap_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime)
        return [p.stem for p in files]

    def delete_snapshot(self, agent_id: str, name: str) -> None:
        """Delete a named snapshot. No-op if it does not exist."""
        snap_path = self._snapshot_dir(agent_id) / f"{name}.yaml"
        with contextlib.suppress(FileNotFoundError):
            snap_path.unlink()
