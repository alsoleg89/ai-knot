"""AuditTrail — write provenance for every memory operation."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal


@dataclasses.dataclass(frozen=True, slots=True)
class AuditEvent:
    event_id: str
    operation: Literal["write", "forget", "consolidate", "read"]
    atom_id: str | None
    agent_id: str
    timestamp: int
    details: dict[str, Any] = dataclasses.field(default_factory=dict, hash=False)


@dataclasses.dataclass(frozen=True, slots=True)
class AuditTrail:
    atom_id: str | None
    events: tuple[AuditEvent, ...]
