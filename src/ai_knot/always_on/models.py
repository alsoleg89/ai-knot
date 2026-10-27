"""Public value objects and validation for durable agent memory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, TypeAlias


class UnsupportedAlwaysOnBackend(Exception):
    pass


class AccessDenied(Exception):
    pass


class CursorExpired(Exception):
    pass


class InvalidDelivery(Exception):
    pass


JSONValue: TypeAlias = "None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]"
JSONObject: TypeAlias = dict[str, JSONValue]


class RevisionConflict(Exception):
    pass


class LeaseLost(Exception):
    pass


class IdempotencyConflict(Exception):
    pass


class StaleEvidence(Exception):
    pass


class IntentState(StrEnum):
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ActionState(StrEnum):
    PREPARED = "prepared"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class VerificationOutcome(StrEnum):
    CONFIRMED = "confirmed"
    CHANGED = "changed"
    UNAVAILABLE = "unavailable"


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timezone-aware UTC required")
    return value


def canonical(value: JSONValue) -> str:
    def check(item: JSONValue) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) or not key for key in item):
                raise ValueError("JSON keys must be nonempty strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (bool, int, float, str):
            raise ValueError("not a JSON value")

    check(value)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    if len(encoded.encode()) > 65536:
        raise ValueError("JSON payload exceeds 65536 bytes")
    return encoded


def digest(value: JSONValue) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class EvidenceRef:
    namespace: str
    fact_id: str
    revision: str

    def __post_init__(self) -> None:
        nonempty(self.namespace, "namespace")
        nonempty(self.fact_id, "fact_id")
        nonempty(self.revision, "revision")


@dataclass(frozen=True)
class WaitUntil:
    at: datetime

    def __post_init__(self) -> None:
        utc(self.at)


@dataclass(frozen=True)
class WaitForSlot:
    namespace: str
    slot_key: str
    value_text: str

    def __post_init__(self) -> None:
        nonempty(self.namespace, "namespace")
        nonempty(self.slot_key, "slot_key")
        if not isinstance(self.value_text, str):
            raise ValueError("value_text must be text")


@dataclass(frozen=True)
class WaitForChange:
    ref: EvidenceRef


WaitCondition: TypeAlias = WaitUntil | WaitForSlot | WaitForChange


@dataclass(frozen=True)
class Lease:
    token: str
    generation: int
    worker_id: str
    expires_at: datetime


@dataclass(frozen=True)
class Intent:
    id: str
    owner_agent_id: str
    client_key: str
    goal: str
    state: IntentState
    progress: JSONObject
    next_step: str
    waiting_for: WaitCondition | None
    revision: int
    basis: tuple[EvidenceRef, ...]
    lease: Lease | None
    deadline: datetime | None
    last_transition_at: datetime
    lease_generation: int = 0
    blocked_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionRecord:
    id: str
    intent_id: str
    action_key: str
    request: JSONObject
    request_digest: str
    basis: tuple[EvidenceRef, ...]
    state: ActionState
    idempotency_key: str
    result: JSONObject | None
    error: JSONObject | None
    created_at: datetime
    updated_at: datetime
    reconciliation_evidence: JSONObject | None = None
    revision: int = 1


@dataclass(frozen=True)
class ActionPermit:
    action_id: str
    idempotency_key: str


@dataclass(frozen=True)
class FreshnessRecord:
    ref: EvidenceRef
    subject_kind: Literal["slot", "fact"]
    subject_key: str
    source_ref: str
    recheck_after_seconds: int
    checked_at: datetime | None
    verified_until: datetime
    retry_at: datetime
    last_outcome: VerificationOutcome | None
    failure_count: int
    evidence: JSONObject | None = None
    source_version: str = ""


@dataclass(frozen=True)
class VerificationRequest:
    ref: EvidenceRef
    source_ref: str


@dataclass(frozen=True)
class VerificationResult:
    outcome: VerificationOutcome
    evidence: JSONObject
    source_version: str


@dataclass(frozen=True)
class BasisIssue:
    reason: str
    ref: EvidenceRef | None  # access_changed must not disclose a closed fact ID


@dataclass(frozen=True)
class ResumePacket:
    intent: Intent
    open_actions: tuple[ActionRecord, ...]
    stale_basis: tuple[BasisIssue, ...]
    freshness_due: tuple[EvidenceRef, ...]
    ready: bool
    reasons: tuple[str, ...]


def nonempty(value: str, field: str = "value") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def page_limit(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 500:
        raise ValueError("limit must be an integer in 1..500")
    return value


def positive_integer(value: int, field: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {field}")
    return value
