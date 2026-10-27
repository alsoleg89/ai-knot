"""Durable intentions, change subscriptions and guarded external actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .changes import ChangeBatch, FactChange
from .models import (
    AccessDenied,
    ActionPermit,
    ActionRecord,
    ActionState,
    BasisIssue,
    CursorExpired,
    EvidenceRef,
    FreshnessRecord,
    IdempotencyConflict,
    Intent,
    IntentState,
    InvalidDelivery,
    JSONObject,
    JSONValue,
    Lease,
    LeaseLost,
    ResumePacket,
    RevisionConflict,
    StaleEvidence,
    UnsupportedAlwaysOnBackend,
    VerificationOutcome,
    VerificationRequest,
    VerificationResult,
    WaitCondition,
    WaitForChange,
    WaitForSlot,
    WaitUntil,
)
from .subscriptions import Subscription, WatchSelector

if TYPE_CHECKING:
    from .maintenance import (
        ConsolidationProposal,
        ConsolidationProvider,
        MaintenanceBudget,
        MaintenanceReport,
        VerificationProvider,
        run_maintenance,
    )
    from .memory import AlwaysOnMemory


__all__ = [
    "AlwaysOnMemory",
    "MaintenanceBudget",
    "MaintenanceReport",
    "ConsolidationProposal",
    "VerificationProvider",
    "ConsolidationProvider",
    "run_maintenance",
    "AccessDenied",
    "ActionPermit",
    "ActionRecord",
    "ActionState",
    "BasisIssue",
    "ChangeBatch",
    "CursorExpired",
    "EvidenceRef",
    "FactChange",
    "FreshnessRecord",
    "IdempotencyConflict",
    "Intent",
    "IntentState",
    "InvalidDelivery",
    "JSONObject",
    "JSONValue",
    "Lease",
    "LeaseLost",
    "ResumePacket",
    "RevisionConflict",
    "StaleEvidence",
    "Subscription",
    "UnsupportedAlwaysOnBackend",
    "VerificationOutcome",
    "VerificationRequest",
    "VerificationResult",
    "WaitCondition",
    "WaitForChange",
    "WaitForSlot",
    "WaitUntil",
    "WatchSelector",
]


def __getattr__(name: str) -> Any:
    if name == "AlwaysOnMemory":
        from .memory import AlwaysOnMemory

        return AlwaysOnMemory
    if name in {
        "MaintenanceBudget",
        "MaintenanceReport",
        "ConsolidationProposal",
        "VerificationProvider",
        "ConsolidationProvider",
        "run_maintenance",
    }:
        from . import maintenance

        return getattr(maintenance, name)
    raise AttributeError(name)
