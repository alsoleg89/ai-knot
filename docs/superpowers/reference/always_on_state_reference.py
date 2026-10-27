"""Executable design reference; pure transitions, no SQL or production imports.

Run: python docs/superpowers/reference/always_on_state_reference.py
Inputs named views must be read under the SQL locks described in 02-agent-state-code.md.
Passing fabricated views to this demonstrator proves nothing about persistence or ACL.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal, TypeAlias, TypedDict, Unpack

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


@dataclass(frozen=True)
class WaitUntil:
    at: datetime


@dataclass(frozen=True)
class WaitForSlot:
    namespace: str
    slot_key: str
    value_text: str


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
class EvidenceView:
    """Transaction-local projection, not a public API or a cached permission."""

    namespace: str
    fact_id: str
    revision: str | None
    readable: bool = True
    active: bool = True
    supported: bool = True
    freshness: FreshnessRecord | None = None


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


TERMINAL = (IntentState.COMPLETED, IntentState.CANCELLED)
UNRESOLVED = (ActionState.STARTED, ActionState.UNKNOWN)


class IntentChanges(TypedDict, total=False):
    state: IntentState
    progress: JSONObject
    next_step: str
    waiting_for: WaitCondition | None
    basis: tuple[EvidenceRef, ...]
    lease: Lease | None
    lease_generation: int
    blocked_reasons: tuple[str, ...]


class ActionChanges(TypedDict, total=False):
    state: ActionState
    basis: tuple[EvidenceRef, ...]
    result: JSONObject | None
    reconciliation_evidence: JSONObject | None


class VerificationFields(TypedDict):
    last_outcome: VerificationOutcome
    evidence: JSONObject
    source_version: str


def union_basis(*groups: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
    refs: dict[tuple[str, str], EvidenceRef] = {}
    for group in groups:
        for ref in group:
            key = (ref.namespace, ref.fact_id)
            if key in refs and refs[key].revision != ref.revision:
                raise StaleEvidence("conflicting revisions of one fact")
            refs[key] = ref
    return tuple(sorted(refs.values()))


def freshness_blocks(policy: FreshnessRecord, ref: EvidenceRef, now: datetime) -> bool:
    utc(now)
    return (
        policy.ref != ref
        or policy.checked_at is None
        or policy.last_outcome != VerificationOutcome.CONFIRMED
        or now >= policy.verified_until
    )


def validate_basis(
    refs: tuple[EvidenceRef, ...], views: tuple[EvidenceView, ...], now: datetime
) -> tuple[BasisIssue, ...]:
    utc(now)
    index = {(view.namespace, view.fact_id): view for view in views}
    issues = []
    for ref in union_basis(refs):
        view = index.get((ref.namespace, ref.fact_id))
        if view is not None and not view.readable:
            issues.append(BasisIssue("access_changed", None))
            continue
        reason = None
        if view is None or view.revision is None:
            reason = "missing"
        elif view.revision != ref.revision:
            reason = "changed"
        elif not view.active:
            reason = "inactive"
        elif not view.supported:
            reason = "unsupported"
        elif view.freshness and freshness_blocks(view.freshness, ref, now):
            reason = "verification_due"
        if reason:
            issues.append(BasisIssue(reason, ref))
    return tuple(issues)


def transition(intent: Intent, now: datetime, **fields: Unpack[IntentChanges]) -> Intent:
    if "state" in fields and fields["state"] != IntentState.WAITING:
        fields["waiting_for"] = None
    return replace(intent, revision=intent.revision + 1, last_transition_at=utc(now), **fields)


def action_transition(
    action: ActionRecord, now: datetime, **fields: Unpack[ActionChanges]
) -> ActionRecord:
    return replace(action, revision=action.revision + 1, updated_at=utc(now), **fields)


def require_lease(intent: Intent, token: str, now: datetime) -> Lease:
    utc(now)
    lease = intent.lease
    if lease is None or lease.token != token or now >= lease.expires_at:
        raise LeaseLost("expired or superseded token")
    return lease


def expire(
    intent: Intent, actions: tuple[ActionRecord, ...], now: datetime
) -> tuple[Intent, tuple[ActionRecord, ...]]:
    utc(now)
    if intent.state in TERMINAL:
        return intent, actions
    expired = intent.lease is not None and now >= intent.lease.expires_at
    if expired:
        actions = tuple(
            action_transition(a, now, state=ActionState.UNKNOWN)
            if a.state == ActionState.STARTED
            else a
            for a in actions
        )
        unresolved = any(a.state in UNRESOLVED for a in actions)
        state = intent.state
        reasons = intent.blocked_reasons
        if unresolved:
            state, reasons = IntentState.BLOCKED, tuple(sorted(set(reasons) | {"unknown_action"}))
        elif state == IntentState.RUNNING:
            state, reasons = IntentState.READY, ()
        intent = transition(intent, now, state=state, lease=None, blocked_reasons=reasons)
    if (
        intent.deadline is not None
        and now >= utc(intent.deadline)
        and "deadline_exceeded" not in intent.blocked_reasons
    ):
        reasons = tuple(sorted(set(intent.blocked_reasons) | {"deadline_exceeded"}))
        intent = transition(intent, now, state=IntentState.BLOCKED, blocked_reasons=reasons)
    return intent, actions


def claim(
    intent: Intent, *, worker_id: str, token: str, ttl_seconds: int, now: datetime
) -> tuple[Intent, Lease]:
    utc(now)
    if not token or not worker_id or not 1 <= ttl_seconds <= 3600:
        raise ValueError("invalid lease arguments")
    if intent.lease is not None:
        raise RevisionConflict("settle expired lease first; live lease cannot be replaced")
    if intent.state not in (IntentState.READY, IntentState.BLOCKED):
        raise RevisionConflict("intent cannot be claimed")
    lease = Lease(
        token, intent.lease_generation + 1, worker_id, now + timedelta(seconds=ttl_seconds)
    )
    state = IntentState.RUNNING if intent.state == IntentState.READY else intent.state
    return transition(
        intent, now, state=state, lease=lease, lease_generation=lease.generation
    ), lease


def renew(intent: Intent, *, token: str, ttl_seconds: int, now: datetime) -> Intent:
    lease = require_lease(intent, token, now)
    if not 1 <= ttl_seconds <= 3600:
        raise ValueError("invalid TTL")
    return replace(intent, lease=replace(lease, expires_at=now + timedelta(seconds=ttl_seconds)))


def checkpoint(
    intent: Intent,
    actions: tuple[ActionRecord, ...],
    *,
    token: str,
    expected_revision: int,
    state: IntentState,
    progress: JSONObject,
    next_step: str,
    waiting_for: WaitCondition | None,
    basis: tuple[EvidenceRef, ...],
    views: tuple[EvidenceView, ...],
    now: datetime,
) -> Intent:
    require_lease(intent, token, now)
    if expected_revision != intent.revision:
        raise RevisionConflict("revision mismatch")
    allowed = {
        IntentState.RUNNING: set(IntentState) - {IntentState.RUNNING},
        IntentState.BLOCKED: {IntentState.READY, IntentState.WAITING, IntentState.CANCELLED},
    }
    if state not in allowed.get(intent.state, set()):
        raise RevisionConflict("invalid checkpoint transition")
    deadline = intent.deadline is not None and now >= utc(intent.deadline)
    if (
        deadline or "deadline_exceeded" in intent.blocked_reasons
    ) and state != IntentState.CANCELLED:
        raise RevisionConflict("deadline permits only cancellation")
    if (state == IntentState.WAITING) != (waiting_for is not None):
        raise ValueError("waiting_for iff waiting")
    if isinstance(waiting_for, WaitUntil):
        utc(waiting_for.at)
    if any(a.state == ActionState.STARTED for a in actions):
        raise RevisionConflict("finish or expire a started action before releasing its lease")
    unknown = any(a.state == ActionState.UNKNOWN for a in actions)
    if unknown and state not in (IntentState.BLOCKED, IntentState.CANCELLED):
        raise RevisionConflict("unknown effects require recovery blocking or explicit cancellation")
    refs = union_basis(basis)
    if validate_basis(refs, views, now):
        raise StaleEvidence("checkpoint basis is not current")
    canonical(progress)
    # JSON round-trip prevents a caller from mutating this saved value by alias.
    copied = json.loads(canonical(progress))
    reasons: tuple[str, ...] = ()
    if state == IntentState.BLOCKED:
        reasons = ("unknown_action",) if unknown else ("host_blocked",)
    return transition(
        intent,
        now,
        state=state,
        progress=copied,
        next_step=next_step,
        waiting_for=waiting_for,
        basis=refs,
        lease=None,
        blocked_reasons=reasons,
    )


def resume(
    intent: Intent,
    actions: tuple[ActionRecord, ...],
    views: tuple[EvidenceView, ...],
    *,
    condition_met: bool,
    now: datetime,
) -> ResumePacket:
    """Host computes condition_met from current ACL-projected slot/change/time data."""
    intent, actions = expire(intent, actions, now)
    issues = validate_basis(intent.basis, views, now)
    if issues and intent.state not in TERMINAL:
        reasons = tuple(sorted(set(intent.blocked_reasons) | {issue.reason for issue in issues}))
        if intent.state != IntentState.BLOCKED or reasons != intent.blocked_reasons:
            intent = transition(intent, now, state=IntentState.BLOCKED, blocked_reasons=reasons)
    elif intent.state == IntentState.WAITING and condition_met:
        intent = transition(intent, now, state=IntentState.READY, waiting_for=None)
    packet_reasons = set(intent.blocked_reasons)
    if intent.lease is not None:
        packet_reasons.add("live_lease")
    if intent.state != IntentState.READY:
        packet_reasons.add(intent.state.value)
    if any(a.state in UNRESOLVED for a in actions):
        packet_reasons.add("unknown_action")
    due = tuple(
        issue.ref
        for issue in issues
        if issue.reason == "verification_due" and issue.ref is not None
    )
    opened = tuple(a for a in actions if a.state in (*UNRESOLVED, ActionState.PREPARED))
    return ResumePacket(
        intent, opened, issues, due, not packet_reasons, tuple(sorted(packet_reasons))
    )


def prepare_action(
    intent: Intent,
    existing: ActionRecord | None,
    *,
    action_id: str,
    token: str,
    action_key: str,
    request: JSONObject,
    basis: tuple[EvidenceRef, ...],
    views: tuple[EvidenceView, ...],
    now: datetime,
) -> ActionRecord:
    require_lease(intent, token, now)
    if not action_key:
        raise ValueError("empty action_key")
    request_digest = digest(request)
    if existing is not None:
        if existing.intent_id != intent.id or existing.action_key != action_key:
            raise RevisionConflict("wrong existing action")
        if existing.request_digest != request_digest:
            raise IdempotencyConflict("action key reused for another request")
        if existing.state != ActionState.PREPARED:
            return existing
    refs = union_basis(intent.basis, basis)
    if validate_basis(refs, views, now):
        raise StaleEvidence("effective basis is not current")
    if existing is not None:
        return existing if existing.basis == refs else action_transition(existing, now, basis=refs)
    key = digest([intent.owner_agent_id, intent.id, action_key])
    return ActionRecord(
        action_id,
        intent.id,
        action_key,
        json.loads(canonical(request)),
        request_digest,
        refs,
        ActionState.PREPARED,
        key,
        None,
        None,
        now,
        now,
    )


def begin_action(
    intent: Intent,
    action: ActionRecord,
    actions: tuple[ActionRecord, ...],
    *,
    token: str,
    views: tuple[EvidenceView, ...],
    now: datetime,
) -> tuple[ActionRecord, ActionPermit]:
    require_lease(intent, token, now)
    if intent.deadline is not None and now >= utc(intent.deadline):
        raise RevisionConflict("deadline exceeded")
    if action.intent_id != intent.id or intent.state != IntentState.RUNNING:
        raise RevisionConflict("running owning intent required")
    if action.state != ActionState.PREPARED or any(a.state in UNRESOLVED for a in actions):
        raise RevisionConflict("another permit or unresolved effect")
    refs = union_basis(intent.basis, action.basis)
    if validate_basis(refs, views, now):
        raise StaleEvidence("pre-action guard rejected basis")
    return (
        action_transition(action, now, state=ActionState.STARTED),
        ActionPermit(action.id, action.idempotency_key),
    )


def finish_action(
    intent: Intent,
    action: ActionRecord,
    *,
    token: str,
    outcome: Literal["succeeded", "failed", "unknown"],
    result: JSONObject,
    now: datetime,
) -> ActionRecord:
    require_lease(intent, token, now)
    if outcome not in ("succeeded", "failed", "unknown"):
        raise ValueError("invalid finish outcome")
    if action.intent_id != intent.id:
        raise RevisionConflict("wrong intent")
    state = ActionState(outcome)
    encoded = canonical(result)
    if action.state == state and action.result is not None and canonical(action.result) == encoded:
        return action
    if action.state != ActionState.STARTED:
        raise RevisionConflict("only started action accepts a new finish")
    return action_transition(action, now, state=state, result=json.loads(encoded))


def reconcile_action(
    intent: Intent,
    action: ActionRecord,
    actions: tuple[ActionRecord, ...],
    *,
    token: str,
    outcome: Literal["succeeded", "failed", "retryable"],
    evidence: JSONObject,
    views: tuple[EvidenceView, ...],
    now: datetime,
) -> tuple[Intent, ActionRecord]:
    require_lease(intent, token, now)
    if outcome not in ("succeeded", "failed", "retryable"):
        raise ValueError("invalid reconciliation outcome")
    canonical(evidence)
    if not evidence or action.intent_id != intent.id:
        raise ValueError("owning intent and nonempty host evidence required")
    state = ActionState.PREPARED if outcome == "retryable" else ActionState(outcome)
    if action.reconciliation_evidence is not None and action.state == state:
        if canonical(action.reconciliation_evidence) == canonical(evidence):
            return intent, action
        if action.state in (ActionState.SUCCEEDED, ActionState.FAILED, ActionState.PREPARED):
            raise RevisionConflict("reconciliation result differs from stored result")
    if action.state not in (ActionState.UNKNOWN, ActionState.FAILED):
        raise RevisionConflict("only unknown/failed can be reconciled")
    action = action_transition(
        action,
        now,
        state=state,
        reconciliation_evidence=json.loads(canonical(evidence)),
        result=None if outcome == "retryable" else json.loads(canonical(evidence)),
    )
    remaining = tuple(action if a.id == action.id else a for a in actions)
    clear = not any(a.state in UNRESOLVED for a in remaining)
    try:
        refs = union_basis(intent.basis, action.basis)
        guards = validate_basis(refs, views, now)
    except StaleEvidence:
        # An historical basis conflict blocks recovery, not recording the proven outcome.
        guards = (BasisIssue("changed", None),)
    deadline_clear = intent.deadline is None or now < utc(intent.deadline)
    if (
        intent.state == IntentState.BLOCKED
        and intent.blocked_reasons == ("unknown_action",)
        and clear
        and not guards
        and deadline_clear
    ):
        intent = transition(intent, now, state=IntentState.RUNNING, blocked_reasons=())
    elif intent.state == IntentState.BLOCKED and clear:
        reasons = set(intent.blocked_reasons) - {"unknown_action"}
        reasons.update(issue.reason for issue in guards)
        if not deadline_clear:
            reasons.add("deadline_exceeded")
        intent = transition(intent, now, blocked_reasons=tuple(sorted(reasons)))
    return intent, action


def set_freshness(
    ref: EvidenceRef, *, slot_key: str, source_ref: str, interval: int, now: datetime
) -> FreshnessRecord:
    utc(now)
    if not source_ref or interval <= 0:
        raise ValueError("source and positive interval required")
    return FreshnessRecord(
        ref,
        "slot" if slot_key else "fact",
        slot_key or ref.fact_id,
        source_ref,
        interval,
        None,
        now,
        now,
        None,
        0,
    )


def verification_due(policy: FreshnessRecord, current: EvidenceRef, now: datetime) -> bool:
    utc(now)
    return policy.ref != current or now >= policy.retry_at


def record_verification(
    policy: FreshnessRecord,
    ref: EvidenceRef,
    *,
    current: EvidenceRef,
    result: VerificationResult,
    now: datetime,
) -> FreshnessRecord:
    utc(now)
    canonical(result.evidence)
    if ref != current:
        raise StaleEvidence("verification was for a previous revision")
    if result.outcome == VerificationOutcome.CONFIRMED and not result.evidence:
        raise ValueError("confirmation requires evidence")
    if policy.ref != current:
        # Rebinding retains the slot policy but drops confirmation of the previous identity.
        policy = replace(
            policy,
            ref=current,
            checked_at=None,
            verified_until=now,
            retry_at=now,
            last_outcome=None,
            failure_count=0,
        )
    common: VerificationFields = dict(
        last_outcome=result.outcome,
        evidence=json.loads(canonical(result.evidence)),
        source_version=result.source_version,
    )
    if result.outcome == VerificationOutcome.UNAVAILABLE:
        count = policy.failure_count + 1
        delay = min(3600, 30 * 2 ** min(count - 1, 7))
        return replace(
            policy, failure_count=count, retry_at=now + timedelta(seconds=delay), **common
        )
    until = now + timedelta(seconds=policy.recheck_after_seconds)
    if result.outcome == VerificationOutcome.CHANGED:
        return replace(
            policy,
            checked_at=now,
            verified_until=min(policy.verified_until, now),
            retry_at=until,
            failure_count=0,
            **common,
        )
    return replace(
        policy, checked_at=now, verified_until=until, retry_at=until, failure_count=0, **common
    )


class ReferenceChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 10, 26, tzinfo=UTC)
        self.ref = EvidenceRef("owner", "f1", "rev1")
        self.views = (EvidenceView("owner", "f1", "rev1"),)
        ready = Intent(
            "i1",
            "owner",
            "key",
            "Ship",
            IntentState.READY,
            {},
            "",
            None,
            1,
            (self.ref,),
            None,
            None,
            self.now,
        )
        self.intent, self.lease = claim(
            ready, worker_id="w1", token="t1", ttl_seconds=60, now=self.now
        )

    def action(self) -> ActionRecord:
        return prepare_action(
            self.intent,
            None,
            action_id="a1",
            token="t1",
            action_key="publish",
            request={"v": 1},
            basis=(),
            views=self.views,
            now=self.now,
        )

    def test_checkpoint_cannot_orphan_started_action(self) -> None:
        started, _ = begin_action(
            self.intent, self.action(), (), token="t1", views=self.views, now=self.now
        )
        for target in IntentState:
            if target == IntentState.RUNNING:
                continue
            with self.subTest(target=target), self.assertRaises(RevisionConflict):
                checkpoint(
                    self.intent,
                    (started,),
                    token="t1",
                    expected_revision=self.intent.revision,
                    state=target,
                    progress={},
                    next_step="",
                    waiting_for=WaitUntil(self.now + timedelta(days=1))
                    if target == IntentState.WAITING
                    else None,
                    basis=self.intent.basis,
                    views=self.views,
                    now=self.now,
                )
        expired, actions = expire(self.intent, (started,), self.now + timedelta(seconds=60))
        self.assertEqual(expired.state, IntentState.BLOCKED)
        self.assertEqual(actions[0].state, ActionState.UNKNOWN)
        _, recovery = claim(
            expired,
            worker_id="recovery",
            token="t2",
            ttl_seconds=60,
            now=self.now + timedelta(seconds=60),
        )
        self.assertEqual(recovery.generation, self.lease.generation + 1)

    def test_unknown_checkpoint_only_blocks_or_explicitly_cancels(self) -> None:
        unknown = replace(self.action(), state=ActionState.UNKNOWN)
        for target in (IntentState.READY, IntentState.WAITING, IntentState.COMPLETED):
            with self.subTest(target=target), self.assertRaises(RevisionConflict):
                checkpoint(
                    self.intent,
                    (unknown,),
                    token="t1",
                    expected_revision=self.intent.revision,
                    state=target,
                    progress={},
                    next_step="",
                    waiting_for=WaitUntil(self.now + timedelta(days=1))
                    if target == IntentState.WAITING
                    else None,
                    basis=self.intent.basis,
                    views=self.views,
                    now=self.now,
                )
        blocked = checkpoint(
            self.intent,
            (unknown,),
            token="t1",
            expected_revision=self.intent.revision,
            state=IntentState.BLOCKED,
            progress={},
            next_step="Reconcile",
            waiting_for=None,
            basis=self.intent.basis,
            views=self.views,
            now=self.now,
        )
        self.assertIsNone(blocked.lease)
        self.assertEqual(blocked.blocked_reasons, ("unknown_action",))
        recovered, _ = claim(
            blocked, worker_id="recovery", token="t2", ttl_seconds=60, now=self.now
        )
        settled, _ = reconcile_action(
            recovered,
            unknown,
            (unknown,),
            token="t2",
            outcome="succeeded",
            evidence={"receipt": "confirmed"},
            views=self.views,
            now=self.now,
        )
        self.assertEqual(settled.state, IntentState.RUNNING)
        cancelled = checkpoint(
            self.intent,
            (unknown,),
            token="t1",
            expected_revision=self.intent.revision,
            state=IntentState.CANCELLED,
            progress={},
            next_step="",
            waiting_for=None,
            basis=self.intent.basis,
            views=self.views,
            now=self.now,
        )
        self.assertEqual(cancelled.state, IntentState.CANCELLED)

    def test_inherited_basis_and_single_permit(self) -> None:
        action = self.action()
        self.assertEqual(action.basis, self.intent.basis)
        with self.assertRaises(StaleEvidence):
            begin_action(
                self.intent,
                action,
                (action,),
                token="t1",
                views=(EvidenceView("owner", "f1", "rev2"),),
                now=self.now,
            )
        started, permit = begin_action(
            self.intent, action, (action,), token="t1", views=self.views, now=self.now
        )
        self.assertEqual(permit.idempotency_key, action.idempotency_key)
        with self.assertRaises(RevisionConflict):
            begin_action(
                self.intent, started, (started,), token="t1", views=self.views, now=self.now
            )

    def test_recovery_lease_unknown_and_stale_token(self) -> None:
        started, _ = begin_action(
            self.intent, self.action(), (), token="t1", views=self.views, now=self.now
        )
        later = self.now + timedelta(seconds=60)
        packet = resume(self.intent, (started,), self.views, condition_met=False, now=later)
        self.assertEqual(packet.open_actions[0].state, ActionState.UNKNOWN)
        self.assertFalse(packet.ready)
        recovered, lease = claim(
            packet.intent, worker_id="w2", token="t2", ttl_seconds=60, now=later
        )
        self.assertEqual(recovered.state, IntentState.BLOCKED)
        self.assertEqual(lease.generation, 2)
        with self.assertRaises(LeaseLost):
            finish_action(recovered, started, token="t1", outcome="succeeded", result={}, now=later)
        with self.assertRaises(RevisionConflict):
            begin_action(
                recovered,
                self.action(),
                packet.open_actions,
                token="t2",
                views=self.views,
                now=later,
            )
        running, reconciled = reconcile_action(
            recovered,
            packet.open_actions[0],
            packet.open_actions,
            token="t2",
            outcome="retryable",
            evidence={"lookup": "absent"},
            views=self.views,
            now=later,
        )
        self.assertEqual(running.state, IntentState.RUNNING)
        self.assertEqual(reconciled.state, ActionState.PREPARED)
        self.assertEqual(reconciled.idempotency_key, started.idempotency_key)
        self.assertEqual(running.lease, lease)

    def test_freshness_expiry_is_not_retry_schedule(self) -> None:
        policy = set_freshness(
            self.ref, slot_key="endpoint", source_ref="config", interval=60, now=self.now
        )
        self.assertTrue(freshness_blocks(policy, self.ref, self.now))
        policy = record_verification(
            policy,
            self.ref,
            current=self.ref,
            now=self.now,
            result=VerificationResult(VerificationOutcome.CONFIRMED, {"etag": "v1"}, "v1"),
        )
        later = self.now + timedelta(seconds=61)
        failed = record_verification(
            policy,
            self.ref,
            current=self.ref,
            now=later,
            result=VerificationResult(VerificationOutcome.UNAVAILABLE, {"timeout": True}, ""),
        )
        self.assertEqual(failed.verified_until, policy.verified_until)
        self.assertEqual(failed.retry_at, later + timedelta(seconds=30))
        self.assertTrue(freshness_blocks(failed, self.ref, later))
        self.assertFalse(verification_due(failed, self.ref, later))
        new_ref = EvidenceRef("owner", "f2", "rev2")
        self.assertTrue(verification_due(failed, new_ref, self.now))
        self.assertTrue(freshness_blocks(failed, new_ref, self.now))
        with self.assertRaises(StaleEvidence):
            record_verification(
                failed,
                self.ref,
                current=new_ref,
                now=later,
                result=VerificationResult(VerificationOutcome.CONFIRMED, {"etag": "v1"}, "v1"),
            )

    def test_checkpoint_cas_and_lease_expiry(self) -> None:
        with self.assertRaises(LeaseLost):
            renew(self.intent, token="t1", ttl_seconds=60, now=self.now + timedelta(seconds=60))
        with self.assertRaises(RevisionConflict):
            checkpoint(
                self.intent,
                (),
                token="t1",
                expected_revision=1,
                state=IntentState.READY,
                progress={},
                next_step="",
                waiting_for=None,
                basis=(self.ref,),
                views=self.views,
                now=self.now,
            )
        packet = resume(
            self.intent, (), self.views, condition_met=False, now=self.now + timedelta(seconds=60)
        )
        self.assertTrue(packet.ready)
        self.assertIsNone(packet.intent.lease)

    def test_blocked_replanning_and_deadline(self) -> None:
        blocked = replace(self.intent, state=IntentState.BLOCKED, blocked_reasons=("changed",))
        replanned = checkpoint(
            blocked,
            (),
            token="t1",
            expected_revision=blocked.revision,
            state=IntentState.READY,
            progress={},
            next_step="new plan",
            waiting_for=None,
            basis=(self.ref,),
            views=self.views,
            now=self.now,
        )
        self.assertEqual(replanned.blocked_reasons, ())
        deadline = replace(self.intent, deadline=self.now)
        with self.assertRaises(RevisionConflict):
            checkpoint(
                deadline,
                (),
                token="t1",
                expected_revision=deadline.revision,
                state=IntentState.READY,
                progress={},
                next_step="",
                waiting_for=None,
                basis=(),
                views=(),
                now=self.now,
            )

    def test_json_and_acl(self) -> None:
        with self.assertRaises(ValueError):
            canonical({"x": float("nan")})
        with self.assertRaises(ValueError):
            canonical({"x": "я" * 32768})
        (issue,) = validate_basis(
            (self.ref,), (EvidenceView("owner", "f1", "rev1", readable=False),), self.now
        )
        self.assertEqual(issue, BasisIssue("access_changed", None))

    def test_reprepare_and_finish_idempotency(self) -> None:
        action = self.action()
        new_ref = EvidenceRef("owner", "f1", "rev2")
        updated = replace(self.intent, basis=(new_ref,))
        views = (EvidenceView("owner", "f1", "rev2"),)
        rebased = prepare_action(
            updated,
            action,
            action_id="unused",
            token="t1",
            action_key="publish",
            request={"v": 1},
            basis=(),
            views=views,
            now=self.now,
        )
        self.assertEqual(rebased.id, action.id)
        self.assertEqual(rebased.basis, (new_ref,))
        self.assertEqual(rebased.revision, action.revision + 1)
        with self.assertRaises(IdempotencyConflict):
            prepare_action(
                updated,
                action,
                action_id="unused",
                token="t1",
                action_key="publish",
                request={"v": 2},
                basis=(),
                views=views,
                now=self.now,
            )
        started, _ = begin_action(
            updated, rebased, (rebased,), token="t1", views=views, now=self.now
        )
        done = finish_action(
            updated, started, token="t1", outcome="succeeded", result={"ok": True}, now=self.now
        )
        self.assertEqual(
            done,
            finish_action(
                updated, done, token="t1", outcome="succeeded", result={"ok": True}, now=self.now
            ),
        )
        with self.assertRaises(RevisionConflict):
            finish_action(
                updated, done, token="t1", outcome="succeeded", result={"ok": False}, now=self.now
            )
        old_basis = prepare_action(
            self.intent,
            done,
            action_id="unused",
            token="t1",
            action_key="publish",
            request={"v": 1},
            basis=(),
            views=self.views,
            now=self.now,
        )
        self.assertEqual(old_basis, done)

    def test_historical_stale_basis_does_not_discard_reconciliation(self) -> None:
        action = replace(self.action(), state=ActionState.UNKNOWN)
        new_ref = EvidenceRef("owner", "f1", "rev2")
        blocked = replace(
            self.intent,
            state=IntentState.BLOCKED,
            basis=(new_ref,),
            blocked_reasons=("unknown_action",),
        )
        current, done = reconcile_action(
            blocked,
            action,
            (action,),
            token="t1",
            outcome="succeeded",
            evidence={"remote": "success"},
            views=(EvidenceView("owner", "f1", "rev2"),),
            now=self.now,
        )
        self.assertEqual(done.state, ActionState.SUCCEEDED)
        self.assertEqual(current.state, IntentState.BLOCKED)
        self.assertEqual(current.blocked_reasons, ("changed",))
        self.assertEqual(done.basis, (self.ref,))
        self.assertEqual(
            (current, done),
            reconcile_action(
                current,
                done,
                (done,),
                token="t1",
                outcome="succeeded",
                evidence={"remote": "success"},
                views=self.views,
                now=self.now,
            ),
        )
        with self.assertRaises(RevisionConflict):
            reconcile_action(
                current,
                done,
                (done,),
                token="t1",
                outcome="succeeded",
                evidence={"remote": "different"},
                views=self.views,
                now=self.now,
            )

    def test_unknown_effect_blocks_another_action_key(self) -> None:
        action = self.action()
        unknown = replace(action, id="other", action_key="previous", state=ActionState.UNKNOWN)
        with self.assertRaises(RevisionConflict):
            begin_action(
                self.intent, action, (unknown, action), token="t1", views=self.views, now=self.now
            )

    def test_waiting_deadline_and_terminals(self) -> None:
        wait = checkpoint(
            self.intent,
            (),
            token="t1",
            expected_revision=self.intent.revision,
            state=IntentState.WAITING,
            progress={"saved": True},
            next_step="CI",
            waiting_for=WaitUntil(self.now + timedelta(seconds=10)),
            basis=(self.ref,),
            views=self.views,
            now=self.now,
        )
        self.assertFalse(resume(wait, (), self.views, condition_met=False, now=self.now).ready)
        self.assertTrue(
            resume(
                wait, (), self.views, condition_met=True, now=self.now + timedelta(seconds=10)
            ).ready
        )
        expired, _ = expire(replace(wait, deadline=self.now), (), self.now)
        self.assertEqual(expired.state, IntentState.BLOCKED)
        self.assertIsNone(expired.waiting_for)
        terminal = replace(wait, state=IntentState.COMPLETED, waiting_for=None)
        self.assertEqual(
            resume(terminal, (), self.views, condition_met=True, now=self.now).intent, terminal
        )

    def test_outage_blocks_before_expiry_and_backoff_caps(self) -> None:
        policy = set_freshness(
            self.ref, slot_key="", source_ref="source", interval=3600, now=self.now
        )
        policy = record_verification(
            policy,
            self.ref,
            current=self.ref,
            now=self.now,
            result=VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "v1"),
        )
        valid_until = policy.verified_until
        failed = VerificationResult(VerificationOutcome.UNAVAILABLE, {"timeout": True}, "")
        for _ in range(20):
            policy = record_verification(
                policy, self.ref, current=self.ref, result=failed, now=self.now
            )
        self.assertEqual(policy.verified_until, valid_until)
        self.assertTrue(freshness_blocks(policy, self.ref, self.now))
        self.assertEqual(policy.retry_at, self.now + timedelta(seconds=3600))


if __name__ == "__main__":
    unittest.main()
