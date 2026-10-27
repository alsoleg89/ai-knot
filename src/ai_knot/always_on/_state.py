"""Pure state transitions; callers provide facts projected within their transaction."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Literal, TypedDict, Unpack

from .models import (
    ActionPermit,
    ActionRecord,
    ActionState,
    BasisIssue,
    EvidenceRef,
    FreshnessRecord,
    IdempotencyConflict,
    Intent,
    IntentState,
    JSONObject,
    Lease,
    LeaseLost,
    ResumePacket,
    RevisionConflict,
    StaleEvidence,
    VerificationOutcome,
    VerificationResult,
    WaitCondition,
    WaitUntil,
    canonical,
    digest,
    utc,
)


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
        IntentState.BLOCKED: {
            IntentState.READY,
            IntentState.WAITING,
            IntentState.CANCELLED,
            *(
                {IntentState.BLOCKED}
                if any(a.state == ActionState.UNKNOWN for a in actions)
                else set()
            ),
        },
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
