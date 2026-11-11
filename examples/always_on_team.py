"""One-shot host recipe for event/timer driven teams; no resident worker or model.

Run: PYTHONPATH=src python examples/always_on_team.py
Inspection is a candidate for host work, never a lease or action permit.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import (
    AlwaysOnMemory,
    ChangeBatch,
    CursorExpired,
    EvidenceRef,
    IntentState,
    ResumePacket,
    RevisionConflict,
    SemanticBudget,
    SemanticCandidates,
    SemanticContext,
    SnapshotTooLarge,
    StaleEvidence,
    TaskContext,
    TaskContextGuard,
    WaitUntil,
    WatchSelector,
    WorkspaceMemory,
)
from ai_knot.always_on.codec import canonical_json, encode_always_on, loads_strict
from ai_knot.always_on.models import canonical, context_label, positive_integer
from ai_knot.storage.base import AtomicUpdateCapable, StorageBackend
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact, MemoryType


class Inbox:
    """Host-owned cache and latest receipt share one durable local transaction."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS inbox (
                owner TEXT, subscription TEXT, delivery_id TEXT, context_json TEXT,
                PRIMARY KEY(owner,subscription))""")

    def saved_refs(self, owner: str, subscription: str) -> tuple[EvidenceRef, ...]:
        with closing(sqlite3.connect(self.path)) as conn:
            row = conn.execute(
                "SELECT context_json FROM inbox WHERE owner=? AND subscription=?",
                (owner, subscription),
            ).fetchone()
        return (
            ()
            if row is None
            else tuple(EvidenceRef(**item["ref"]) for item in json.loads(row[0])["facts"])
        )

    def save(
        self, owner: str, subscription: str, context: TaskContext, delivery_id: str | None = None
    ) -> None:
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                """INSERT INTO inbox VALUES(?,?,?,?) ON CONFLICT(owner,subscription)
                DO UPDATE SET context_json=excluded.context_json,
                delivery_id=COALESCE(excluded.delivery_id,inbox.delivery_id)""",
                (owner, subscription, delivery_id, json.dumps(encode_always_on(context))),
            )


@dataclass(frozen=True)
class TurnObservation:
    status: str
    context: TaskContext
    packet: ResumePacket
    pages: int
    snapshots: int
    rejected_saved: int
    through_seq: int
    head_seq: int

    @property
    def wake_model(self) -> bool:
        return self.status in ("ready", "replan")

    def diagnostics(self) -> dict[str, object]:
        """No source content, IDs, goal text, lease tokens or credentials."""
        return {
            "status": self.status,
            "wake_model": self.wake_model,
            "pages": self.pages,
            "snapshots": self.snapshots,
            "rejected_saved": self.rejected_saved,
            "context_facts": len(self.context.facts),
            "missing_requirements": len(self.context.missing_requirements),
            "budget_limited": self.context.budget_limited,
            "through_seq": self.through_seq,
            "head_seq": self.head_seq,
            "reasons": self.packet.reasons,
        }


def inspect_turn(
    memory: AlwaysOnMemory,
    inbox: Inbox,
    *,
    intent_id: str,
    subscription_id: str,
    requirements: tuple[str, ...],
    task_key: str = "",
    role: str = "",
    max_pages: int = 4,
    max_facts: int = 3,
    max_chars: int = 600,
) -> TurnObservation:
    if type(max_pages) is not int or not 1 <= max_pages <= 20:
        raise ValueError("max_pages must be an integer in 1..20")
    owner = memory.get_intent(intent_id).owner_agent_id
    saved = memory.review_context(inbox.saved_refs(owner, subscription_id), max_chars=max_chars)

    def select() -> TaskContext:
        return memory.select_context(
            requirements, task_key=task_key, role=role, max_facts=max_facts, max_chars=max_chars
        )

    # Validate the task/budget before touching delivery progress.
    context = select()
    last = ChangeBatch(None, ())
    status = "catching_up"
    snapshots = 0
    pages = 0
    for _ in range(max_pages):
        pages += 1
        try:
            try:
                last = memory.poll(subscription_id)
            except CursorExpired:
                last = memory.resync(subscription_id)
            if last.resync_required:
                last = memory.resync(subscription_id)
        except SnapshotTooLarge:
            status = "snapshot_too_large"
            break
        snapshots += int(last.is_snapshot)
        context = select()
        # Replacing context is idempotent; never ack before this host commit.
        inbox.save(owner, subscription_id, context, last.delivery_id)
        if last.delivery_id is not None:
            memory.ack(subscription_id, delivery_id=last.delivery_id)
        if not last.has_more:
            status = "caught_up"
            break
        if last.delivery_id is None:
            status = "debounced"
            break

    packet = memory.resume(intent_id)
    context = select()
    if status != "snapshot_too_large":
        inbox.save(owner, subscription_id, context)
    if status == "caught_up":
        if packet.intent.state in ("completed", "cancelled"):
            status = packet.intent.state.value
        elif packet.intent.lease is not None:
            status = "lease_busy"
        elif "unknown_action" in packet.reasons:
            status = "reconcile"
        elif "deadline_exceeded" in packet.reasons:
            status = "deadline_exceeded"
        elif "host_blocked" in packet.reasons:
            status = "blocked"
        elif packet.freshness_due:
            status = "verification_due"
        elif not context.complete:
            status = "missing_context"
        elif packet.stale_basis:
            status = "replan"
        elif packet.ready:
            status = "ready"
        else:
            status = packet.intent.state.value
    return TurnObservation(
        status,
        context,
        packet,
        pages,
        snapshots,
        len(saved.issues),
        last.through_seq,
        last.head_seq,
    )


async def acquire_missing_context(
    turn: TurnObservation,
    selector: SemanticContext,
    *,
    relevant_event: bool,
    budget: SemanticBudget,
    task_key: str = "",
    role: str = "",
    max_facts: int = 3,
    max_chars: int = 600,
) -> SemanticCandidates | None:
    """Opt-in candidates only; preserve lexical context, status and action guards."""
    if turn.status != "missing_context" or relevant_event is not True:
        return None
    return await selector.acquire(
        turn.context.missing_requirements,
        budget=budget,
        task_key=task_key,
        role=role,
        max_facts=max_facts,
        max_chars=max_chars,
    )


@dataclass(frozen=True)
class PlanningReport:
    status: str
    model_calls_reserved: int = 0
    proposal: dict[str, object] | None = None
    error_type: str | None = None
    context_guard: TaskContextGuard | None = None
    acquisition_calls_reserved: int = 0


_PLANNING = "_ai_knot_planning"


def _planning_allowed(packet: ResumePacket, token: str | None = None) -> bool:
    lease = packet.intent.lease
    if (lease is None and token is not None) or (lease is not None and lease.token != token):
        return False
    if packet.open_actions or packet.freshness_due:
        return False
    states = (IntentState.READY, IntentState.BLOCKED)
    if packet.intent.state not in (*states, *((IntentState.RUNNING,) if token else ())):
        return False
    allowed = {
        "blocked",
        "changed",
        "missing",
        "inactive",
        "unsupported",
        "access_changed",
        "verification_due",
    }
    if token:
        allowed.update(("live_lease", "running"))
    return set(packet.reasons) <= allowed


def _planning_proposal(raw: str, facts):
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 16384:
        raise ValueError("Invalid planning output")
    try:
        value = loads_strict(raw)
        if not isinstance(value, dict) or set(value) != {"action", "next_step", "basis", "data"}:
            raise ValueError("Invalid planning output")
        if value["action"] not in ("plan", "wait") or not isinstance(value["data"], dict):
            raise ValueError("Invalid planning output")
        if (
            not isinstance(value["next_step"], str)
            or not value["next_step"].strip()
            or len(value["next_step"]) > 4096
        ):
            raise ValueError("Invalid next step")
        indices = value["basis"]
        if (
            not isinstance(indices, list)
            or any(type(i) is not int or not 1 <= i <= len(facts) for i in indices)
            or len(set(indices)) != len(indices)
            or (value["action"] == "plan" and not indices)
        ):
            raise ValueError("Invalid planning basis")
        canonical(value)
        return value, tuple(facts[i - 1].ref for i in indices)
    except (RecursionError, TypeError):
        raise ValueError("Invalid planning output") from None


async def plan_once(
    memory: AlwaysOnMemory,
    turn: TurnObservation,
    planner: Callable[[dict[str, object]], Awaitable[str]],
    *,
    trigger_key: str | None,
    requirements: tuple[str, ...],
    worker_id: str,
    task_key: str = "",
    max_model_calls: int = 0,
    max_facts: int = 8,
    max_chars: int = 12000,
    timeout_seconds: float = 30,
    selector: SemanticContext | None = None,
    semantic_budget: SemanticBudget | None = None,
    required_refs: tuple[EvidenceRef, ...] = (),
) -> PlanningReport:
    """Persist one proposed step, never an effect or a semantic-readiness verdict.

    The callback owns aggregate token/request metering. A crash before the
    attempted-input marker is saved can repeat inference after lease recovery.
    """
    if type(max_model_calls) is not int or max_model_calls not in (0, 1):
        raise ValueError("max_model_calls must be 0 or 1")
    if (
        type(timeout_seconds) not in (float, int)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 60
    ):
        raise ValueError("timeout_seconds must be finite in (0,60]")
    positive_integer(max_facts, "max_facts", 128)
    positive_integer(max_chars, "max_chars", 65536)
    context_label(task_key, "task_key")
    if selector is not None and not isinstance(selector, SemanticContext):
        raise ValueError("selector must be SemanticContext")
    semantic_budget = SemanticBudget() if semantic_budget is None else semantic_budget
    if not isinstance(semantic_budget, SemanticBudget) or semantic_budget.timeout > 60:
        raise ValueError("SemanticBudget with timeout at most60seconds required")
    from ai_knot.always_on.context import validate_required_refs

    required_refs = validate_required_refs(required_refs)
    if (
        not isinstance(requirements, (tuple, list))
        or not 1 <= len(requirements) <= 5
        or any(not isinstance(r, str) or not r.strip() or len(r) > 4096 for r in requirements)
        or len(set(requirements)) != len(requirements)
        or not callable(planner)
    ):
        raise ValueError("Invalid planning requirements or callback")
    if trigger_key is None:
        return PlanningReport("idle")
    context_label(trigger_key, "trigger_key")
    if not trigger_key:
        raise ValueError("trigger_key must be nonempty")
    if turn.status not in ("ready", "replan", "missing_context"):
        return PlanningReport("skipped")
    if not max_model_calls:
        return PlanningReport("budget_exhausted")
    intent_id = turn.packet.intent.id
    packet = memory.resume(intent_id)
    if not _planning_allowed(packet):
        return PlanningReport("blocked")
    try:
        lease = memory.claim(intent_id, worker_id=worker_id, ttl_seconds=300)
    except RevisionConflict:
        return PlanningReport("lease_busy")
    reserved = 0
    acquired = 0
    input_key = None
    guard = None

    async def work() -> PlanningReport:
        nonlocal reserved, acquired, input_key, guard
        current = memory.resume(intent_id)
        if not _planning_allowed(current, lease.token):
            return PlanningReport("blocked")
        try:
            guard = memory.task_guard(task_key=task_key)
        except SnapshotTooLarge:
            return PlanningReport("snapshot_too_large")
        progress = {k: v for k, v in current.intent.progress.items() if k != _PLANNING}
        identity = {
            "policy": "guarded-planning-v2",
            "goal": current.intent.goal,
            "requirements": list(requirements),
            "progress": progress,
            "guard": guard,
            "trigger": trigger_key,
            "selection": selector is not None,
            "max_facts": max_facts,
            "max_chars": max_chars,
        }
        if required_refs:
            identity["required_refs"] = required_refs
        input_key = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        previous = current.intent.progress.get(_PLANNING)
        if isinstance(previous, dict) and previous.get("input_key") == input_key:
            return PlanningReport("unchanged")
        try:
            facts = memory.task_snapshot(
                task_key=task_key, max_facts=max_facts, max_chars=max_chars
            )
        except SnapshotTooLarge:
            if selector is None:
                return PlanningReport("snapshot_too_large")
            if not _planning_allowed(memory.resume(intent_id), lease.token):
                return PlanningReport("blocked")
            # Unknown cancellation keeps this conservative reservation. A normal
            # return replaces it with the selector's actual local reservation.
            acquired = int(semantic_budget.max_provider_calls > 0)
            try:
                candidates = await selector.acquire(
                    requirements,
                    budget=semantic_budget,
                    task_key=task_key,
                    max_facts=max_facts,
                    max_chars=max_chars,
                    required_refs=required_refs,
                )
            except (StaleEvidence, SnapshotTooLarge) as exc:
                # Native selector raises these required-input refusals before I/O.
                acquired = 0
                input_key = None
                return PlanningReport(
                    "context_changed" if isinstance(exc, StaleEvidence) else "snapshot_too_large"
                )
            acquired = candidates.provider_calls_reserved
            if candidates.status not in ("snapshot", "ok") or candidates.review.issues:
                if not acquired:
                    input_key = None  # A zero-call refusal did not spend the event.
                return PlanningReport(
                    "context_changed"
                    if candidates.status == "changed"
                    else "acquisition_" + candidates.status
                )
            try:
                if memory.task_guard(task_key=task_key) != guard:
                    return PlanningReport("context_changed")
            except SnapshotTooLarge:
                return PlanningReport("context_changed")
            facts = candidates.review.accepted
        if any(ref not in {f.ref for f in facts} for ref in required_refs):
            input_key = None
            return PlanningReport("context_changed")
        if not facts:
            return PlanningReport("no_context")
        checked = memory.review_context(tuple(f.ref for f in facts), max_chars=max_chars)
        if not checked.complete:
            return PlanningReport("context_changed")
        facts = checked.accepted
        payload = {
            "goal": current.intent.goal,
            "requirements": list(requirements),
            "progress": progress,
            "facts": [{"index": i + 1, "content": f.content} for i, f in enumerate(facts)],
        }
        if len(canonical_json(payload).encode("utf-8")) > 65536:
            return PlanningReport("input_limit")
        if not _planning_allowed(memory.resume(intent_id), lease.token):
            return PlanningReport("blocked")
        reserved = 1
        try:
            raw = await asyncio.wait_for(
                planner(json.loads(canonical_json(payload))), timeout_seconds
            )
        except Exception as exc:
            return PlanningReport("unavailable", reserved, error_type=type(exc).__name__)
        try:
            proposal, refs = _planning_proposal(raw, facts)
            refs = tuple(dict.fromkeys((*required_refs, *refs)))
        except (ValueError, UnicodeError) as exc:
            return PlanningReport("invalid_proposal", reserved, error_type=type(exc).__name__)
        current = memory.resume(intent_id)
        if not _planning_allowed(current, lease.token):
            return PlanningReport("blocked", reserved)
        if {k: v for k, v in current.intent.progress.items() if k != _PLANNING} != progress:
            return PlanningReport("context_changed", reserved)
        checked = memory.review_context(refs, max_chars=max_chars)
        if not checked.complete:
            return PlanningReport("context_changed", reserved)
        progress = dict(current.intent.progress)
        progress[_PLANNING] = {
            "input_key": input_key,
            "trigger_key": trigger_key,
            "proposal": proposal,
            "context_guard": encode_always_on(guard),
        }
        waiting = proposal["action"] == "wait"
        try:
            memory.checkpoint(
                intent_id,
                lease_token=lease.token,
                expected_revision=current.intent.revision,
                state=IntentState.WAITING if waiting else IntentState.READY,
                progress=progress,
                next_step=proposal["next_step"],
                basis=refs,
                waiting_for=WaitUntil(checked.checked_at + timedelta(seconds=60))
                if waiting
                else None,
                context_guard=guard,
            )
        except StaleEvidence:
            return PlanningReport("context_changed", reserved)
        return PlanningReport(
            "waiting" if waiting else "planned", reserved, proposal, context_guard=guard
        )

    def settle(report: PlanningReport) -> PlanningReport:
        if report.status in ("planned", "waiting"):
            return report
        try:
            current = memory.resume(intent_id)
            if not _planning_allowed(current, lease.token):
                return report
            progress = dict(current.intent.progress)
            unchanged = report.status == "unchanged"
            if not unchanged:
                progress[_PLANNING] = {
                    "input_key": input_key,
                    "trigger_key": trigger_key,
                    "status": report.status,
                    "model_calls_reserved": reserved,
                    "acquisition_calls_reserved": acquired,
                }
            memory.checkpoint(
                intent_id,
                lease_token=lease.token,
                expected_revision=current.intent.revision,
                state=IntentState.READY,
                progress=progress,
                next_step=current.intent.next_step if unchanged else "",
                basis=current.intent.basis if unchanged else (),
                context_guard=guard if unchanged else None,
            )
        except Exception as exc:
            return PlanningReport("failed", reserved, error_type=type(exc).__name__)
        return report

    try:
        report = await work()
    except asyncio.CancelledError:
        if not reserved and not acquired:
            input_key = None
        settle(PlanningReport("cancelled", reserved))
        raise
    except Exception as exc:
        if not reserved and not acquired:
            input_key = None
        report = PlanningReport("failed", reserved, error_type=type(exc).__name__)
    return replace(settle(report), acquisition_calls_reserved=acquired)


def example(
    directory: Path, *, make_store: Callable[[], StorageBackend] | None = None
) -> dict[str, object]:
    """Scripted host decisions exercise one team corpus; no real model/effect."""
    with (
        patch("ai_knot.embedder.embed_texts", side_effect=AssertionError("embedding call")),
        patch("httpx.Client.send", side_effect=AssertionError("inference HTTP call")),
        patch("httpx.AsyncClient.send", side_effect=AssertionError("inference HTTP call")),
    ):
        return _example(
            directory, make_store or (lambda: SQLiteStorage(str(directory / "team.db")))
        )


def _example(directory: Path, make_store: Callable[[], StorageBackend]) -> dict[str, object]:
    now = [datetime.now(UTC)]

    def clock() -> datetime:
        return now[0]

    store = make_store()
    assert isinstance(store, AtomicUpdateCapable)
    workspace = WorkspaceMemory(store, "rollout-team")
    workspace.grant("planner", "execute")
    workspace.grant("planner", "maintain")
    grants = SharedMemoryPool(store, persist_stats=True)
    for owner in ("planner", "reviewer"):
        grants.grant_read(owner, "team:ops")

    def agent(owner: str) -> AlwaysOnMemory:
        independent = make_store()
        return AlwaysOnMemory(
            KnowledgeBase(owner, storage=independent, embed_url=""),
            pool=SharedMemoryPool(independent, persist_stats=True),
            workspace=WorkspaceMemory(independent, "rollout-team"),
            clock=clock,
        )

    def source(content: str, identity: str, slot: str = "") -> Fact:
        return Fact(
            content,
            id=identity,
            slot_key=slot,
            created_at=clock(),
            last_accessed=clock(),
            valid_from=clock(),
            origin_agent_id="researcher",
            visibility_scope="team:ops",
        )

    def put(*facts: Fact) -> None:
        # A trusted connector publishes observations under the shared SQL writer lock.
        ids = {f.id for f in facts}
        store.atomic_update(
            "__shared__", lambda old: [f for f in old if f.id not in ids] + list(facts)
        )

    target = source("Deployment target EU", "target", "deployment::target")
    backup = source("Backup receipt verified", "backup", "backup::receipt")
    put(target, backup)
    planner = agent("planner")
    intent = planner.create_intent(client_key="rollout42", goal="Inspect rollout after timer")
    planner.subscribe(
        subscription_id="team",
        selector=WatchSelector("__shared__", slot_keys=("deployment::target", "backup::receipt")),
    )
    inbox = Inbox(directory / "host-inbox.db")
    basis = planner.capture_evidence((("__shared__", target.id), ("__shared__", backup.id)))
    lease = planner.claim(intent.id, worker_id="host", ttl_seconds=300)
    planner.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=planner.get_intent(intent.id).revision,
        state=IntentState.WAITING,
        progress={},
        waiting_for=WaitUntil(clock() + timedelta(seconds=60)),
        basis=basis,
    )

    def inspect(**kwargs: int) -> TurnObservation:
        return inspect_turn(
            planner,
            inbox,
            intent_id=intent.id,
            subscription_id="team",
            requirements=("deployment target", "backup receipt"),
            task_key="rollout42",
            role="operator",
            **kwargs,
        )

    quiet = [inspect() for _ in range(5)]
    assert all(t.status == "waiting" and not t.wake_model for t in quiet)
    put(*(source(f"Inventory bulletin {i}", f"noise-{i}") for i in range(1200)))
    target = replace(target, content="Deployment target US")
    put(target)
    now[0] += timedelta(seconds=61)
    planner = agent("planner")  # Fresh storage/facade after sleep; cache refs are durable.
    backlog = inspect(max_pages=1)
    assert backlog.status == "catching_up" and not backlog.wake_model
    replan = inspect()
    assert replan.status == "replan" and replan.wake_model
    target = replace(target, content="Deployment target US confirmed")
    put(target)
    planner.prune_changes("__shared__", force=True)
    recovered = inspect()
    assert recovered.snapshots == 1 and recovered.status == "replan"

    def checkpoint_ready(token: str, context: TaskContext) -> None:
        planner.checkpoint(
            intent.id,
            lease_token=token,
            expected_revision=planner.get_intent(intent.id).revision,
            state=IntentState.READY,
            progress={"scripted_replan": True},
            basis=tuple(f.ref for f in context.facts),
        )

    # A real host claims before paying for a model. Here the decision is scripted.
    lease = planner.claim(intent.id, worker_id="host", ttl_seconds=300)
    checkpoint_ready(lease.token, recovered.context)
    lease = planner.claim(intent.id, worker_id="host", ttl_seconds=300)
    action = planner.prepare_action(
        intent.id,
        lease_token=lease.token,
        action_key="rollout",
        request={"target": "US"},
        operation_key="rollout42",
        resource_key="production",
    )
    backup = replace(backup, content="Backup receipt verified revision two")
    put(backup)
    try:
        planner.begin_action(action.id, lease_token=lease.token)
    except StaleEvidence:
        stale_rejected = True
    else:
        raise AssertionError("Changed backup evidence must reject the prepared action")

    fresh = inspect()
    # The host still owns its lease; it can checkpoint fresh evidence and re-prepare.
    assert fresh.status == "lease_busy"
    checkpoint_ready(lease.token, fresh.context)
    lease = planner.claim(intent.id, worker_id="host", ttl_seconds=300)
    action = planner.prepare_action(
        intent.id,
        lease_token=lease.token,
        action_key="rollout",
        request={"target": "US"},
        operation_key="rollout42",
        resource_key="production",
    )
    planner.begin_action(action.id, lease_token=lease.token)
    action = planner.finish_action(
        action.id,
        lease_token=lease.token,
        outcome="succeeded",
        result={"scripted_connector": "verified"},
    )
    receipt = replace(
        source("Host receipt: rollout verification succeeded", "receipt"), type=MemoryType.EPISODIC
    )
    put(receipt)
    experience = planner.record_experience(
        action.id,
        expected_action_revision=action.revision,
        experience_key="verified-rollout",
        content="Rollout lesson: verify backup before traffic.",
        basis=planner.capture_evidence((("__shared__", receipt.id),)),
        namespace="__shared__",
        visibility_scope="team:ops",
        task_key="rollout42",
        for_role="reviewer",
    )
    reviewer = agent("reviewer")
    peer_context = reviewer.select_context(
        ("rollout lesson",), task_key="rollout42", role="reviewer"
    )
    assert experience.lesson.ref in {item.ref for item in peer_context.facts}
    put(replace(receipt, content="Host receipt withdrawn after an independent check"))
    source_retracted = not reviewer.review_context((experience.lesson.ref,)).accepted

    probe = planner.prepare_action(
        intent.id, lease_token=lease.token, action_key="probe", request={}
    )
    planner.begin_action(probe.id, lease_token=lease.token)
    probe = planner.finish_action(
        probe.id, lease_token=lease.token, outcome="failed", result={"scripted_probe": "failed"}
    )
    warning = planner.record_experience(
        probe.id,
        expected_action_revision=probe.revision,
        experience_key="probe",
        content="Probe warning: inspect the recorded failure.",
        basis=planner.capture_evidence((("__shared__", backup.id),)),
        namespace="__shared__",
        visibility_scope="team:ops",
        task_key="rollout42",
    )
    assert reviewer.review_context((warning.lesson.ref,)).complete
    planner.reconcile_action(
        probe.id,
        lease_token=lease.token,
        outcome="succeeded",
        evidence={"scripted_independent_check": "corrected"},
    )
    outcome_retracted = not reviewer.review_context((warning.lesson.ref,)).accepted
    planner.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=planner.get_intent(intent.id).revision,
        state=IntentState.COMPLETED,
        progress={"scripted_checks": True},
    )
    assert source_retracted and outcome_retracted
    return {
        "quiet_ticks": len(quiet),
        "idle_model_wakes": sum(t.wake_model for t in quiet),
        "backlog_deferred": not backlog.wake_model,
        "resyncs": recovered.snapshots,
        "stale_action_rejected": stale_rejected,
        "peer_experience_reused": True,
        "source_retraction": source_retracted,
        "outcome_retraction": outcome_retracted,
        "completed": inspect().status == "completed",
        "model_calls": 0,
        "diagnostics": [backlog.diagnostics(), replan.diagnostics(), recovered.diagnostics()],
        "scope": "Scripted host decisions; remote effects and model quality unmeasured",
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="knot-team-cycle-") as temporary:
        print(json.dumps(example(Path(temporary)), sort_keys=True, indent=2))
