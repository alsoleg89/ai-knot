"""Exercise durable state through public APIs, including abrupt process death."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_knot.always_on import (
    ActionState,
    AlwaysOnMemory,
    IntentState,
    LeaseLost,
    StaleEvidence,
    VerificationOutcome,
    VerificationResult,
    WaitUntil,
    WatchSelector,
)
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.base import StorageBackend
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact

NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)


def external_effect(path: Path, key: str) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY)")
        conn.execute("INSERT OR IGNORE INTO effects(key) VALUES (?)", (key,))


def external_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    with closing(sqlite3.connect(path)) as conn:
        return [row[0] for row in conn.execute("SELECT key FROM effects")]


def storage_config(storage: StorageBackend) -> dict[str, str]:
    if isinstance(storage, SQLiteStorage):
        return {"kind": "sqlite", "location": storage._db_path}
    return {"kind": "postgres", "location": storage._dsn}  # type: ignore[attr-defined]


def open_storage(config: dict[str, str]) -> StorageBackend:
    if config["kind"] == "sqlite":
        return SQLiteStorage(config["location"])
    from ai_knot.storage.postgres_storage import PostgresStorage

    return PostgresStorage(config["location"])


def crash_worker(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    storage = open_storage(config["storage"])
    memory = AlwaysOnMemory(
        KnowledgeBase(config["owner"], storage=storage, embed_url=""), clock=lambda: NOW
    )
    audit_path = Path(config["audit_path"])
    audit: dict[str, Any] = {}

    def save() -> None:
        audit_path.write_text(json.dumps(audit))

    def crash(point: str) -> None:
        if config["crash_at"] == point:
            save()
            os._exit(73)

    batch = memory.poll("watch")
    audit["delivery_id"] = batch.delivery_id
    audit["event_ids"] = [event.fact_id for event in batch.events]
    save()  # represents a durable host inbox before acknowledgement
    crash("after_poll")
    if batch.delivery_id:
        memory.ack("watch", delivery_id=batch.delivery_id)
    intent = memory.get_intent(config["intent_id"])
    lease = memory.claim(intent.id, worker_id="crashing-worker", ttl_seconds=1)
    audit["old_token"] = lease.token
    save()
    if config["crash_at"] == "after_checkpoint":
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=memory.get_intent(intent.id).revision,
            state=IntentState.WAITING,
            progress={"phase": "waiting"},
            waiting_for=WaitUntil(NOW + timedelta(seconds=1)),
            basis=intent.basis,
        )
        crash("after_checkpoint")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="deploy", request={"target": "staging"}
    )
    permit = memory.begin_action(action.id, lease_token=lease.token)
    audit["action_id"] = action.id
    audit["external_key"] = permit.idempotency_key
    save()
    crash("after_begin")
    external_effect(Path(config["external_path"]), permit.idempotency_key)
    crash("after_external")
    memory.finish_action(
        action.id, lease_token=lease.token, outcome="succeeded", result={"ok": True}
    )
    crash("after_finish")
    raise AssertionError("declared crash point was not reached")


def crash_recovery(
    storage: StorageBackend, work: Path, *, crash_at: str, reconcile: bool = True
) -> dict[str, Any]:
    owner = "crash-" + crash_at + ("-proof" if reconcile else "-no-proof")
    kb = KnowledgeBase(owner, storage=storage, embed_url="")
    memory = AlwaysOnMemory(kb, clock=lambda: NOW)
    memory.subscribe(subscription_id="watch", selector=WatchSelector(namespace=owner))
    fact = Fact(
        id="basis",
        content="Deployment uses staging",
        created_at=NOW,
        last_accessed=NOW,
        valid_from=NOW,
    )
    storage.save(owner, [fact])
    refs = memory.capture_evidence(((owner, fact.id),))
    intent = memory.create_intent(client_key="deployment", goal="Deploy with recovery")
    lease = memory.claim(intent.id, worker_id="planner")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state=IntentState.READY,
        progress={},
        basis=refs,
    )
    work.mkdir(parents=True, exist_ok=True)
    audit_path = work / (owner + "-audit.json")
    external_path = work / (owner + "-external.db")
    config_path = work / (owner + "-config.json")
    config_path.write_text(
        json.dumps(
            dict(
                storage=storage_config(storage),
                owner=owner,
                crash_at=crash_at,
                intent_id=intent.id,
                audit_path=str(audit_path),
                external_path=str(external_path),
            )
        )
    )
    child = subprocess.run(
        [sys.executable, "-m", "tests.eval.always_on.worker", str(config_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if child.returncode != 73:
        raise AssertionError(f"Crash worker failed ({child.returncode}): {child.stderr}")
    audit = json.loads(audit_path.read_text())
    restored = AlwaysOnMemory(
        KnowledgeBase(owner, storage=open_storage(storage_config(storage)), embed_url=""),
        clock=lambda: NOW + timedelta(seconds=2),
    )
    replay = restored.poll("watch")
    missing = int(fact.id not in audit["event_ids"])
    if crash_at == "after_poll":
        missing += int(replay.delivery_id != audit["delivery_id"])
        missing += int(fact.id not in {event.fact_id for event in replay.events})
    if audit["delivery_id"]:
        restored.ack("watch", delivery_id=audit["delivery_id"])
        restored.ack("watch", delivery_id=audit["delivery_id"])
    packet = restored.resume(intent.id)
    stale_mutations = 0
    if audit.get("old_token"):
        try:
            restored.renew(intent.id, lease_token=audit["old_token"])
            stale_mutations += 1
        except LeaseLost:
            pass
    unknown = [a for a in packet.open_actions if a.state == ActionState.UNKNOWN]
    before_keys = external_keys(external_path)
    if unknown and not reconcile:
        again = restored.resume(intent.id)
        return dict(
            missing_committed_changes=missing,
            unintended_duplicate_effects=max(0, len(before_keys) - 1),
            accepted_stale_lease_mutations=stale_mutations,
            false_auto_retries_unknown_actions=int(external_keys(external_path) != before_keys),
            completed=again.intent.state == IntentState.COMPLETED,
            unknown_actions=len(unknown),
            external_effects=len(before_keys),
        )
    lease = restored.claim(intent.id, worker_id="recovery-worker")
    for action in unknown:
        proven = action.idempotency_key in before_keys
        restored.reconcile_action(
            action.id,
            lease_token=lease.token,
            outcome="succeeded" if proven else "retryable",
            evidence={"lookup": "external-store", "found": proven},
        )
    packet = restored.resume(intent.id)
    action = next((a for a in packet.open_actions if a.state == ActionState.PREPARED), None)
    if not before_keys:
        if action is None:
            action = restored.prepare_action(
                intent.id,
                lease_token=lease.token,
                action_key="deploy",
                request={"target": "staging"},
            )
        permit = restored.begin_action(action.id, lease_token=lease.token)
        external_effect(external_path, permit.idempotency_key)
        restored.finish_action(
            action.id, lease_token=lease.token, outcome="succeeded", result={"ok": True}
        )
    restored.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=restored.get_intent(intent.id).revision,
        state=IntentState.COMPLETED,
        progress={"deployed": True},
        basis=refs,
    )
    effects = external_keys(external_path)
    return dict(
        missing_committed_changes=missing,
        unintended_duplicate_effects=max(0, len(effects) - 1),
        accepted_stale_lease_mutations=stale_mutations,
        false_auto_retries_unknown_actions=0,
        completed=restored.get_intent(intent.id).state == IntentState.COMPLETED,
        unknown_actions=0,
        external_effects=len(effects),
    )


def snapshot_history(storage: StorageBackend) -> dict[str, Any]:
    owner = "snapshot-owner"
    kb = KnowledgeBase(owner, storage=storage, embed_url="")
    memory = AlwaysOnMemory(kb, clock=lambda: NOW)
    original = Fact(
        id="snap", content="old config", created_at=NOW, last_accessed=NOW, valid_from=NOW
    )
    storage.save(owner, [original])
    kb.snapshot("before-change")
    memory.subscribe(subscription_id="snap-watch", selector=WatchSelector(namespace=owner))
    storage.save(owner, [replace(original, content="new config")])
    batch = memory.poll("snap-watch")
    assert batch.delivery_id
    memory.ack("snap-watch", delivery_id=batch.delivery_id)
    intent = memory.create_intent(client_key="snap", goal="Keep history")
    lease = memory.claim(intent.id, worker_id="host")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="effect", request={}
    )
    memory.begin_action(action.id, lease_token=lease.token)
    memory.finish_action(
        action.id, lease_token=lease.token, outcome="succeeded", result={"ok": True}
    )
    kb.restore("before-change")
    restored = AlwaysOnMemory(KnowledgeBase(owner, storage=storage), clock=lambda: NOW)
    same = restored.prepare_action(
        intent.id, lease_token=lease.token, action_key="effect", request={}
    )
    restored.ack("snap-watch", delivery_id=batch.delivery_id)
    changes = restored.poll("snap-watch")
    return dict(
        action_survived=same.id == action.id and same.state == ActionState.SUCCEEDED,
        old_ack_noop=True,
        restore_change_delivered=original.id in {e.fact_id for e in changes.events},
    )


def stale_plan(storage: StorageBackend, *, deleted: bool = False) -> dict[str, Any]:
    owner = "deleted-plan" if deleted else "stale-plan"
    memory = AlwaysOnMemory(KnowledgeBase(owner, storage=storage), clock=lambda: NOW)
    fact = Fact(id="basis", content="old", created_at=NOW, last_accessed=NOW, valid_from=NOW)
    storage.save(owner, [fact])
    refs = memory.capture_evidence(((owner, fact.id),))
    intent = memory.create_intent(client_key="plan", goal="Guard action")
    lease = memory.claim(intent.id, worker_id="planner")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state=IntentState.READY,
        progress={},
        basis=refs,
    )
    lease = memory.claim(intent.id, worker_id="executor")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="effect", request={}
    )
    storage.save(owner, [] if deleted else [replace(fact, content="corrected")])
    starts = 0
    try:
        memory.begin_action(action.id, lease_token=lease.token)
        starts += 1
    except StaleEvidence:
        pass
    return dict(
        stale_action_starts=starts,
        blocked=memory.get_intent(intent.id).state == IntentState.BLOCKED,
    )


def verifier_outage(storage: StorageBackend) -> dict[str, Any]:
    owner = "outage"
    memory = AlwaysOnMemory(KnowledgeBase(owner, storage=storage), clock=lambda: NOW)
    fact = Fact(id="basis", content="guarded", created_at=NOW, last_accessed=NOW, valid_from=NOW)
    storage.save(owner, [fact])
    ref = memory.capture_evidence(((owner, fact.id),))[0]
    memory.set_freshness(ref, source_ref="configured-source", recheck_after_seconds=300)
    ok = memory.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"etag": "v1"}, "v1")
    )
    unavailable = memory.record_verification(
        ref, result=VerificationResult(VerificationOutcome.UNAVAILABLE, {"error": "offline"}, "v1")
    )
    return dict(
        fact_preserved=storage.load(owner)[0].content == "guarded",
        validity_not_extended=unavailable.verified_until == ok.verified_until,
        outcome=unavailable.last_outcome.value,
        failure_count=unavailable.failure_count,
    )


def consolidation_race(storage: StorageBackend) -> dict[str, Any]:
    import asyncio

    from ai_knot.always_on.maintenance import (
        ConsolidationProposal,
        MaintenanceBudget,
        run_maintenance,
    )
    from ai_knot.types import MemoryType

    owner = "race-owner"
    memory = AlwaysOnMemory(KnowledgeBase(owner, storage=storage), clock=lambda: NOW)
    source = Fact(
        id="episode",
        content="Old deployment observation",
        type=MemoryType.EPISODIC,
        created_at=NOW - timedelta(days=2),
        valid_from=NOW - timedelta(days=2),
    )
    target = Fact(id="current", content="Deployment v1", slot_key="deploy::target", value_text="v1")
    target.created_at = target.valid_from = NOW
    storage.save(owner, [source, target])

    class RacingProvider:
        config_key = "race-test-v1"
        model_cost = 1

        async def propose(self, episodes: tuple[Fact, ...]) -> ConsolidationProposal:
            refs = memory.capture_evidence(tuple((owner, f.id) for f in episodes))
            current = storage.load(owner)
            for fact in current:
                if fact.id == target.id:
                    fact.content = "Foreground deployment v2"
                    fact.value_text = "v2"
            storage.save(owner, current)
            summary = Fact(
                id="summary",
                content="Old summarized deployment",
                slot_key="deploy::target",
                value_text="v1",
                qualifiers={"consolidation_ids": json.dumps([source.id])},
            )
            return ConsolidationProposal((summary,), (source.id,), refs)

    report = asyncio.run(
        run_maintenance(
            memory,
            run_key="race",
            budget=MaintenanceBudget(max_model_calls=1),
            consolidator=RacingProvider(),
        )
    )
    current = {fact.id: fact for fact in storage.load(owner)}
    return dict(
        consolidated=report.consolidated,
        foreground_preserved=current[target.id].value_text == "v2"
        and current[target.id].is_active(NOW),
        source_preserved=current[source.id].is_active(NOW),
        stale_summary_absent="summary" not in current,
    )


def owner_and_scope_isolation(storage: StorageBackend) -> dict[str, Any]:
    from ai_knot.always_on import AccessDenied
    from ai_knot.pool import SharedMemoryPool

    pool = SharedMemoryPool(storage=storage, persist_stats=True)
    alice = AlwaysOnMemory(KnowledgeBase("acl-alice", storage=storage), pool=pool)
    bob = AlwaysOnMemory(KnowledgeBase("acl-bob", storage=storage), pool=pool)
    intent = alice.create_intent(client_key="private", goal="Private intent")
    leaks = 0
    try:
        bob.get_intent(intent.id)
        leaks += 1
    except AccessDenied:
        pass
    pool.grant_read("acl-alice", "team:ops")
    alice.subscribe(subscription_id="acl-watch", selector=WatchSelector("__shared__"))
    fact = Fact(
        id="acl-secret",
        content="Private shared target",
        origin_agent_id="publisher",
        visibility_scope="team:ops",
        memory_tier="pool",
    )
    storage.save("__shared__", [fact])
    batch = alice.poll("acl-watch")
    if fact.id not in {event.fact_id for event in batch.events}:
        raise AssertionError("authorized event was lost")
    pool.revoke_read("acl-alice", "team:ops")
    replay = alice.poll("acl-watch")
    leaks += len(replay.events)
    try:
        alice.capture_evidence((("__shared__", fact.id),))
        leaks += 1
    except AccessDenied:
        pass
    return dict(
        cross_owner_scope_leaks=leaks,
        replay_identity_preserved=replay.delivery_id == batch.delivery_id,
        revoked_resync=replay.resync_required,
    )
