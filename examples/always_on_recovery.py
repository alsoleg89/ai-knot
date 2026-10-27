"""Offline recovery across real process stops, with a separate external-effect DB.

Run twice: python examples/always_on_recovery.py --data-dir /tmp/knot-recovery
No model, network service, scheduler or real deployment is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_knot.always_on import AlwaysOnMemory, IntentState, WaitForSlot, WatchSelector
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact

OWNER = "deploy-agent"
EPOCH = datetime(2026, 10, 27, 12, tzinfo=UTC)


class ExternalAPI:
    """The external effect and inbox are durable independently of AI Knot."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY,result TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS inbox (delivery_id TEXT PRIMARY KEY)")
            conn.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY,value TEXT)")

    def put(self, key: str, value: str) -> None:
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get(self, key: str) -> str | None:
        with closing(sqlite3.connect(self.path)) as conn, conn:
            row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
            return row[0] if row else None

    def execute_once(self, key: str) -> dict[str, str]:
        result = {"deployment": "build-42"}
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO effects VALUES(?,?) ON CONFLICT(key) DO NOTHING",
                (key, json.dumps(result)),
            )
        return result

    def lookup(self, key: str) -> dict[str, str] | None:
        with closing(sqlite3.connect(self.path)) as conn, conn:
            row = conn.execute("SELECT result FROM effects WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def receive_once(self, delivery_id: str) -> None:
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO inbox VALUES(?) ON CONFLICT(delivery_id) DO NOTHING", (delivery_id,)
            )

    def count(self) -> int:
        with closing(sqlite3.connect(self.path)) as conn, conn:
            return int(conn.execute("SELECT COUNT(*) FROM effects").fetchone()[0])


def phase(directory: Path, name: str) -> None:
    external = ExternalAPI(directory / "external.db")
    seconds = {"prepare": 0, "effect": 10, "recover": 100}[name]
    now = EPOCH + timedelta(seconds=seconds)
    kb = KnowledgeBase(OWNER, storage=SQLiteStorage(str(directory / "memory.db")), embed_url="")
    ao = AlwaysOnMemory(kb, clock=lambda: now)
    if name == "prepare":
        kb.add_resolved(
            [
                Fact("CI pending", slot_key="ci::status", value_text="pending"),
                Fact("Old target", slot_key="deploy::target", value_text="old"),
            ]
        )
        intent = ao.create_intent(client_key="deploy:42", goal="Deploy after CI")
        external.put("intent", intent.id)
        ao.subscribe(subscription_id="deploy-watch", selector=WatchSelector(OWNER))
        basis = ao.capture_evidence(
            tuple((OWNER, f.id) for f in kb.list_facts() if f.is_active(now))
        )
        lease = ao.claim(intent.id, worker_id="host-1")
        current = ao.get_intent(intent.id)
        ao.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=current.revision,
            state=IntentState.WAITING,
            progress={"stage": "ci"},
            waiting_for=WaitForSlot(OWNER, "ci::status", "green"),
            basis=basis,
        )
        return
    intent_id = external.get("intent")
    assert intent_id is not None
    if name == "effect":
        kb.add_resolved(
            [
                Fact("CI green", slot_key="ci::status", value_text="green"),
                Fact("New target", slot_key="deploy::target", value_text="new"),
            ]
        )
        batch = ao.poll("deploy-watch")
        assert batch.delivery_id is not None
        external.receive_once(batch.delivery_id)
        # Retry before ack yields the same delivery after reopening the facade.
        assert AlwaysOnMemory(kb, clock=lambda: now).poll("deploy-watch") == batch
        ao.ack("deploy-watch", delivery_id=batch.delivery_id)
        packet = ao.resume(intent_id)
        assert packet.stale_basis
        external.put("stale_plan_blocked", "true")
        lease = ao.claim(intent_id, worker_id="host-2")
        current = ao.get_intent(intent_id)
        fresh = ao.capture_evidence(
            tuple((OWNER, f.id) for f in kb.list_facts() if f.is_active(now))
        )
        ao.checkpoint(
            intent_id,
            lease_token=lease.token,
            expected_revision=current.revision,
            state=IntentState.READY,
            progress={"stage": "replanned"},
            basis=fresh,
        )
        lease = ao.claim(intent_id, worker_id="host-2")
        action = ao.prepare_action(
            intent_id,
            lease_token=lease.token,
            action_key="deploy-build-42",
            request={"build": "42"},
        )
        permit = ao.begin_action(action.id, lease_token=lease.token)
        external.execute_once(permit.idempotency_key)
        os._exit(73)  # Real crash window: effect committed; finish was never called.
    packet = ao.resume(intent_id)
    assert packet.open_actions and not packet.ready
    recovery = ao.claim(intent_id, worker_id="host-recovery")
    for action in packet.open_actions:
        result = external.lookup(action.idempotency_key)
        assert result is not None
        ao.reconcile_action(
            action.id,
            lease_token=recovery.token,
            outcome="succeeded",
            evidence={"connector": "offline-external-db", "result": result},
        )
    current = ao.get_intent(intent_id)
    ao.checkpoint(
        intent_id,
        lease_token=recovery.token,
        expected_revision=current.revision,
        state=IntentState.COMPLETED,
        progress={"reconciled": True},
    )
    # A service without idempotency or lookup evidence stays blocked after expiry.
    unknown = ao.create_intent(client_key="unqueryable:42", goal="Unknown remote outcome")
    lease = ao.claim(unknown.id, worker_id="host")
    action = ao.prepare_action(
        unknown.id, lease_token=lease.token, action_key="unqueryable", request={}
    )
    ao.begin_action(action.id, lease_token=lease.token)
    later = AlwaysOnMemory(kb, clock=lambda: now + timedelta(seconds=61))
    assert later.resume(unknown.id).intent.state == IntentState.BLOCKED
    external.put("unknown_branch", "blocked")
    external.put("completed", "true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "effect", "recover"])
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.phase:
        phase(args.data_dir, args.phase)
        return
    external = ExternalAPI(args.data_dir / "external.db")
    if external.get("completed") != "true":
        for name, expected in (("prepare", 0), ("effect", 73), ("recover", 0)):
            result = subprocess.run(
                [sys.executable, __file__, "--data-dir", str(args.data_dir), "--phase", name],
                check=False,
            )
            if result.returncode != expected:
                raise SystemExit(f"Phase {name} exited {result.returncode}; expected {expected}")
    print(
        json.dumps(
            {
                "completed": external.get("completed") == "true",
                "effects": external.count(),
                "stale_plan_blocked": external.get("stale_plan_blocked") == "true",
                "model_calls": 0,
                "no_idempotency_branch": external.get("unknown_branch"),
            }
        )
    )


if __name__ == "__main__":
    main()
