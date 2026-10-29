"""A bounded planner over shared observations, followed by host-guarded work.

Run: PYTHONPATH=src python -m examples.always_on_planning
The default callback and connector are scripted; no network/model is started.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from examples.always_on_team import Inbox, inspect_turn, plan_once

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory, IntentState, WatchSelector
from ai_knot.storage.base import StorageBackend
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact

REQUIREMENTS = ("activation destination", "approval receipt")


async def scripted_planner(payload: dict[str, object]) -> str:
    """Fixture decision only; a real host supplies its metered async callback."""
    pending = any("outstanding" in f["content"] for f in payload["facts"])
    return json.dumps(
        {
            "action": "wait" if pending else "plan",
            "next_step": "Await sign-off." if pending else "Activate Boreal.",
            "basis": [f["index"] for f in payload["facts"]],
            "data": {} if pending else {"operation": "activate", "target": "Boreal"},
        }
    )


async def example(
    directory: Path,
    *,
    make_store: Callable[[], StorageBackend] | None = None,
    planner: Callable[[dict[str, object]], Awaitable[str]] = scripted_planner,
) -> dict[str, object]:
    """The connector publishes truth; a model proposal cannot grant permission."""
    make_store = make_store or (lambda: SQLiteStorage(str(directory / "planning.db")))
    now = [datetime(2026, 10, 28, tzinfo=UTC)]
    store = make_store()
    SharedMemoryPool(store, persist_stats=True).grant_read("operator", "ops")

    def agent() -> AlwaysOnMemory:
        independent = make_store()
        return AlwaysOnMemory(
            KnowledgeBase("operator", storage=independent, embed_url=""),
            pool=SharedMemoryPool(independent, persist_stats=True),
            clock=lambda: now[0],
        )

    destination = Fact(
        "The designated next cluster is Boreal.",
        id="destination",
        slot_key="rollout::cluster",
        value_text="Boreal",
        created_at=now[0],
        last_accessed=now[0],
        valid_from=now[0],
        origin_agent_id="researcher",
        visibility_scope="ops",
        qualifiers={"task_key": "rollout"},
    )
    signoff = replace(
        destination,
        id="signoff",
        slot_key="rollout::signoff",
        value_text="pending",
        content="Controller sign-off remains outstanding for Boreal.",
    )

    def publish() -> None:
        # Trusted connector observations; never copy a model proposal into truth.
        store.save("__shared__", [destination, signoff])

    publish()
    memory = agent()
    intent = memory.create_intent(
        client_key="rollout", goal="Activate only the signed-off cluster."
    )
    memory.subscribe(subscription_id="rollout", selector=WatchSelector("__shared__"))
    inbox = Inbox(directory / "planning-inbox.db")

    def inspect():
        return inspect_turn(
            memory,
            inbox,
            intent_id=intent.id,
            subscription_id="rollout",
            requirements=REQUIREMENTS,
            task_key="rollout",
        )

    async def plan(trigger):
        return await plan_once(
            memory,
            inspect(),
            planner,
            trigger_key=trigger,
            requirements=REQUIREMENTS,
            worker_id="event-host",
            task_key="rollout",
            max_model_calls=1,
        )

    initial = inspect()
    first = await plan("observed-pending")
    quiet = [await plan(None) for _ in range(5)]
    now[0] += timedelta(seconds=61)
    signoff = replace(
        signoff,
        value_text="confirmed:Boreal",
        content="Controller sign-off is completed for Boreal.",
    )
    publish()
    memory = agent()  # Independent connection/facade after host sleep/restart.
    second = await plan("observed-confirmed")
    repeated = await plan("observed-confirmed")
    result = {
        "initial_inspection": initial.status,
        "initial_missing": initial.context.missing_requirements,
        "proposals": [asdict(first), asdict(second)],
        "quiet_statuses": [q.status for q in quiet],
        "quiet_model_calls_reserved": sum(q.model_calls_reserved for q in quiet),
        "repeat_status": repeated.status,
        "repeat_model_calls_reserved": repeated.model_calls_reserved,
        "effect_count": 0,
        "reconciliation_status": None,
        "final_state": memory.get_intent(intent.id).state.value,
    }

    # This is application policy over trusted structured observations, not an LLM
    # verdict. All hosts in the smoke comparison use this identical gate.
    current = {f.slot_key: f.value_text for f in store.load("__shared__")}
    data = second.proposal.get("data") if second.proposal else None
    if not (
        first.status == "waiting"
        and second.status == "planned"
        and data == {"operation": "activate", "target": current.get("rollout::cluster")}
        and current.get("rollout::signoff") == "confirmed:" + current.get("rollout::cluster", "")
        and len(memory.get_intent(intent.id).basis) == 2
    ):
        return result

    lease = memory.claim(intent.id, worker_id="executor")
    action = memory.prepare_action(
        intent.id,
        lease_token=lease.token,
        action_key="activate",
        request=data,
    )
    memory.begin_action(action.id, lease_token=lease.token, context_guard=second.context_guard)
    # A scripted connector loses its response after one effect. Do not infer success.
    result["effect_count"] = 1
    memory.finish_action(action.id, lease_token=lease.token, outcome="unknown", result={})
    now[0] += timedelta(seconds=61)
    unresolved = await plan("unknown-response")
    result["unknown_model_calls_reserved"] = unresolved.model_calls_reserved
    result["reconciliation_status"] = inspect().status
    lease = memory.claim(intent.id, worker_id="reconciler")
    memory.reconcile_action(
        action.id,
        lease_token=lease.token,
        outcome="succeeded",
        evidence={"scripted_connector_receipt": "activated:Boreal"},
    )
    current_intent = memory.get_intent(intent.id)
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=current_intent.revision,
        state=IntentState.COMPLETED,
        progress={"connector_receipt_checked": True},
        basis=current_intent.basis,
    )
    result["final_state"] = memory.get_intent(intent.id).state.value
    return result


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        print(json.dumps(asyncio.run(example(Path(directory))), indent=2))
