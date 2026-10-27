"""Durability, lease fencing and pre-action safety through the public facade."""

from datetime import UTC, datetime, timedelta

import pytest

from ai_knot import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage


def test_durable_checkpoint_and_lease_fencing(tmp_path):
    from ai_knot.always_on import AlwaysOnMemory, IntentState, LeaseLost, WaitUntil

    now = [datetime.now(UTC)]
    path = str(tmp_path / "state.db")

    def factory():
        return AlwaysOnMemory(
            KnowledgeBase(agent_id="owner", storage=SQLiteStorage(path)), clock=lambda: now[0]
        )

    ao = factory()
    intent = ao.create_intent(client_key="ship", goal="Ship release")
    lease = ao.claim(intent.id, worker_id="first")
    current = ao.get_intent(intent.id)
    ao.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=current.revision,
        state=IntentState.WAITING,
        progress={"done": 1},
        waiting_for=WaitUntil(now[0] + timedelta(seconds=60)),
    )
    assert not factory().resume(intent.id).ready
    now[0] += timedelta(seconds=60)
    assert factory().resume(intent.id).ready
    new_lease = factory().claim(intent.id, worker_id="second")
    assert new_lease.generation > lease.generation
    with pytest.raises(LeaseLost):
        ao.renew(intent.id, lease_token=lease.token)


def test_started_action_cannot_release_lease_and_recovers_unknown(tmp_path):
    from ai_knot.always_on import AlwaysOnMemory, IntentState, RevisionConflict

    now = [datetime.now(UTC)]
    ao = AlwaysOnMemory(
        KnowledgeBase(agent_id="owner", storage=SQLiteStorage(str(tmp_path / "a.db"))),
        clock=lambda: now[0],
    )
    intent = ao.create_intent(client_key="ship", goal="Ship")
    lease = ao.claim(intent.id, worker_id="w")
    action = ao.prepare_action(
        intent.id, lease_token=lease.token, action_key="deploy", request={"v": 1}
    )
    permit = ao.begin_action(action.id, lease_token=lease.token)
    assert permit.idempotency_key == action.idempotency_key
    for state in (
        IntentState.READY,
        IntentState.BLOCKED,
        IntentState.COMPLETED,
        IntentState.CANCELLED,
    ):
        with pytest.raises(RevisionConflict):
            ao.checkpoint(
                intent.id,
                lease_token=lease.token,
                expected_revision=ao.get_intent(intent.id).revision,
                state=state,
                progress={},
            )
    now[0] += timedelta(seconds=60)
    packet = ao.resume(intent.id)
    assert packet.open_actions[0].state == "unknown"
    recovered = ao.claim(intent.id, worker_id="recover")
    ao.reconcile_action(
        action.id, lease_token=recovered.token, outcome="succeeded", evidence={"remote_id": "1"}
    )
    assert ao.get_intent(intent.id).state == IntentState.RUNNING
    ao.checkpoint(
        intent.id,
        lease_token=recovered.token,
        expected_revision=ao.get_intent(intent.id).revision,
        state=IntentState.COMPLETED,
        progress={},
    )
    assert ao.get_intent(intent.id).state == IntentState.COMPLETED
