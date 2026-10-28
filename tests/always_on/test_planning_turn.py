"""A planning proposal uses current evidence without becoming an action permit."""

import asyncio
import importlib
import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta

import pytest
from tests.always_on.test_team_cycle import setup

from ai_knot import KnowledgeBase
from ai_knot.always_on import (
    AlwaysOnMemory,
    IntentState,
    LeaseLost,
    StaleEvidence,
    VerificationOutcome,
    VerificationResult,
    WaitUntil,
    WatchSelector,
    WorkspaceMemory,
)
from ai_knot.types import Fact


@pytest.fixture
def recipe():
    return importlib.import_module("examples.always_on_team")


def available(recipe, sql_storage, clock, tmp_path):
    kb, memory, source, intent, inbox, inspect = setup(recipe, sql_storage, clock, tmp_path)
    source = replace(
        source, content="A validated recovery copy is available", qualifiers={"task_key": "restore"}
    )
    kb.replace_facts([source])
    view = inspect(task_key="restore")
    assert view.status == "missing_context" and not view.wake_model
    return kb, memory, source, intent, view


def proposal(action="plan", basis=None):
    return json.dumps(
        {
            "action": action,
            "next_step": "Inspect the recovery copy.",
            "basis": [1] if basis is None else basis,
            "data": {"choice": "copy"},
        }
    )


async def invoke(recipe, memory, view, callback, **kwargs):
    arguments = {
        "trigger_key": "delivery-1",
        "requirements": ("backup receipt",),
        "worker_id": "host",
        "task_key": "restore",
    }
    arguments.update(kwargs)
    return await recipe.plan_once(memory, view, callback, **arguments)


@pytest.mark.anyio
async def test_paraphrase_evidence_can_produce_a_persisted_bounded_plan(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, source, intent, view = available(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def decide(payload):
        calls.append(payload)
        assert payload["goal"] == intent.goal
        assert payload["requirements"] == ["backup receipt"]
        assert payload["facts"] == [{"index": 1, "content": source.content}]
        assert "lease" not in json.dumps(payload) and source.id not in json.dumps(payload)
        return proposal()

    result = await invoke(recipe, memory, view, decide, max_model_calls=1)
    assert result.status == "planned" and result.model_calls_reserved == 1 and len(calls) == 1
    current = memory.get_intent(intent.id)
    assert current.state == IntentState.READY and current.lease is None
    assert current.next_step == "Inspect the recovery copy."
    assert [r.fact_id for r in current.basis] == [source.id]
    assert result.proposal["data"] == {"choice": "copy"}
    assert memory.resume(intent.id).open_actions == ()
    assert view.status == "missing_context" and not view.wake_model
    with pytest.raises(LeaseLost):
        memory.prepare_action(intent.id, lease_token="unclaimed", action_key="copy", request={})


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status",
    [
        "catching_up",
        "debounced",
        "snapshot_too_large",
        "waiting",
        "verification_due",
        "lease_busy",
        "reconcile",
        "blocked",
        "deadline_exceeded",
        "completed",
        "cancelled",
    ],
)
async def test_ineligible_inspection_never_wakes_a_planner(
    recipe, sql_storage, clock, tmp_path, status
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)

    async def forbidden(payload):
        raise AssertionError("Ineligible turn")

    result = await invoke(
        recipe, memory, replace(view, status=status), forbidden, max_model_calls=1
    )
    assert result.status == "skipped" and result.model_calls_reserved == 0
    assert memory.get_intent(intent.id).lease is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "condition",
    ["manual", "waiting", "terminal", "unknown", "prepared", "lease", "freshness", "deadline"],
)
async def test_fresh_authoritative_state_cannot_be_bypassed_by_old_turn(
    recipe, sql_storage, clock, tmp_path, condition
):
    _, memory, source, intent, view = available(recipe, sql_storage, clock, tmp_path)
    refs = memory.capture_evidence((("owner", source.id),))
    if condition == "deadline":
        intent = memory.create_intent(
            client_key="deadline", goal="Restore", deadline=clock() + timedelta(seconds=1)
        )
        view = replace(view, packet=memory.resume(intent.id))
        clock.advance(seconds=2)
    else:
        lease = memory.claim(intent.id, worker_id="other")
        if condition in ("unknown", "prepared"):
            action = memory.prepare_action(
                intent.id, lease_token=lease.token, action_key="pending", request={}
            )
            if condition == "unknown":
                memory.begin_action(action.id, lease_token=lease.token)
                memory.finish_action(
                    action.id, lease_token=lease.token, outcome="unknown", result={}
                )
                clock.advance(seconds=61)
            else:
                memory.checkpoint(
                    intent.id,
                    lease_token=lease.token,
                    expected_revision=memory.get_intent(intent.id).revision,
                    state=IntentState.READY,
                    progress={},
                )
        elif condition != "lease":
            state = {
                "manual": IntentState.BLOCKED,
                "waiting": IntentState.WAITING,
                "terminal": IntentState.COMPLETED,
                "freshness": IntentState.READY,
            }[condition]
            memory.checkpoint(
                intent.id,
                lease_token=lease.token,
                expected_revision=memory.get_intent(intent.id).revision,
                state=state,
                progress={},
                basis=refs,
                waiting_for=WaitUntil(clock() + timedelta(seconds=60))
                if condition == "waiting"
                else None,
            )
            if condition == "freshness":
                memory.set_freshness(refs[0], source_ref="copy-service", recheck_after_seconds=1)
                memory.record_verification(
                    refs[0],
                    result=VerificationResult(
                        VerificationOutcome.CONFIRMED, {"verified": True}, "v1"
                    ),
                )
                clock.advance(seconds=2)

    async def forbidden(payload):
        raise AssertionError("Authoritatively blocked")

    result = await invoke(recipe, memory, view, forbidden, max_model_calls=1)
    assert result.status == "blocked" and result.model_calls_reserved == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "not json",
        json.dumps({"action": "complete", "next_step": "done", "basis": [1], "data": {}}),
        *[
            json.dumps({"action": "plan", "next_step": "inspect", "basis": b, "data": {}})
            for b in ([True], [], [0], [2], [1, 1], "1")
        ],
        json.dumps({"action": "plan", "next_step": " " * 5000 + "x", "basis": [1], "data": {}}),
        json.dumps({"action": "plan", "next_step": "inspect", "basis": [1], "data": []}),
        '{"action":"plan","next_step":"inspect","basis":[],"basis":[1],"data":{}}',
        '{"action":"plan","next_step":"inspect","basis":[1],"data":{"value":1e999}}',
        '{"action":"plan","next_step":"inspect","basis":[1],"data":'
        + "[" * 2000
        + "]" * 2000
        + "}",
        json.dumps(
            {"action": "plan", "next_step": "inspect", "basis": [1], "data": {"long": "🙂" * 5000}},
            ensure_ascii=False,
        ),
    ],
    ids=[
        "empty",
        "non_json",
        "complete_action",
        "bool_index",
        "no_basis",
        "zero_index",
        "out_of_range",
        "duplicate_index",
        "string_basis",
        "oversize_step",
        "array_data",
        "duplicate_json_key",
        "nonfinite_number",
        "deep_json",
        "utf8_output_limit",
    ],
)
async def test_invalid_response_is_charged_and_cannot_create_a_plan(
    recipe, sql_storage, clock, tmp_path, raw
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def decide(payload):
        calls.append(payload)
        return raw

    result = await invoke(recipe, memory, view, decide, max_model_calls=1)
    assert result.status == "invalid_proposal" and result.model_calls_reserved == 1
    assert result.proposal is None and memory.get_intent(intent.id).next_step == ""
    assert memory.get_intent(intent.id).lease is None
    assert (await invoke(recipe, memory, view, decide, max_model_calls=1)).status == "unchanged"
    assert len(calls) == 1


@pytest.mark.anyio
async def test_provider_failure_and_timeout_keep_reservation_and_deduplicate(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def unavailable(payload):
        calls.append(payload)
        raise OSError("private provider failure detail")

    result = await invoke(recipe, memory, view, unavailable, max_model_calls=1)
    assert result.status == "unavailable" and result.model_calls_reserved == 1
    assert result.error_type == "OSError"
    assert "private provider" not in json.dumps(memory.get_intent(intent.id).progress)
    assert (
        await invoke(recipe, memory, view, unavailable, max_model_calls=1)
    ).status == "unchanged"

    async def stalled(payload):
        calls.append(payload)
        await asyncio.Event().wait()

    result = await invoke(
        recipe,
        memory,
        view,
        stalled,
        trigger_key="retry-1",
        max_model_calls=1,
        timeout_seconds=0.001,
    )
    assert result.status == "unavailable" and result.model_calls_reserved == 1
    assert result.error_type == "TimeoutError" and len(calls) == 2
    assert memory.get_intent(intent.id).lease is None


@pytest.mark.anyio
async def test_cancellation_remains_cancellation_with_safe_attempt_cleanup(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def cancelled(payload):
        calls.append(payload)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await invoke(recipe, memory, view, cancelled, max_model_calls=1)
    assert memory.get_intent(intent.id).lease is None
    assert (await invoke(recipe, memory, view, cancelled, max_model_calls=1)).status == "unchanged"
    assert len(calls) == 1


@pytest.mark.anyio
async def test_lease_expiry_during_acquisition_prevents_even_the_paid_callback(
    recipe, sql_storage, clock, tmp_path, monkeypatch
):
    _, memory, _, _, view = available(recipe, sql_storage, clock, tmp_path)
    snapshot = memory.task_snapshot
    calls = []

    def late(**kwargs):
        facts = snapshot(**kwargs)
        clock.advance(seconds=301)
        return facts

    monkeypatch.setattr(memory, "task_snapshot", late)

    async def forbidden(payload):
        calls.append(payload)
        return proposal()

    result = await invoke(recipe, memory, view, forbidden, max_model_calls=1)
    assert result.status == "blocked" and result.model_calls_reserved == 0
    assert not calls


@pytest.mark.anyio
async def test_repeated_event_and_reopened_host_do_not_repay_for_same_input(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def decide(payload):
        calls.append(payload)
        return proposal()

    assert (await invoke(recipe, memory, view, decide, max_model_calls=1)).status == "planned"
    reopened = AlwaysOnMemory(
        KnowledgeBase("owner", storage=sql_storage, embed_url=""), clock=clock
    )
    assert (await invoke(recipe, reopened, view, decide, max_model_calls=1)).status == "unchanged"
    assert len(calls) == 1 and reopened.get_intent(intent.id).lease is None


@pytest.mark.anyio
async def test_default_budget_and_idle_trigger_make_no_callback(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)

    async def forbidden(payload):
        raise AssertionError("No model invocation authorized")

    assert (await invoke(recipe, memory, view, forbidden)).status == "budget_exhausted"
    result = await recipe.plan_once(
        memory,
        view,
        forbidden,
        trigger_key=None,
        requirements=("backup receipt",),
        worker_id="host",
        task_key="restore",
        max_model_calls=1,
    )
    assert result.status == "idle" and result.model_calls_reserved == 0
    assert memory.get_intent(intent.id).lease is None


@pytest.mark.anyio
async def test_wait_preserves_a_quiet_timer_and_does_not_mark_work_completed(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)

    async def decide(payload):
        return proposal("wait", [])

    result = await invoke(recipe, memory, view, decide, max_model_calls=1)
    current = memory.get_intent(intent.id)
    assert result.status == "waiting" and result.model_calls_reserved == 1
    assert current.state == IntentState.WAITING and current.waiting_for.at > clock()
    assert not memory.resume(intent.id).ready


@pytest.mark.anyio
async def test_source_change_during_model_work_rejects_the_proposal(
    recipe, sql_storage, clock, tmp_path
):
    kb, memory, source, intent, view = available(recipe, sql_storage, clock, tmp_path)

    async def decide(payload):
        kb.replace_facts([replace(source, content="Recovery copy withdrawn")])
        return proposal()

    result = await invoke(recipe, memory, view, decide, max_model_calls=1)
    current = memory.get_intent(intent.id)
    assert result.status == "context_changed" and result.model_calls_reserved == 1
    assert current.next_step == "" and current.basis == () and current.lease is None
    assert memory.resume(intent.id).open_actions == ()


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["manual", "deadline", "prepared", "unknown", "foreign_lease"])
async def test_callback_cannot_overwrite_control_state_or_a_replacement_lease(
    recipe, sql_storage, clock, tmp_path, change
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)
    if change == "deadline":
        intent = memory.create_intent(
            client_key="expires", goal="Restore", deadline=clock() + timedelta(seconds=1)
        )
        view = replace(view, packet=memory.resume(intent.id))
    foreign = []

    async def decide(payload):
        current = memory.get_intent(intent.id)
        if change == "manual":
            memory.checkpoint(
                intent.id,
                lease_token=current.lease.token,
                expected_revision=current.revision,
                state=IntentState.BLOCKED,
                progress={"manual": "stop"},
            )
        elif change == "deadline":
            clock.advance(seconds=2)
        elif change == "foreign_lease":
            clock.advance(seconds=301)
            memory.resume(intent.id)
            foreign.append(memory.claim(intent.id, worker_id="replacement"))
        else:
            action = memory.prepare_action(
                intent.id, lease_token=current.lease.token, action_key="pending", request={}
            )
            if change == "unknown":
                memory.begin_action(action.id, lease_token=current.lease.token)
                memory.finish_action(
                    action.id, lease_token=current.lease.token, outcome="unknown", result={}
                )
        return proposal()

    result = await invoke(recipe, memory, view, decide, max_model_calls=1)
    assert (
        result.status == "blocked" and result.model_calls_reserved == 1 and result.proposal is None
    )
    current = memory.get_intent(intent.id)
    assert current.next_step == "" and "_ai_knot_planning" not in current.progress
    if change == "manual":
        assert current.state == IntentState.BLOCKED and current.progress == {"manual": "stop"}
    elif change == "deadline":
        assert "deadline_exceeded" in memory.resume(intent.id).reasons
    elif change == "foreign_lease":
        assert current.lease == foreign[0]
    else:
        assert memory.resume(intent.id).open_actions[0].state == (
            "prepared" if change == "prepared" else "unknown"
        )


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["addition", "deletion", "overflow"])
async def test_complete_corpus_change_during_callback_rejects_even_unchanged_selected_ref(
    recipe, sql_storage, clock, tmp_path, change
):
    kb, memory, source, intent, view = available(recipe, sql_storage, clock, tmp_path)

    async def decide(payload):
        if change == "deletion":
            kb.forget(source.id)
        else:
            additions = [
                Fact(
                    "Another applicable condition",
                    id=f"added-{i}",
                    qualifiers={"task_key": "restore"},
                )
                for i in range(8 if change == "overflow" else 1)
            ]
            kb.replace_facts([source, *additions])
        return proposal()

    result = await invoke(recipe, memory, view, decide, max_model_calls=1)
    assert result.status == "context_changed" and result.model_calls_reserved == 1
    assert (
        memory.get_intent(intent.id).next_step == "" and memory.get_intent(intent.id).lease is None
    )


def shared_available(recipe, sql_storage, clock, tmp_path):
    from tests.always_on.test_task_snapshot import team

    kb, pool, memory = team(sql_storage, clock)
    workspace = WorkspaceMemory(sql_storage, "shared-planning")
    workspace.grant("owner", "maintain")
    memory = AlwaysOnMemory(kb, pool=pool, workspace=workspace, clock=clock)
    own = Fact("Local recovery material is present", id="same", qualifiers={"task_key": "restore"})
    shared = Fact(
        "Reviewed offsite recovery material is present",
        id="same",
        visibility_scope="ops",
        qualifiers={"task_key": "restore"},
    )
    kb.replace_facts([own])
    sql_storage.save("__shared__", [shared])
    pool.grant_read("owner", "ops")
    intent = memory.create_intent(client_key="shared", goal="Restore after observations")
    memory.subscribe(subscription_id="shared", selector=WatchSelector("__shared__"))
    view = recipe.inspect_turn(
        memory,
        recipe.Inbox(tmp_path / "shared-inbox.db"),
        intent_id=intent.id,
        subscription_id="shared",
        requirements=("backup receipt",),
        task_key="restore",
    )
    assert view.status == "missing_context"
    return kb, pool, memory, shared, intent, view


@pytest.mark.anyio
async def test_namespaced_same_id_sources_keep_distinct_citations(
    recipe, sql_storage, clock, tmp_path
):
    _, _, memory, _, intent, view = shared_available(recipe, sql_storage, clock, tmp_path)

    async def decide(payload):
        assert len(payload["facts"]) == 2
        assert all(set(f) == {"index", "content"} for f in payload["facts"])
        return proposal(basis=[1, 2])

    result = await invoke(recipe, memory, view, decide, max_model_calls=1)
    assert result.status == "planned"
    assert {(r.namespace, r.fact_id) for r in memory.get_intent(intent.id).basis} == {
        ("owner", "same"),
        ("__shared__", "same"),
    }


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["acl", "freshness"])
async def test_shared_source_withdrawal_during_callback_rejects_plan(
    recipe, sql_storage, clock, tmp_path, change
):
    _, pool, memory, shared, intent, view = shared_available(recipe, sql_storage, clock, tmp_path)
    ref = memory.capture_evidence((("__shared__", shared.id),))[0]
    if change == "freshness":
        memory.set_freshness(ref, source_ref="copy-service", recheck_after_seconds=1)
        memory.record_verification(
            ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"checked": True}, "v1")
        )

    async def decide(payload):
        if change == "acl":
            pool.revoke_read("owner", "ops")
        else:
            clock.advance(seconds=2)
        return proposal(basis=[1])

    result = await invoke(recipe, memory, view, decide, max_model_calls=1)
    assert result.status == "context_changed" and result.model_calls_reserved == 1
    assert memory.get_intent(intent.id).basis == () and memory.get_intent(intent.id).lease is None


@pytest.mark.anyio
async def test_no_sql_transaction_is_held_during_callback(
    recipe, sql_storage, clock, tmp_path, monkeypatch
):
    _, memory, _, _, view = available(recipe, sql_storage, clock, tmp_path)
    original = memory._transaction
    active = []

    @contextmanager
    def measured(*args, **kwargs):
        with original(*args, **kwargs) as tx:
            active.append(tx)
            try:
                yield tx
            finally:
                active.pop()

    monkeypatch.setattr(memory, "_transaction", measured)

    async def decide(payload):
        assert not active
        return proposal()

    assert (await invoke(recipe, memory, view, decide, max_model_calls=1)).status == "planned"


@pytest.mark.anyio
async def test_new_host_progress_and_trigger_can_plan_again_without_reusing_old_output(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def decide(payload):
        assert "previous_proposal" not in payload and "_ai_knot_planning" not in payload["progress"]
        calls.append(payload)
        return proposal()

    assert (await invoke(recipe, memory, view, decide, max_model_calls=1)).status == "planned"
    assert (
        await invoke(recipe, memory, view, decide, max_model_calls=1, trigger_key="event-2")
    ).status == "planned"
    lease = memory.claim(intent.id, worker_id="application")
    current = memory.get_intent(intent.id)
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=current.revision,
        state=IntentState.READY,
        progress={**current.progress, "phase": "after-work"},
        basis=current.basis,
    )
    assert (
        await invoke(recipe, memory, view, decide, max_model_calls=1, trigger_key="event-2")
    ).status == "planned"
    assert len(calls) == 3


@pytest.mark.anyio
async def test_two_hosts_do_not_both_pay_while_a_lease_is_live(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def decide(payload):
        calls.append(payload)
        started.set()
        await release.wait()
        return proposal()

    first = asyncio.create_task(invoke(recipe, memory, view, decide, max_model_calls=1))
    await started.wait()
    independent = AlwaysOnMemory(
        KnowledgeBase("owner", storage=sql_storage, embed_url=""), clock=clock
    )
    second = await invoke(recipe, independent, view, decide, max_model_calls=1)
    assert second.status == "blocked" and second.model_calls_reserved == 0
    release.set()
    assert (await first).status == "planned"
    assert (
        await invoke(recipe, independent, view, decide, max_model_calls=1)
    ).status == "unchanged"
    assert len(calls) == 1 and memory.get_intent(intent.id).lease is None


@pytest.mark.anyio
@pytest.mark.parametrize("condition", ["empty", "overflow", "input_bytes"])
async def test_unusable_input_never_invokes_model(recipe, sql_storage, clock, tmp_path, condition):
    kb, memory, source, intent, view = available(recipe, sql_storage, clock, tmp_path)
    if condition == "empty":
        kb.forget(source.id)
    elif condition == "overflow":
        kb.replace_facts(
            [
                source,
                *[Fact("Other condition", qualifiers={"task_key": "restore"}) for _ in range(8)],
            ]
        )
    else:
        kb.replace_facts([source, Fact("n" * 2000, qualifiers={"task_key": "restore"})])
        lease = memory.claim(intent.id, worker_id="application")
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=memory.get_intent(intent.id).revision,
            state=IntentState.READY,
            progress={"blob": "🙂" * 16000},
        )

    async def forbidden(payload):
        raise AssertionError("No callback for unusable input")

    result = await invoke(recipe, memory, view, forbidden, max_model_calls=1)
    assert (
        result.status
        == {"empty": "no_context", "overflow": "snapshot_too_large", "input_bytes": "input_limit"}[
            condition
        ]
    )
    assert result.model_calls_reserved == 0 and memory.get_intent(intent.id).lease is None


@pytest.mark.anyio
async def test_a_saved_proposal_still_needs_fresh_action_admission(
    recipe, sql_storage, clock, tmp_path
):
    kb, memory, source, intent, view = available(recipe, sql_storage, clock, tmp_path)

    async def decide(payload):
        return proposal()

    assert (await invoke(recipe, memory, view, decide, max_model_calls=1)).status == "planned"
    lease = memory.claim(intent.id, worker_id="executor")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="copy", request={"choice": "copy"}
    )
    kb.replace_facts([replace(source, content="Recovery material withdrawn")])
    with pytest.raises(StaleEvidence):
        memory.begin_action(action.id, lease_token=lease.token)
    assert (
        next(a for a in memory.resume(intent.id).open_actions if a.id == action.id).state
        == "prepared"
    )


@pytest.mark.anyio
async def test_checkpoint_conflict_preserves_a_new_manual_block_without_retry(
    recipe, sql_storage, clock, tmp_path, monkeypatch
):
    _, memory, _, intent, view = available(recipe, sql_storage, clock, tmp_path)
    checkpoint = memory.checkpoint
    attempts = []

    def race(*args, **kwargs):
        attempts.append(kwargs)
        current = memory.get_intent(intent.id)
        checkpoint(
            intent.id,
            lease_token=current.lease.token,
            expected_revision=current.revision,
            state=IntentState.BLOCKED,
            progress={"operator": "stop"},
        )
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(memory, "checkpoint", race)

    async def decide(payload):
        return proposal()

    result = await invoke(recipe, memory, view, decide, max_model_calls=1)
    current = memory.get_intent(intent.id)
    assert result.status == "failed" and result.model_calls_reserved == 1
    assert len(attempts) == 1 and result.proposal is None
    assert current.state == IntentState.BLOCKED and current.progress == {"operator": "stop"}


@pytest.mark.anyio
async def test_restored_verification_can_replan_a_source_block(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, source, intent, view = available(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def decide(payload):
        calls.append(payload)
        return proposal()

    assert (await invoke(recipe, memory, view, decide, max_model_calls=1)).status == "planned"
    refs = memory.get_intent(intent.id).basis
    memory.set_freshness(refs[0], source_ref="copy-service", recheck_after_seconds=1)
    memory.record_verification(
        refs[0], result=VerificationResult(VerificationOutcome.CONFIRMED, {"receipt": "v1"}, "v1")
    )
    clock.advance(seconds=2)
    assert memory.resume(intent.id).freshness_due
    memory.record_verification(
        refs[0], result=VerificationResult(VerificationOutcome.CONFIRMED, {"receipt": "v2"}, "v2")
    )
    assert not memory.resume(intent.id).freshness_due
    result = await invoke(recipe, memory, view, decide, max_model_calls=1, trigger_key="restored")
    assert result.status == "planned" and len(calls) == 2
    assert calls[-1]["facts"] == [{"index": 1, "content": source.content}]
    assert memory.review_context(memory.get_intent(intent.id).basis).complete
