"""An exact selected basis cannot detect a newly added, unselected condition."""

from dataclasses import replace
from datetime import timedelta

import pytest

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import (
    AlwaysOnMemory,
    IntentState,
    LeaseLost,
    RevisionConflict,
    SnapshotTooLarge,
    StaleEvidence,
    TaskContextGuard,
    VerificationOutcome,
    VerificationResult,
)
from ai_knot.types import Fact, MemoryType


def setup(sql_storage, clock):
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    source = Fact(
        "Current deployment destination", id="destination", qualifiers={"task_key": "rollout"}
    )
    kb.replace_facts([source])
    memory = AlwaysOnMemory(kb, clock=clock)
    intent = memory.create_intent(
        client_key="rollout", goal="Act only when current conditions permit"
    )
    lease = memory.claim(intent.id, worker_id="host")
    basis = memory.capture_evidence((("owner", source.id),))
    return kb, memory, source, intent, lease, basis


def test_guarded_checkpoint_rejects_a_new_unselected_condition_without_mutation(sql_storage, clock):
    kb, memory, source, intent, lease, basis = setup(sql_storage, clock)
    guard = memory.task_guard(task_key="rollout")
    before = memory.get_intent(intent.id)
    kb.replace_facts([source, Fact("Do not deploy until the incident is cleared", id="condition")])
    assert memory.review_context(basis).complete
    with pytest.raises(StaleEvidence):
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=before.revision,
            state=IntentState.READY,
            progress={"planned": True},
            basis=basis,
            context_guard=guard,
        )
    assert memory.get_intent(intent.id) == before


def test_matching_guard_keeps_selected_basis_and_ignores_other_tasks_or_episodes(
    sql_storage, clock
):
    kb, memory, source, intent, lease, basis = setup(sql_storage, clock)
    guard = memory.task_guard(task_key="rollout")
    kb.replace_facts(
        [
            source,
            Fact("Inventory observation", id="other", qualifiers={"task_key": "inventory"}),
            Fact("Raw prior event", id="event", type=MemoryType.EPISODIC),
        ]
    )
    assert memory.task_guard(task_key="rollout") == guard
    current = memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state=IntentState.READY,
        progress={"planned": True},
        basis=basis,
        context_guard=guard,
    )
    assert current.basis == basis and current.lease is None


def test_explicit_pre_action_guard_rejects_later_addition_and_keeps_action_prepared(
    sql_storage, clock
):
    kb, memory, source, intent, lease, basis = setup(sql_storage, clock)
    guard = memory.task_guard(task_key="rollout")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state=IntentState.READY,
        progress={},
        basis=basis,
        context_guard=guard,
    )
    lease = memory.claim(intent.id, worker_id="executor")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="deploy", request={}
    )
    kb.replace_facts([source, Fact("A separate sign-off is now needed", id="new")])
    with pytest.raises(StaleEvidence):
        memory.begin_action(action.id, lease_token=lease.token, context_guard=guard)
    packet = memory.resume(intent.id)
    assert packet.intent.state == IntentState.BLOCKED and packet.open_actions[0].state == "prepared"


def test_guard_has_no_content_budget_and_empty_set_is_valid(sql_storage, clock):
    kb, memory, source, *_ = setup(sql_storage, clock)
    kb.replace_facts([])
    empty = memory.task_guard(task_key="rollout")
    assert len(empty.digest) == 64 and empty.max_facts == 128
    kb.replace_facts([replace(source, content="Large source " * 10000)])
    assert memory.task_guard(task_key="rollout") != empty


@pytest.mark.parametrize(
    "change", ["add", "delete", "revise", "activate", "expire", "support", "slot"]
)
def test_guard_detects_unselected_membership_or_revision_changes(sql_storage, clock, change):
    kb, memory, source, intent, lease, basis = setup(sql_storage, clock)
    extra = Fact("Another applicable prerequisite", id="extra", slot_key="guard::extra", version=1)
    if change == "activate":
        extra = replace(extra, valid_from=clock() + timedelta(seconds=1))
    if change == "expire":
        extra = replace(extra, valid_until=clock() + timedelta(seconds=1))
    if change == "support":
        extra = replace(extra, supported=False)
    kb.replace_facts([source, extra])
    guard = memory.task_guard(task_key="rollout")
    if change == "delete":
        kb.replace_facts([source])
    elif change == "revise":
        kb.replace_facts([source, replace(extra, content="Revised prerequisite")])
    elif change == "support":
        kb.replace_facts([source, replace(extra, supported=True)])
    elif change in ("activate", "expire"):
        clock.advance(seconds=2)
    elif change == "slot":
        kb.replace_facts(
            [source, extra, replace(extra, id="peer", version=2, qualifiers={"task_key": "other"})]
        )
    else:
        kb.replace_facts([source, extra, Fact("New condition", id="added")])
    assert memory.review_context(basis).complete
    before = memory.get_intent(intent.id)
    with pytest.raises(StaleEvidence):
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=before.revision,
            state=IntentState.READY,
            progress={},
            basis=basis,
            context_guard=guard,
        )
    assert memory.get_intent(intent.id) == before


def test_descriptor_and_source_limit_are_strict(sql_storage, clock):
    kb, memory, source, *_ = setup(sql_storage, clock)
    for bad in (True, 0, 129, "1"):
        with pytest.raises(ValueError):
            memory.task_guard(task_key="rollout", max_facts=bad)
        with pytest.raises(ValueError):
            TaskContextGuard("rollout", "a" * 64, bad)
    for bad in ("", "A" * 64, "x" * 64, 1, None):
        with pytest.raises(ValueError):
            TaskContextGuard("rollout", bad)
    assert memory.task_guard(task_key="rollout", max_facts=1)
    guard = memory.task_guard(task_key="rollout", max_facts=1)
    kb.replace_facts([source, Fact("New condition", id="new")])
    with pytest.raises(SnapshotTooLarge):
        memory.task_guard(task_key="rollout", max_facts=1)
    intent = memory.list_intents()[0]
    with pytest.raises(StaleEvidence):
        memory.checkpoint(
            intent.id,
            lease_token=intent.lease.token,
            expected_revision=intent.revision,
            state=IntentState.READY,
            progress={},
            context_guard=guard,
        )


@pytest.mark.parametrize(
    "field,value", [("task_key", "other"), ("max_facts", 127), ("digest", "0" * 64)]
)
def test_tampered_descriptor_cannot_match(sql_storage, clock, field, value):
    _, memory, _, intent, lease, basis = setup(sql_storage, clock)
    guard = replace(memory.task_guard(task_key="rollout"), **{field: value})
    with pytest.raises(StaleEvidence):
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=memory.get_intent(intent.id).revision,
            state=IntentState.READY,
            progress={},
            basis=basis,
            context_guard=guard,
        )


def test_wrong_lease_or_revision_cannot_mutate_on_guard_mismatch(sql_storage, clock):
    kb, memory, source, intent, lease, basis = setup(sql_storage, clock)
    guard = memory.task_guard(task_key="rollout")
    kb.replace_facts([source, Fact("New condition", id="new")])
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="work", request={}
    )
    before = memory.get_intent(intent.id)
    with pytest.raises(LeaseLost):
        memory.begin_action(action.id, lease_token="wrong", context_guard=guard)
    with pytest.raises(RevisionConflict):
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=before.revision + 1,
            state=IntentState.READY,
            progress={},
            basis=basis,
            context_guard=guard,
        )
    assert memory.get_intent(intent.id) == before


def test_unguarded_existing_calls_and_matching_pre_action_guard_remain_compatible(
    sql_storage, clock
):
    _, memory, _, intent, lease, basis = setup(sql_storage, clock)
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state=IntentState.READY,
        progress={},
        basis=basis,
    )
    lease = memory.claim(intent.id, worker_id="executor")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="work", request={}
    )
    permit = memory.begin_action(
        action.id, lease_token=lease.token, context_guard=memory.task_guard(task_key="rollout")
    )
    assert permit.action_id == action.id


def test_guard_is_order_independent_and_binds_logical_owner(sql_storage, clock):
    pool = SharedMemoryPool(sql_storage, persist_stats=True, embed_url="", clock=clock)
    for owner in ("owner", "other"):
        pool.grant_read(owner, "ops")
    shared = [
        Fact("First shared observation", id="z", visibility_scope="ops"),
        Fact("Second shared observation", id="a", visibility_scope="ops"),
    ]
    sql_storage.save("__shared__", shared)
    first = AlwaysOnMemory(
        KnowledgeBase("owner", storage=sql_storage, embed_url=""), pool=pool, clock=clock
    )
    other = AlwaysOnMemory(
        KnowledgeBase("other", storage=sql_storage, embed_url=""), pool=pool, clock=clock
    )
    guard = first.task_guard(task_key="rollout")
    assert guard != other.task_guard(task_key="rollout")
    sql_storage.save("__shared__", list(reversed(shared)))
    assert first.task_guard(task_key="rollout") == guard


@pytest.mark.parametrize("change", ["revoke", "freshness", "lineage"])
def test_shared_access_or_unselected_source_admission_change_is_detected(
    sql_storage, clock, change
):
    kb, memory, source, intent, lease, basis = setup(sql_storage, clock)
    pool = SharedMemoryPool(sql_storage, persist_stats=True, embed_url="", clock=clock)
    pool.grant_read("owner", "ops")
    shared = Fact("Shared prerequisite", id="shared", visibility_scope="ops")
    sql_storage.save("__shared__", [shared])
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    ref = memory.capture_evidence((("__shared__", shared.id),))[0]
    if change == "freshness":
        extra = Fact("Independent local prerequisite", id="extra")
        kb.replace_facts([source, extra])
        private = memory.capture_evidence((("owner", extra.id),))[0]
        memory.set_freshness(private, source_ref="host", recheck_after_seconds=1)
        memory.record_verification(
            private,
            result=VerificationResult(VerificationOutcome.CONFIRMED, {"receipt": True}, "v1"),
        )
    if change == "lineage":
        memory.derive_fact("Current projection", basis=(ref,), task_key="rollout")
    guard = memory.task_guard(task_key="rollout")
    if change == "revoke":
        pool.revoke_read("owner", "ops")
    elif change == "freshness":
        clock.advance(seconds=2)
    else:
        sql_storage.save("__shared__", [replace(shared, content="Corrected prerequisite")])
    assert memory.review_context(basis).complete
    before = memory.get_intent(intent.id)
    with pytest.raises(StaleEvidence):
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=before.revision,
            state=IntentState.READY,
            progress={},
            basis=basis,
            context_guard=guard,
        )
    assert memory.get_intent(intent.id) == before


def test_inaccessible_new_fact_does_not_change_admissible_guard(sql_storage, clock):
    kb, _, _, _, _, _ = setup(sql_storage, clock)
    pool = SharedMemoryPool(sql_storage, persist_stats=True, embed_url="", clock=clock)
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    guard = memory.task_guard(task_key="rollout")
    sql_storage.save(
        "__shared__", [Fact("Unreadable note", id="hidden", visibility_scope="ungranted")]
    )
    assert memory.task_guard(task_key="rollout") == guard


def test_same_id_in_private_and_shared_are_distinct_candidates(sql_storage, clock):
    kb, memory, source, *_ = setup(sql_storage, clock)
    pool = SharedMemoryPool(sql_storage, persist_stats=True, embed_url="", clock=clock)
    pool.grant_read("owner", "ops")
    sql_storage.save(
        "__shared__", [replace(source, visibility_scope="ops", content="Shared condition")]
    )
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    with pytest.raises(SnapshotTooLarge):
        memory.task_guard(task_key="rollout", max_facts=1)
    assert memory.task_guard(task_key="rollout", max_facts=2)


def test_guard_has_only_the_documented_ref_count_cap(sql_storage, clock):
    kb, memory, *_ = setup(sql_storage, clock)
    kb.replace_facts([Fact("Small source", id=f"{i:03d}-" + "r" * 520) for i in range(128)])
    assert len(memory.task_guard(task_key="rollout").digest) == 64


@pytest.mark.parametrize("condition", ["deadline", "manual", "unknown"])
def test_a_matching_guard_never_bypasses_existing_action_control(sql_storage, clock, condition):
    _, memory, _, intent, lease, _ = setup(sql_storage, clock)
    if condition == "deadline":
        intent = memory.create_intent(
            client_key="due", goal="Plan", deadline=clock() + timedelta(seconds=1)
        )
        lease = memory.claim(intent.id, worker_id="host")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="work", request={}
    )
    if condition == "deadline":
        clock.advance(seconds=2)
    elif condition == "manual":
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=memory.get_intent(intent.id).revision,
            state=IntentState.BLOCKED,
            progress={},
        )
        lease = memory.claim(intent.id, worker_id="host")
    else:
        memory.begin_action(action.id, lease_token=lease.token)
        memory.finish_action(action.id, lease_token=lease.token, outcome="unknown", result={})
    guard = memory.task_guard(task_key="rollout")
    with pytest.raises(RevisionConflict):
        memory.begin_action(action.id, lease_token=lease.token, context_guard=guard)
    state = memory.resume(intent.id).open_actions[0].state
    assert state == ("unknown" if condition == "unknown" else "prepared")


@pytest.mark.parametrize("bad", [{}, False, "not-a-descriptor"])
def test_python_writes_reject_untyped_guard_without_mutating(sql_storage, clock, bad):
    _, memory, _, intent, lease, _ = setup(sql_storage, clock)
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="work", request={}
    )
    before = memory.get_intent(intent.id)
    with pytest.raises(ValueError):
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=before.revision,
            state=IntentState.READY,
            progress={},
            context_guard=bad,
        )
    with pytest.raises(ValueError):
        memory.begin_action(action.id, lease_token=lease.token, context_guard=bad)
    assert memory.get_intent(intent.id) == before
