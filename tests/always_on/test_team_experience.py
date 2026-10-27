"""Public-path SQL contracts for outcome-bound procedural memories."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import (
    AccessDenied,
    AlwaysOnMemory,
    IdempotencyConflict,
    RevisionConflict,
    StaleEvidence,
    VerificationOutcome,
    VerificationResult,
    WatchSelector,
    WorkspaceMemory,
)
from ai_knot.always_on.storage import SQLSession, sql
from ai_knot.types import Fact, MemoryType


def completed(memory, outcome="succeeded", result=None, key="run", shared=False):
    intent = memory.create_intent(client_key=key, goal="Restore deployment")
    lease = memory.claim(intent.id, worker_id="host", ttl_seconds=300)
    action = memory.prepare_action(
        intent.id,
        lease_token=lease.token,
        action_key="restore",
        request={"private": "secret"},
        **({"operation_key": "restore", "resource_key": "deployment"} if shared else {}),
    )
    memory.begin_action(action.id, lease_token=lease.token)
    action = memory.finish_action(
        action.id,
        lease_token=lease.token,
        outcome=outcome,
        result={"receipt": "private-tool-output"} if result is None else result,
    )
    return action, lease


def record(memory, action, basis, **extra):
    return memory.record_experience(
        action.id,
        expected_action_revision=action.revision,
        experience_key="lesson",
        content="Restore checkpoint: verify backup before switching traffic.",
        basis=basis,
        **extra,
    )


@pytest.mark.parametrize("outcome", ["succeeded", "failed"])
def test_recorded_outcomes_keep_sources_types_and_replay_without_writes(kb, clock, outcome):
    memory = AlwaysOnMemory(kb, clock=clock)
    source = kb.add("Tool receipt: backup verified for rollout42")
    basis = memory.capture_evidence((("owner", source.id),))
    action, _ = completed(memory, outcome)
    result = record(memory, action, basis, task_key="rollout42", for_role="operator")
    lesson = kb.get(result.lesson.ref.fact_id)
    assert result.outcome == outcome and result.action_revision == action.revision
    assert lesson.type == MemoryType.PROCEDURAL
    assert lesson.content.startswith(f"Recorded outcome: {outcome}.")
    assert lesson.qualifiers["task_key"] == "rollout42"
    assert memory.independent_evidence((result.lesson.ref,)) == basis
    memory.subscribe(subscription_id="experience", selector=WatchSelector("owner"))
    assert record(memory, action, basis, task_key="rollout42", for_role="operator") == result
    assert not memory.poll("experience").events
    assert len(kb.list_facts()) == 2
    context = memory.select_context(("restore checkpoint",), task_key="rollout42")
    assert context.complete and context.facts == (result.lesson,)


@pytest.mark.parametrize("invalid", ["unknown", "empty_result", "old_revision", "no_basis"])
def test_recording_rejects_unresolved_unobserved_or_stale_inputs(kb, clock, invalid):
    memory = AlwaysOnMemory(kb, clock=clock)
    source = kb.add("Receipt")
    basis = memory.capture_evidence((("owner", source.id),))
    action, _ = completed(
        memory,
        "unknown" if invalid == "unknown" else "succeeded",
        {} if invalid == "empty_result" else None,
    )
    if invalid == "old_revision":
        action = replace(action, revision=action.revision - 1)
    with pytest.raises((RevisionConflict, ValueError)):
        record(memory, action, () if invalid == "no_basis" else basis)
    assert len(kb.list_facts()) == 1


@pytest.mark.parametrize("change", ["different_content", "amend", "delete"])
def test_stable_key_never_changes_or_resurrects_output(kb, clock, change):
    memory = AlwaysOnMemory(kb, clock=clock)
    source = kb.add("Receipt")
    basis = memory.capture_evidence((("owner", source.id),))
    action, _ = completed(memory)
    result = record(memory, action, basis)
    if change == "different_content":
        with pytest.raises(IdempotencyConflict):
            memory.record_experience(
                action.id,
                expected_action_revision=action.revision,
                experience_key="lesson",
                content="Changed instruction",
                basis=basis,
            )
    else:
        if change == "amend":
            kb.replace_facts(
                [
                    replace(f, content="Amended") if f.id == result.lesson.ref.fact_id else f
                    for f in kb.list_facts()
                ]
            )
        else:
            kb.forget(result.lesson.ref.fact_id)
        with pytest.raises((StaleEvidence, RevisionConflict)):
            record(memory, action, basis)


@pytest.mark.parametrize("change", ["correct", "delete", "freshness"])
def test_source_changes_retract_experience_and_copies(kb, clock, change):
    memory = AlwaysOnMemory(kb, clock=clock)
    source = kb.add("Receipt: restore verified")
    basis = memory.capture_evidence((("owner", source.id),))
    action, _ = completed(memory)
    lesson = record(memory, action, basis)
    copy = memory.derive_fact("Restore checkpoint: copied guidance", basis=(lesson.lesson.ref,))
    refs = (lesson.lesson.ref, *memory.capture_evidence((("owner", copy.id),)))
    assert memory.independent_evidence(refs) == basis
    if change == "delete":
        kb.forget(source.id)
    elif change == "correct":
        kb.replace_facts(
            [
                replace(f, content="Receipt withdrawn") if f.id == source.id else f
                for f in kb.list_facts()
            ]
        )
    else:
        memory.set_freshness(basis[0], source_ref="remote", recheck_after_seconds=10)
        memory.record_verification(
            basis[0],
            result=VerificationResult(VerificationOutcome.CONFIRMED, {"receipt": "checked"}, "v1"),
        )
        clock.advance(seconds=11)
    assert not memory.review_context(refs).accepted
    assert not memory.select_context(("restore checkpoint",)).facts
    with pytest.raises((StaleEvidence, RevisionConflict)):
        record(memory, action, basis)


@pytest.mark.parametrize("outcome", ["succeeded", "retryable"])
def test_action_reconciliation_retracts_lesson_and_derived_copies(kb, clock, outcome):
    memory = AlwaysOnMemory(kb, clock=clock)
    source = kb.add("Receipt")
    basis = memory.capture_evidence((("owner", source.id),))
    action, lease = completed(memory, "failed")
    result = record(memory, action, basis)
    copy = memory.derive_fact("Restore checkpoint copy", basis=(result.lesson.ref,))
    refs = (result.lesson.ref, *memory.capture_evidence((("owner", copy.id),)))
    memory.subscribe(
        subscription_id="outcome-changes",
        selector=WatchSelector("owner", fact_ids=(result.lesson.ref.fact_id,)),
    )
    memory.reconcile_action(
        action.id,
        lease_token=lease.token,
        outcome=outcome,
        evidence={"independent_check": "corrected"},
    )
    assert not memory.review_context(refs).accepted
    assert not memory.select_context(("restore checkpoint",)).facts
    assert not kb.get(result.lesson.ref.fact_id).supported
    assert any(
        event.fact_id == result.lesson.ref.fact_id
        for event in memory.poll("outcome-changes").events
    )
    with pytest.raises(RevisionConflict):
        record(memory, action, basis)


def team(sql_storage, clock):
    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock)
    workspace = WorkspaceMemory(sql_storage, "team")
    workspace.grant("owner", "maintain")
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    memory = AlwaysOnMemory(kb, pool=pool, workspace=workspace, clock=clock)
    source = Fact("Receipt from tool", origin_agent_id="writer", visibility_scope="team:ops")
    sql_storage.save("__shared__", [source])
    pool.grant_read("owner", "team:ops")
    pool.grant_read("peer", "team:ops")
    basis = memory.capture_evidence((("__shared__", source.id),))
    peer = AlwaysOnMemory(
        KnowledgeBase("peer", storage=sql_storage, embed_url=""), pool=pool, clock=clock
    )
    return kb, memory, peer, pool, workspace, basis


def test_shared_record_requires_authority_hides_payload_and_retracts_without_pool(
    sql_storage, clock
):
    kb, memory, peer, pool, workspace, basis = team(sql_storage, clock)
    action, lease = completed(memory, "failed")
    workspace.revoke("owner", "maintain")
    with pytest.raises(AccessDenied):
        record(memory, action, basis, namespace="__shared__", visibility_scope="team:ops")
    workspace.grant("owner", "maintain")
    result = record(memory, action, basis, namespace="__shared__", visibility_scope="team:ops")
    assert peer.review_context((result.lesson.ref,)).complete
    shared = next(f for f in sql_storage.load("__shared__") if f.id == result.lesson.ref.fact_id)
    assert all(
        private not in repr(shared)
        for private in (action.id, action.intent_id, "secret", "private-tool-output")
    )
    reopened = AlwaysOnMemory(
        KnowledgeBase("owner", storage=sql_storage, embed_url=""), workspace=workspace, clock=clock
    )
    reopened.reconcile_action(
        action.id, lease_token=lease.token, outcome="succeeded", evidence={"checked": True}
    )
    assert not peer.review_context((result.lesson.ref,)).accepted
    assert not peer.select_context(("restore checkpoint",)).facts


def test_shared_experience_cannot_launder_private_roots_and_revocation_retracts(sql_storage, clock):
    kb, memory, peer, pool, _, basis = team(sql_storage, clock)
    action, _ = completed(memory)
    private = kb.add("Private receipt")
    private_basis = memory.capture_evidence((("owner", private.id),))
    hidden = record(memory, action, private_basis, namespace="__shared__")
    assert not peer.review_context((hidden.lesson.ref,)).accepted
    result = memory.record_experience(
        action.id,
        expected_action_revision=action.revision,
        experience_key="public",
        content="Restore checkpoint",
        basis=basis,
        namespace="__shared__",
        visibility_scope="team:ops",
    )
    assert peer.review_context((result.lesson.ref,)).complete
    pool.revoke_read("peer", "team:ops")
    assert not peer.review_context((result.lesson.ref,)).accepted
    assert not peer.select_context(("restore checkpoint",)).facts


def test_action_and_experience_invalidation_roll_back_together(kb, clock, monkeypatch):
    memory = AlwaysOnMemory(kb, clock=clock)
    basis = memory.capture_evidence((("owner", kb.add("Receipt").id),))
    action, lease = completed(memory, "failed")
    result = record(memory, action, basis)
    original = SQLSession.replace_facts

    def fail_on_invalidation(session, namespace, facts):
        if any(f.id == result.lesson.ref.fact_id and not f.supported for f in facts):
            raise RuntimeError("Injected write failure")
        return original(session, namespace, facts)

    monkeypatch.setattr(SQLSession, "replace_facts", fail_on_invalidation)
    with pytest.raises(RuntimeError, match="Injected write failure"):
        memory.reconcile_action(
            action.id, lease_token=lease.token, outcome="succeeded", evidence={"checked": True}
        )
    assert record(memory, action, basis) == result
    assert memory.review_context((result.lesson.ref,)).complete


def test_recording_raced_with_reconciliation_never_leaves_live_old_advice(kb, clock):
    memory = AlwaysOnMemory(kb, clock=clock)
    basis = memory.capture_evidence((("owner", kb.add("Receipt").id),))
    action, lease = completed(memory, "failed")
    barrier = Barrier(2)

    def recorder():
        barrier.wait()
        try:
            return record(memory, action, basis)
        except RevisionConflict:
            return None

    def corrector():
        barrier.wait()
        memory.reconcile_action(
            action.id, lease_token=lease.token, outcome="succeeded", evidence={"checked": True}
        )

    with ThreadPoolExecutor(2) as workers:
        recorded, corrected = workers.submit(recorder), workers.submit(corrector)
        result = recorded.result(timeout=10)
        corrected.result(timeout=10)
    if result:
        assert not memory.review_context((result.lesson.ref,)).accepted
    assert not memory.select_context(("restore checkpoint",)).facts


@pytest.mark.parametrize("state", ["prepared", "started"])
def test_nonterminal_action_cannot_publish_experience(kb, clock, state):
    memory = AlwaysOnMemory(kb, clock=clock)
    basis = memory.capture_evidence((("owner", kb.add("Receipt").id),))
    intent = memory.create_intent(client_key="run", goal="Restore")
    lease = memory.claim(intent.id, worker_id="host")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="restore", request={}
    )
    if state == "started":
        memory.begin_action(action.id, lease_token=lease.token)
    with pytest.raises(RevisionConflict):
        record(memory, action, basis)


def test_replay_finish_keeps_lesson_and_new_known_revision_can_form_new_experience(kb, clock):
    memory = AlwaysOnMemory(kb, clock=clock)
    basis = memory.capture_evidence((("owner", kb.add("Receipt").id),))
    action, lease = completed(memory, "failed")
    old = record(memory, action, basis)
    memory.finish_action(action.id, lease_token=lease.token, outcome="failed", result=action.result)
    assert memory.review_context((old.lesson.ref,)).complete
    current = memory.reconcile_action(
        action.id, lease_token=lease.token, outcome="succeeded", evidence={"receipt": "corrected"}
    )
    new = memory.record_experience(
        action.id,
        expected_action_revision=current.revision,
        experience_key="corrected",
        content="Restore checkpoint: inspect independent receipt",
        basis=basis,
    )
    assert new.outcome == "succeeded" and new.lesson.ref != old.lesson.ref
    assert memory.review_context((new.lesson.ref,)).complete
    assert not memory.review_context((old.lesson.ref,)).accepted
    assert memory.independent_evidence((new.lesson.ref,)) == basis


def test_foreign_owner_and_workspace_cannot_capture_action(kb, sql_storage, clock):
    memory = AlwaysOnMemory(kb, clock=clock)
    basis = memory.capture_evidence((("owner", kb.add("Receipt").id),))
    action, _ = completed(memory)
    peer = AlwaysOnMemory(KnowledgeBase("peer", storage=sql_storage), clock=clock)
    with pytest.raises(AccessDenied):
        record(peer, action, basis)
    _, bound, _, _, workspace, shared_basis = team(sql_storage, clock)
    workspace.grant("owner", "execute")
    workspace_action, _ = completed(bound, key="workspace", shared=True)
    with pytest.raises(AccessDenied):
        record(memory, workspace_action, shared_basis)


def test_schema7_upgrade_preserves_work_and_allows_reopen(kb, sql_storage, clock):
    memory = AlwaysOnMemory(kb, clock=clock)
    action, _ = completed(memory)
    with memory._transaction() as session:
        session.execute(sql("DROP TABLE ao_experiences"))
        session.execute(sql("UPDATE ao_schema SET version=7 WHERE id=1"))
    address = getattr(sql_storage, "_db_path", None) or sql_storage._dsn
    reopened = AlwaysOnMemory(
        KnowledgeBase("owner", storage=type(sql_storage)(address)), clock=clock
    )
    basis = reopened.capture_evidence((("owner", kb.add("Receipt").id),))
    assert record(reopened, action, basis).outcome == "succeeded"
    with reopened._transaction(write=False) as session:
        assert session.fetch_one(sql("SELECT version FROM ao_schema WHERE id=1"))["version"] == 8


def test_unbound_action_never_loads_or_write_locks_shared_facts(kb, clock, monkeypatch):
    memory = AlwaysOnMemory(kb, clock=clock)
    original = SQLSession.load_facts

    def load(session, namespace, **kwargs):
        assert namespace != "__shared__"
        if session._write:
            assert "__shared__" not in session._namespaces
        return original(session, namespace, **kwargs)

    monkeypatch.setattr(SQLSession, "load_facts", load)
    action, _ = completed(memory)
    assert action.state == "succeeded"


@pytest.mark.parametrize("persistent", [True, False])
def test_publication_preserves_action_binding_or_rejects_untracked_copy(
    sql_storage, clock, persistent
):
    kb, memory, peer, _, _, basis = team(sql_storage, clock)
    action, lease = completed(memory, "failed")
    original = record(memory, action, basis)
    prior = Fact("Current inventory state", slot_key="restore::checkpoint", valid_from=clock())
    sql_storage.save("__shared__", [*sql_storage.load("__shared__"), prior])
    kb.replace_facts(
        [
            replace(f, slot_key=prior.slot_key) if f.id == original.lesson.ref.fact_id else f
            for f in kb.list_facts()
        ]
    )
    publisher = SharedMemoryPool(sql_storage, persist_stats=persistent, clock=clock)
    publisher.register("owner")
    publisher.grant_read("owner", "team:ops")
    copies = publisher.publish("owner", [original.lesson.ref.fact_id], kb=kb)
    # Shared experiences use the guarded writer/maintain path, not ordinary publication.
    assert not copies
    unchanged = next(f for f in sql_storage.load("__shared__") if f.id == prior.id)
    assert unchanged.valid_until is None
    memory.reconcile_action(
        action.id, lease_token=lease.token, outcome="succeeded", evidence={"receipt": "corrected"}
    )
    assert not peer.select_context(("restore checkpoint",)).facts


def test_terminal_workspace_alias_cannot_form_current_experience(sql_storage, clock):
    _, memory, _, pool, workspace, basis = team(sql_storage, clock)
    workspace.grant("owner", "execute")
    workspace.grant("peer", "execute")
    action, lease = completed(memory, "failed", shared=True)
    peer = AlwaysOnMemory(
        KnowledgeBase("peer", storage=sql_storage), pool=pool, workspace=workspace, clock=clock
    )
    intent = peer.create_intent(client_key="alias", goal="Restore")
    other_lease = peer.claim(intent.id, worker_id="host")
    alias = peer.prepare_action(
        intent.id,
        lease_token=other_lease.token,
        action_key="restore",
        request={"private": "secret"},
        operation_key="restore",
        resource_key="deployment",
    )
    assert alias.state == "failed"
    with pytest.raises(RevisionConflict):
        record(peer, alias, basis)
    memory.reconcile_action(
        action.id, lease_token=lease.token, outcome="succeeded", evidence={"receipt": "corrected"}
    )
    with pytest.raises(RevisionConflict):
        record(peer, alias, basis)


def test_snapshot_restore_cannot_resurrect_action_bound_lesson_or_copy(kb, clock):
    memory = AlwaysOnMemory(kb, clock=clock)
    basis = memory.capture_evidence((("owner", kb.add("Receipt").id),))
    action, lease = completed(memory, "failed")
    result = record(memory, action, basis)
    copy = memory.derive_fact("Restore checkpoint copy", basis=(result.lesson.ref,))
    refs = (result.lesson.ref, *memory.capture_evidence((("owner", copy.id),)))
    kb.snapshot("before-correction")
    memory.reconcile_action(
        action.id, lease_token=lease.token, outcome="succeeded", evidence={"receipt": "corrected"}
    )
    kb.restore("before-correction")
    assert not memory.review_context(refs).accepted
    assert not memory.select_context(("restore checkpoint",)).facts
    assert not {ref.fact_id for ref in refs} & {
        f.id for f in kb.recall_facts("Restore checkpoint", top_k=10)
    }
