"""Queryless context is an all-or-error source snapshot, never a relevance claim."""

from dataclasses import replace
from datetime import timedelta

import pytest

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import (
    AlwaysOnMemory,
    SnapshotTooLarge,
    VerificationOutcome,
    VerificationResult,
    WatchSelector,
    WorkspaceMemory,
)
from ai_knot.always_on.codec import canonical_json
from ai_knot.types import Fact, MemoryType


def team(sql_storage, clock):
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    pool = SharedMemoryPool(sql_storage, persist_stats=True, embed_url="", clock=clock)
    return kb, pool, AlwaysOnMemory(kb, pool=pool, clock=clock)


def source(clock, text="Verified recovery copy", *, fact_id="one", task="restore", **kwargs):
    return Fact(
        text,
        id=fact_id,
        created_at=clock(),
        valid_from=clock(),
        qualifiers={"task_key": task} if task else {},
        **kwargs,
    )


def test_snapshot_discovers_scoped_sources_without_a_query_or_inference(
    sql_storage, clock, monkeypatch
):
    kb, pool, memory = team(sql_storage, clock)
    own = source(clock)
    generic = source(clock, "Generic runbook", fact_id="generic", task="")
    shared = source(clock, "Shared receipt", visibility_scope="ops", origin_agent_id="peer")
    rejected = [
        replace(shared, id="denied", visibility_scope="closed"),
        replace(shared, id="wrong", qualifiers={"task_key": "other"}),
        replace(shared, id="unsupported", supported=False),
        replace(shared, id="expired", valid_until=clock()),
        replace(shared, id="future", valid_from=clock() + timedelta(days=1)),
        replace(shared, id="episode", type=MemoryType.EPISODIC),
    ]
    kb.replace_facts([own, generic])
    sql_storage.save("__shared__", [shared, *rejected])
    pool.grant_read("owner", "ops")

    def forbidden(*args, **kwargs):
        raise AssertionError("Task snapshot must not invoke inference")

    monkeypatch.setattr("ai_knot.embedder.embed_texts", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    result = memory.task_snapshot(task_key="restore")
    assert [(f.ref.namespace, f.ref.fact_id) for f in result] == [
        ("__shared__", "one"),
        ("owner", "generic"),
        ("owner", "one"),
    ]
    assert {f.content for f in result} == {own.content, shared.content, generic.content}
    assert memory.review_context(tuple(f.ref for f in result)).complete
    assert not hasattr(result, "complete")
    assert [f.ref.fact_id for f in memory.task_snapshot(task_key="")] == ["generic"]
    pool.revoke_read("owner", "ops")
    assert len(memory.task_snapshot(task_key="restore")) == 2


@pytest.mark.parametrize("limits", [{"max_facts": 1}, {"max_chars": 3}])
def test_overflow_returns_no_partial_prefix_and_unicode_limits_are_exact(
    sql_storage, clock, limits
):
    kb, _, memory = team(sql_storage, clock)
    kb.replace_facts([source(clock, "я🙂", fact_id="b"), source(clock, "é界", fact_id="a")])
    result = memory.task_snapshot(task_key="restore", max_facts=2, max_chars=4)
    assert [f.content for f in result] == ["é界", "я🙂"]
    with pytest.raises(SnapshotTooLarge):
        memory.task_snapshot(task_key="restore", **limits)
    assert len(memory.task_snapshot(task_key="restore", max_facts=2, max_chars=4)) == 2


def test_snapshot_is_read_only_and_does_not_ack_delivery(sql_storage, clock):
    kb, _, memory = team(sql_storage, clock)
    memory.subscribe(subscription_id="watch", selector=WatchSelector("owner"), start="beginning")
    kb.replace_facts([source(clock)])
    pending = memory.poll("watch")
    assert pending.delivery_id is not None
    intent = memory.create_intent(client_key="restore", goal="Restore")

    def state():
        return canonical_json(
            {
                "facts": sql_storage.load("owner"),
                "shared": sql_storage.load("__shared__"),
                "usage": sql_storage.load_usage_events(),
                "stats": sql_storage.load_pool_stats(),
                "intent": memory.get_intent(intent.id),
            }
        )

    before = state()
    refs = memory.task_snapshot(task_key="restore")
    assert refs and state() == before
    assert memory.poll("watch") == pending
    assert memory.get_intent(intent.id).lease is None


@pytest.mark.parametrize("withdrawal", ["root_change", "root_delete", "freshness", "scope"])
def test_snapshot_rechecks_inherited_source_conditions(sql_storage, clock, withdrawal):
    kb, pool, memory = team(sql_storage, clock)
    root = source(clock, visibility_scope="ops", origin_agent_id="peer")
    sql_storage.save("__shared__", [root])
    pool.grant_read("owner", "ops")
    basis = memory.capture_evidence((("__shared__", root.id),))
    if withdrawal == "freshness":
        workspace = WorkspaceMemory(sql_storage, "repair")
        workspace.grant("owner", "maintain")
        memory = AlwaysOnMemory(kb, pool=pool, workspace=workspace, clock=clock)
        memory.set_freshness(basis[0], source_ref="remote", recheck_after_seconds=1)
        memory.record_verification(
            basis[0],
            result=VerificationResult(VerificationOutcome.CONFIRMED, {"checked": True}, "v1"),
        )
    derived = memory.derive_fact("Recovery procedure", basis=basis, task_key="restore")
    assert derived.id in {f.ref.fact_id for f in memory.task_snapshot(task_key="restore")}
    if withdrawal == "root_change":
        sql_storage.save("__shared__", [replace(root, content="Recovery withdrawn")])
    elif withdrawal == "root_delete":
        sql_storage.save("__shared__", [])
    elif withdrawal == "scope":
        pool.revoke_read("owner", "ops")
    else:
        clock.advance(seconds=2)
    assert derived.id not in {f.ref.fact_id for f in memory.task_snapshot(task_key="restore")}
    assert kb.list_facts()[0].id == derived.id  # Reading hides; it does not delete history.


def test_snapshot_reloads_exact_revision_and_fits_an_empty_corpus(sql_storage, clock):
    kb, _, memory = team(sql_storage, clock)
    assert memory.task_snapshot(task_key="restore") == ()
    first = source(clock)
    kb.replace_facts([first])
    before = memory.task_snapshot(task_key="restore")[0]
    kb.replace_facts([replace(first, content="Corrected recovery copy")])
    after = memory.task_snapshot(task_key="restore")[0]
    assert after.ref.revision != before.ref.revision and after.content != before.content
    assert not memory.review_context((before.ref,)).complete


def test_shared_named_owner_does_not_duplicate_one_source(sql_storage, clock):
    kb = KnowledgeBase("__shared__", storage=sql_storage, embed_url="")
    kb.replace_facts([source(clock)])
    pool = SharedMemoryPool(sql_storage, persist_stats=True, embed_url="", clock=clock)
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    assert len(memory.task_snapshot(task_key="restore", max_facts=1)) == 1


def test_ambiguous_slot_is_not_a_source_or_budget_omission(sql_storage, clock):
    kb, _, memory = team(sql_storage, clock)
    first = source(clock, slot_key="restore::target", value_text="east")
    second = replace(first, id="two", content="West target", value_text="west", version=2)
    kb.replace_facts([first, second])
    assert memory.task_snapshot(task_key="restore", max_facts=1) == ()
    kb.replace_facts([first, replace(second, value_text="east")])
    assert [f.ref.fact_id for f in memory.task_snapshot(task_key="restore", max_facts=1)] == ["two"]


def test_outcome_correction_retracts_lesson_and_copy_from_snapshot(sql_storage, clock):
    from tests.always_on.test_team_experience import completed, record

    kb, _, memory = team(sql_storage, clock)
    root = source(clock)
    kb.replace_facts([root])
    basis = memory.capture_evidence((("owner", root.id),))
    action, lease = completed(memory, "failed")
    lesson = record(memory, action, basis, task_key="restore")
    copy = memory.derive_fact(
        "Conditional recovery procedure", basis=(lesson.lesson.ref,), task_key="restore"
    )
    ids = {lesson.lesson.ref.fact_id, copy.id}
    assert ids <= {f.ref.fact_id for f in memory.task_snapshot(task_key="restore")}
    memory.reconcile_action(
        action.id,
        lease_token=lease.token,
        outcome="succeeded",
        evidence={"independent_check": "corrected"},
    )
    assert not ids & {f.ref.fact_id for f in memory.task_snapshot(task_key="restore")}


def test_runnable_task_example_handles_snapshot_overflow_without_inference(tmp_path, monkeypatch):
    from examples.task_context import example

    def forbidden(*args, **kwargs):
        raise AssertionError("Task example must stay model-free")

    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    result = example(tmp_path)
    assert result["snapshot_facts"] == 6 and result["snapshot_overflow_handled"]
    assert result["model_calls"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_key": " restore"},
        {"task_key": "x" * 257},
        {"task_key": None},
        {"max_facts": True},
        {"max_facts": 0},
        {"max_facts": 129},
        {"max_chars": 0},
        {"max_chars": 65537},
        {"max_chars": 2.5},
    ],
)
def test_snapshot_validates_before_reading(sql_storage, clock, kwargs):
    _, _, memory = team(sql_storage, clock)
    with pytest.raises(ValueError):
        memory.task_snapshot(**{"task_key": "restore", **kwargs})
