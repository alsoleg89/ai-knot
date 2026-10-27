"""Every agent-facing pool pull enforces current lineage and ACLs."""

from datetime import UTC, datetime

import pytest

from ai_knot import KnowledgeBase
from ai_knot.always_on import AlwaysOnMemory, WorkspaceMemory
from ai_knot.always_on.lineage import ROOTS_KEY
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact, MESIState


def team(store, clock):
    pool = SharedMemoryPool(store, persist_stats=True, clock=clock)
    workspace = WorkspaceMemory(store, "sync-test")
    workspace.grant("writer", "maintain")
    writer = AlwaysOnMemory(
        KnowledgeBase("writer", storage=store, embed_url=""),
        pool=pool,
        workspace=workspace,
        clock=clock,
    )
    pool.register("writer")
    return pool, writer


@pytest.mark.parametrize("method", ["sync_dirty", "sync_slot_deltas"])
def test_sync_hides_private_derivations_and_advances_cursor(sql_storage, clock, method):
    pool, writer = team(sql_storage, clock)
    public = writer._kb.add("Public deployment status")
    pool.publish("writer", [public.id], kb=writer._kb)
    private = writer._kb.add("Secret private credential")
    derived = writer.derive_fact(
        "Secret derived credential", basis=writer.capture_evidence((("writer", private.id),))
    )
    pool.publish("writer", [derived.id], kb=writer._kb)
    result = getattr(pool, method)("reader")
    assert [item.content for item in result] == [public.content]
    assert pool._known_version["reader"] == 2
    assert getattr(pool, method)("reader") == []


@pytest.mark.parametrize("method", ["sync_dirty", "sync_slot_deltas"])
@pytest.mark.parametrize("change", ["revoke", "changed-root", "removed-root", "malformed"])
def test_sync_invalidates_cached_derivation_without_disclosing_text(
    sql_storage, clock, method, change
):
    pool, writer = team(sql_storage, clock)
    root = Fact(
        "Scoped source",
        origin_agent_id="writer",
        visibility_scope="team:ops",
        version=1,
        created_at=clock(),
        valid_from=clock(),
    )
    sql_storage.save("__shared__", [root])
    pool.grant_read("writer", "team:ops")
    pool.grant_read("reader", "team:ops")
    derived = writer.derive_fact(
        "Derived secret text", basis=writer.capture_evidence((("__shared__", root.id),))
    )
    facts = writer._kb.list_facts()
    next(item for item in facts if item.id == derived.id).slot_key = "deployment::summary"
    writer._kb.replace_facts(facts)
    pool.publish("writer", [derived.id], kb=writer._kb)
    assert derived.id in {
        getattr(item, "fact_id", getattr(item, "id", None))
        for item in getattr(pool, method)("reader")
    }
    cursor = pool._known_version["reader"]
    if change == "revoke":
        pool.revoke_read("reader", "team:ops")
    else:
        facts = sql_storage.load("__shared__")
        if change == "changed-root":
            next(item for item in facts if item.id == root.id).content = "Changed source"
        elif change == "removed-root":
            facts = [item for item in facts if item.id != root.id]
        else:
            next(item for item in facts if item.id == derived.id).qualifiers[ROOTS_KEY] = "not-json"
        sql_storage.save("__shared__", facts)
    result = getattr(pool, method)("reader")
    invalidated = next(
        item for item in result if getattr(item, "fact_id", getattr(item, "id", None)) == derived.id
    )
    assert invalidated.content == ""
    assert invalidated.prompt_surface == ""
    if method == "sync_dirty":
        assert invalidated.mesi_state == MESIState.INVALID
    else:
        assert invalidated.op == "invalidate"
    assert pool._known_version["reader"] >= cursor
    assert getattr(pool, method)("reader") == []
    assert derived.id not in {fact.id for fact, _ in pool.recall("Derived secret text", "reader")}


def test_publication_and_private_original_are_one_observation(sql_storage, clock):
    pool, writer = team(sql_storage, clock)
    writer._clock = lambda: datetime.now(UTC)
    source = writer._kb.add("One original observation")
    pool.publish("writer", [source.id], kb=writer._kb)
    refs = writer.capture_evidence((("writer", source.id), ("__shared__", source.id)))
    roots = writer.independent_evidence(refs)
    assert len(roots) == 1
    assert roots[0].namespace == "__shared__"
    reader = AlwaysOnMemory(
        KnowledgeBase("reader", storage=sql_storage, embed_url=""), pool=pool, clock=writer._clock
    )
    shared_refs = reader.capture_evidence((("__shared__", source.id),))
    assert reader.independent_evidence(shared_refs) == shared_refs
    assert pool.recall("One original observation", "reader")


@pytest.mark.parametrize("method", ["sync_dirty", "sync_slot_deltas"])
def test_sync_grant_restores_skipped_version_and_cross_api_cursor(sql_storage, clock, method):
    pool, writer = team(sql_storage, clock)
    fact = writer._kb.add("Scoped fact")
    pool.publish("writer", [fact.id], kb=writer._kb, visibility_scope="team:private")
    assert getattr(pool, method)("reader") == []
    assert pool._known_version["reader"] == 1
    pool.grant_read("reader", "team:private")
    restored = getattr(pool, method)("reader")
    assert [item.content for item in restored] == [fact.content]
    other = "sync_dirty" if method == "sync_slot_deltas" else "sync_slot_deltas"
    assert getattr(pool, other)("reader") == []
    pool.revoke_read("reader", "team:private")
    revoked = getattr(pool, other)("reader")
    assert len(revoked) == 1
    assert revoked[0].content == ""
    assert pool._known_version["reader"] == 1


def test_publication_and_correlated_private_summary_count_once(sql_storage, clock):
    pool, writer = team(sql_storage, clock)
    writer._clock = lambda: datetime.now(UTC)
    original = writer._kb.add("Original observation")
    original_refs = writer.capture_evidence((("writer", original.id),))
    summary = writer.derive_fact("Correlated summary", basis=original_refs)
    pool.publish("writer", [original.id], kb=writer._kb)
    refs = writer.capture_evidence((("writer", summary.id), ("__shared__", original.id)))
    assert len(writer.independent_evidence(refs)) == 1
    separate = Fact("Original observation")
    writer._kb.replace_facts([*writer._kb.list_facts(), separate])
    # Equal text from a distinct source is a distinct recorded observation.
    separate_refs = writer.capture_evidence((("writer", separate.id),))
    assert len(writer.independent_evidence((*refs, *separate_refs))) == 2
