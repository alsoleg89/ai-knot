"""Public feed behavior against real, independent SQL connections."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ai_knot.always_on.changes import evidence_revision
from ai_knot.always_on.models import AccessDenied, CursorExpired, InvalidDelivery, RevisionConflict
from ai_knot.always_on.subscriptions import WatchSelector
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact


def test_restart_replay_ack_and_incarnation(ao_factory, kb):
    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    fact = kb.add("Pipeline migrated to Prefect")
    first = ao.poll("watch")
    assert [event.fact_id for event in first.events] == [fact.id]
    assert ao_factory().poll("watch") == first
    with pytest.raises(InvalidDelivery):
        ao.ack("watch", delivery_id="future")
    ao_factory().ack("watch", delivery_id=first.delivery_id)
    kb.add("Database runs on PostgreSQL")
    second = ao.poll("watch")
    ao.ack("watch", delivery_id=second.delivery_id)
    ao_factory().ack("watch", delivery_id=first.delivery_id)
    assert ao.poll("watch").delivery_id is None
    ao.unsubscribe("watch")
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    with pytest.raises(InvalidDelivery):
        ao.ack("watch", delivery_id=first.delivery_id)


def test_change_digest_telemetry_delete_restore(ao_factory, sql_storage, clock):
    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    fact = Fact(
        id="one", content="A", created_at=clock(), last_accessed=clock(), valid_from=clock()
    )
    sql_storage.save("owner", [fact])
    batch = ao.poll("watch")
    ao.ack("watch", delivery_id=batch.delivery_id)
    sql_storage.save("owner", [replace(fact, access_count=10, importance=0.2)])
    assert ao.poll("watch").delivery_id is None
    sql_storage.save("owner", [replace(fact, slot_key="B", content="changed")])
    batch = ao.poll("watch")
    assert batch.events[0].kind == "update"
    assert batch.events[0].before_revision == evidence_revision(fact)
    ao.ack("watch", delivery_id=batch.delivery_id)
    sql_storage.delete("owner", fact.id)
    batch = ao.poll("watch")
    assert batch.events[0].kind == "delete"
    ao.ack("watch", delivery_id=batch.delivery_id)
    sql_storage.save("owner", [fact])
    assert ao.poll("watch").events[0].kind == "insert"


def test_limit_slot_moves_debounce_and_contract(ao_factory, sql_storage, clock):
    ao = ao_factory()
    ao.subscribe(
        subscription_id="watch",
        selector=WatchSelector("owner", slot_keys=("A",)),
        debounce_seconds=30,
    )
    f = Fact(id="f1", content="A", slot_key="A")
    sql_storage.save("owner", [f])
    assert ao.poll("watch").delivery_id is None
    clock.advance(seconds=29)
    assert ao_factory().poll("watch").delivery_id is None
    sql_storage.save("owner", [replace(f, slot_key="B")])
    clock.advance(seconds=1)
    batch = ao_factory().poll("watch")
    assert [e.fact_id for e in batch.events] == ["f1"]
    assert batch.events[0].slot_key == "B"
    assert batch.events[0].before_revision is None
    with pytest.raises(RevisionConflict):
        ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    for limit in (0, 501, True):
        with pytest.raises(ValueError):
            ao.poll("watch", limit=limit)


def test_hidden_page_ack_and_limit_boundary(ao_factory, sql_storage):
    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True)
    ao = ao_factory(pool=pool)
    ao.subscribe(subscription_id="watch", selector=WatchSelector("__shared__"))
    hidden = [
        Fact(id=f"h{i}", content="secret", origin_agent_id="peer", visibility_scope="closed")
        for i in range(1000)
    ]
    sql_storage.save("__shared__", hidden)
    sql_storage.save(
        "__shared__",
        hidden + [Fact(id="public1", content="visible"), Fact(id="public2", content="visible")],
    )
    first = ao.poll("watch", limit=1)
    assert first.delivery_id and first.events == ()
    ao.ack("watch", delivery_id=first.delivery_id)
    next_batch = ao.poll("watch", limit=1)
    assert [e.fact_id for e in next_batch.events] == ["public1"]
    assert ao.poll("watch", limit=500) == next_batch
    ao.ack("watch", delivery_id=next_batch.delivery_id)
    assert [e.fact_id for e in ao.poll("watch").events] == ["public2"]


def test_shared_replay_redacts_revoke_and_private_tombstone(ao_factory, sql_storage):
    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True)
    ao = ao_factory(pool=pool)
    pool.grant_read("owner", "team")
    ao.subscribe(subscription_id="watch", selector=WatchSelector("__shared__"))
    fact = Fact(id="secret", content="secret", visibility_scope="team", origin_agent_id="peer")
    sql_storage.save("__shared__", [fact])
    batch = ao.poll("watch")
    assert [e.fact_id for e in batch.events] == ["secret"]
    pool.revoke_read("owner", "team")
    replay = ao_factory(pool=pool).poll("watch")
    assert (
        replay.delivery_id == batch.delivery_id and replay.events == () and replay.resync_required
    )
    ao.ack("watch", delivery_id=batch.delivery_id)
    sql_storage.save("__shared__", [replace(fact, visibility_scope="global")])
    batch = ao.poll("watch")
    sql_storage.save("__shared__", [replace(fact, visibility_scope="private")])
    sql_storage.delete("__shared__", "secret")
    replay = ao.poll("watch")
    assert (
        replay.delivery_id == batch.delivery_id and replay.events == () and replay.resync_required
    )


def test_pruning_explicit_gap_and_subscribe_beginning(ao_factory, kb):
    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    kb.add("Application uses durable SQLite state")
    assert ao.prune_changes() == 0
    assert ao.prune_changes(force=True) == 1
    with pytest.raises(CursorExpired):
        ao.poll("watch")
    with pytest.raises(CursorExpired):
        ao.subscribe(subscription_id="old", selector=WatchSelector("owner"), start="beginning")
    ao.unsubscribe("watch")
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"), start="now")
    assert ao.poll("watch").delivery_id is None


def test_baseline_and_invalid_namespace(ao_factory, kb):
    kb.add("Existing knowledge before enabling stream")
    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"), start="beginning")
    assert ao.poll("watch").resync_required
    with pytest.raises(AccessDenied):
        ao.subscribe(subscription_id="bad", selector=WatchSelector("other"))
    with pytest.raises(AccessDenied):
        ao.poll("missing")


def test_shared_poll_does_not_materialize_namespace(ao_factory, sql_storage, monkeypatch):
    from ai_knot.always_on.storage import SQLSession

    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True)
    ao = ao_factory(pool=pool)
    ao.subscribe(subscription_id="watch", selector=WatchSelector("__shared__"))
    sql_storage.save("__shared__", [Fact(id="one", content="public")])

    def disallow_full_load(self, namespace):
        raise AssertionError("poll must use bounded ID projection queries")

    monkeypatch.setattr(SQLSession, "load_facts", disallow_full_load)
    batch = ao.poll("watch")
    assert [event.fact_id for event in batch.events] == ["one"]
    assert ao.poll("watch") == batch
    ao.ack("watch", delivery_id=batch.delivery_id)
    assert ao.poll("watch").delivery_id is None
