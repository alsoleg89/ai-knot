"""Sleeping agents distinguish filtered pages, backlog and durable recovery."""

from dataclasses import replace

import pytest

from ai_knot.always_on import AlwaysOnMemory, WatchSelector
from ai_knot.knowledge import KnowledgeBase
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact


def test_resync_bridges_pruned_history_and_preserves_later_writes(ao_factory, kb):
    from ai_knot.always_on import CursorExpired, InvalidDelivery

    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    fact = kb.add("Production is in EU")
    old = ao.poll("watch")
    kb.add("Backup is in US")
    ao.prune_changes(force=True)
    with pytest.raises(CursorExpired):
        ao.poll("watch")
    snapshot = ao.resync("watch")
    assert snapshot.is_snapshot and not snapshot.resync_required and not snapshot.has_more
    assert {event.fact_id for event in snapshot.events} == {f.id for f in kb.list_facts()}
    assert snapshot.through_seq == snapshot.head_seq == 2
    assert ao_factory().poll("watch") == snapshot
    with pytest.raises(InvalidDelivery):
        ao.ack("watch", delivery_id=old.delivery_id)
    later = kb.add("The deployment window is Friday")
    replay = ao_factory().poll("watch")
    assert replay.events == snapshot.events and replay.is_snapshot
    assert replay.has_more and replay.through_seq == 2 and replay.head_seq == 3
    ao.ack("watch", delivery_id=snapshot.delivery_id)
    delta = ao.poll("watch")
    assert [event.fact_id for event in delta.events] == [later.id]
    assert fact.id not in [event.fact_id for event in delta.events]


def test_resync_overflow_keeps_cursor_and_pending_delivery(ao_factory, sql_storage):
    from ai_knot.always_on import SnapshotTooLarge

    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    sql_storage.save("owner", [Fact(id="one", content="one"), Fact(id="two", content="two")])
    pending = ao.poll("watch", limit=1)
    with pytest.raises(SnapshotTooLarge):
        ao.resync("watch", limit=1)
    assert ao_factory().poll("watch") == pending
    ao.ack("watch", delivery_id=pending.delivery_id)
    assert [e.fact_id for e in ao.poll("watch").events] == ["two"]


@pytest.mark.parametrize("mutation", ["revision", "expiry", "delete", "revoke", "root"])
def test_snapshot_replay_invalidates_instead_of_returning_old_manifest(
    ao_factory, sql_storage, clock, mutation
):
    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True)
    ao = ao_factory(pool=pool)
    pool.grant_read("owner", "team")
    ao.subscribe(subscription_id="watch", selector=WatchSelector("__shared__"))
    source = Fact(
        id="root", content="EU deployment", visibility_scope="team", origin_agent_id="peer"
    )
    sql_storage.save("__shared__", [source])
    if mutation == "root":
        ao.derive_fact("Deploy in EU", basis=ao.capture_evidence((("__shared__", "root"),)))
        ao.subscribe(subscription_id="watch-private", selector=WatchSelector("owner"))
        watch = "watch-private"
    else:
        watch = "watch"
    snapshot = ao.resync(watch)
    assert snapshot.is_snapshot and snapshot.events
    assert ao_factory(pool=pool).poll(watch).is_snapshot
    if mutation in ("revision", "root"):
        sql_storage.save("__shared__", [replace(source, content="US deployment")])
    elif mutation == "delete":
        sql_storage.delete("__shared__", source.id)
    elif mutation == "revoke":
        pool.revoke_read("owner", "team")
    else:
        sql_storage.save("__shared__", [replace(source, valid_until=clock())])
    replay = ao_factory(pool=pool).poll(watch)
    assert replay.delivery_id == snapshot.delivery_id
    assert not replay.events and replay.resync_required and not replay.is_snapshot
    assert ao_factory(pool=pool).poll(watch).resync_required


def test_resync_respects_selectors_support_time_and_independent_ack(ao_factory, sql_storage, clock):
    ao = ao_factory()
    ao.subscribe(
        subscription_id="watch", selector=WatchSelector("owner", slot_keys=("deploy::region",))
    )
    facts = [
        Fact(id="selected", content="EU", slot_key="deploy::region", value_text="EU"),
        Fact(id="other", content="Unrelated"),
        Fact(id="old", content="US", slot_key="deploy::region", valid_until=clock()),
        Fact(id="unsupported", content="Maybe", supported=False, slot_key="other::region"),
    ]
    sql_storage.save("owner", facts)
    snapshot = ao.resync("watch")
    assert [event.fact_id for event in snapshot.events] == ["selected"]
    assert all(event.after_revision for event in snapshot.events)
    newer = ao.resync("watch")
    from ai_knot.always_on import InvalidDelivery

    with pytest.raises(InvalidDelivery):
        ao.ack("watch", delivery_id=snapshot.delivery_id)
    ao.ack("watch", delivery_id=newer.delivery_id)
    assert ao.poll("watch").delivery_id is None


def test_empty_snapshot_is_acknowledgeable_and_limits_are_strict(ao_factory):
    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    for limit in (0, True, 501):
        with pytest.raises(ValueError):
            ao.resync("watch", limit=limit)
    snapshot = ao.resync("watch")
    assert snapshot.is_snapshot and snapshot.delivery_id and not snapshot.events
    ao.ack("watch", delivery_id=snapshot.delivery_id)
    assert not ao.poll("watch").is_snapshot


def test_snapshot_expiry_without_a_journal_write_requires_recovery(ao_factory, sql_storage, clock):
    from datetime import timedelta

    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    fact = Fact(id="expires", content="Temporary", valid_until=clock() + timedelta(seconds=10))
    sql_storage.save("owner", [fact])
    snapshot = ao.resync("watch")
    clock.advance(seconds=10)
    replay = ao_factory().poll("watch")
    assert replay.head_seq == snapshot.head_seq
    assert not replay.events and replay.resync_required and not replay.is_snapshot


def test_upgrade_keeps_legacy_pending_delivery_and_receipts(ao_factory, sql_storage, kb, clock):
    from ai_knot.always_on.storage import sql

    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    kb.add("Before migration")
    first = ao.poll("watch")
    ao.ack("watch", delivery_id=first.delivery_id)
    kb.add("Pending across migration")
    pending = ao.poll("watch")
    with ao._transaction() as tx:
        tx.execute(sql("ALTER TABLE ao_subscriptions DROP COLUMN inflight_snapshot"))
        tx.execute(sql("UPDATE ao_schema SET version=6 WHERE id=1"))
    address = getattr(sql_storage, "_db_path", None) or sql_storage._dsn
    fresh_storage = type(sql_storage)(address)
    reopened = AlwaysOnMemory(KnowledgeBase("owner", storage=fresh_storage), clock=clock)
    assert reopened.poll("watch") == pending
    reopened.ack("watch", delivery_id=first.delivery_id)
    assert reopened.poll("watch").delivery_id == pending.delivery_id
    reopened.ack("watch", delivery_id=pending.delivery_id)
    assert reopened.resync("watch").is_snapshot


def test_snapshot_long_slot_history_does_not_repeat_full_history_validation(
    ao_factory, sql_storage, clock, monkeypatch
):
    from ai_knot.always_on.storage import SQLSession

    ao = ao_factory()
    ao.subscribe(
        subscription_id="watch", selector=WatchSelector("owner", slot_keys=("deploy::region",))
    )
    facts = [
        Fact(id=f"old-{i:03}", content="old", slot_key="deploy::region", valid_until=clock())
        for i in range(100)
    ]
    facts.append(Fact(id="current", content="EU", slot_key="deploy::region", value_text="EU"))
    sql_storage.save("owner", facts)
    decoded = 0
    load = SQLSession.load_facts

    def measured(self, *args, **kwargs):
        nonlocal decoded
        result = load(self, *args, **kwargs)
        decoded += len(result)
        return result

    monkeypatch.setattr(SQLSession, "load_facts", measured)
    assert [e.fact_id for e in ao.resync("watch").events] == ["current"]
    # A regression revalidating every historical row decodes >10,000 facts.
    assert decoded < 500


def test_pending_delivery_bridges_pruned_covered_history(ao_factory, kb):
    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    kb.add("Already captured")
    pending = ao.poll("watch")
    ao.prune_changes(force=True, through_seq=pending.through_seq)
    assert ao_factory().poll("watch") == pending
    ao.ack("watch", delivery_id=pending.delivery_id)
    later = kb.add("Committed after the covered frontier")
    assert [e.fact_id for e in ao.poll("watch").events] == [later.id]


def test_snapshot_and_concurrent_writer_share_one_commit_boundary(
    ao_factory, sql_storage, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from ai_knot.always_on.changes import evidence_revision

    pool = SharedMemoryPool(sql_storage, persist_stats=True)
    ao = ao_factory(pool=pool)
    ao.subscribe(subscription_id="watch", selector=WatchSelector("__shared__"))
    fact = Fact(id="target", content="old")
    sql_storage.save("__shared__", [fact])
    captured, release, writer_started = Event(), Event(), Event()
    original = ao._snapshot_current

    def paused(session, event, now):
        captured.set()
        assert release.wait(5)
        return original(session, event, now)

    def write():
        writer_started.set()
        sql_storage.atomic_update("__shared__", lambda facts: [replace(facts[0], content="new")])

    monkeypatch.setattr(ao, "_snapshot_current", paused)
    with ThreadPoolExecutor(max_workers=2) as workers:
        recovery = workers.submit(ao.resync, "watch")
        try:
            assert captured.wait(5)
            writing = workers.submit(write)
            assert writer_started.wait(5)
            assert not writing.done()
        finally:
            release.set()
        snapshot = recovery.result(timeout=5)
        writing.result(timeout=5)
    assert snapshot.events[0].after_revision == evidence_revision(fact)
    ao.ack("watch", delivery_id=snapshot.delivery_id)
    delta = ao.poll("watch")
    assert [event.fact_id for event in delta.events] == [fact.id]
    assert delta.events[0].after_revision != snapshot.events[0].after_revision
    assert delta.through_seq == snapshot.through_seq + 1


def test_empty_filtered_page_is_not_caught_up(ao_factory, sql_storage):
    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True)
    ao = ao_factory(pool=pool)
    ao.subscribe(subscription_id="watch", selector=WatchSelector("__shared__"))
    hidden = [
        Fact(id=f"h{i:04}", content="hidden", visibility_scope="closed", origin_agent_id="peer")
        for i in range(1000)
    ]
    target = Fact(id="target", content="deployment changed")
    sql_storage.save("__shared__", [*hidden, target])
    first = ao.poll("watch", limit=1)
    assert first.delivery_id and not first.events
    assert first.through_seq == 1000
    assert first.head_seq == 1001
    assert first.has_more and not first.is_snapshot
    ao.ack("watch", delivery_id=first.delivery_id)
    second = ao.poll("watch", limit=1)
    assert [event.fact_id for event in second.events] == ["target"]
    assert second.through_seq == second.head_seq == 1001
    assert not second.has_more
    ao.ack("watch", delivery_id=second.delivery_id)
    idle = ao.poll("watch")
    assert idle.delivery_id is None and not idle.events and not idle.has_more
    assert idle.through_seq == idle.head_seq == 1001


def test_pending_delivery_tracks_new_backlog_without_changing_payload(ao_factory, kb):
    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    fact = kb.add("The production region is EU")
    first = ao.poll("watch")
    kb.add("The backup region is US")
    replay = ao_factory().poll("watch")
    assert replay.delivery_id == first.delivery_id
    assert replay.events == first.events
    assert replay.through_seq == 1 and replay.head_seq == 2 and replay.has_more
    assert replay.events[0].fact_id == fact.id
    ao.ack("watch", delivery_id=replay.delivery_id)
    assert ao.poll("watch").through_seq == 2


def test_limit_frontier_and_independent_agent_ack(ao_factory, sql_storage, clock):
    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True)
    a = ao_factory(pool=pool)
    b = AlwaysOnMemory(KnowledgeBase("peer", storage=sql_storage), pool=pool, clock=clock)
    for agent in (a, b):
        agent.subscribe(subscription_id="watch", selector=WatchSelector("__shared__"))
    one, two = Fact(id="a", content="one"), Fact(id="b", content="two")
    sql_storage.save("__shared__", [one, two])
    first = a.poll("watch", limit=1)
    assert first.through_seq == 1 and first.head_seq == 2 and first.has_more
    a.ack("watch", delivery_id=first.delivery_id)
    assert b.poll("watch", limit=1).events == first.events
    assert [event.fact_id for event in a.poll("watch").events] == ["b"]
