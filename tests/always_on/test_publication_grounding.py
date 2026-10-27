"""Generated publication is fenced before shared CAS and trust side effects."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import Event

import pytest

from ai_knot import KnowledgeBase
from ai_knot.pool import SharedMemoryPool
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact


@pytest.fixture(autouse=True)
def publication_time(clock, monkeypatch):
    class PublicationClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock()

    monkeypatch.setattr("ai_knot.pool.datetime", PublicationClock)


def field(value, attribute):
    return Fact(
        f"Deployment {attribute} {value}",
        entity="deployment",
        attribute=attribute,
        value_text=value,
    )


def projection(kb):
    return next(
        f
        for f in kb.add_resolved([field("42", "build"), field("Europe", "region")])
        if f.attribute == "_aggregate"
    )


def unchanged_state(pool, storage):
    return (
        storage.load("__shared__"),
        pool._stats_snapshot(),
        storage.load_pool_stats(),
        storage.load_trust_events(),
    )


def assert_archived_replay_rejected(kb, pool, storage, clock):
    pool.register("owner")
    old = projection(kb)
    assert pool.publish("owner", [old.id], kb=kb)
    clock.advance(seconds=1)
    current = next(
        f for f in kb.add_resolved([field("43", "build")]) if f.attribute == "_aggregate"
    )
    assert pool.publish("owner", [current.id], kb=kb)
    assert not kb.get(old.id).is_active(clock())
    before = unchanged_state(pool, storage)
    assert pool.publish("owner", [old.id], kb=kb) == []
    assert unchanged_state(pool, storage) == before
    assert current.id in {
        f.id for f, _ in pool.recall("Deployment build 43 Europe", "owner", now=clock())
    }


@pytest.mark.parametrize("persist_stats", [False, True])
def test_archived_generated_replay_cannot_close_current_shared_slot(
    sql_storage, clock, persist_stats
):
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    pool = SharedMemoryPool(sql_storage, persist_stats=persist_stats, clock=clock)
    assert_archived_replay_rejected(kb, pool, sql_storage, clock)


@pytest.mark.parametrize("persist_stats", [False, True])
def test_yaml_archived_generated_replay_cannot_close_current_slot(tmp_path, clock, persist_stats):
    storage = YAMLStorage(str(tmp_path))
    kb = KnowledgeBase("owner", storage=storage, embed_url="")
    pool = SharedMemoryPool(storage, persist_stats=persist_stats, clock=clock)
    assert_archived_replay_rejected(kb, pool, storage, clock)


def candidate_and_current(storage, clock, persist_stats):
    kb = KnowledgeBase("owner", storage=storage, embed_url="")
    pool = SharedMemoryPool(storage, persist_stats=persist_stats, clock=clock)
    pool.register("owner")
    pool.register("peer")
    candidate = projection(kb)
    peer = KnowledgeBase("peer", storage=storage, embed_url="")
    (current,) = peer.add_resolved(
        [Fact("Accepted replacement build 43", slot_key=candidate.slot_key)]
    )
    assert pool.publish("peer", [current.id], kb=peer)
    return kb, pool, candidate, current


@pytest.mark.parametrize("persist_stats", [False, True])
@pytest.mark.parametrize("change", ["future", "stale-root"])
def test_unusable_generated_candidate_cannot_close_valid_replacement(
    sql_storage, clock, persist_stats, change
):
    kb, pool, candidate, current = candidate_and_current(sql_storage, clock, persist_stats)
    facts = kb.list_facts()
    if change == "future":
        next(f for f in facts if f.id == candidate.id).valid_from = clock() + timedelta(days=1)
    else:
        next(f for f in facts if f.attribute == "build").content = "Corrected root build 44"
        assert next(f for f in facts if f.id == candidate.id).is_active(clock())
    kb.replace_facts(facts)
    before = unchanged_state(pool, sql_storage)
    assert pool.publish("owner", [candidate.id], kb=kb) == []
    assert unchanged_state(pool, sql_storage) == before
    assert next(f for f in sql_storage.load("__shared__") if f.id == current.id).is_active(clock())


@pytest.mark.parametrize("persist_stats", [False, True])
@pytest.mark.parametrize("change", ["expire", "archive-candidate", "correct-root"])
def test_generated_publication_rechecks_after_sql_writer_lock(
    sql_storage, clock, monkeypatch, persist_stats, change
):
    kb, pool, candidate, current = candidate_and_current(sql_storage, clock, persist_stats)
    facts = kb.list_facts()
    next(f for f in facts if f.id == candidate.id).valid_until = clock() + timedelta(seconds=1)
    kb.replace_facts(facts)
    before = unchanged_state(pool, sql_storage)
    original = sql_storage.always_on_transaction
    attempted = Event()

    @contextmanager
    def transaction(namespaces, *, write):
        if write and "__shared__" in namespaces:
            attempted.set()
        with original(namespaces, write=write) as session:
            yield session

    monkeypatch.setattr(sql_storage, "always_on_transaction", transaction)
    with ThreadPoolExecutor(max_workers=1) as executor:
        # The publisher has prepared its candidate, then encounters this writer.
        with original(("owner",), write=True) as writer:
            future = executor.submit(pool.publish, "owner", [candidate.id], kb=kb)
            assert attempted.wait(10), "publisher never reached its transaction"
            private = writer.load_facts("owner")
            if change == "expire":
                clock.advance(seconds=2)
            elif change == "archive-candidate":
                next(f for f in private if f.id == candidate.id).valid_until = clock()
                writer.replace_facts("owner", private)
            else:
                next(
                    f for f in private if f.attribute == "build"
                ).content = "Changed root while waiting"
                writer.replace_facts("owner", private)
        assert future.result(timeout=15) == []
    assert unchanged_state(pool, sql_storage) == before
    assert next(f for f in sql_storage.load("__shared__") if f.id == current.id).is_active(clock())


@pytest.mark.parametrize("persist_stats", [False, True])
def test_trusted_manual_same_owner_republication_contract_remains(
    sql_storage, clock, persist_stats
):
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    pool = SharedMemoryPool(sql_storage, persist_stats=persist_stats, clock=clock)
    pool.register("owner")
    (old,) = kb.add_resolved([Fact("Manual original", slot_key="manual::status")])
    assert pool.publish("owner", [old.id], kb=kb)
    clock.advance(seconds=1)
    (current,) = kb.add_resolved([Fact("Manual correction", slot_key="manual::status")])
    assert pool.publish("owner", [current.id], kb=kb)
    (republished,) = pool.publish("owner", [old.id], kb=kb)
    assert republished.id == old.id
    assert republished.is_active(clock())
