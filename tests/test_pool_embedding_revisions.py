"""A long-lived shared pool must rank the text it actually embedded."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ai_knot import SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact

_AT = datetime(2026, 1, 1, tzinfo=UTC)
_QUERY = "backup restoration"
_OLD = "The Boreal vault holds recovery snapshots"
_NEW = "The garden shed holds watering cans"
_STABLE = "The Alpine archive preserves disaster copies"
_FILLERS = [f"Unrelated office inventory item {i}" for i in range(8)]
_VECTORS = {
    _QUERY: [1.0, 0.0],
    _OLD: [1.0, 0.0],
    _NEW: [-1.0, 0.0],
    _STABLE: [0.8, 0.6],
    **{text: [0.0, 1.0] for text in _FILLERS},
}


def _facts():
    return [
        Fact(id="changing", content=_OLD, created_at=_AT, valid_from=_AT),
        Fact(id="stable", content=_STABLE, created_at=_AT, valid_from=_AT),
        *[
            Fact(id=f"filler{i}", content=text, created_at=_AT, valid_from=_AT)
            for i, text in enumerate(_FILLERS)
        ],
    ]


def test_same_id_content_revision_matches_a_reopened_pool(tmp_path, monkeypatch):
    calls = []

    async def embed(texts, **kwargs):
        calls.append(texts)
        return [_VECTORS[text] for text in texts]

    monkeypatch.setattr("ai_knot.embedder.embed_texts", embed)
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    original, *unchanged = _facts()
    storage.save("__shared__", [original, *unchanged])
    pool = SharedMemoryPool(storage)
    first = asyncio.run(pool.arecall(_QUERY, "reader", top_k=10))
    assert first[0][0].id == "changing"

    # Content can be corrected without a reliable version increment by an importer.
    storage.save("__shared__", [replace(original, content=_NEW), *unchanged])
    revised = asyncio.run(pool.arecall(_QUERY, "reader", top_k=10))
    reopened = asyncio.run(SharedMemoryPool(storage).arecall(_QUERY, "reader", top_k=10))

    assert reopened[0][0].id == "stable"
    assert revised[0][0].id == "stable"
    assert calls == [
        [_OLD, _STABLE, *_FILLERS],
        [_QUERY],
        [_NEW],
        [_QUERY],
        [_NEW, _STABLE, *_FILLERS],
        [_QUERY],
    ]


@pytest.mark.parametrize("during", ["documents", "query"])
def test_source_revision_during_embedding_never_ranks_new_text_by_old_semantics(
    tmp_path, monkeypatch, during
):
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    writer = SQLiteStorage(str(tmp_path / "pool.db"))
    original, *unchanged = _facts()
    storage.save("__shared__", [original, *unchanged])
    pool = SharedMemoryPool(storage)

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def embed(texts, **kwargs):
            if not entered.is_set() and (
                (during == "documents" and _OLD in texts)
                or (during == "query" and texts == [_QUERY])
            ):
                entered.set()
                await release.wait()
            return [_VECTORS[text] for text in texts]

        monkeypatch.setattr("ai_knot.embedder.embed_texts", embed)
        pending = asyncio.create_task(pool.arecall(_QUERY, "reader", top_k=10))
        await asyncio.wait_for(entered.wait(), timeout=5)
        writer.save("__shared__", [replace(original, content=_NEW), *unchanged])
        release.set()
        result = await asyncio.wait_for(pending, timeout=5)
        assert result[0][0].id == "stable"
        # A discarded response must leave the new revision needing an embedding.
        assert await pool.embed_pool_facts() == 1

    asyncio.run(run())


@pytest.mark.parametrize("failed_response", [[], [[]], [[1.0, 0.0], [1.0, 0.0]]])
def test_failed_revision_embedding_is_retried_without_using_the_old_vector(
    tmp_path, monkeypatch, failed_response
):
    fail = False

    async def embed(texts, **kwargs):
        if fail and texts == [_NEW]:
            return failed_response
        return [_VECTORS[text] for text in texts]

    monkeypatch.setattr("ai_knot.embedder.embed_texts", embed)
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    original, *unchanged = _facts()
    storage.save("__shared__", [original, *unchanged])
    pool = SharedMemoryPool(storage)
    assert asyncio.run(pool.embed_pool_facts()) == 10
    storage.save("__shared__", [replace(original, content=_NEW), *unchanged])
    fail = True
    assert asyncio.run(pool.embed_pool_facts()) == 0
    result = asyncio.run(pool.arecall(_QUERY, "reader", top_k=10))
    assert result[0][0].id == "stable"
    fail = False
    assert asyncio.run(pool.embed_pool_facts()) == 1


@pytest.mark.parametrize("inactive", ["deleted", "expired", "future"])
def test_inactive_vectors_are_reclaimed_before_same_id_is_recreated(
    tmp_path, monkeypatch, inactive
):
    calls = []

    async def embed(texts, **kwargs):
        calls.append(texts)
        return [_VECTORS[text] for text in texts]

    monkeypatch.setattr("ai_knot.embedder.embed_texts", embed)
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    original = _facts()[0]
    storage.save("__shared__", [original])
    pool = SharedMemoryPool(storage)
    assert asyncio.run(pool.arecall(_QUERY, "reader"))
    if inactive == "deleted":
        storage.delete("__shared__", original.id)
    elif inactive == "expired":
        storage.save("__shared__", [replace(original, valid_until=_AT)])
    else:
        storage.save(
            "__shared__", [replace(original, valid_from=datetime.now(UTC) + timedelta(days=1))]
        )
    assert asyncio.run(pool.embed_pool_facts()) == 0
    assert asyncio.run(pool.arecall(_QUERY, "reader")) == []
    # No query call once all cached vectors have been reclaimed.
    assert calls == [[_OLD], [_QUERY]]
    storage.save("__shared__", [replace(original, content=_NEW)])
    assert asyncio.run(pool.embed_pool_facts()) == 1
    assert calls[-1] == [_NEW]


def test_still_valid_expiring_text_is_embedded(tmp_path, monkeypatch):
    async def embed(texts, **kwargs):
        return [_VECTORS[text] for text in texts]

    monkeypatch.setattr("ai_knot.embedder.embed_texts", embed)
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    storage.save(
        "__shared__", [replace(_facts()[0], valid_until=datetime.now(UTC) + timedelta(days=1))]
    )
    assert asyncio.run(SharedMemoryPool(storage).embed_pool_facts()) == 1


def test_late_old_response_cannot_overwrite_a_newer_revision(tmp_path, monkeypatch):
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    original, *unchanged = _facts()
    storage.save("__shared__", [original, *unchanged])
    pool = SharedMemoryPool(storage)

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def embed(texts, **kwargs):
            if _OLD in texts:
                entered.set()
                await release.wait()
            return [_VECTORS[text] for text in texts]

        monkeypatch.setattr("ai_knot.embedder.embed_texts", embed)
        old_response = asyncio.create_task(pool.embed_pool_facts())
        await asyncio.wait_for(entered.wait(), timeout=5)
        storage.save("__shared__", [replace(original, content=_NEW), *unchanged])
        assert await pool.embed_pool_facts() == 10
        release.set()
        assert await asyncio.wait_for(old_response, timeout=5) == 9
        assert await pool.embed_pool_facts() == 0
        result = await pool.arecall(_QUERY, "reader", top_k=10)
        assert result[0][0].id == "stable"

    asyncio.run(run())


def test_only_hidden_cached_vectors_preserve_visible_lexical_fallback(tmp_path, monkeypatch):
    fail = False

    async def embed(texts, **kwargs):
        return [] if fail and texts != [_QUERY] else [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("ai_knot.embedder.embed_texts", embed)
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    hidden = Fact(content=_STABLE, id="hidden", visibility_scope="private", origin_agent_id="owner")
    original = _facts()[0]
    storage.save("__shared__", [original, hidden])
    pool = SharedMemoryPool(storage)
    assert asyncio.run(pool.embed_pool_facts()) == 2
    storage.save(
        "__shared__",
        [replace(original, content="Backup readiness confirmed"), hidden, Fact(content=_NEW)],
    )
    fail = True
    result = asyncio.run(pool.arecall(_QUERY, "reader", top_k=10))
    assert [fact.id for fact, _ in result] == ["changing"]


@pytest.mark.parametrize("offset_days", [-2, 2])
def test_async_embeddings_use_the_explicit_recall_time(tmp_path, monkeypatch, offset_days):
    calls = []

    async def embed(texts, **kwargs):
        calls.append(texts)
        return [_VECTORS[text] for text in texts]

    monkeypatch.setattr("ai_knot.embedder.embed_texts", embed)
    at = datetime.now(UTC) + timedelta(days=offset_days)
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    storage.save(
        "__shared__",
        [replace(f, valid_from=at, valid_until=at + timedelta(hours=12)) for f in _facts()],
    )
    pool = SharedMemoryPool(storage)
    assert asyncio.run(pool.embed_pool_facts()) == 0
    assert calls == []
    result = asyncio.run(pool.arecall(_QUERY, "reader", top_k=10, now=at))
    assert calls == [[_OLD, _STABLE, *_FILLERS], [_QUERY]]
    assert result[0][0].id == "changing"
    assert asyncio.run(pool.embed_pool_facts(now=at)) == 0
