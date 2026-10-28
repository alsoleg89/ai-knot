"""Shared async retrieval must use the embedding provider the host selected."""

import asyncio

import pytest

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage


def test_shared_documents_and_query_use_explicit_openai_configuration(tmp_path, monkeypatch):
    calls = []

    async def embed(texts, **kwargs):
        calls.append((texts, kwargs))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("ai_knot.embedder.embed_texts", embed)
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    kb = KnowledgeBase("writer", storage=storage, embed_url="")
    pool = SharedMemoryPool(
        storage,
        embed_url="https://api.openai.com",
        embed_model="text-embedding-3-small",
        embed_api_key="test-openai-key",
    )
    pool.register("writer")
    fact = kb.add("Backup receipt verified")
    pool.publish("writer", [fact.id], kb=kb)
    assert asyncio.run(pool.arecall("Backup receipt", "reader"))
    expected = {
        "base_url": "https://api.openai.com",
        "model": "text-embedding-3-small",
        "api_key": "test-openai-key",
    }
    assert calls == [(["Backup receipt verified"], expected), (["Backup receipt"], expected)]


def test_unavailable_openai_embeddings_keep_lexical_recall(tmp_path, monkeypatch):
    calls = []

    async def unavailable(texts, **kwargs):
        calls.append(texts)
        return []

    monkeypatch.setattr("ai_knot.embedder.embed_texts", unavailable)
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    kb = KnowledgeBase("writer", storage=storage, embed_url="")
    pool = SharedMemoryPool(
        storage,
        embed_url="https://api.openai.com",
        embed_model="text-embedding-3-small",
        embed_api_key="test-openai-key",
    )
    pool.register("writer")
    fact = kb.add("Backup receipt verified")
    pool.publish("writer", [fact.id], kb=kb)
    result = asyncio.run(pool.arecall("Backup receipt", "reader"))
    assert result and result[0][0].id == fact.id
    assert calls == [["Backup receipt verified"]]


@pytest.mark.real_embedder
def test_disabled_shared_embeddings_never_contact_a_provider(tmp_path, monkeypatch):
    import httpx

    def unexpected(*args, **kwargs):
        raise AssertionError("Disabled embeddings contacted a provider")

    monkeypatch.setattr(httpx, "AsyncClient", unexpected)
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    kb = KnowledgeBase("writer", storage=storage, embed_url="")
    pool = SharedMemoryPool(storage, embed_url="")
    pool.register("writer")
    fact = kb.add("Backup receipt verified")
    pool.publish("writer", [fact.id], kb=kb)
    result = asyncio.run(pool.arecall("Backup receipt", "reader"))
    assert result and result[0][0].id == fact.id
