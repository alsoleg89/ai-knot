"""Shared fixtures for ai-knot test suite."""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import ConversationTurn, Fact, MemoryType

# ---------------------------------------------------------------------------
# Deterministic offline embedder
# ---------------------------------------------------------------------------
#
# Production code talks to an OpenAI-compatible /v1/embeddings endpoint
# (Ollama by default, OpenAI when configured). CI does not run an embed
# server, and we deliberately do not ship the OpenAI key as a CI secret.
# Without a stub, recall falls back to BM25-only and any test relying on
# semantic recall (e.g. "what language?" → "Python") fails.
#
# This autouse fixture monkey-patches `embed_texts` to return reproducible
# pseudo-vectors derived from MD5 of the input. The hashing is content-
# stable across runs, so dense similarity is well-defined; identical texts
# get identical vectors, and unrelated texts get near-orthogonal vectors.
# Tests that need real embeddings can opt-out via the `real_embedder`
# marker (none currently do — production behaviour is exercised by the
# benchmark harness, not by unit tests).


def _fake_embed_texts(
    texts: list[str],
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> list[list[float]]:
    """Return a deterministic 16-dim vector per input text (MD5-derived)."""

    async def _impl() -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            h = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).digest()
            # 16 bytes → 16 floats in [0, 1).
            vectors.append([b / 255.0 for b in h])
        return vectors

    return _impl()  # type: ignore[return-value]  # awaited by caller


@pytest.fixture(autouse=True)
def _stub_embedder(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Replace the network embedder with the deterministic stub for the suite."""
    if "real_embedder" in request.keywords:
        yield
        return
    monkeypatch.setattr("ai_knot.embedder.embed_texts", _fake_embed_texts)
    yield


@pytest.fixture
def tmp_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Temporary directory for storage tests."""
    return tmp_path


@pytest.fixture
def sample_fact() -> Fact:
    """A single representative fact for testing."""
    return Fact(
        content="User works at Sber as Operations Director",
        type=MemoryType.SEMANTIC,
        importance=0.95,
    )


@pytest.fixture
def sample_facts() -> list[Fact]:
    """A collection of diverse facts for testing."""
    return [
        Fact(
            content="User works at Sber as Operations Director",
            type=MemoryType.SEMANTIC,
            importance=0.95,
            tags=["user_profile", "work"],
        ),
        Fact(
            content="User prefers Python, dislikes async code",
            type=MemoryType.PROCEDURAL,
            importance=0.85,
            tags=["preferences", "coding"],
        ),
        Fact(
            content="User deploys everything in Docker on Kubernetes",
            type=MemoryType.SEMANTIC,
            importance=0.80,
            tags=["infrastructure"],
        ),
        Fact(
            content="Deploy failed last Tuesday due to OOM",
            type=MemoryType.EPISODIC,
            importance=0.50,
            tags=["incidents"],
        ),
        Fact(
            content="User prefers concise responses without emoji",
            type=MemoryType.PROCEDURAL,
            importance=0.70,
            tags=["preferences", "communication"],
        ),
    ]


@pytest.fixture
def sample_turns() -> list[ConversationTurn]:
    """A representative conversation for extraction tests."""
    return [
        ConversationTurn(role="user", content="I deploy everything in Docker"),
        ConversationTurn(role="assistant", content="Got it, I'll use Docker examples"),
        ConversationTurn(role="user", content="I really hate async code, prefer sync"),
        ConversationTurn(role="assistant", content="Understood, sync-first approach"),
        ConversationTurn(role="user", content="By the way, I work at Sber"),
        ConversationTurn(role="assistant", content="Nice, noted"),
        ConversationTurn(role="user", content="ok thanks"),
    ]


@pytest.fixture
def yaml_storage(tmp_dir: pathlib.Path) -> YAMLStorage:
    """YAML storage backed by a temporary directory."""
    return YAMLStorage(base_dir=str(tmp_dir))


@pytest.fixture
def sqlite_storage(tmp_dir: pathlib.Path) -> SQLiteStorage:
    """SQLite storage backed by a temporary database file."""
    return SQLiteStorage(db_path=str(tmp_dir / "test.db"))


@pytest.fixture
def old_fact() -> Fact:
    """A fact created a long time ago with no recent access — should decay."""
    return Fact(
        content="User mentioned liking Java once",
        type=MemoryType.EPISODIC,
        importance=0.3,
        access_count=0,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        last_accessed=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def fresh_fact() -> Fact:
    """A recently accessed, high-importance fact — should retain."""
    return Fact(
        content="User's primary language is Python",
        type=MemoryType.PROCEDURAL,
        importance=0.95,
        access_count=20,
    )
