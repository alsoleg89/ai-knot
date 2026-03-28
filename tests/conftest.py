"""Shared fixtures for agentmemo test suite."""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone

import pytest

from agentmemo.types import ConversationTurn, Fact, MemoryType
from agentmemo.storage.yaml_storage import YAMLStorage
from agentmemo.storage.sqlite_storage import SQLiteStorage


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
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        last_accessed=datetime(2025, 1, 1, tzinfo=timezone.utc),
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
