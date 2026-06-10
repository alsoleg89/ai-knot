"""Tests for the optional LLM-backed SemanticConflictResolver adapter.

The adapter takes a caller-supplied ``complete`` callable, so these tests use a
deterministic stub — no network, no model dependency.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from ai_knot.integrations.semantic_resolver_llm import LLMSemanticConflictResolver
from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact


def _facts(*contents: str) -> list[Fact]:
    return [Fact(content=c) for c in contents]


class TestAdapterParsing:
    def test_parses_indices_to_ids(self) -> None:
        resolver = LLMSemanticConflictResolver(lambda _p: "1")
        facts = _facts("stale claim", "current claim")
        assert resolver(facts) == {facts[0].id}

    def test_none_keeps_all(self) -> None:
        resolver = LLMSemanticConflictResolver(lambda _p: "none")
        facts = _facts("a", "b")
        assert resolver(facts) == set()

    def test_single_candidate_is_noop(self) -> None:
        called = False

        def complete(_p: str) -> str:
            nonlocal called
            called = True
            return "1"

        resolver = LLMSemanticConflictResolver(complete)
        assert resolver(_facts("solo")) == set()
        assert not called  # never even asks the model for <2 candidates

    def test_resolver_failure_keeps_all(self) -> None:
        def boom(_p: str) -> str:
            raise RuntimeError("model down")

        resolver = LLMSemanticConflictResolver(boom)
        assert resolver(_facts("a", "b")) == set()

    def test_all_stale_verdict_is_rejected(self) -> None:
        # Marking every candidate stale would erase the subject → treat as
        # no-confidence and keep everything.
        resolver = LLMSemanticConflictResolver(lambda _p: "1, 2")
        assert resolver(_facts("a", "b")) == set()

    def test_out_of_range_indices_ignored(self) -> None:
        resolver = LLMSemanticConflictResolver(lambda _p: "2, 9")
        facts = _facts("a", "b", "c")
        assert resolver(facts) == {facts[1].id}


def _stub_llm(prompt: str) -> str:
    """Deterministic stand-in for an LLM: a 'deprecated' claim supersedes the
    competing 'backward compatibility' claim about the same endpoint."""
    if "deprecated" not in prompt.lower():
        return "none"
    for line in prompt.splitlines():
        m = re.match(r"\s*(\d+)\.\s", line)
        if m and "backward compatibility" in line.lower():
            return m.group(1)
    return "none"


class TestAdapterInPool:
    """The adapter plugs into the SharedMemoryPool seam and resolves a
    lexically-divergent value conflict the deterministic resolver cannot.
    """

    _STALE = "The REST collector endpoint supports both protocols for backward compatibility."
    _CURRENT = "The REST collector endpoint has been deprecated since the April 2025 release."
    _QUERY = "Is the REST collector endpoint still supported?"

    @pytest.fixture
    def sqlite_db(self, tmp_path: pathlib.Path) -> SQLiteStorage:
        return SQLiteStorage(str(tmp_path / "test.db"))

    def _seed(self, pool: SharedMemoryPool, db: SQLiteStorage) -> None:
        pool.register("agent_a")
        pool.register("agent_b")
        kb_a = KnowledgeBase("agent_a", storage=db)
        pool.publish("agent_a", [kb_a.add(self._STALE).id], kb=kb_a)
        kb_b = KnowledgeBase("agent_b", storage=db)
        pool.publish("agent_b", [kb_b.add(self._CURRENT).id], kb=kb_b)

    def test_adapter_drops_superseded_claim(self, sqlite_db: SQLiteStorage) -> None:
        pool = SharedMemoryPool(
            storage=sqlite_db,
            semantic_resolver=LLMSemanticConflictResolver(_stub_llm),
        )
        self._seed(pool, sqlite_db)
        texts = [f.content.lower() for f, _ in pool.recall(self._QUERY, "querier", top_k=5)]
        assert any("deprecated" in t for t in texts)
        assert not any("backward compatibility" in t for t in texts)

    def test_default_pool_keeps_both(self, sqlite_db: SQLiteStorage) -> None:
        # Without the adapter the deterministic path cannot group these.
        pool = SharedMemoryPool(storage=sqlite_db)
        self._seed(pool, sqlite_db)
        texts = [f.content.lower() for f, _ in pool.recall(self._QUERY, "querier", top_k=5)]
        assert any("backward compatibility" in t for t in texts)
        assert any("deprecated" in t for t in texts)
