"""Tests for the CrewAI adapter.

No real CrewAI import or network calls. The adapter is exercised both without
CrewAI installed and with a minimal fake CrewAI surface injected via
``sys.modules``.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="crewai_test", storage=YAMLStorage(base_dir=str(tmp_path)))


def _reload_crewai_module() -> Any:
    import ai_knot.integrations.crewai as crewai_module

    return importlib.reload(crewai_module)


@dataclass
class _FakeMemoryRecord:
    id: str
    content: str
    scope: str = "/"
    categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    embedding: list[float] | None = None
    source: str | None = None
    private: bool = False


@dataclass
class _FakeMemoryMatch:
    record: _FakeMemoryRecord
    score: float
    match_reasons: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)

    def format(self) -> str:
        return f"- (score={self.score:.2f}) {self.record.content}"


@dataclass
class _FakeScopeInfo:
    path: str
    record_count: int = 0
    categories: list[str] = field(default_factory=list)
    oldest_record: datetime | None = None
    newest_record: datetime | None = None
    child_scopes: list[str] = field(default_factory=list)


class _FakeMemory:
    def __init__(
        self,
        *,
        storage: Any = None,
        embedder: Any = None,
        read_only: bool = False,
        root_scope: str | None = None,
        **_: Any,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.read_only = read_only
        self.root_scope = root_scope
        self.memory_kind = "memory"


class _FakeMemoryScope:
    def __init__(self, *, memory: Any = None, root_path: str = "/", **_: Any) -> None:
        self.memory = memory
        self.root_path = root_path


def _install_fake_crewai(monkeypatch: pytest.MonkeyPatch) -> None:
    crewai_mod = types.ModuleType("crewai")
    memory_pkg = types.ModuleType("crewai.memory")
    unified_memory_mod = types.ModuleType("crewai.memory.unified_memory")
    memory_scope_mod = types.ModuleType("crewai.memory.memory_scope")
    memory_types_mod = types.ModuleType("crewai.memory.types")

    unified_memory_mod.Memory = _FakeMemory
    memory_scope_mod.MemoryScope = _FakeMemoryScope
    memory_types_mod.MemoryRecord = _FakeMemoryRecord
    memory_types_mod.MemoryMatch = _FakeMemoryMatch
    memory_types_mod.ScopeInfo = _FakeScopeInfo

    monkeypatch.setitem(sys.modules, "crewai", crewai_mod)
    monkeypatch.setitem(sys.modules, "crewai.memory", memory_pkg)
    monkeypatch.setitem(sys.modules, "crewai.memory.unified_memory", unified_memory_mod)
    monkeypatch.setitem(sys.modules, "crewai.memory.memory_scope", memory_scope_mod)
    monkeypatch.setitem(sys.modules, "crewai.memory.types", memory_types_mod)


def _remove_crewai_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == "crewai" or name.startswith("crewai."):
            monkeypatch.delitem(sys.modules, name, raising=False)


class TestAiKnotCrewAIMemory:
    def test_import_safe_without_crewai_and_basic_recall(
        self,
        monkeypatch: pytest.MonkeyPatch,
        kb: KnowledgeBase,
    ) -> None:
        with monkeypatch.context() as mp:
            _remove_crewai_modules(mp)
            crewai_module = _reload_crewai_module()

            memory = crewai_module.AiKnotCrewAIMemory(kb, top_k=3)
            record = memory.remember(
                "User prefers Python",
                scope="/project/demo",
                categories=["stack"],
                metadata={"source_doc": "notes.md"},
            )
            matches = memory.recall("python", scope="/project", categories=["stack"])

            assert record is not None
            assert record.scope == "/project/demo"
            assert matches
            assert matches[0].record.content == "User prefers Python"
            assert matches[0].record.metadata["source_doc"] == "notes.md"
            assert memory.list_scopes("/") == ["/project"]
            assert memory.info("/project").record_count == 1

        _reload_crewai_module()

    def test_scope_views_filter_to_subtree_without_crewai(
        self,
        monkeypatch: pytest.MonkeyPatch,
        kb: KnowledgeBase,
    ) -> None:
        with monkeypatch.context() as mp:
            _remove_crewai_modules(mp)
            crewai_module = _reload_crewai_module()

            memory = crewai_module.AiKnotCrewAIMemory(kb)
            researcher = memory.scope("/agent/researcher")
            writer = memory.scope("/agent/writer")

            researcher.remember("Use PostgreSQL for the billing service", categories=["database"])
            writer.remember("Publish the changelog before release", categories=["docs"])

            research_matches = researcher.recall("postgresql")
            writer_matches = writer.recall("postgresql")

            assert research_matches
            assert "PostgreSQL" in research_matches[0].record.content
            assert all(match.record.scope == "/agent/researcher" for match in research_matches)
            assert writer_matches
            assert all(match.record.scope == "/agent/writer" for match in writer_matches)

        _reload_crewai_module()

    def test_private_memories_require_matching_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
        kb: KnowledgeBase,
    ) -> None:
        with monkeypatch.context() as mp:
            _remove_crewai_modules(mp)
            crewai_module = _reload_crewai_module()

            memory = crewai_module.AiKnotCrewAIMemory(kb)
            memory.remember(
                "API key rotation happens on Fridays",
                source="alice",
                private=True,
            )

            assert memory.recall("API key rotation") == []
            assert memory.recall("API key rotation", source="alice")
            assert memory.recall("API key rotation", include_private=True)

        _reload_crewai_module()

    def test_extract_memories_fallback_splits_bullets(
        self,
        monkeypatch: pytest.MonkeyPatch,
        kb: KnowledgeBase,
    ) -> None:
        with monkeypatch.context() as mp:
            _remove_crewai_modules(mp)
            crewai_module = _reload_crewai_module()

            memory = crewai_module.AiKnotCrewAIMemory(kb)
            extracted = memory.extract_memories(
                "- Use PostgreSQL for primary storage\n- Ship changes behind feature flags"
            )

            assert extracted == [
                "Use PostgreSQL for primary storage",
                "Ship changes behind feature flags",
            ]

        _reload_crewai_module()

    def test_subclasses_fake_crewai_memory_when_crewai_is_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        kb: KnowledgeBase,
    ) -> None:
        with monkeypatch.context() as mp:
            _install_fake_crewai(mp)
            crewai_module = _reload_crewai_module()

            memory = crewai_module.AiKnotCrewAIMemory(kb, top_k=2)
            record = memory.remember("User deploys with Docker", scope="/team/platform")
            scope = memory.scope("/team/platform")
            matches = memory.recall("docker")

            assert isinstance(memory, _FakeMemory)
            assert isinstance(record, _FakeMemoryRecord)
            assert isinstance(scope, _FakeMemoryScope)
            assert scope.root_path == "/team/platform"
            assert matches
            assert isinstance(matches[0], _FakeMemoryMatch)

        _reload_crewai_module()
