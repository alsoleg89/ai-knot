"""Tests for the AutoGen adapter.

No real AutoGen import or network calls. The required runtime surface is faked
via ``sys.modules``.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from ai_knot.integrations.autogen import AiKnotAutoGenMemory
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="autogen_test", storage=YAMLStorage(base_dir=str(tmp_path)))


@pytest.fixture
def memory(kb: KnowledgeBase) -> AiKnotAutoGenMemory:
    return AiKnotAutoGenMemory(kb, top_k=3)


@dataclass
class _FakeMemoryContent:
    content: Any
    mime_type: str
    metadata: dict[str, Any] | None = None


@dataclass
class _FakeMemoryQueryResult:
    results: list[_FakeMemoryContent]


@dataclass
class _FakeUpdateContextResult:
    memories: _FakeMemoryQueryResult


class _FakeMemoryMimeType:
    TEXT = "text/plain"


@dataclass
class _FakeSystemMessage:
    content: str


@dataclass
class _FakeUserMessage:
    content: Any
    source: str = "user"


@dataclass
class _FakeAssistantMessage:
    content: Any
    source: str = "assistant"


class _FakeModelContext:
    def __init__(self, messages: list[object]) -> None:
        self._messages = list(messages)

    async def get_messages(self) -> list[object]:
        return list(self._messages)

    async def add_message(self, message: object) -> None:
        self._messages.append(message)


def _install_fake_autogen(monkeypatch: pytest.MonkeyPatch) -> None:
    autogen_core = types.ModuleType("autogen_core")
    memory_mod = types.ModuleType("autogen_core.memory")
    models_mod = types.ModuleType("autogen_core.models")

    memory_mod.MemoryContent = _FakeMemoryContent
    memory_mod.MemoryQueryResult = _FakeMemoryQueryResult
    memory_mod.UpdateContextResult = _FakeUpdateContextResult
    memory_mod.MemoryMimeType = _FakeMemoryMimeType
    models_mod.SystemMessage = _FakeSystemMessage

    monkeypatch.setitem(sys.modules, "autogen_core", autogen_core)
    monkeypatch.setitem(sys.modules, "autogen_core.memory", memory_mod)
    monkeypatch.setitem(sys.modules, "autogen_core.models", models_mod)


class TestAiKnotAutoGenMemory:
    def test_extract_query_from_latest_user_message(self, memory: AiKnotAutoGenMemory) -> None:
        messages = [
            _FakeAssistantMessage(content="Noted."),
            _FakeUserMessage(content="I deploy with Docker"),
            _FakeUserMessage(
                content=[
                    {"text": "I use Kubernetes"},
                    "in production",
                ]
            ),
        ]
        assert memory.extract_query(messages) == "I use Kubernetes in production"

    def test_query_requires_autogen(self, memory: AiKnotAutoGenMemory) -> None:
        with pytest.raises(ImportError, match=r"ai-knot\[autogen\]"):
            asyncio.run(memory.query("deployment"))

    def test_query_returns_ranked_memory_contents(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory: AiKnotAutoGenMemory,
        kb: KnowledgeBase,
    ) -> None:
        _install_fake_autogen(monkeypatch)
        kb.add("User prefers Python")
        kb.add("Team standup is at 10am")

        result = asyncio.run(memory.query("python preferences"))

        assert isinstance(result, _FakeMemoryQueryResult)
        assert result.results
        assert any("Python" in item.content for item in result.results)
        assert result.results[0].mime_type == "text/plain"
        assert "type" in (result.results[0].metadata or {})
        assert "score" in (result.results[0].metadata or {})

    def test_query_accepts_memory_content_as_query(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory: AiKnotAutoGenMemory,
        kb: KnowledgeBase,
    ) -> None:
        _install_fake_autogen(monkeypatch)
        kb.add("User deploys with Docker Compose")

        result = asyncio.run(
            memory.query(_FakeMemoryContent(content="deployment tooling", mime_type="text/plain"))
        )

        assert any("Docker Compose" in item.content for item in result.results)

    def test_update_context_injects_system_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory: AiKnotAutoGenMemory,
        kb: KnowledgeBase,
    ) -> None:
        _install_fake_autogen(monkeypatch)
        kb.add("User deploys APIs with Docker and Kubernetes")
        kb.add("User prefers Python")

        model_context = _FakeModelContext(
            [_FakeUserMessage(content="How should I deploy this Python API?")]
        )
        result = asyncio.run(memory.update_context(model_context))

        assert isinstance(result, _FakeUpdateContextResult)
        assert result.memories.results
        assert isinstance(model_context._messages[-1], _FakeSystemMessage)
        assert "Relevant memory content" in model_context._messages[-1].content
        assert "Docker" in model_context._messages[-1].content

    def test_update_context_no_user_message_noops(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory: AiKnotAutoGenMemory,
    ) -> None:
        _install_fake_autogen(monkeypatch)
        model_context = _FakeModelContext([_FakeAssistantMessage(content="No user input")])

        result = asyncio.run(memory.update_context(model_context))

        assert isinstance(result, _FakeUpdateContextResult)
        assert result.memories.results == []
        assert len(model_context._messages) == 1

    def test_add_persists_fact_with_metadata(
        self,
        memory: AiKnotAutoGenMemory,
        kb: KnowledgeBase,
    ) -> None:
        asyncio.run(
            memory.add(
                _FakeMemoryContent(
                    content="Always use pytest",
                    mime_type="text/plain",
                    metadata={
                        "type": "procedural",
                        "importance": 0.92,
                        "tags": ["testing", "python"],
                    },
                )
            )
        )

        facts = kb.list_facts()
        assert len(facts) == 1
        assert facts[0].type.value == "procedural"
        assert facts[0].importance == pytest.approx(0.92)
        assert facts[0].tags == ["testing", "python"]

    def test_clear_forgets_everything(self, memory: AiKnotAutoGenMemory, kb: KnowledgeBase) -> None:
        kb.add("User likes dark mode")
        kb.add("User deploys on Fridays")

        asyncio.run(memory.clear())

        assert kb.list_facts() == []
