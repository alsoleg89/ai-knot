"""Tests for the PydanticAI adapter.

No real framework import or network calls. The adapter is validated against a
fake agent surface that mirrors PydanticAI's documented ``instructions=``
runtime hook on ``run`` / ``run_sync`` / ``run_stream``.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import pytest

from ai_knot.integrations.pydanticai import AiKnotPydanticAIMemory
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="pydanticai_test", storage=YAMLStorage(base_dir=str(tmp_path)))


@pytest.fixture
def memory(kb: KnowledgeBase) -> AiKnotPydanticAIMemory:
    return AiKnotPydanticAIMemory(kb, top_k=3)


@dataclass
class _FakeResult:
    output: str
    user_prompt: str
    instructions: Any
    kwargs: dict[str, Any]


class _FakeAgent:
    def run_sync(self, user_prompt: str, /, **kwargs: Any) -> _FakeResult:
        return _FakeResult(
            output="sync",
            user_prompt=user_prompt,
            instructions=kwargs.get("instructions"),
            kwargs=kwargs,
        )

    async def run(self, user_prompt: str, /, **kwargs: Any) -> _FakeResult:
        return _FakeResult(
            output="async",
            user_prompt=user_prompt,
            instructions=kwargs.get("instructions"),
            kwargs=kwargs,
        )

    def run_stream(self, user_prompt: str, /, **kwargs: Any) -> _FakeResult:
        return _FakeResult(
            output="stream",
            user_prompt=user_prompt,
            instructions=kwargs.get("instructions"),
            kwargs=kwargs,
        )

    def run_stream_sync(self, user_prompt: str, /, **kwargs: Any) -> _FakeResult:
        return _FakeResult(
            output="stream_sync",
            user_prompt=user_prompt,
            instructions=kwargs.get("instructions"),
            kwargs=kwargs,
        )


class TestAiKnotPydanticAIMemory:
    def test_augment_instructions_appends_memory_block(
        self,
        memory: AiKnotPydanticAIMemory,
        kb: KnowledgeBase,
    ) -> None:
        kb.add("User prefers Python")
        augmented = memory.augment_instructions("You are helpful.", "what stack should I use?")

        assert isinstance(augmented, str)
        assert "You are helpful." in augmented
        assert "Agent Memory" in augmented
        assert "Python" in augmented

    def test_augment_instructions_supports_sequence_payload(
        self,
        memory: AiKnotPydanticAIMemory,
        kb: KnowledgeBase,
    ) -> None:
        kb.add("User deploys with Docker Compose")
        augmented = memory.augment_instructions(
            ["Be concise.", "Return bullet points."],
            "how should I deploy this?",
        )

        assert isinstance(augmented, list)
        assert augmented[:2] == ["Be concise.", "Return bullet points."]
        assert any("Docker Compose" in item for item in augmented)

    def test_run_sync_forwards_query_aware_runtime_instructions(
        self,
        memory: AiKnotPydanticAIMemory,
        kb: KnowledgeBase,
    ) -> None:
        kb.add("User prefers pytest over unittest")
        agent = _FakeAgent()

        result = memory.run_sync(
            agent,
            "How should I write tests?",
            instructions="You are a concise engineer.",
            deps={"tenant": "acme"},
        )

        assert result.user_prompt == "How should I write tests?"
        assert isinstance(result.instructions, str)
        assert "You are a concise engineer." in result.instructions
        assert "pytest" in result.instructions
        assert result.kwargs["deps"] == {"tenant": "acme"}

    @pytest.mark.anyio
    async def test_run_async_forwards_to_agent(
        self,
        memory: AiKnotPydanticAIMemory,
        kb: KnowledgeBase,
    ) -> None:
        kb.add("User prefers TypeScript")
        agent = _FakeAgent()

        result = await memory.run(agent, "What language should I use?")

        assert result.output == "async"
        assert isinstance(result.instructions, str)
        assert "TypeScript" in result.instructions

    def test_run_stream_and_stream_sync_keep_instructions_when_no_memory(
        self,
        memory: AiKnotPydanticAIMemory,
    ) -> None:
        agent = _FakeAgent()

        stream_result = memory.run_stream(
            agent,
            "No matching memory here.",
            instructions="Keep answers short.",
        )
        stream_sync_result = memory.run_stream_sync(
            agent,
            "Still no matching memory.",
            instructions="Keep answers short.",
        )

        assert stream_result.instructions == "Keep answers short."
        assert stream_sync_result.instructions == "Keep answers short."
