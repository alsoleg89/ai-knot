"""Tests for OpenAI integration — mocked, no real API calls."""

from __future__ import annotations

import pathlib

import pytest

from agentmemo.integrations.openai import MemoryEnabledOpenAI
from agentmemo.knowledge import KnowledgeBase
from agentmemo.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="openai_test", storage=YAMLStorage(base_dir=str(tmp_path)))


class TestMemoryEnabledOpenAI:
    """MemoryEnabledOpenAI wraps OpenAI client with automatic memory."""

    def test_init(self, kb: KnowledgeBase) -> None:
        client = MemoryEnabledOpenAI(knowledge_base=kb)
        assert client._kb is kb

    def test_inject_context_into_system_prompt(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")

        client = MemoryEnabledOpenAI(knowledge_base=kb)
        messages = [{"role": "user", "content": "Write me a script"}]

        enriched = client._enrich_messages(messages)

        # Should have a system message with memory context
        system_msgs = [m for m in enriched if m["role"] == "system"]
        assert len(system_msgs) >= 1
        assert "Python" in system_msgs[0]["content"]

    def test_preserves_existing_system_prompt(self, kb: KnowledgeBase) -> None:
        kb.add("User likes Docker")

        client = MemoryEnabledOpenAI(knowledge_base=kb)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Deploy this"},
        ]

        enriched = client._enrich_messages(messages)
        system_content = enriched[0]["content"]

        assert "helpful assistant" in system_content
        assert "Docker" in system_content

    def test_no_context_on_empty_kb(self, kb: KnowledgeBase) -> None:
        client = MemoryEnabledOpenAI(knowledge_base=kb)
        messages = [{"role": "user", "content": "Hello"}]

        enriched = client._enrich_messages(messages)
        # Should not inject empty context
        system_msgs = [m for m in enriched if m["role"] == "system"]
        if system_msgs:
            assert system_msgs[0]["content"].strip() != ""

    def test_learn_from_response(self, kb: KnowledgeBase) -> None:
        client = MemoryEnabledOpenAI(knowledge_base=kb, auto_learn=False)
        # auto_learn=False means we don't call LLM extraction
        # Just verify the interface exists
        assert hasattr(client, "_kb")
        assert hasattr(client, "_enrich_messages")
