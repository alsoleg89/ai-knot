"""Tests for the LlamaIndex adapter — no llama-index dependency required."""

from __future__ import annotations

import pathlib

from ai_knot.integrations.llamaindex import AiKnotLlamaIndexMemory
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


def _message_role(message: object) -> str:
    if isinstance(message, dict):
        return str(message["role"])
    return str(message.role)


def _message_content(message: object) -> str:
    if isinstance(message, dict):
        return str(message["content"])
    return str(message.content)


def _kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="llamaindex_test", storage=YAMLStorage(base_dir=str(tmp_path)))


class TestAiKnotLlamaIndexMemory:
    def test_import_safe_without_llamaindex_and_injects_memory(
        self, tmp_path: pathlib.Path
    ) -> None:
        kb = _kb(tmp_path)
        kb.add("User deploys APIs with Docker Compose")
        kb.add("User prefers Python over Java")

        memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, top_k=3)
        memory.put({"role": "user", "content": "Please remember release notes stay concise."})

        messages = memory.get("Write a deployment checklist for my API stack.")

        assert messages
        assert _message_role(messages[0]).lower().endswith("system")
        injected = _message_content(messages[0])
        assert "Agent Memory" in injected
        assert "Docker Compose" in injected or "Python" in injected
        assert len(memory.get_all()) == 1
        assert any("release notes" in fact.content.lower() for fact in kb.list_facts())

    def test_existing_system_message_is_merged(self, tmp_path: pathlib.Path) -> None:
        kb = _kb(tmp_path)
        kb.add("User deploys APIs with Docker Compose")

        memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, top_k=3)
        memory.set(
            [
                {"role": "system", "content": "You are a concise staff engineer."},
                {"role": "user", "content": "Write a deployment checklist."},
            ]
        )

        messages = memory.get()

        assert len(messages) == 2
        assert _message_role(messages[0]).lower().endswith("system")
        merged = _message_content(messages[0])
        assert "You are a concise staff engineer." in merged
        assert "Agent Memory" in merged
        assert "Docker Compose" in merged

    def test_set_only_stores_new_messages(self, tmp_path: pathlib.Path) -> None:
        kb = _kb(tmp_path)
        memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb)

        initial = [
            {"role": "user", "content": "User prefers Python"},
            {"role": "assistant", "content": "Noted."},
        ]
        extended = initial + [{"role": "user", "content": "User deploys with Docker Compose"}]

        memory.set(initial)
        memory.set(extended)

        contents = [fact.content for fact in kb.list_facts()]
        assert contents.count("User prefers Python") == 1
        assert contents.count("User deploys with Docker Compose") == 1

    def test_extract_on_write_routes_through_learn(self, tmp_path: pathlib.Path) -> None:
        kb = _kb(tmp_path)
        captured: list[list[object]] = []

        def fake_learn(turns: list[object], **kwargs: object) -> list[object]:
            captured.append(list(turns))
            assert kwargs["provider"] == "openai"
            return []

        kb.learn = fake_learn  # type: ignore[method-assign]

        memory = AiKnotLlamaIndexMemory.from_defaults(
            knowledge_base=kb,
            extract_on_write=True,
            store_assistant_messages=True,
            provider="openai",
        )
        memory.set(
            [
                {"role": "user", "content": "I deploy everything in Docker."},
                {"role": "assistant", "content": "Noted."},
            ]
        )

        assert len(captured) == 1
        assert len(captured[0]) == 2

    def test_dedup_exact_skips_duplicate_raw_messages(self, tmp_path: pathlib.Path) -> None:
        kb = _kb(tmp_path)
        memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, dedup_exact=True)

        memory.put({"role": "user", "content": "User prefers Python"})
        memory.put({"role": "user", "content": "User prefers Python"})

        assert len(kb.list_facts()) == 1
