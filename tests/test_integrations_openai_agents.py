"""Tests for the OpenAI Agents SDK adapter.

No real SDK import or network calls. The SDK surface is faked via ``sys.modules``.
"""

from __future__ import annotations

import pathlib
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from ai_knot.integrations.openai_agents import AiKnotAgentsMemory
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="openai_agents_test", storage=YAMLStorage(base_dir=str(tmp_path)))


@pytest.fixture
def memory(kb: KnowledgeBase) -> AiKnotAgentsMemory:
    return AiKnotAgentsMemory(kb, top_k=3)


@dataclass
class _FakeModelInputData:
    input: list[Any]
    instructions: str | None = None


@dataclass
class _FakeCallModelData:
    model_data: _FakeModelInputData


@dataclass
class _FakeRunConfig:
    call_model_input_filter: Any
    workflow_name: str | None = None


def _install_fake_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    agents_mod = types.ModuleType("agents")
    agents_mod.RunConfig = _FakeRunConfig

    run_mod = types.ModuleType("agents.run")
    run_mod.ModelInputData = _FakeModelInputData

    monkeypatch.setitem(sys.modules, "agents", agents_mod)
    monkeypatch.setitem(sys.modules, "agents.run", run_mod)


class TestAiKnotAgentsMemory:
    def test_extract_query_from_responses_items(self, memory: AiKnotAgentsMemory) -> None:
        items = [
            {"role": "developer", "content": "Ignore previous habits."},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "I deploy with Docker Compose"}],
            },
        ]
        assert memory.extract_query(items) == "I deploy with Docker Compose"

    def test_augment_instructions_appends_memory(
        self, memory: AiKnotAgentsMemory, kb: KnowledgeBase
    ) -> None:
        kb.add("User prefers Python")
        augmented = memory.augment_instructions("You are helpful.", "write some code")
        assert augmented is not None
        assert "You are helpful." in augmented
        assert "Agent Memory" in augmented
        assert "Python" in augmented

    def test_build_run_config_requires_sdk(self, memory: AiKnotAgentsMemory) -> None:
        with pytest.raises(ImportError, match=r"ai-knot\[agents\]"):
            memory.build_run_config()

    def test_build_run_config_installs_filter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory: AiKnotAgentsMemory,
        kb: KnowledgeBase,
    ) -> None:
        _install_fake_agents(monkeypatch)
        kb.add("User prefers Python")
        run_config = memory.build_run_config(workflow_name="test-flow")

        data = _FakeCallModelData(
            model_data=_FakeModelInputData(
                input=[{"role": "user", "content": "Write a script for my stack"}],
                instructions="You are helpful.",
            )
        )
        filtered = run_config.call_model_input_filter(data)

        assert isinstance(run_config, _FakeRunConfig)
        assert run_config.workflow_name == "test-flow"
        assert "You are helpful." in (filtered.instructions or "")
        assert "Agent Memory" in (filtered.instructions or "")
        assert "Python" in (filtered.instructions or "")

    def test_existing_filter_is_composed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory: AiKnotAgentsMemory,
        kb: KnowledgeBase,
    ) -> None:
        _install_fake_agents(monkeypatch)
        kb.add("User deploys with Docker")

        def existing(data: _FakeCallModelData) -> _FakeModelInputData:
            return _FakeModelInputData(
                input=data.model_data.input,
                instructions="Existing instructions.",
            )

        run_config = memory.build_run_config(call_model_input_filter=existing)
        filtered = run_config.call_model_input_filter(
            _FakeCallModelData(
                model_data=_FakeModelInputData(
                    input=[{"role": "user", "content": "How should I deploy this API?"}],
                    instructions=None,
                )
            )
        )

        assert "Existing instructions." in (filtered.instructions or "")
        assert "Docker" in (filtered.instructions or "")

    def test_no_query_keeps_instructions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        memory: AiKnotAgentsMemory,
    ) -> None:
        _install_fake_agents(monkeypatch)
        run_config = memory.build_run_config()
        filtered = run_config.call_model_input_filter(
            _FakeCallModelData(
                model_data=_FakeModelInputData(
                    input=[{"role": "assistant", "content": "No user message here"}],
                    instructions="Existing instructions.",
                )
            )
        )
        assert filtered.instructions == "Existing instructions."
