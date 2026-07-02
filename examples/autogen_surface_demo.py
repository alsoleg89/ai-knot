"""Zero-network demo of the AutoGen-shaped ai-knot memory surface.

This example does **not** import AutoGen or call a model. It shows the exact
``memory=[AiKnotAutoGenMemory(...)]`` seam and the ``SystemMessage`` payload
that ai-knot injects into an AutoGen-style model context for the next turn.

Run::

    python examples/autogen_surface_demo.py
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any

from ai_knot import KnowledgeBase
from ai_knot.integrations.autogen import AiKnotAutoGenMemory
from ai_knot.storage import YAMLStorage


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


class _FakeModelContext:
    def __init__(self, messages: list[object]) -> None:
        self._messages = list(messages)

    async def get_messages(self) -> list[object]:
        return list(self._messages)

    async def add_message(self, message: object) -> None:
        self._messages.append(message)


@dataclass
class DemoResult:
    user_prompt: str
    recalled_items: list[str]
    injected_system_message: str
    simulated_answer: str


@contextmanager
def _fake_autogen_runtime() -> Iterator[None]:
    autogen_core = types.ModuleType("autogen_core")
    memory_mod = types.ModuleType("autogen_core.memory")
    models_mod = types.ModuleType("autogen_core.models")

    memory_mod.MemoryContent = _FakeMemoryContent
    memory_mod.MemoryQueryResult = _FakeMemoryQueryResult
    memory_mod.UpdateContextResult = _FakeUpdateContextResult
    memory_mod.MemoryMimeType = _FakeMemoryMimeType
    models_mod.SystemMessage = _FakeSystemMessage

    original_modules = {
        "autogen_core": sys.modules.get("autogen_core"),
        "autogen_core.memory": sys.modules.get("autogen_core.memory"),
        "autogen_core.models": sys.modules.get("autogen_core.models"),
    }
    sys.modules["autogen_core"] = autogen_core
    sys.modules["autogen_core.memory"] = memory_mod
    sys.modules["autogen_core.models"] = models_mod
    try:
        yield
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _build_simulated_answer(recalled_items: list[str]) -> str:
    steps: list[str] = []
    if any("Python" in item for item in recalled_items):
        steps.append("Keep the deployment checklist aligned with a Python API.")
    if any("Docker" in item for item in recalled_items):
        steps.append("Include container build and orchestration steps.")
    if not steps:
        steps.append("Use the recalled memory items before answering.")
    return "Simulated next-turn answer:\n" + "\n".join(
        f"{index}. {step}" for index, step in enumerate(steps, start=1)
    )


def build_demo_result() -> DemoResult:
    with TemporaryDirectory(prefix="ai-knot-autogen-") as tmpdir:
        kb = KnowledgeBase(
            agent_id="autogen-surface-demo",
            storage=YAMLStorage(base_dir=tmpdir),
            embed_url="",
        )
        kb.add("User prefers Python over Java")
        kb.add("User deploys APIs with Docker and Kubernetes")

        memory = AiKnotAutoGenMemory(kb, top_k=3)
        user_prompt = "Write a deployment checklist for my Python API."
        model_context = _FakeModelContext([_FakeUserMessage(content=user_prompt)])

        with _fake_autogen_runtime():
            update_result = asyncio.run(memory.update_context(model_context))

        injected = model_context._messages[-1]
        return DemoResult(
            user_prompt=user_prompt,
            recalled_items=[item.content for item in update_result.memories.results],
            injected_system_message=getattr(injected, "content", ""),
            simulated_answer=_build_simulated_answer(
                [item.content for item in update_result.memories.results]
            ),
        )


def main() -> None:
    result = build_demo_result()

    print("=== AutoGen memory surface (no API call) ===")
    print("Pass memory=[AiKnotAutoGenMemory(kb)] into AssistantAgent(...)")
    print()
    print("User prompt:")
    print(f"  {result.user_prompt}")
    print()
    print("Recalled memory items:")
    for item in result.recalled_items:
        print(f"  - {item}")
    print()
    print("SystemMessage injected into the AutoGen model context:")
    print(result.injected_system_message)
    print()
    print(result.simulated_answer)
    print("(No model call was made in this demo.)")


if __name__ == "__main__":
    main()
