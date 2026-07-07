"""Zero-network demo of the OpenAI Agents SDK-shaped ai-knot memory surface.

This example does **not** import the real SDK or call a model. It shows the
exact ``run_config=memory.build_run_config()`` seam that ai-knot uses and the
instructions payload that reaches the next model turn after memory injection.

Run::

    python examples/openai_agents_surface_demo.py
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any

from ai_knot import KnowledgeBase
from ai_knot.integrations.openai_agents import AiKnotAgentsMemory
from ai_knot.storage import YAMLStorage


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


@dataclass
class DemoResult:
    user_prompt: str
    workflow_name: str | None
    instructions: str | None
    simulated_answer: str


@contextmanager
def _fake_agents_sdk() -> Iterator[None]:
    agents_mod = types.ModuleType("agents")
    agents_mod.RunConfig = _FakeRunConfig

    run_mod = types.ModuleType("agents.run")
    run_mod.ModelInputData = _FakeModelInputData

    original_agents = sys.modules.get("agents")
    original_run = sys.modules.get("agents.run")
    sys.modules["agents"] = agents_mod
    sys.modules["agents.run"] = run_mod
    try:
        yield
    finally:
        if original_agents is None:
            sys.modules.pop("agents", None)
        else:
            sys.modules["agents"] = original_agents
        if original_run is None:
            sys.modules.pop("agents.run", None)
        else:
            sys.modules["agents.run"] = original_run


def _build_simulated_answer(instructions: str | None) -> str:
    text = instructions or ""
    steps: list[str] = []
    if "Python" in text:
        steps.append("Keep the API checklist Python-focused.")
    if "Docker Compose" in text:
        steps.append("Include Docker Compose steps for deployment.")
    if not steps:
        steps.append("Use the recalled memory block before planning the answer.")
    return "Simulated next-turn answer:\n" + "\n".join(
        f"{index}. {step}" for index, step in enumerate(steps, start=1)
    )


def build_demo_result() -> DemoResult:
    with TemporaryDirectory(prefix="ai-knot-openai-agents-") as tmpdir:
        kb = KnowledgeBase(
            agent_id="openai-agents-surface-demo",
            storage=YAMLStorage(base_dir=tmpdir),
            embed_url="",
        )
        kb.add("User prefers Python over Java")
        kb.add("User deploys APIs with Docker Compose")

        memory = AiKnotAgentsMemory(kb, top_k=3)
        user_prompt = "Write a deployment checklist for my API stack."
        with _fake_agents_sdk():
            run_config = memory.build_run_config(workflow_name="surface-demo")
            filtered = run_config.call_model_input_filter(
                _FakeCallModelData(
                    model_data=_FakeModelInputData(
                        input=[{"role": "user", "content": user_prompt}],
                        instructions="You are a concise staff engineer.",
                    )
                )
            )

        return DemoResult(
            user_prompt=user_prompt,
            workflow_name=run_config.workflow_name,
            instructions=filtered.instructions,
            simulated_answer=_build_simulated_answer(filtered.instructions),
        )


def main() -> None:
    result = build_demo_result()

    print("=== OpenAI Agents SDK memory surface (no API call) ===")
    print("Pass run_config=memory.build_run_config() into Runner.run(...) or Runner.run_sync(...)")
    print()
    print("Workflow name:")
    print(f"  {result.workflow_name}")
    print()
    print("User prompt:")
    print(f"  {result.user_prompt}")
    print()
    print("Instructions sent to the next model turn:")
    print(result.instructions)
    print()
    print(result.simulated_answer)
    print("(No model call was made in this demo.)")


if __name__ == "__main__":
    main()
