"""Zero-network demo of the PydanticAI-shaped ai-knot memory surface.

This example does **not** call a model. It shows the exact runtime
``instructions=...`` hook that PydanticAI exposes and how ``ai-knot`` appends
query-relevant long-term memory to it.

Run::

    python examples/pydanticai_surface_demo.py
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import cast

from ai_knot import KnowledgeBase
from ai_knot.integrations.pydanticai import AiKnotPydanticAIMemory


@dataclass
class FakeResult:
    output: str
    user_prompt: str
    instructions: str | list[str] | None


class FakeAgent:
    def run_sync(self, user_prompt: str, /, **kwargs: object) -> FakeResult:
        return FakeResult(
            output="No model call was made.",
            user_prompt=user_prompt,
            instructions=cast(str | list[str] | None, kwargs.get("instructions")),
        )


def build_demo_result() -> FakeResult:
    kb = KnowledgeBase(agent_id="pydanticai-surface-demo")
    kb.add("User prefers Python over Java")
    kb.add("User deploys APIs with Docker Compose")

    memory = AiKnotPydanticAIMemory(kb, top_k=3)
    agent = FakeAgent()
    return memory.run_sync(
        agent,
        "Write a deployment checklist for my stack.",
        instructions="You are a concise staff engineer.",
    )


def main() -> None:
    result = build_demo_result()

    print("=== PydanticAI memory surface (no API call) ===")
    print("Pass your normal Agent into AiKnotPydanticAIMemory.run_sync(...)")
    print()
    print("User prompt:")
    print(f"  {result.user_prompt}")
    print()
    print("Runtime instructions sent to the agent:")
    print(result.instructions)
    print()
    print("Result placeholder:")
    print(f"  {result.output}")

    shutil.rmtree(".ai_knot/pydanticai-surface-demo", ignore_errors=True)


if __name__ == "__main__":
    main()
