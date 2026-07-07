"""Zero-network demo of the PydanticAI-shaped ai-knot memory surface.

This example does **not** call a model. It shows the exact runtime
``instructions=...`` hook that PydanticAI exposes and how ``ai-knot`` appends
query-relevant long-term memory to it.

Run::

    python examples/pydanticai_surface_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import cast

from ai_knot import KnowledgeBase
from ai_knot.integrations.pydanticai import AiKnotPydanticAIMemory
from ai_knot.storage import YAMLStorage


@dataclass
class FakeResult:
    simulated_answer: str
    user_prompt: str
    instructions: str | list[str] | None


def _build_simulated_answer(instructions: str | list[str] | None) -> str:
    text = "\n".join(instructions) if isinstance(instructions, list) else (instructions or "")
    steps: list[str] = []
    if "Python" in text:
        steps.append("Use Python for the service implementation.")
    if "Docker Compose" in text:
        steps.append("Package the API with Docker Compose for local deployment.")
    if not steps:
        steps.append("Review the injected memory before planning the next step.")
    return "Simulated next-turn answer:\n" + "\n".join(
        f"{index}. {step}" for index, step in enumerate(steps, start=1)
    )


class FakeAgent:
    def run_sync(self, user_prompt: str, /, **kwargs: object) -> FakeResult:
        instructions = cast(str | list[str] | None, kwargs.get("instructions"))
        return FakeResult(
            simulated_answer=_build_simulated_answer(instructions),
            user_prompt=user_prompt,
            instructions=instructions,
        )


def build_demo_result() -> FakeResult:
    with TemporaryDirectory(prefix="ai-knot-pydanticai-") as tmpdir:
        kb = KnowledgeBase(
            agent_id="pydanticai-surface-demo",
            storage=YAMLStorage(base_dir=tmpdir),
            embed_url="",
        )
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
    print(result.simulated_answer)
    print("(No model call was made in this demo.)")


if __name__ == "__main__":
    main()
