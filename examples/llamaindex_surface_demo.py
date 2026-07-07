"""Zero-network demo of the LlamaIndex-shaped ai-knot memory surface.

This example does **not** require LlamaIndex or call a model. It shows the
same `memory=...` seam LlamaIndex agents and chat engines expect, and the
system-style memory block that ai-knot injects on `get(...)`.

Run::

    python examples/llamaindex_surface_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory

from ai_knot import KnowledgeBase
from ai_knot.integrations.llamaindex import AiKnotLlamaIndexMemory
from ai_knot.storage import YAMLStorage


@dataclass
class DemoResult:
    user_prompt: str
    injected_role: str
    injected_content: str
    history_length: int
    stored_fact_count: int
    simulated_answer: str


def _normalise_role(message: object) -> str:
    if isinstance(message, dict):
        return str(message["role"]).lower()
    role = message.role
    value = getattr(role, "value", role)
    return str(value).lower()


def _build_simulated_answer(injected_content: str) -> str:
    steps: list[str] = []
    if "Python" in injected_content:
        steps.append("Keep the checklist centered on a Python API.")
    if "Docker Compose" in injected_content:
        steps.append("Add Docker Compose deployment steps to the answer.")
    if not steps:
        steps.append("Use the injected memory block before answering.")
    return "Simulated next-turn answer:\n" + "\n".join(
        f"{index}. {step}" for index, step in enumerate(steps, start=1)
    )


def build_demo_result() -> DemoResult:
    with TemporaryDirectory(prefix="ai-knot-llamaindex-") as tmpdir:
        kb = KnowledgeBase(
            agent_id="llamaindex-surface-demo",
            storage=YAMLStorage(base_dir=tmpdir),
            embed_url="",
        )
        kb.add("User prefers Python over Java")
        kb.add("User deploys APIs with Docker Compose")

        memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, top_k=3)
        memory.put({"role": "user", "content": "Please remember that release notes stay concise."})

        user_prompt = "Write a deployment checklist for my API stack."
        messages = memory.get(user_prompt)

        injected = messages[0]
        injected_role = _normalise_role(injected)
        injected_content = (
            injected["content"] if isinstance(injected, dict) else str(injected.content)
        )

        return DemoResult(
            user_prompt=user_prompt,
            injected_role=injected_role,
            injected_content=injected_content,
            history_length=len(memory.get_all()),
            stored_fact_count=len(kb.list_facts()),
            simulated_answer=_build_simulated_answer(injected_content),
        )


def main() -> None:
    result = build_demo_result()

    print("=== LlamaIndex memory surface (no API call) ===")
    print("Pass memory=AiKnotLlamaIndexMemory.from_defaults(...) into")
    print("SimpleChatEngine.from_defaults(...), FunctionAgent.run(...), or ReActAgent.run(...).")
    print()
    print("User prompt:")
    print(f"  {result.user_prompt}")
    print()
    print("Injected message role:")
    print(f"  {result.injected_role}")
    print()
    print("Injected system-style memory block:")
    print(result.injected_content)
    print()
    print("Short-term history length:")
    print(f"  {result.history_length}")
    print()
    print("Stored fact count:")
    print(f"  {result.stored_fact_count}")
    print()
    print(result.simulated_answer)
    print("(No model call was made in this demo.)")


if __name__ == "__main__":
    main()
