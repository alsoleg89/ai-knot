"""Zero-network demo of ai-knot's plain function-calling memory surface.

This example does **not** require LangChain, LangGraph, or a model call. It
shows the exact `add -> search -> list -> get -> delete` seam that ai-knot can
expose to any runtime that accepts ordinary Python callables as tools.

Run::

    python examples/function_calling_surface_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory

from ai_knot import KnowledgeBase
from ai_knot.integrations import create_basic_memory_functions
from ai_knot.storage import YAMLStorage


@dataclass
class DemoResult:
    function_names: list[str]
    add_output: str
    search_output: str
    listed_output: str
    fetched_output: str
    deleted_output: str
    remaining_facts: int
    simulated_answer: str


def _build_simulated_answer(search_output: str) -> str:
    lines: list[str] = []
    if "Python" in search_output:
        lines.append("Use Python for the deployment automation.")
    if "Docker Compose" in search_output:
        lines.append("Include Docker Compose in the rollout checklist.")
    if not lines:
        lines.append("Use the recalled memory facts before planning the next turn.")
    return "Simulated next-turn answer:\n" + "\n".join(
        f"{index}. {line}" for index, line in enumerate(lines, start=1)
    )


def build_demo_result() -> DemoResult:
    with TemporaryDirectory(prefix="ai-knot-function-tools-") as tmpdir:
        kb = KnowledgeBase(
            agent_id="function-calling-surface-demo",
            storage=YAMLStorage(base_dir=tmpdir),
            embed_url="",
        )
        functions = create_basic_memory_functions(kb, top_k=3, include_get=True)
        by_name = {func.__name__: func for func in functions}
        add_memory = by_name["add_memory"]
        search_memory = by_name["search_memory"]
        list_memory = by_name["list_memory"]
        get_memory = by_name["get_memory"]
        delete_memory = by_name["delete_memory"]

        add_output = add_memory("User deploys APIs with Docker Compose")
        add_memory("User prefers Python over Java")

        search_output = search_memory("what stack should I use for the deployment script?")
        listed_output = list_memory()
        fact_id = kb.list_facts()[0].id
        fetched_output = get_memory(fact_id)
        deleted_output = delete_memory(fact_id)

        return DemoResult(
            function_names=[func.__name__ for func in functions],
            add_output=add_output,
            search_output=search_output,
            listed_output=listed_output,
            fetched_output=fetched_output,
            deleted_output=deleted_output,
            remaining_facts=len(kb.list_facts()),
            simulated_answer=_build_simulated_answer(search_output),
        )


def main() -> None:
    result = build_demo_result()

    print("=== Function-calling memory surface (no API call) ===")
    print("Pass these plain Python callables into any runtime that accepts")
    print("ordinary functions as tools.")
    print()
    print("Function names:")
    for function_name in result.function_names:
        print(f"  {function_name}")
    print()
    print("Add function output:")
    print(result.add_output)
    print()
    print("Search function output:")
    print(result.search_output)
    print()
    print("List function output:")
    print(result.listed_output)
    print()
    print("Get function output:")
    print(result.fetched_output)
    print()
    print("Delete function output:")
    print(result.deleted_output)
    print()
    print("Remaining facts after delete:")
    print(f"  {result.remaining_facts}")
    print()
    print(result.simulated_answer)
    print("(No model call was made in this demo.)")


if __name__ == "__main__":
    main()
