"""Zero-network demo of the LangGraph-shaped ai-knot memory-tool surface.

This example does **not** import LangGraph or call a model. It shows the exact
explicit `add -> search -> list -> delete` seam that ai-knot exposes for
LangGraph-style agent memory flows.

Run::

    python examples/langgraph_surface_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory

from ai_knot import KnowledgeBase
from ai_knot.integrations.langchain import (
    create_basic_memory_tools,
)
from ai_knot.storage import YAMLStorage


@dataclass
class DemoResult:
    tool_names: list[str]
    add_output: str
    search_output: str
    listed_output: str
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
        lines.append("Use the search results before planning the next turn.")
    return "Simulated next-turn answer:\n" + "\n".join(
        f"{index}. {line}" for index, line in enumerate(lines, start=1)
    )


def build_demo_result() -> DemoResult:
    with TemporaryDirectory(prefix="ai-knot-langgraph-") as tmpdir:
        kb = KnowledgeBase(
            agent_id="langgraph-surface-demo",
            storage=YAMLStorage(base_dir=tmpdir),
            embed_url="",
        )
        tools = create_basic_memory_tools(kb, top_k=3)
        by_name = {getattr(tool, "name", ""): tool for tool in tools}
        add_tool = by_name["add_memory"]
        search_tool = by_name["search_memory"]
        list_tool = by_name["list_memory"]
        delete_tool = by_name["delete_memory"]

        add_output = add_tool.invoke({"content": "User deploys APIs with Docker Compose"})
        add_tool.invoke({"content": "User prefers Python over Java"})

        search_output = search_tool.invoke(
            {"query": "what stack should I use for the deployment script?"}
        )
        listed_output = list_tool.invoke({})
        fact_id = kb.list_facts()[0].id
        deleted_output = delete_tool.invoke({"fact_id": fact_id})

        return DemoResult(
            tool_names=[getattr(tool, "name", "") for tool in tools],
            add_output=add_output,
            search_output=search_output,
            listed_output=listed_output,
            deleted_output=deleted_output,
            remaining_facts=len(kb.list_facts()),
            simulated_answer=_build_simulated_answer(search_output),
        )


def main() -> None:
    result = build_demo_result()

    print("=== LangGraph memory tools surface (no API call) ===")
    print("Pass create_basic_memory_tools(kb) into create_react_agent(...)")
    print("or unpack the explicit add/search/list/delete tools yourself.")
    print()
    print("Tool names:")
    for tool_name in result.tool_names:
        print(f"  {tool_name}")
    print()
    print("Add tool output:")
    print(result.add_output)
    print()
    print("Search tool output:")
    print(result.search_output)
    print()
    print("List tool output:")
    print(result.listed_output)
    print()
    print("Delete tool output:")
    print(result.deleted_output)
    print()
    print("Remaining facts after delete:")
    print(f"  {result.remaining_facts}")
    print()
    print(result.simulated_answer)
    print("(No model call was made in this demo.)")


if __name__ == "__main__":
    main()
