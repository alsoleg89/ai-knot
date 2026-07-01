"""Zero-network demo of the CrewAI-shaped ai-knot memory surface.

This example does **not** run a Crew or call a model. It shows the exact memory
object you pass into ``Crew(memory=...)`` and the scoped views you can pass into
``Agent(memory=memory.scope(...))``.

It runs without CrewAI installed because ``AiKnotCrewAIMemory`` safely falls
back to lightweight shims outside a real CrewAI runtime.

Run::

    python examples/crewai_surface_demo.py
"""

from __future__ import annotations

import shutil

from ai_knot import KnowledgeBase
from ai_knot.integrations.crewai import AiKnotCrewAIMemory


def main() -> None:
    kb = KnowledgeBase(agent_id="crew-surface-demo")
    memory = AiKnotCrewAIMemory(kb, top_k=3)
    researcher = memory.scope("/agent/researcher")
    writer = memory.scope("/agent/writer")

    researcher.remember(
        "User deploys APIs with Docker and Kubernetes",
        categories=["ops"],
    )
    researcher.remember(
        "Primary database is PostgreSQL",
        categories=["database"],
    )
    writer.remember(
        "Release notes should stay concise and numbered",
        categories=["docs"],
    )

    print("=== CrewAI memory surface (no API call) ===")
    print("Pass root memory into Crew(memory=memory)")
    print("Pass a scoped view into Agent(memory=memory.scope('/agent/researcher'))")
    print()
    print("Visible top-level scopes:", memory.list_scopes("/"))
    print()
    print("Researcher recall for 'database and deployment':")
    for match in researcher.recall("database and deployment"):
        print(
            f"  - {match.record.content} "
            f"[scope={match.record.scope}, categories={match.record.categories}]"
        )
    print()
    print("Writer recall for 'release notes':")
    for match in writer.recall("release notes"):
        print(
            f"  - {match.record.content} "
            f"[scope={match.record.scope}, categories={match.record.categories}]"
        )
    print()
    agent_info = memory.info("/agent")
    print(f"/agent subtree facts: {agent_info.record_count}")

    shutil.rmtree(".ai_knot/crew-surface-demo", ignore_errors=True)


if __name__ == "__main__":
    main()
