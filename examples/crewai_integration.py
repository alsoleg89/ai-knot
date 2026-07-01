"""CrewAI integration example.

Shows how to pass ai-knot into CrewAI's native ``Crew(memory=...)`` and
``Agent(memory=...)`` surfaces.

Run::

    pip install "ai-knot[crewai]"
    OPENAI_API_KEY=... python examples/crewai_integration.py
"""

from __future__ import annotations

from ai_knot import KnowledgeBase
from ai_knot.integrations.crewai import AiKnotCrewAIMemory

try:
    from crewai import LLM, Agent, Crew, Process, Task
except ImportError as exc:  # pragma: no cover - example only
    raise SystemExit(
        'Install CrewAI first: pip install "ai-knot[crewai]" (or pip install crewai)'
    ) from exc


kb = KnowledgeBase(agent_id="crew-demo", provider="openai")
kb.add("User prefers Python over Java")
kb.add("User deploys APIs with Docker and Kubernetes")

memory = AiKnotCrewAIMemory(kb, top_k=5)
llm = LLM(model="gpt-4o-mini", temperature=0)

researcher = Agent(
    role="Researcher",
    goal="Identify the user's stack and operational constraints.",
    backstory="A careful backend researcher who keeps durable notes.",
    llm=llm,
    memory=memory.scope("/agent/researcher"),
)

writer = Agent(
    role="Writer",
    goal="Produce a concise deployment checklist.",
    backstory="A senior engineer who writes crisp runbooks.",
    llm=llm,
)

task = Task(
    description="Write a deployment checklist for the user's API stack.",
    expected_output="A short numbered checklist.",
    agent=writer,
)

crew = Crew(
    name="deployment-checklist",
    agents=[researcher, writer],
    tasks=[task],
    process=Process.sequential,
    memory=memory,
    verbose=True,
)

result = crew.kickoff()
print(result)
