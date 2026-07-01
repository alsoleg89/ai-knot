"""PydanticAI integration example.

Shows how to add ai-knot long-term memory to a PydanticAI run via runtime
``instructions=...``. Requires ``pydantic-ai`` and a model API key for an
actual networked run.

Run::

    pip install "ai-knot[pydanticai,openai]"
    OPENAI_API_KEY=... python examples/pydanticai_integration.py
"""

from __future__ import annotations

from ai_knot import KnowledgeBase
from ai_knot.integrations.pydanticai import AiKnotPydanticAIMemory

try:
    from pydantic_ai import Agent
except ImportError as exc:  # pragma: no cover - example only
    raise SystemExit(
        'Install PydanticAI first: pip install "ai-knot[pydanticai]" '
        "(or pip install pydantic-ai)"
    ) from exc


kb = KnowledgeBase(agent_id="assistant")
kb.add("User prefers Python over Java")
kb.add("User deploys APIs with Docker Compose")

memory = AiKnotPydanticAIMemory(kb)
agent = Agent(
    "openai:gpt-5.2",
    instructions="You are a concise senior backend engineer.",
)

result = memory.run_sync(agent, "Write a local deployment checklist for my API stack.")
print(result.output)
