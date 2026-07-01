"""AutoGen integration example.

Shows how to add ai-knot long-term memory to an AutoGen ``AssistantAgent`` via
the ``memory=[...]`` constructor hook.

Run::

    pip install "ai-knot[autogen]"
    OPENAI_API_KEY=... python examples/autogen_integration.py
"""

from __future__ import annotations

import asyncio

from ai_knot import KnowledgeBase
from ai_knot.integrations.autogen import AiKnotAutoGenMemory

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError as exc:  # pragma: no cover - example only
    raise SystemExit(
        'Install AutoGen first: pip install "ai-knot[autogen]" '
        '(or pip install autogen-agentchat "autogen-ext[openai]")'
    ) from exc


async def main() -> None:
    kb = KnowledgeBase(agent_id="assistant")
    kb.add("User prefers Python over Java")
    kb.add("User deploys APIs with Docker and Kubernetes")

    memory = AiKnotAutoGenMemory(kb, top_k=5)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent(
        name="coding_assistant",
        model_client=model_client,
        memory=[memory],
    )

    stream = agent.run_stream(task="Write a deployment checklist for my API stack.")
    await Console(stream)


asyncio.run(main())
