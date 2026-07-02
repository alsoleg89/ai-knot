"""OpenAI Agents SDK integration example.

Shows how to add ai-knot long-term memory to an OpenAI Agents SDK run via the
``call_model_input_filter`` hook. Requires the SDK and an OpenAI API key for an
actual networked run.

Run::

    pip install "ai-knot[agents]"
    OPENAI_API_KEY=... python examples/openai_agents_integration.py
"""

from __future__ import annotations

from ai_knot import KnowledgeBase
from ai_knot.integrations.openai_agents import AiKnotAgentsMemory

try:
    from agents import Agent, Runner
except ImportError as exc:  # pragma: no cover - example only
    raise SystemExit(
        'Install the OpenAI Agents SDK first: pip install "ai-knot[agents]" '
        "(or pip install openai-agents)"
    ) from exc


kb = KnowledgeBase(agent_id="assistant")
kb.add("User prefers Python over Java")
kb.add("User deploys APIs with Docker and Kubernetes")

memory = AiKnotAgentsMemory(kb)
run_config = memory.build_run_config()

agent = Agent(
    name="Coding assistant",
    instructions="You are a concise senior backend engineer.",
)

result = Runner.run_sync(
    agent,
    "Write a deployment checklist for my API stack.",
    run_config=run_config,
)

print(result.final_output)
