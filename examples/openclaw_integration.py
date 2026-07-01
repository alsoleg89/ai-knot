"""OpenClaw integration example.

Shows both ways to use ai-knot with OpenClaw:
  1. generate a paste-ready MCP config for the OpenClaw app;
  2. use the Python-side compatibility adapter in a custom agent.

This is a zero-network proof: no OpenClaw app, subprocess, or API key required.

Run::

    python examples/openclaw_integration.py
"""

from __future__ import annotations

import json
import shutil

from ai_knot import KnowledgeBase
from ai_knot.integrations.openclaw import OpenClawMemoryAdapter, generate_mcp_config

print("=== MCP config for the OpenClaw app ===")
print(json.dumps(generate_mcp_config("openclaw-demo"), indent=2))

print("\n=== Python adapter demo ===")
kb = KnowledgeBase(agent_id="openclaw-demo", embed_url="")
memory = OpenClawMemoryAdapter(kb)
memory.add([{"role": "user", "content": "Deploy on Fridays with Docker Compose"}])

results = memory.search("deployment schedule")
for item in results:
    print(f"  [{item['score']:.2f}] {item['memory']}")

shutil.rmtree(".ai_knot", ignore_errors=True)
print("\nDemo complete.")
