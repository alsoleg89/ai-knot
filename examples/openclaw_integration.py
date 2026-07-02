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
from pathlib import Path

from ai_knot import Fact, KnowledgeBase
from ai_knot.integrations.openclaw import OpenClawMemoryAdapter, generate_mcp_config

print("=== Shortest CLI path on supported platforms ===")
print("ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config")
print("ai-knot doctor --json")
print()
print("Inside OpenClaw, the memory loop stays add/search/list/delete.")
print("Inside Python provider-compat mode, the adapter supports both add/search/get_all/delete")
print("and the ai-knot aliases add/search/list/delete.")
print()
print("=== Manual MCP config fallback ===")
print("macOS / Linux: ~/.openclaw/openclaw.json")
print("Windows:       %APPDATA%\\OpenClaw\\openclaw.json")
print()
print(json.dumps(generate_mcp_config("openclaw-demo"), indent=2))

print("\n=== Python adapter demo ===")
data_dir = Path(".ai_knot").resolve()
kb = KnowledgeBase(agent_id="openclaw-demo", data_dir=str(data_dir), embed_url="")
memory = OpenClawMemoryAdapter(kb)
created = memory.add([{"role": "user", "content": "Deploy on Fridays with Docker Compose"}])
memory.add([{"role": "user", "content": "Primary API stack is FastAPI + PostgreSQL"}])

results = memory.search("deployment schedule")
for item in results:
    print(f"  [{item['score']:.2f}] {item['memory']}")

print("\nAll stored memories (provider-compatible get_all):")
for item in memory.get_all():
    print(f"  - {item['id']}: {item['memory']}")

structured = kb.add_resolved(
    [
        Fact(
            content="Current employer is Acme",
            entity="user",
            attribute="employer",
            value_text="Acme",
        )
    ]
)[0]
updated = memory.update(structured.id, "Current employer is Globex")
print("\nStructured update via provider-compatible memory.update(...):")
print(f"  New active id: {updated['id']}")
print(f"  Active list count: {len(memory.get_all())}")
print(f"  History count: {len(memory.get_all(include_inactive=True))}")
print("  Structured lineage:")
for item in memory.lineage(updated["id"]):
    state = "active" if item["metadata"]["active"] else "inactive"
    print(f"    - [{state}] {item['memory']}")

created_id = created["results"][0]["id"]
memory.forget(created_id)
print(f"\nRemoved {created_id} via memory.forget(...) alias.")

print("Remaining memories (ai-knot list alias):")
for item in memory.list():
    print(f"  - {item['id']}: {item['memory']}")

shutil.rmtree(".ai_knot", ignore_errors=True)
print("\nDemo complete.")
