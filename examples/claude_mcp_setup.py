"""Zero-network setup demo for the Claude Desktop / Claude Code MCP path.

This example prints the exact MCP config block you can paste into Claude's MCP
config, using the same underlying config shape as ``ai-knot setup claude``.

Run::

    python examples/claude_mcp_setup.py
"""

from __future__ import annotations

import json

from ai_knot.integrations.openclaw import generate_mcp_config

config = generate_mcp_config(agent_id="claude-demo", storage="sqlite")

print("=== Shortest CLI path on supported platforms ===")
print("ai-knot setup claude --agent-id assistant --storage sqlite --write-default-config")
print("ai-knot doctor --json")
print()
print("Inside Claude, the memory loop stays add/search/list/delete.")
print()
print("=== Manual MCP config fallback ===")
print("macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json")
print("Windows: %APPDATA%\\Claude\\claude_desktop_config.json")
print()
print(json.dumps(config, indent=2))
