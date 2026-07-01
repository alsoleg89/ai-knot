# Claude MCP case study / proof asset

Updated: **July 1, 2026**

Use this file when you want one concrete `ai-knot` integration story that starts
from **Claude Desktop / Claude Code** rather than from a Python framework. The
integration path is short, the underlying MCP surface is already in the product,
and the proof can start from a zero-network setup demo.

Related official ecosystem references:

- MCP servers repo: https://github.com/modelcontextprotocol/servers
- MCP site: https://modelcontextprotocol.io

As of **July 1, 2026**, the official `modelcontextprotocol/servers` repo shows
**87,892 GitHub stars**. That is large enough to treat MCP not as an implementation
detail, but as a real distribution channel.

---

## The angle

Do not pitch this as "an SDK." Pitch it as:

> **Give Claude persistent memory through one MCP server config.**

That means:

- one command prints the config,
- the config drops into Claude's MCP config,
- ai-knot provides self-hosted, deterministic recall underneath.

The hook is not "memory research." The hook is that Claude users can try it with
copy/paste setup and no custom runtime adoption.

---

## Fastest proof paths

### Zero-network proof

Runs without Claude and without an API key:

```bash
python examples/claude_mcp_setup.py
```

What it proves:

- the MCP config block is ready now,
- the Claude path is a real setup surface, not a promise,
- the user can inspect the exact config before wiring anything.

### Real Claude path

```bash
ai-knot setup claude --agent-id my_agent --storage sqlite
```

Copy the printed JSON into:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

What it proves:

- ai-knot is ready for Claude's MCP tool path,
- setup friction is low enough for a copy/paste trial,
- the integration story does not require project-level Python glue code.

---

## What to emphasize

### Problem

Claude users want persistent memory without stuffing transcripts into every turn
and without standing up a hosted memory service first.

### What ai-knot adds

- self-hosted memory behind a paste-ready MCP config,
- deterministic recall with no LLM on the read path,
- SQLite / PostgreSQL / YAML storage control,
- a tool path Claude can call directly over MCP.

### What not to claim

- Do not say Claude requires ai-knot.
- Do not make the post about every MCP client at once.
- Do not lead with architecture when the config path is the sharper hook.

---

## Copy blocks

### GitHub discussion / follow-up comment

> A third concrete surface that is ready today: Claude over MCP.
>
> `ai-knot setup claude --agent-id my_agent --storage sqlite`
>
> Paste the printed JSON into Claude's MCP config and Claude gets persistent,
> self-hosted memory tools. There is also a zero-network proof in
> `examples/claude_mcp_setup.py` if you want to inspect the exact config first.

### X / LinkedIn

> Claude Desktop / Claude Code path for `ai-knot` is ready.
>
> `ai-knot setup claude --agent-id my_agent --storage sqlite`
>
> Paste the config into Claude's MCP config and you get self-hosted,
> deterministic memory over MCP.
>
> Shortest proof: `python examples/claude_mcp_setup.py`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you want the Claude-first route instead of a framework adapter, start with
> the MCP path. `ai-knot setup claude` prints the config, and
> `examples/claude_mcp_setup.py` gives a zero-network proof before you wire Claude.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/claude_mcp_setup.py`
2. `ai-knot setup claude --agent-id my_agent --storage sqlite`
3. [docs/integrations.md](integrations.md)

Do not send people to the whitepaper first. Send them to the setup path.
