# Claude MCP case study / proof asset

Updated: **July 2, 2026**

Use this file when you want one concrete `ai-knot` integration story that starts
from **Claude Desktop / Claude Code** rather than from a Python framework. The
integration path is short, the underlying MCP surface is already in the product,
and the proof can start from a zero-network setup demo.

Related official ecosystem references:

- MCP servers repo: https://github.com/modelcontextprotocol/servers
- MCP site: https://modelcontextprotocol.io

MCP is large enough to treat not as an implementation detail, but as a real
distribution channel. What matters for this asset is that Claude users can try
memory through a familiar config-driven tool path, not that a star count moved
by a few hundred.

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
ai-knot setup claude --agent-id my_agent --storage sqlite --write-default-config
# or merge directly into a non-default plain-JSON config:
ai-knot setup claude --agent-id my_agent --storage sqlite --write-config ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

If you stay on the manual path instead of `--write-config`, copy the printed JSON into:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

What it proves:

- ai-knot is ready for Claude's MCP tool path,
- setup friction is low enough for either a direct config merge or a copy/paste trial,
- the integration story does not require project-level Python glue code.

### Remote MCP host path

If the MCP host speaks Streamable HTTP instead of consuming a local stdio
config, the same repo now also exposes:

```bash
ai-knot serve-mcp assistant --port 8765
```

That is not the default Claude Desktop path, but it matters for adjacent MCP
hosts that want the same memory server over HTTP.

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
> `ai-knot setup claude --agent-id my_agent --storage sqlite --write-default-config`
>
> `ai-knot` resolves Claude's default config path, merges `mcpServers.ai-knot`, and Claude gets persistent,
> self-hosted memory tools. There is also a zero-network proof in
> `examples/claude_mcp_setup.py` if you want to inspect the exact config first.

### X / LinkedIn

> Claude Desktop / Claude Code path for `ai-knot` is ready.
>
> `ai-knot setup claude --agent-id my_agent --storage sqlite --write-default-config`
>
> Merge Claude's default MCP config and you get self-hosted,
> deterministic memory over MCP.
>
> Shortest proof: `python examples/claude_mcp_setup.py`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you want the Claude-first route instead of a framework adapter, start with
> the MCP path. `ai-knot setup claude --write-default-config` can merge the config, and
> `examples/claude_mcp_setup.py` gives a zero-network proof before you wire Claude.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/claude_mcp_setup.py`
2. `ai-knot setup claude --agent-id my_agent --storage sqlite --write-default-config`
3. `ai-knot serve-mcp assistant --port 8765` for HTTP-capable MCP hosts
4. [docs/integrations.md](integrations.md)

Do not send people to the whitepaper first. Send them to the setup path.
