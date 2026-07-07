# OpenClaw case study / proof asset

Updated: **July 2, 2026**

One concrete `ai-knot` integration story that doesn't start from Python framework code. OpenClaw is the best-prepared app/MCP follow-up surface: the integration path is one config block, and the repo already has a zero-network proof.

Official references:

- OpenClaw repo: https://github.com/openclaw/openclaw
- OpenClaw site: https://openclaw.ai
- MCP servers repo: https://github.com/modelcontextprotocol/servers

OpenClaw and the broader MCP ecosystem are both large public distribution
surfaces. The implication is simple: app/MCP channels can be larger
top-of-funnel surfaces than a Python adapter alone.

---

## The angle

Not "another MCP server" — instead:

> **Add persistent memory to OpenClaw with one config block.**

That means:

- one command prints the config,
- the config drops into `~/.openclaw/openclaw.json`,
- ai-knot provides self-hosted, deterministic recall underneath.

Not "infrastructure" — the user can try memory in an app-shaped surface without adopting a new runtime.

---

## Fastest proof paths

### Zero-network proof

Runs without OpenClaw installed and without an API key:

```bash
python examples/openclaw_integration.py
```

What it proves:

- the MCP config block is ready now,
- the Python-side compatibility adapter works now,
- the provider-compat loop is concrete: `add/search/get_all/delete` with
  `list/forget` aliases available too,
- structured memories also expose `memory.lineage(current_id)` for a direct
  newest -> oldest audit trail,
- structured updates can preserve lineage instead of only doing delete+add,
- the storage/retrieval path is local and deterministic.

### Real OpenClaw app path

```bash
ai-knot setup openclaw --agent-id my_agent --storage sqlite --write-default-config
# or merge directly into a non-default plain-JSON config:
ai-knot setup openclaw --agent-id my_agent --storage sqlite --write-config ~/.openclaw/openclaw.json
```

If you stay on the manual path instead of `--write-config`, copy the printed JSON into:

- macOS / Linux: `~/.openclaw/openclaw.json`
- Windows: `%APPDATA%\OpenClaw\openclaw.json`

What it proves:

- ai-knot is ready for the app-facing MCP route,
- setup friction is low enough for either a direct config merge or a copy/paste trial,
- the integration story does not require Python glue code.

### Remote MCP host path

If the MCP host speaks Streamable HTTP instead of a local stdio config, the
same repo now also exposes:

```bash
ai-knot serve-mcp assistant --port 8765
```

That is not the main OpenClaw wedge, but it matters for adjacent MCP hosts that
want the same memory server over HTTP rather than a local config file.

---

## What to emphasize

### Problem

App-first agent users want persistent memory without wiring a full backend stack or adopting a hosted memory platform.

### What ai-knot adds

- self-hosted memory behind a paste-ready MCP config,
- deterministic recall with no LLM on the read path,
- SQLite / PostgreSQL / YAML storage control,
- a second path for Python-native provider compatibility when needed.

### What not to claim

- Don't say OpenClaw requires ai-knot.
- Don't blur the path with every MCP client at once; keep the post focused.
- Don't turn this into a generic MCP explainer first.

---

## Copy blocks

### GitHub discussion / follow-up comment

> A second concrete surface that is ready today: OpenClaw.
>
> `ai-knot setup openclaw --agent-id my_agent --storage sqlite --write-default-config`
>
> `ai-knot` resolves the default OpenClaw config path, merges `mcpServers.ai-knot`, and you get
> persistent, self-hosted memory over MCP. There is also a zero-network proof in
> `examples/openclaw_integration.py` if you want to see both the config path and
> the Python adapter path without running the app.

### X / LinkedIn

> OpenClaw path for `ai-knot` is ready.
>
> `ai-knot setup openclaw --agent-id my_agent --storage sqlite --write-default-config`
>
> Merge the default OpenClaw config and you get self-hosted,
> deterministic memory over MCP.
>
> Shortest proof: `python examples/openclaw_integration.py`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you want the app/MCP route instead of a Python framework adapter, start with
> the OpenClaw path. The config is now one command, and the repo has a local
> `examples/openclaw_integration.py` proof before you touch the app.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/openclaw_integration.py`
2. `ai-knot setup openclaw --agent-id my_agent --storage sqlite --write-default-config`
3. `ai-knot serve-mcp assistant --port 8765` for HTTP-capable MCP hosts
4. [docs/integrations.md](integrations.md)

Don't send people to the benchmark page first. Send them to the config flow.
