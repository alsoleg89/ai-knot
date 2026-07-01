# OpenClaw case study / proof asset

Updated: **July 1, 2026**

Use this file when you want one concrete `ai-knot` integration story that does
not start from Python framework code. Right now OpenClaw is the best prepared
**app/MCP** follow-up surface: the integration path is one config block, the
repo already has a zero-network proof, and the user story is easy to repeat.

Official references:

- OpenClaw repo: https://github.com/openclaw/openclaw
- OpenClaw site: https://openclaw.ai
- MCP servers repo: https://github.com/modelcontextprotocol/servers

As of **July 1, 2026**, the official `openclaw/openclaw` repo shows
**381,166 GitHub stars**, and the official `modelcontextprotocol/servers` repo
shows **87,892 GitHub stars**. The implication is simple: app/MCP channels can
be larger top-of-funnel surfaces than a Python adapter alone.

---

## The angle

Do not pitch this as "another MCP server." Pitch it as:

> **Add persistent memory to OpenClaw with one config block.**

That means:

- one command prints the config,
- the config drops into `~/.openclaw/openclaw.json`,
- ai-knot provides self-hosted, deterministic recall underneath.

The hook is not "infrastructure." The hook is that the user can try memory in an
app-shaped surface without adopting a new runtime.

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
- the storage/retrieval path is local and deterministic.

### Real OpenClaw app path

```bash
ai-knot setup openclaw --agent-id my_agent --storage sqlite
```

Copy the printed JSON into:

- macOS / Linux: `~/.openclaw/openclaw.json`
- Windows: `%APPDATA%\OpenClaw\openclaw.json`

What it proves:

- ai-knot is ready for the app-facing MCP route,
- setup friction is low enough for a copy/paste trial,
- the integration story does not require Python glue code.

---

## What to emphasize

### Problem

App-first agent users want persistent memory without wiring a full backend stack
or adopting a hosted memory platform.

### What ai-knot adds

- self-hosted memory behind a paste-ready MCP config,
- deterministic recall with no LLM on the read path,
- SQLite / PostgreSQL / YAML storage control,
- a second path for Python-native provider compatibility when needed.

### What not to claim

- Do not say OpenClaw requires ai-knot.
- Do not blur the path with every MCP client at once; keep the post focused.
- Do not turn this into a generic MCP explainer first.

---

## Copy blocks

### GitHub discussion / follow-up comment

> A second concrete surface that is ready today: OpenClaw.
>
> `ai-knot setup openclaw --agent-id my_agent --storage sqlite`
>
> Paste the printed JSON into `~/.openclaw/openclaw.json` and you get
> persistent, self-hosted memory over MCP. There is also a zero-network proof in
> `examples/openclaw_integration.py` if you want to see both the config path and
> the Python adapter path without running the app.

### X / LinkedIn

> OpenClaw path for `ai-knot` is ready.
>
> `ai-knot setup openclaw --agent-id my_agent --storage sqlite`
>
> Paste the config into `~/.openclaw/openclaw.json` and you get self-hosted,
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
2. `ai-knot setup openclaw --agent-id my_agent --storage sqlite`
3. [docs/integrations.md](integrations.md)

Do not send people to the benchmark page first. Send them to the config flow.
