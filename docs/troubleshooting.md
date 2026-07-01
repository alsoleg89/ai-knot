# Troubleshooting

Updated: **July 1, 2026**

Use this page when `ai-knot` does not install, the first run is unclear, or an
integration surface does not behave the way the docs suggest.

---

## Start here

If you are on the machine where the failure happened, run:

```bash
ai-knot doctor --json
```

That prints:

- `ai_knot_version`
- Python version and executable
- storage/backend config
- whether key modules are importable (`mcp`, `crewai`, `autogen`, `agents`, `fastapi`, `psycopg`, etc.)
- whether important env vars are present
- whether `ai-knot-mcp` is on `PATH`

The output avoids printing secret values and is designed to be pasted into the
install bug issue template.

If the console script is not on `PATH`, the module entrypoint works too:

```bash
python -m ai_knot.cli doctor --json
```

---

## Public release looks stale

If the README on GitHub or the npm package still looks older than this branch,
check the actual public state:

```bash
./.venv/bin/python scripts/check_public_release.py
```

This verifies:

- local version sync,
- public PyPI latest,
- public npm latest,
- whether public `main` already exposes the launch-ready README/docs markers.

If this fails on `npm latest` or missing public README/docs markers, the issue is
not your local install. The public release simply has not caught up yet.

---

## Common install and setup failures

### `pip install "ai-knot[...]"` worked, but an integration still says a package is missing

Run `ai-knot doctor --json` and inspect the `modules` section.

Common mappings:

- `crewai=false` → install `pip install "ai-knot[crewai]"`
- `autogen_agentchat=false` or `autogen_ext=false` → install `pip install "ai-knot[autogen]"`
- `openai_agents_sdk=false` → install `pip install "ai-knot[agents]"`
- `mcp=false` → install `pip install "ai-knot[mcp]"`
- `fastapi=false` or `uvicorn=false` → install `pip install "ai-knot[server]"`

### `npm install ai-knot` succeeded, but the runtime cannot find Python or MCP

The npm client still needs:

- Python `3.11+`
- `pip`
- the `ai-knot[mcp]` install path underneath

Run `ai-knot doctor --json` and check:

- `python_executable`
- `commands.ai_knot_mcp_on_path`
- `modules.mcp`

### Claude Desktop / Claude Code cannot see memory

Check:

1. you used `ai-knot setup claude --agent-id ... --storage sqlite`
2. the printed JSON is under the `mcpServers` key in Claude's config
3. the storage path is absolute when needed

Fastest sanity check:

```bash
python examples/claude_mcp_setup.py
```

### OpenClaw cannot see memory

Check:

1. you used `ai-knot setup openclaw --agent-id ... --storage sqlite`
2. the JSON was pasted into `~/.openclaw/openclaw.json`
3. the config is nested under `mcpServers`

Fastest local proof:

```bash
python examples/openclaw_integration.py
```

### PostgreSQL backend fails immediately

Check:

- `--storage postgres` is actually set
- `AI_KNOT_DSN` exists, or `--dsn` is passed explicitly
- `modules.psycopg` is `true` in `ai-knot doctor --json`

---

## Filing a useful issue

Use the dedicated templates:

- install/setup failure → `Install bug report`
- wanted runtime/framework surface → `Integration request`
- reproducibility/methodology question → `Benchmark question`

Best payload to include:

1. exact command that failed
2. full error output
3. `ai-knot doctor --json`
4. the surface you were trying to use: CrewAI, OpenClaw, Claude/MCP, AutoGen, OpenAI Agents SDK, LangChain, TypeScript, or HTTP

---

## Fast path references

- integration routing: [integrations.md](integrations.md)
- production/deployment: [deployment.md](deployment.md)
- public launch status: [publish-ready-audit.md](publish-ready-audit.md)
- release flow: [RELEASE.md](RELEASE.md)
