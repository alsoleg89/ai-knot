# Troubleshooting

Updated: **July 2, 2026**

Use this page when `ai-knot` does not install, the first run is unclear, or an
integration surface does not behave the way the docs suggest.

---

## Start here

If you are on the machine where the failure happened, run:

```bash
ai-knot doctor --json
```

If the failure is on the npm / TypeScript path, run this first instead:

```bash
npx ai-knot-doctor --json
```

That prints:

- `ai_knot_version`
- Python version and executable
- storage/backend config
- `mcp_clients.claude` / `mcp_clients.openclaw` with the default config path,
  whether the file exists, whether it parses, and whether `mcpServers.ai-knot`
  is already registered there
- whether key modules are importable (`mcp`, `crewai`, `autogen`, `agents`, `fastapi`, `psycopg`, etc.)
- whether important env vars are present
- whether `ai-knot-mcp` is on `PATH`

The output avoids printing secret values and is designed to be pasted into the
install bug issue template.

If the console script is not on `PATH`, the module entrypoint works too:

```bash
python -m ai_knot.cli doctor --json
```

The npm doctor checks the whole Node-to-Python bridge:

- Node version
- Python `3.11+`
- `pip`
- whether Python can import `ai_knot`
- whether the npm package version matches the installed Python package version
- whether `ai-knot-mcp` is on `PATH`
- the Python-side `ai-knot doctor --json` payload underneath

---

## Public release looks stale

If the README on GitHub or the npm package still looks older than this branch,
check the actual public state:

```bash
./.venv/bin/python scripts/check_public_release.py
```

If the problem is specifically the Pages landing/article URL, require the live
Pages surface too:

```bash
./.venv/bin/python scripts/check_public_release.py --require-pages
```

This verifies:

- local version sync,
- public PyPI latest,
- public npm latest,
- public npm package-page metadata (description, repository URL, README markers),
- whether public `main` already exposes the current README/docs markers.

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

Run `npx ai-knot-doctor --json` and check:

- `checks` for `python`, `pip`, `python_package`, `version_parity`, and `mcp_binary`
- `nextActions` for the exact repair command
- `pythonDoctor.commands.ai_knot_mcp_on_path`
- `pythonDoctor.modules.mcp`

If `mcp_binary` is still missing after install, either fix `PATH` or pass
`command` explicitly in the npm client:

```typescript
const kb = new KnowledgeBase({
  agentId: "assistant",
  command: "/absolute/path/to/ai-knot-mcp",
});
```

The underlying Python-side check is still useful too. Run `ai-knot doctor --json`
or `python -m ai_knot.cli doctor --json` and check:

- `python_executable`
- `commands.ai_knot_mcp_on_path`
- `modules.mcp`

### Claude Desktop / Claude Code cannot see memory

Check:

1. you used `ai-knot setup claude --agent-id ... --storage sqlite`
2. the printed JSON is under the `mcpServers` key in Claude's config, or you used
   `--write-default-config` / `--write-config <path>` on a plain-JSON config file
3. the storage path is absolute when needed

In `ai-knot doctor --json`, inspect:

- `mcp_clients.claude.default_config_path`
- `mcp_clients.claude.exists`
- `mcp_clients.claude.parseable`
- `mcp_clients.claude.ai_knot_registered`

Fastest sanity check:

```bash
python examples/claude_mcp_setup.py
```

### OpenClaw cannot see memory

Check:

1. you used `ai-knot setup openclaw --agent-id ... --storage sqlite`
2. the JSON was pasted into `~/.openclaw/openclaw.json`, or you used
   `--write-default-config` / `--write-config ~/.openclaw/openclaw.json` on a plain-JSON config file
3. the config is nested under `mcpServers`

If `setup openclaw --write-default-config` or `--write-config` succeeded, the CLI
already printed the next step: restart OpenClaw, then run `ai-knot doctor --json`.

In `ai-knot doctor --json`, inspect:

- `mcp_clients.openclaw.default_config_path`
- `mcp_clients.openclaw.exists`
- `mcp_clients.openclaw.parseable`
- `mcp_clients.openclaw.ai_knot_registered`

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
- release path and registry workflow: [RELEASE.md](RELEASE.md)
- release flow: [RELEASE.md](RELEASE.md)
