# Codespaces quickstart

Use this path when you want to try `ai-knot` without installing Python, Node,
or local dependencies first.

Open the repo in GitHub Codespaces:

- https://codespaces.new/alsoleg89/ai-knot

The repo already ships `.devcontainer/devcontainer.json`, which installs:

- Python dev deps plus `mcp`, `postgres`, and `server` extras
- the npm workspace dependencies in `npm/`

Wait for the post-create step to finish, then start with one of these proofs.

## Fastest first runs

### Core memory loop

```bash
python examples/quickstart.py
```

What you see:

- direct `add` / `search` / `recall` on a local store
- deterministic retrieval with no LLM call
- the shape of the core Python API in under a minute

### Visual/browser proof

```bash
python examples/browser_inspector_demo.py
```

What you see:

- the HTTP sidecar starts locally
- sample facts are seeded automatically
- you can open `/inspect` from the forwarded port and inspect the memory store in a browser

### TypeScript / Vercel AI SDK proof

```bash
cd npm
npm run example:vercel-ai-sdk-surface
```

What you see:

- the exact `system` / `messages` surface built by `AiKnotAISDKMemory`
- no Python-side MCP process required
- no model call required

### CLI proof

```bash
ai-knot add assistant "User prefers Python and deploys with Docker"
ai-knot search assistant "what stack does the user use?"
ai-knot list assistant
```

What you see:

- the market-standard memory loop from the terminal
- the same CRUD/search language shown in the README

## If something fails

Start with:

```bash
ai-knot doctor --json
```

Then check:

- [troubleshooting.md](troubleshooting.md)
- [integrations.md](integrations.md)
- [usage.md](usage.md)

## Next surfaces after the first proof

- MCP / Claude / OpenClaw: [deployment.md#4-run-the-mcp-server](deployment.md#4-run-the-mcp-server)
- CrewAI: [../examples/crewai_surface_demo.py](../examples/crewai_surface_demo.py)
- PydanticAI: [../examples/pydanticai_surface_demo.py](../examples/pydanticai_surface_demo.py)
- Vercel AI SDK: [../npm/README.md](../npm/README.md)
