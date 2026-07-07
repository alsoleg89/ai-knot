# Codespaces quickstart

Try `ai-knot` without installing Python, Node, or local dependencies first.

Open the repo in GitHub Codespaces:

- https://codespaces.new/alsoleg89/ai-knot

The repo already ships `.devcontainer/devcontainer.json`, which installs:

- Python dev deps plus `mcp`, `postgres`, and `server` extras
- the npm workspace dependencies in `npm/`

Wait for the post-create step to finish, then run one of these proofs.

## Fastest first runs

### Core memory loop

```bash
ai-knot demo
```

What you see:

- the installed product-level `add` / `search` / `list` / `get` / `delete` loop
- temporary local storage, so you can prove the behavior without cleanup
- the shortest path from "repo opened" to "memory works"

For the raw Python API:

```bash
python examples/quickstart.py
```

What you see:

- direct `add` / `search` / `recall` on a local store
- deterministic retrieval with no LLM call
- the shape of the core Python API in under a minute

### HTTP JSON proof

```bash
python examples/http_sidecar_surface_demo.py
```

What you see:

- the HTTP sidecar JSON routes exercised without binding a real port
- `/health`, `POST /v1/facts`, `POST /v1/search`, `GET /v1/facts`, `GET /v1/facts/{fact_id}`, and delete
- the same surface `ai-knot serve` exposes to polyglot runtimes

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
npx ai-knot-demo
cd npm
npm run example:basic-memory-loop
```

What you see:

- the packaged npm bridge can route into the built-in proof command
- the repo-native Node example shows the same `add` / `search` / `list` / `delete` loop from TypeScript

For the Vercel AI SDK surface specifically:

```bash
cd npm
npm run example:vercel-ai-sdk-surface
```

What you see:

- the exact `system` / `messages` surface built by `AiKnotAISDKMemory`
- no Python-side MCP process required
- no model call required

If the real npm client path fails, run:

```bash
cd npm
npm run doctor
```

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
