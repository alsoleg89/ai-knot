# HTTP sidecar case study / proof asset

Updated: **July 2, 2026**

Use this file when you want one concrete `ai-knot` integration story that
starts from a **polyglot runtime boundary** instead of a Python framework or an
MCP desktop client.

Official references:

- FastAPI repo: https://github.com/fastapi/fastapi
- FastAPI docs: https://fastapi.tiangolo.com/

The point of this surface is not "we also have an API." The point is that a
team can keep its existing runtime, service boundary, or language mix and add
deterministic long-term memory through plain HTTP.

---

## The angle

Do not pitch this as "yet another wrapper around the Python library." Pitch it
as:

> **Keep your runtime, add deterministic long-term memory over plain HTTP.**

That means:

- the same memory loop is available over JSON routes,
- the runtime does not need to spawn the local MCP subprocess,
- Node / TypeScript can talk to the same sidecar through `HttpKnowledgeBase`,
- support and debugging stay easier because `/inspect` exposes a browser view of
  the same store.

The hook is not "infrastructure for infrastructure people." The hook is that a
polyglot team can trial memory quickly without changing how the rest of the app
is deployed.

---

## Fastest proof paths

### Zero-network JSON-loop proof

Runs without a real socket bind and without external services:

```bash
python examples/http_sidecar_surface_demo.py
```

What it proves:

- the HTTP surface is real now, not just documented,
- the familiar JSON loop is concrete:
  `GET /health` → `POST /v1/facts` → `POST /v1/search` → `GET /v1/facts`
  → `GET /v1/facts/{fact_id}` → `DELETE /v1/facts/{fact_id}`,
- the same JSON API can be evaluated before you wire a real client or bind a
  port.

### Browser/debug proof

Runs a real local sidecar and seeds example data:

```bash
python examples/browser_inspector_demo.py
```

What it proves:

- the sidecar starts locally,
- `/inspect` is ready for support and demo flows,
- the same store can be inspected in a browser without building a custom UI.

### Real socket path

```bash
pip install "ai-knot[server]"
ai-knot --storage sqlite serve my_agent --port 8000
```

Then use the same routes on `http://127.0.0.1:8000` or point a runtime at them
through:

```ts
import { HttpKnowledgeBase } from "ai-knot";
```

If you want the repo-native Node path, use:

```bash
cd npm
npm run example:http-sidecar
```

---

## What to emphasize

### Problem

Many teams want persistent memory, but they do not want every runtime to depend
on a local Python subprocess or an MCP-specific setup path.

### What ai-knot adds

- deterministic recalled facts over a plain HTTP boundary,
- self-hosted storage behind SQLite / PostgreSQL / YAML,
- `HttpKnowledgeBase` for Node / TypeScript when local MCP spawn is the wrong
  fit,
- `/inspect` for a read-only browser debugging surface,
- richer write paths on the same sidecar:
  `learn([...])` and `addResolved([...])` / `POST /v1/facts/resolved`.

### What not to claim

- Do not say this replaces the core Python API.
- Do not lead with MCP if the audience is explicitly polyglot/app-service
  oriented.
- Do not overcomplicate the message with every backend detail; lead with the
  JSON loop and debugging surface.

---

## Copy blocks

### GitHub discussion / follow-up comment

> A good non-framework path for `ai-knot` is the HTTP sidecar.
>
> Fastest proof:
> - `python examples/http_sidecar_surface_demo.py` for the zero-network JSON loop
> - `python examples/browser_inspector_demo.py` for the real local sidecar + browser inspector
> - `cd npm && npm run example:http-sidecar` for the Node / TypeScript client path
>
> The point is simple: keep your runtime, call deterministic memory over plain
> HTTP, and use `/inspect` when you need to debug what is actually stored.

### X / LinkedIn

> `ai-knot` is not only a Python/MCP path.
>
> The HTTP sidecar is ready too:
> - `python examples/http_sidecar_surface_demo.py`
> - `python examples/browser_inspector_demo.py`
> - `cd npm && npm run example:http-sidecar`
>
> Same deterministic memory loop, plain HTTP boundary, browser inspector for
> support/debug, and no LLM on the read path.
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you want the polyglot/service boundary instead of the local MCP path,
> start with the HTTP sidecar. There is now a local zero-network proof in
> `python examples/http_sidecar_surface_demo.py`, a browser path in
> `python examples/browser_inspector_demo.py`, and a real Node client example in
> `cd npm && npm run example:http-sidecar`.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/http_sidecar_surface_demo.py`
2. `python examples/browser_inspector_demo.py`
3. `cd npm && npm run example:http-sidecar`
4. [deployment.md](deployment.md)

Do not send polyglot/runtime teams to framework-specific examples first. Send
them to the JSON boundary they can adopt without changing the rest of the stack.
