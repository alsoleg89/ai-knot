# PydanticAI case study / proof asset

Updated: **July 1, 2026**

Use this file when you want one concrete `ai-knot` integration story that starts
from a modern Python agent framework rather than from MCP or a TypeScript app.
PydanticAI is a strong follow-up surface because the framework is already large,
its runtime `instructions=...` seam maps cleanly onto ai-knot's recall model,
and the repo now has both a zero-network proof and a real wiring example.

Official references:

- PydanticAI repo: https://github.com/pydantic/pydantic-ai
- PydanticAI docs: https://ai.pydantic.dev

As of **July 1, 2026**, the official `pydantic/pydantic-ai` repo shows
**18,121 GitHub stars**. That is large enough to treat PydanticAI as a real
top-of-funnel framework surface, not just a nice-to-have adapter.

---

## The angle

Do not pitch this as "another wrapper." Pitch it as:

> **Keep PydanticAI's runtime ergonomics, add deterministic long-term memory.**

That means:

- your `Agent(...)` stays the host object,
- ai-knot appends only the query-relevant memory block per run,
- PydanticAI keeps short-term history while ai-knot handles durable facts.

The hook is not generic "memory." The hook is that the developer does **not**
have to replace PydanticAI or move to a hosted memory layer to get persistence.

---

## Fastest proof paths

### Zero-network proof

Runs without an API key and without `pydantic-ai` installed:

```bash
python examples/pydanticai_surface_demo.py
```

What it proves:

- the adapter shape exists now,
- the runtime `instructions=...` payload is clear and inspectable,
- recall stays local and query-aware.

### Real PydanticAI wiring

```bash
pip install "ai-knot[pydanticai]"
OPENAI_API_KEY=... python examples/pydanticai_integration.py
```

What it proves:

- `AiKnotPydanticAIMemory` plugs into a real PydanticAI run,
- ai-knot appends recalled facts through the framework's native runtime seam,
- the integration path is lightweight and framework-native.

---

## What to emphasize

### Problem

PydanticAI users want a durable memory layer without replaying transcripts into
every run and without replacing the framework they already chose.

### What ai-knot adds

- persistent facts across sessions,
- deterministic recall with no LLM on the read path,
- query-aware memory injection via runtime `instructions=...`,
- self-hosted storage (SQLite / PostgreSQL / YAML),
- no hard dependency on `pydantic-ai` just to import the adapter.

### What not to claim

- Do not say it replaces PydanticAI.
- Do not pitch it as a hosted platform alternative first.
- Do not overcomplicate the story with every Python framework at once.

---

## Copy blocks

### GitHub discussion / follow-up comment

> A concrete Python-framework surface that is ready today: PydanticAI.
>
> `ai-knot` now plugs into the framework's runtime `instructions=...` seam
> through `AiKnotPydanticAIMemory`, so you can keep the host `Agent(...)`
> object and add deterministic long-term memory underneath.
>
> Fastest proof:
> - `python examples/pydanticai_surface_demo.py` for the zero-network surface proof
> - `examples/pydanticai_integration.py` for the real wiring path

### X / LinkedIn

> PydanticAI users: `ai-knot` now plugs into runtime `instructions=...`.
>
> `AiKnotPydanticAIMemory` appends deterministic recalled facts per run, while
> PydanticAI keeps its normal agent flow.
>
> Zero-network proof: `python examples/pydanticai_surface_demo.py`
> Full wiring: `examples/pydanticai_integration.py`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you want one concrete Python-framework path instead of the whole repo,
> start with the PydanticAI surface. The adapter now fits the framework's
> runtime `instructions=...` seam, and there is a zero-network proof in
> `examples/pydanticai_surface_demo.py` before you wire a real model.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/pydanticai_surface_demo.py`
2. `examples/pydanticai_integration.py`
3. [docs/integrations.md](integrations.md)

Do not send people to the whitepaper first. Send them to the shortest proof.
