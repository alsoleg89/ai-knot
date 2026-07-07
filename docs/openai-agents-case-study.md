# OpenAI Agents SDK case study / proof asset

Updated: **July 2, 2026**

One concrete `ai-knot` integration story that starts from a modern Python agent runtime rather than MCP or a full app stack. The OpenAI Agents SDK is a strong follow-up surface: the host framework is active, its `RunConfig` seam is explicit, and `ai-knot` adds long-term memory without replacing sessions, tracing, tools, or handoffs.

Official references:

- OpenAI Agents SDK repo: https://github.com/openai/openai-agents-python
- OpenAI Agents SDK docs: https://openai.github.io/openai-agents-python/

The OpenAI Agents SDK is a real framework distribution surface, not just a
niche adapter target. The important part for `ai-knot` is that `RunConfig`
gives a clean insertion seam for recalled memory without taking over the rest of
the runtime.

---

## The angle

Not "another wrapper" — instead:

> **Keep the OpenAI Agents SDK flow, add deterministic long-term memory.**

That means:

- your `Agent(...)` object stays the host object,
- your `Runner.run(...)` / `Runner.run_sync(...)` flow stays in place,
- ai-knot appends only the recalled fact block through `RunConfig`,
- SDK sessions and tracing stay responsible for short-term history and run visibility.

Not generic "memory" — the developer does not have to replace the SDK or move to a hosted memory platform to get durable facts.

---

## Fastest proof paths

### Zero-network proof

Runs without the SDK and without an API key:

```bash
python examples/openai_agents_surface_demo.py
```

What it proves:

- the `RunConfig` seam exists now,
- the injected instructions block is clear and inspectable,
- recall stays local and deterministic before any real model call.

### Real OpenAI Agents SDK wiring

```bash
pip install "ai-knot[agents]"
OPENAI_API_KEY=... python examples/openai_agents_integration.py
```

What it proves:

- `AiKnotAgentsMemory` plugs into a real SDK run,
- ai-knot appends recalled facts through the native `call_model_input_filter` path,
- the integration stays lightweight and framework-native.

---

## What to emphasize

### Problem

OpenAI Agents SDK users often want durable memory without turning sessions into full transcript replay or giving up the SDK's own runtime primitives.

### What ai-knot adds

- persistent facts across sessions,
- deterministic recall with no LLM on the read path,
- memory injection through the native `RunConfig` seam,
- self-hosted storage (SQLite / PostgreSQL / YAML),
- no hard `openai-agents` dependency just to import the adapter.

### What not to claim

- Don't say it replaces the OpenAI Agents SDK.
- Don't frame it as "better than sessions"; it complements sessions.
- Don't lead with MCP or benchmarks when talking to SDK-native users.

---

## Copy blocks

### GitHub discussion / follow-up comment

> A concrete framework surface that is ready today: OpenAI Agents SDK.
>
> `ai-knot` now plugs into the SDK's native `RunConfig` seam through
> `AiKnotAgentsMemory`, so you can keep your `Agent(...)` + `Runner.run(...)`
> flow and add deterministic long-term memory underneath.
>
> Fastest proof:
> - `python examples/openai_agents_surface_demo.py` for the zero-network surface proof
> - `examples/openai_agents_integration.py` for the real wiring path

### X / LinkedIn

> OpenAI Agents SDK path for `ai-knot` is ready.
>
> `AiKnotAgentsMemory` plugs into `RunConfig`, appending deterministic recalled
> facts right before the next model turn while the SDK keeps sessions, tracing,
> tools, and handoffs.
>
> Zero-network proof: `python examples/openai_agents_surface_demo.py`
> Full wiring: `OPENAI_API_KEY=... python examples/openai_agents_integration.py`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you want one concrete Python-framework path instead of the whole repo,
> start with the OpenAI Agents SDK surface. The adapter now fits the native
> `RunConfig` seam, and `examples/openai_agents_surface_demo.py` shows the exact
> instructions payload before you wire a real model call.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/openai_agents_surface_demo.py`
2. `examples/openai_agents_integration.py`
3. [integrations.md](integrations.md)

Don't send SDK users to the whitepaper first. Send them to the shortest proof.
