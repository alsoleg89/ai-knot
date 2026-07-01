# CrewAI case study / proof asset

Updated: **July 1, 2026**

Use this file when someone asks for one concrete `ai-knot` integration instead
of the full product story. Right now the best prepared follow-up surface is
**CrewAI**: the framework has large existing pull, memory is already a native
concept in its docs, and `ai-knot` can now plug into the exact objects CrewAI
users already touch.

Official CrewAI references:

- Repo: https://github.com/crewAIInc/crewAI
- Memory docs: https://docs.crewai.com/concepts/memory

---

## The angle

Do not pitch this as "yet another adapter." Pitch it as:

> **Keep CrewAI's ergonomics, swap in deterministic long-term memory.**

That means:

- `Crew(memory=memory)` stays the integration point,
- `Agent(memory=memory.scope("/agent/researcher"))` stays the agent-level shape,
- ai-knot takes over long-term storage and ranked recall underneath.

The hook is not abstract "memory." The hook is that the developer does **not**
have to replace CrewAI or move to a hosted memory platform to get persistence.

---

## Fastest proof paths

### Zero-network proof

Runs without an API key and without CrewAI installed:

```bash
python examples/crewai_surface_demo.py
```

What it proves:

- the root memory object exists and is usable now,
- scoped views behave like per-agent memory slices,
- recall is deterministic and local.

### Real CrewAI wiring

```bash
pip install "ai-knot[crewai]"
OPENAI_API_KEY=... python examples/crewai_integration.py
```

If you want ai-knot itself to extract memories from raw CrewAI task output via
an LLM-backed provider, use:

```bash
pip install "ai-knot[crewai,openai]"
```

What it proves:

- `AiKnotCrewAIMemory` plugs into `Crew(memory=...)`,
- agent-scoped views work through `memory.scope(...)`,
- the integration path is native-feeling, not bolt-on.

---

## What to emphasize

### Problem

CrewAI users can orchestrate agents well, but long-term memory still needs a
clear persistence story that does not turn into full transcript replay or a
hosted dependency.

### What ai-knot adds

- persistent facts across sessions,
- deterministic recall with no LLM on the read path,
- self-hosted storage (SQLite / PostgreSQL / YAML),
- scoped memory views that map cleanly onto agent roles,
- optional ai-knot extraction when you want CrewAI task output distilled into facts.

### What not to claim

- Do not say it replaces CrewAI.
- Do not say it is better because it is "more autonomous."
- Do not oversell semantic richness; the wedge is reproducibility and storage control.

---

## Copy blocks

### GitHub discussion / follow-up comment

> A concrete surface that is ready today: CrewAI.
>
> `ai-knot` now plugs into `Crew(memory=...)` and agent-scoped
> `memory.scope(...)` views, so you can keep CrewAI's runtime ergonomics and add
> deterministic long-term memory underneath.
>
> Fastest proof:
> - `python examples/crewai_surface_demo.py` for the zero-network memory surface
> - `examples/crewai_integration.py` for the real Crew wiring path

### X / LinkedIn

> CrewAI users: `ai-knot` now plugs into the native memory surface.
>
> `Crew(memory=...)`
> `Agent(memory=memory.scope(...))`
>
> Deterministic long-term memory, self-hosted storage, no LLM on the retrieval path.
> Zero-network proof: `python examples/crewai_surface_demo.py`
> Full wiring: `examples/crewai_integration.py`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you want one concrete thing to try instead of the whole repo, start with the
> CrewAI path. The adapter now fits the native `Crew(memory=...)` / scoped-agent
> shape, and there is a zero-network demo in `examples/crewai_surface_demo.py`
> before you even wire a real model.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/crewai_surface_demo.py`
2. `examples/crewai_integration.py`
3. [docs/integrations.md](integrations.md)

Do not send people to the whitepaper first. Send them to the shortest proof.
