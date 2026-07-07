# AutoGen case study / proof asset

Updated: **July 2, 2026**

One concrete `ai-knot` integration story for teams already on AutoGen who want persistent memory without rewriting their agent stack.

Official references:

- AutoGen repo: https://github.com/microsoft/autogen
- AutoGen docs: https://microsoft.github.io/autogen/
- AutoGen migration guide: https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/

The official README marks AutoGen as **maintenance mode** and points new users
toward Microsoft Agent Framework.

AutoGen is still a large installed-base channel, but frame it as a migration-friendly / existing-user surface, not the main greenfield framework wedge.

---

## The angle

Not "start a new AutoGen project" — instead:

> **If you already use AutoGen, add deterministic long-term memory without rewriting your assistant.**

That means:

- `AssistantAgent(...)` stays the host object,
- `memory=[AiKnotAutoGenMemory(...)]` stays the integration seam,
- AutoGen keeps short-term context management,
- ai-knot handles durable fact storage and ranked recall underneath.

Not "another memory library" — an existing AutoGen team can add persistence locally before making bigger framework decisions.

---

## Fastest proof paths

### Zero-network proof

Runs without AutoGen installed and without an API key:

```bash
python examples/autogen_surface_demo.py
```

What it proves:

- the `memory=[...]` seam exists now,
- ai-knot injects a real AutoGen-style `SystemMessage` payload,
- recall stays local and deterministic.

### Real AutoGen wiring

```bash
pip install "ai-knot[autogen]"
OPENAI_API_KEY=... python examples/autogen_integration.py
```

What it proves:

- `AiKnotAutoGenMemory` plugs into a real `AssistantAgent`,
- ai-knot appends only the memory relevant to the current turn,
- the integration path works without replacing AutoGen's own runtime flow.

---

## What to emphasize

### Problem

Existing AutoGen users want durable memory without transcript replay on every turn, and without rewriting their stack.

### What ai-knot adds

- persistent facts across sessions,
- deterministic recall with no LLM on the read path,
- a native `memory=[...]` integration shape,
- self-hosted storage (SQLite / PostgreSQL / YAML),
- no hard AutoGen dependency just to import the adapter.

### What not to claim

- Don't recommend AutoGen as the main greenfield framework in 2026.
- Don't hide the maintenance-mode status from the official README.
- Don't pitch this as a replacement for AutoGen's orchestration primitives.

---

## Copy blocks

### GitHub discussion / follow-up comment

> If you're already on AutoGen, `ai-knot` now plugs into the native
> `memory=[...]` seam through `AiKnotAutoGenMemory`, so you can add
> deterministic long-term memory without rewriting the assistant first.
>
> Fastest proof:
> - `python examples/autogen_surface_demo.py` for the zero-network surface proof
> - `examples/autogen_integration.py` for the real wiring path
>
> Note: the official AutoGen README currently marks the framework as maintenance
> mode, so this is best positioned for existing users rather than greenfield builds.

### X / LinkedIn

> Existing AutoGen users: `ai-knot` now fits the native `memory=[...]` seam.
>
> `AiKnotAutoGenMemory` adds deterministic, self-hosted long-term memory under
> `AssistantAgent(...)` without replacing the rest of your stack.
>
> Zero-network proof: `python examples/autogen_surface_demo.py`
> Full wiring: `OPENAI_API_KEY=... python examples/autogen_integration.py`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you're already running AutoGen and just want a memory layer, start with
> `examples/autogen_surface_demo.py`. It shows the exact `memory=[...]` surface
> locally before you wire a real model call. I would frame this for existing
> AutoGen teams, not for new greenfield framework decisions.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/autogen_surface_demo.py`
2. `examples/autogen_integration.py`
3. [integrations.md](integrations.md)

Don't use AutoGen as the lead launch wedge. Use it as an honest installed-base follow-up.
