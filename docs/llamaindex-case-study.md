# LlamaIndex case study / proof asset

Updated: **July 2, 2026**

One concrete `ai-knot` integration story that starts from the LlamaIndex ecosystem rather than MCP or a framework-neutral Python snippet.

Official references:

- LlamaIndex repo: https://github.com/run-llama/llama_index
- LlamaIndex OSS docs: https://developers.llamaindex.ai/python/framework/

LlamaIndex is large enough to treat as a real distribution channel, not as a
nice-to-have adapter. The important adoption fact is not the exact star count,
but that its ecosystem already normalizes `memory=...` seams and add-on
packages.

---

## The angle

Not "another retriever" — instead:

> **Keep the LlamaIndex `memory=...` flow, add deterministic long-term memory.**

That means:

- your `SimpleChatEngine`, `FunctionAgent`, or `ReActAgent` can keep the same
  `memory=...` seam,
- ai-knot handles durable fact storage and query-aware recall,
- the read path stays deterministic and self-hosted instead of turning memory
  into a hosted/vector-first dependency.

This is the important adoption detail: LlamaIndex users already understand the
`memory=...` seam because the ecosystem documents it and already ships a Mem0
integration there. `ai-knot` now meets them in that exact place.

---

## Fastest proof paths

### Zero-network proof

Runs without LlamaIndex and without an API key:

```bash
python examples/llamaindex_surface_demo.py
```

What it proves:

- the `memory=...` shape exists now,
- ai-knot injects a system-style memory block on `get(...)`,
- write/read behavior is inspectable before any real model call.

### Real install path

```bash
pip install "ai-knot[llamaindex]" "llama-index-llms-openai"
```

Then wire:

```python
from ai_knot.integrations.llamaindex import AiKnotLlamaIndexMemory

memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, top_k=5)
chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=memory)
```

For write-time fact extraction instead of raw user messages, set `extract_on_write=True` and give the `KnowledgeBase` a configured provider.

Repo-native real run:

```bash
OPENAI_API_KEY=... python examples/llamaindex_integration.py
```

---

## What to emphasize

### Problem

LlamaIndex users often have good short-term chat memory but still need a durable, inspectable long-term memory layer that doesn't require replaying the whole transcript or adopting another hosted memory product.

### What ai-knot adds

- a native-feeling `memory=...` adapter,
- deterministic recall with no LLM on the read path,
- self-hosted storage (SQLite / PostgreSQL / YAML),
- explicit memory correction workflows (`list` / `delete`) outside the agent loop,
- optional write-time extraction through `kb.learn(...)` when you want higher
  fidelity than raw-message storage.

### What not to claim

- Don't say it replaces LlamaIndex.
- Don't say it is more native to LlamaIndex than the framework's own building blocks.
- Don't hide the trade-off: the default write path is intentionally simple and
  deterministic; richer extraction is opt-in.

---

## Copy blocks

### GitHub discussion / follow-up comment

> A concrete new framework surface for `ai-knot`: LlamaIndex.
>
> The repo now has `AiKnotLlamaIndexMemory`, which fits the familiar
> `memory=...` seam for chat engines and agents.
>
> That means you can keep the LlamaIndex runtime you already use, but switch the
> long-term memory layer to deterministic, self-hosted fact recall.
>
> Fastest proof:
> - `python examples/llamaindex_surface_demo.py`
> - then `OPENAI_API_KEY=... python examples/llamaindex_integration.py`

### X / LinkedIn

> LlamaIndex users: `ai-knot` now plugs into the native `memory=...` seam.
>
> `AiKnotLlamaIndexMemory` keeps short-term chat history in LlamaIndex and adds
> deterministic long-term memory from ai-knot on read.
>
> Shortest proof: `python examples/llamaindex_surface_demo.py`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you're already on LlamaIndex, the new `AiKnotLlamaIndexMemory` adapter is
> the shortest path. It uses the same `memory=...` seam LlamaIndex already
> expects, but the long-term memory itself stays self-hosted and deterministic.
> Start with `python examples/llamaindex_surface_demo.py`.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/llamaindex_surface_demo.py`
2. `pip install "ai-knot[llamaindex]"`
3. [integrations.md](integrations.md)

Don't send LlamaIndex users to the whitepaper first. Send them to the shortest
proof.
