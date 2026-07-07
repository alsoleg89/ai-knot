# Basic memory commands

Updated: **July 2, 2026**

If you only need the base `ai-knot` loop, start here.

If `ai-knot` is already installed and you want the shortest product-level proof,
run `ai-knot demo`. It executes the same loop below against temporary local
storage and cleans up afterward unless you pass `--keep-data`.

## Default loop

`ai-knot` keeps one market-standard loop visible across surfaces:

`add -> search -> list -> delete`

It also keeps the agent-memory aliases:

- `learn` when you want `ai-knot` to extract facts from raw text with an LLM
- `recall` as an alias for `search`
- `show` as an alias for `list`
- `forget` as an alias for `delete`
- `clear` only when you want to wipe the whole agent namespace

This is deliberate. The strongest comparable projects do not all use the same
words, but they all surface a small first-run verb set early:

- [Mem0](https://github.com/mem0ai/mem0) leads with `init`, `add`, and `search`
- [Graphiti](https://github.com/getzep/graphiti) frames the loop as `add episodes`
  plus `search` over nodes and relationships
- [LangMem](https://github.com/langchain-ai/langmem) exposes
  `create_manage_memory_tool(...)` and `create_search_memory_tool(...)`
- [Letta](https://github.com/letta-ai/letta) treats memory as `memory_blocks` and
  agent/message flows rather than CRUD verbs

`ai-knot` keeps the common `add/search/list/delete` shape so a new user can try
the product without learning repo-specific vocabulary first.

Notice what many memory READMEs do not foreground: they often stop at
`add/search`, `retrieve`, or `manage/search`. That is enough to prove recall,
but not enough to debug or correct a persistent memory store. `ai-knot` keeps
`list` and `delete` visible early because local-first memory needs an
inspection/correction loop on day one, not only a write/read loop.

## How other systems expose the loop

As of **July 1, 2026**, the official quickstarts lead with these first-run
verbs:

| Project | What the official quickstart / README surfaces first | What that means for `ai-knot` |
|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | agent-signup quickstart literally goes `mem0 init --agent ...` → `mem0 add` → `mem0 search` | keep the first CLI proof equally literal instead of hiding it under framework vocabulary |
| [Graphiti](https://github.com/getzep/graphiti) | quickstart proves `add episodes` plus `search relationships` / `search nodes`; MCP docs expose episode `add`, `retrieve`, `delete` | keep CRUD-style verbs easy to discover even when the underlying model is richer than plain facts |
| [LangMem](https://github.com/langchain-ai/langmem) | the first agent block wires `create_manage_memory_tool(...)` and `create_search_memory_tool(...)` | keep a familiar search verb even when the adapter surface is agent-native |
| [Letta](https://github.com/letta-ai/letta) | onboarding splits `letta` CLI from API creation with `memory_blocks` on the agent | `ai-knot` should stay simpler than a full runtime and expose explicit memory verbs directly |
| [OpenClaw](https://github.com/openclaw/openclaw) | app path leads with `openclaw onboard --install-daemon`, `openclaw gateway status`, `openclaw doctor` | keep `setup` and `doctor` obvious for the app/MCP handoff, while the memory verbs stay `add/search/list/delete` inside the client |

## Cross-surface command map

| Surface | Add | Search | List | Delete |
|---|---|---|---|---|
| Core Python | `kb.add(...)` | `kb.search(...)` or `kb.recall(...)` | `kb.list()` or `kb.list_facts()` | `kb.delete(id)` or `kb.forget(id)` |
| TypeScript / npm | `await kb.add(...)` | `await kb.search(...)` or `await kb.recall(...)` | `await kb.list()` or `await kb.listFacts()` | `await kb.delete(id)` or `await kb.forget(id)` |
| CLI | `ai-knot add ...` | `ai-knot search ...` or `ai-knot recall ...` | `ai-knot list ...` or `ai-knot show ...` | `ai-knot delete ...` or `ai-knot forget ...` |
| MCP | `add` | `search` or `recall` | `list` or `list_facts` | `delete` or `forget` |
| HTTP sidecar | `POST /v1/facts` | `POST /v1/search` | `GET /v1/facts` | `DELETE /v1/facts/{fact_id}` |

For the user-facing inspect path, `list` now means **current active memory** on
CLI, MCP, npm, and HTTP. Ask for history explicitly when you need it:

- CLI: `ai-knot list assistant --include-inactive`
- TypeScript / npm: `await kb.list({ includeInactive: true })`
- MCP: `list_facts(include_inactive=true)`
- HTTP sidecar: `GET /v1/facts?include_inactive=true`

## Structured correction when memory changes

Most official memory quickstarts still prove only the write/read loop. Mem0's
current README still leads with `init -> add -> search`, and its April 2026
algorithm note describes an add-only extraction path with no explicit
`UPDATE`/`DELETE` lifecycle. LangMem exposes `manage` / `search` tools and lets
the agent decide when to store or look up memory. Graphiti exposes
`add` / `retrieve` / `delete`, but at the episode/context-graph layer. `ai-knot`
keeps the beginner loop simple and also exposes a direct correction seam when a
stored claim itself changes.

| Surface | Structured correction |
|---|---|
| Core Python | `kb.add_resolved([Fact(..., op=MemoryOp.UPDATE)])` |
| TypeScript / npm | `await kb.addResolved([{ ..., op: "update" }])` |
| CLI | `ai-knot add-resolved ... --op update` |
| MCP | `add_resolved` with `facts[].op` |
| HTTP sidecar | `POST /v1/facts/resolved` with `facts[].op` |

Use `op="update"` when a slot changes and you want a new version with lineage.
Use `op="delete"` when the memory should be closed without inserting a
replacement. Use `op="noop"` when a conversation merely confirms what is
already known and you want to skip mutation entirely.

```python
from ai_knot import Fact, MemoryOp

kb.add_resolved([
    Fact(content="User works at Acme", entity="user", attribute="employer", value_text="Acme")
])
current = kb.add_resolved([
    Fact(
        content="User now works at Globex",
        entity="user",
        attribute="employer",
        value_text="Globex",
        op=MemoryOp.UPDATE,
    )
])[0]
kb.add_resolved([
    Fact(
        content="User no longer works at Globex",
        entity="user",
        attribute="employer",
        slot_key="user::employer",
        op=MemoryOp.DELETE,
    )
])
print(kb.lineage(current.id))
```

```bash
ai-knot add-resolved assistant "User now works at Globex" \
  --entity user --attribute employer --value-text Globex --op update
ai-knot add-resolved assistant "User no longer works at Globex" \
  --entity user --attribute employer --slot-key user::employer --op delete
ai-knot lineage assistant <fact_id>
```

Runnable repo-native proof: [`examples/structured_correction.py`](../examples/structured_correction.py)

## When you already have a fact ID

The default loop stays `add -> search -> list -> delete`. But once `list`,
`recall_json`, or another surface has already given you a `fact_id`, you often
want a tighter inspect-before-delete step. `ai-knot` now keeps that lookup
explicit too:

| Surface | Targeted inspection |
|---|---|
| Core Python | `kb.get(fact_id)` |
| TypeScript / npm | `await kb.get(factId)` |
| CLI | `ai-knot get assistant <fact_id>` |
| MCP | `get` |
| HTTP sidecar | `GET /v1/facts/{fact_id}` |
| OpenClaw Python adapter | `memory.get(fact_id)` |

Use `get(...)` when you want to inspect one stored fact, check whether it is
still active, or confirm metadata before you delete or supersede it.

When the fact has versions and you want the supersession audit trail instead of
one record, use:

| Surface | Supersession audit trail |
|---|---|
| Core Python | `kb.lineage(fact_id)` |
| TypeScript / npm | `await kb.lineage(factId)` |
| CLI | `ai-knot lineage assistant <fact_id>` |
| MCP | `memory_lineage` |
| HTTP sidecar | `GET /v1/facts/{fact_id}/lineage` |
| OpenClaw Python adapter | `memory.lineage(fact_id)` |

For LangGraph / LangChain agent-tool flows, `ai-knot` now also exposes the same
loop as explicit tool builders:

- `create_basic_memory_tools(kb)` for the full `add/search/list/delete` set
- `create_basic_memory_functions(kb, include_get=True)` when your runtime wants
  plain Python callables instead of LangChain tool objects
- `create_add_memory_tool(kb)`, `create_search_memory_tool(kb)`,
  `create_list_memory_tool(kb)`, `create_get_memory_tool(kb)`,
  `create_delete_memory_tool(kb)` when you want to wire only selected verbs
- `create_basic_memory_tools(kb, include_get=True)` when you want the same
  first-run loop plus targeted by-id inspection once the agent already has a
  `fact_id`
- `create_manage_memory_tool(kb)` + `create_search_memory_tool(kb)` when you
  want the more compact LangMem-style shape, with `manage_memory(action="get")`
  available for targeted inspection

If you're arriving through provider-compat surfaces, keep one small translation
in mind:

| Compatibility surface | Add | Search | List | Delete |
|---|---|---|---|---|
| OpenClaw Python adapter | `memory.add(...)` | `memory.search(...)` or `memory.recall(...)` | `memory.get_all()` or `memory.list()` | `memory.delete(id)` or `memory.forget(id)` |

That keeps the external provider shape intact without forcing a different
mental model once you move back to the core `ai-knot` loop.

On the TypeScript side, those same verbs now work on both transports:

- `KnowledgeBase` when Node spawns the local `ai-knot-mcp` subprocess
- `HttpKnowledgeBase` when your app talks to a running `ai-knot serve` sidecar

## Copy-paste examples

### Python

```python
from ai_knot import KnowledgeBase

kb = KnowledgeBase(agent_id="assistant")

fact = kb.add("User deploys APIs with Docker and Kubernetes")
print(kb.search("what does the user deploy with?"))
print(kb.list())
kb.delete(fact.id)
```

If you already have raw conversation text instead of a pre-extracted fact, use
`learn(...)` instead of `add(...)`.

### OpenClaw Python adapter

```python
from ai_knot import KnowledgeBase
from ai_knot.integrations.openclaw import OpenClawMemoryAdapter

memory = OpenClawMemoryAdapter(KnowledgeBase("assistant"))
created = memory.add([{"role": "user", "content": "User deploys APIs with Docker Compose"}])
print(memory.search("what does the user deploy with?"))
print(memory.get_all())   # provider-compatible active listing
memory.forget(created["results"][0]["id"])   # alias: memory.delete(...)
```

Use `memory.get_all(include_inactive=True)` when you want superseded history in
the OpenClaw-compatible adapter instead of only the current active memories.

### TypeScript / npm

```typescript
import { KnowledgeBase } from "ai-knot";

const kb = new KnowledgeBase({ agentId: "assistant", storage: "sqlite" });

const fact = await kb.add("User prefers TypeScript for frontend work");
console.log(await kb.search("what language should I use on the frontend?"));
console.log(await kb.list());
console.log(await kb.list({ includeInactive: true })); // optional audit history
console.log(await kb.get(fact.id));
await kb.delete(fact.id);
```

### CLI

```bash
ai-knot add assistant "User deploys APIs with Docker and Kubernetes"
ai-knot search assistant "what does the user deploy with?"
ai-knot list assistant
ai-knot delete assistant <fact_id>
```

Fastest installed proof of that exact loop:

```bash
ai-knot demo
```

Repo-native proof: [`examples/cli_memory_loop.py`](../examples/cli_memory_loop.py)

If you want `ai-knot` to extract facts from a raw note or message:

```bash
ai-knot learn assistant "I deploy APIs with Docker and Kubernetes"
```

### MCP

Once `ai-knot-mcp` is running, MCP clients see the same loop:

- `add`
- `search` or `recall`
- `list` or `list_facts`
- `get`
- `delete` or `forget`

That keeps Claude Desktop, Claude Code, OpenClaw, and other MCP clients aligned
with the same verbs shown in the README and CLI.

### HTTP sidecar

```bash
curl -X POST http://127.0.0.1:8000/v1/facts \
  -H 'Content-Type: application/json' \
  -d '{"content":"User deploys APIs with Docker and Kubernetes","importance":0.9}'

curl -X POST http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"what does the user deploy with?","top_k":5}'

curl http://127.0.0.1:8000/v1/facts
curl http://127.0.0.1:8000/v1/facts/<fact_id>
curl http://127.0.0.1:8000/v1/facts/<fact_id>/lineage
curl -X DELETE http://127.0.0.1:8000/v1/facts/<fact_id>
```

Repo-native proof of the same surface without binding a real port:
[`examples/http_sidecar_surface_demo.py`](../examples/http_sidecar_surface_demo.py)

## Which verb to use when

- Use `add` when you already know the fact you want to store.
- Use `learn` when you have raw conversation text and want `ai-knot` to extract
  facts for you.
- Use `search` when you want the familiar retrieval verb used by memory/RAG tools.
- Use `recall` when you want your code to read like "retrieve next-turn memory."
- Use `list` when you want to inspect active facts quickly.
- Use `get` when you already have a `fact_id` and want to inspect one fact precisely.
- Use `show` if that reads more naturally in your terminal habits.
- Use `delete` when you want CRUD-style language.
- Use `forget` when you want the agent-memory framing.
- Use `add-resolved --op update|delete|noop` when a stored value itself changes.
- Use `clear` only for full namespace resets, not single-fact removal.

## OpenClaw and Claude paths

OpenClaw itself leads with app/setup verbs like `openclaw onboard`,
`openclaw gateway status`, and `openclaw doctor`. `ai-knot` mirrors that for
the MCP handoff:

```bash
ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config
ai-knot setup claude --agent-id assistant --storage sqlite --write-default-config
ai-knot doctor --json
```

On supported platforms, `ai-knot` can resolve the client config path for you:

```bash
ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config
ai-knot setup claude --agent-id assistant --storage sqlite --write-default-config
```

If you need a non-default path, use `--write-config <path>`.

That is why `ai-knot` keeps two tiny verb sets:

- operator/setup verbs: `setup`, `doctor`
- memory verbs: `add`, `search`, `list`, `delete`

After the MCP config is in place, the memory verbs inside the client remain the
same `add/search/list/delete` loop described above.

## Related docs

- [README.md](../README.md)
- [usage.md](usage.md)
- [integrations.md](integrations.md)
- [codespaces-quickstart.md](codespaces-quickstart.md)
