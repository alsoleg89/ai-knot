# ai-knot examples

Updated: **July 2, 2026**

Use this page when you want the shortest path to a runnable proof.

If you already installed `ai-knot` and just want the fastest installed proof,
run `ai-knot demo`.

The examples are grouped by **what you are trying to prove first**:

- the core memory loop,
- a framework or app integration surface,
- MCP / assistant setup,
- browser / notebook / team-memory demos.

## Start here

| If you want to… | Run this first | What it proves |
|---|---|---|
| Prove the installed product in one command | `ai-knot demo` | the product-level `add` / `search` / `list` / `get` / `delete` loop with temporary local storage |
| Prove the core Python memory loop in code | `python examples/quickstart.py` | `add` / `search` / `list` / `delete` with local persistence |
| Prove the generic function-calling memory surface in code | `python examples/function_calling_surface_demo.py` | plain Python `add` / `search` / `list` / `get` / `delete` callables for runtimes that register ordinary functions as tools |
| Prove structured updates and memory history in code | `python examples/structured_correction.py` | `add_resolved` / `MemoryOp.UPDATE` / `MemoryOp.DELETE` / `lineage()` with active-vs-history output |
| Prove the installed npm bridge in one command | `npx ai-knot-demo` | the same built-in proof routed through the packaged Node bridge |
| Prove the core TypeScript memory loop in under 2 minutes | `cd npm && npm run example:basic-memory-loop` | the same `add` / `search` / `list` / `delete` loop from Node |
| Prove the HTTP sidecar memory loop without binding a real port | `python examples/http_sidecar_surface_demo.py` | `/health`, `POST /v1/facts`, `POST /v1/search`, `GET /v1/facts`, `GET /v1/facts/{id}`, and delete via the same JSON surface as `ai-knot serve` |
| Call a running HTTP sidecar from TypeScript | `cd npm && npm run example:http-sidecar` | `HttpKnowledgeBase` over `/v1/*` with no local MCP spawn, plus `learn(...)` / `addResolved(...)` on the same path |
| Triage the TypeScript bridge before debugging app code | `cd npm && npm run doctor` | whether Node, Python, pip, `ai_knot`, and `ai-knot-mcp` are aligned |
| See the repo-native CLI transcript | `python examples/cli_memory_loop.py` | the same `ai-knot add` / `search` / `list` / `delete` loop through the CLI |
| Prove the Claude / OpenClaw MCP setup path | `ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config` then `ai-knot doctor --json` | one-command client config merge plus registration sanity check |
| Try without local setup | open the repo in Codespaces, then follow [`docs/codespaces-quickstart.md`](../docs/codespaces-quickstart.md) | install-free first run |
| See every major surface in one place | [`docs/integrations.md`](../docs/integrations.md) | ecosystem routing by stack |

## Zero-network surface proofs

These examples do **not** require a real model call. They are the fastest proof
paths for launch threads, issue replies, and first-time evaluators.

| Surface | Example | What you see |
|---|---|---|
| CrewAI | [`crewai_surface_demo.py`](crewai_surface_demo.py) | root memory + scoped agent views |
| Function-calling agent | [`function_calling_surface_demo.py`](function_calling_surface_demo.py) | plain Python `add/search/list/get/delete` callables plus a simulated next-turn answer |
| LangGraph | [`langgraph_surface_demo.py`](langgraph_surface_demo.py) | `create_basic_memory_tools(...)` with explicit `add/search/list/delete` tools plus a simulated next-turn answer |
| LlamaIndex | [`llamaindex_surface_demo.py`](llamaindex_surface_demo.py) | `memory=...` seam with an injected system-style memory block plus a simulated next-turn answer |
| OpenAI Agents SDK | [`openai_agents_surface_demo.py`](openai_agents_surface_demo.py) | `RunConfig` seam, injected instructions, and a simulated next-turn answer |
| AutoGen | [`autogen_surface_demo.py`](autogen_surface_demo.py) | `memory=[...]` seam, injected `SystemMessage`, and a simulated next-turn answer |
| PydanticAI | [`pydanticai_surface_demo.py`](pydanticai_surface_demo.py) | runtime `instructions=...` memory injection plus a simulated next-turn answer |
| OpenClaw | [`openclaw_integration.py`](openclaw_integration.py) | MCP config flow plus Python compatibility adapter with `add/search/get_all/delete`, `list/forget` aliases, and structured lineage |
| Claude MCP | [`claude_mcp_setup.py`](claude_mcp_setup.py) | exact MCP config block for Claude |
| HTTP sidecar | [`http_sidecar_surface_demo.py`](http_sidecar_surface_demo.py) | the JSON `add/search/list/get/delete` loop over the same routes exposed by `ai-knot serve` |
| Browser inspector | [`browser_inspector_demo.py`](browser_inspector_demo.py) | seeded `/inspect` flow over a local store |

## Real integration examples

Use these once the zero-network surface looks right and you want actual runtime wiring.

| Surface | Example | Typical install |
|---|---|---|
| CrewAI | [`crewai_integration.py`](crewai_integration.py) | `pip install "ai-knot[crewai]"` |
| LangChain retriever / chat memory | [`langchain_integration.py`](langchain_integration.py) | `pip install "ai-knot[langchain]"` |
| OpenAI Agents SDK | [`openai_agents_integration.py`](openai_agents_integration.py) | `pip install "ai-knot[agents]"` |
| AutoGen | [`autogen_integration.py`](autogen_integration.py) | `pip install "ai-knot[autogen]"` |
| PydanticAI | [`pydanticai_integration.py`](pydanticai_integration.py) | `pip install "ai-knot[pydanticai]"` |
| LlamaIndex | [`llamaindex_integration.py`](llamaindex_integration.py) | `pip install "ai-knot[llamaindex]" "llama-index-llms-openai"` |
| OpenAI extraction path | [`openai_integration.py`](openai_integration.py) | `pip install "ai-knot[openai]"` |

## Core product and team-memory examples

| Example | Best for |
|---|---|
| [`quickstart.py`](quickstart.py) | first proof of the core memory loop |
| [`structured_correction.py`](structured_correction.py) | explicit update/delete memory lifecycle with lineage and active-vs-history output |
| [`shared_pool.py`](shared_pool.py) | multi-agent shared memory with trust / visibility |
| [`coding_agent.py`](coding_agent.py) | project-memory and coding-agent context |
| [`multilingual.py`](multilingual.py) | multilingual fact storage and recall |
| [`hero_demo.py`](hero_demo.py) | the deterministic demo used for launch assets |

## Notebook and browser-first paths

| Example | Best for |
|---|---|
| [`notebook_walkthrough.ipynb`](notebook_walkthrough.ipynb) | rendered walkthrough you can inspect on GitHub |
| [`browser_inspector_demo.py`](browser_inspector_demo.py) | launch the HTTP sidecar with seeded sample data |

## Recommended next step after each proof

- If the core loop looks right, go to [`docs/memory-commands.md`](../docs/memory-commands.md).
- If a surface demo looks right, go to the matching row in [`docs/integrations.md`](../docs/integrations.md).
- If you want benchmark credibility next, go to [`docs/benchmarks.md`](../docs/benchmarks.md).
- If you want public launch copy for that surface, go to the matching case-study doc in [`docs/README.md`](../docs/README.md).
