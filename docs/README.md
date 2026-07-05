# ai-knot documentation

## Core docs

| Guide | What it covers |
|---|---|
| [../README.md](../README.md) | Product overview, quickstart, install-by-surface, integration snippets, and examples. |
| [usage.md](usage.md) | Full API reference: storage, learning, recall, MCP, integrations, multi-agent memory, and bi-temporal recall. |
| [memory-commands.md](memory-commands.md) | Cross-surface command map for `add`, `search`, `list`, `get`, `delete`, `learn`, and structured correction. |
| [integrations.md](integrations.md) | Fast routing by ecosystem surface: CrewAI, LangGraph, LangChain, LlamaIndex, AutoGen, OpenAI Agents SDK, PydanticAI, MCP, HTTP, and TypeScript. |
| [deployment.md](deployment.md) | Install, storage backends, MCP server, HTTP sidecar, browser inspector, observability, security, and scaling. |
| [troubleshooting.md](troubleshooting.md) | First-run, install, MCP, npm bridge, and Pages troubleshooting. |
| [codespaces-quickstart.md](codespaces-quickstart.md) | Install-free first-run flow for GitHub Codespaces. |
| [../examples/README.md](../examples/README.md) | Runnable example index across the core loop, integrations, MCP, HTTP, browser, and TypeScript paths. |
| [../skills/README.md](../skills/README.md) | Assistant-facing skill surface for coding tools that support repo skills. |

## Product framing

| Guide | What it covers |
|---|---|
| [positioning.md](positioning.md) | One-liner, ICP, pain solved, why now, differentiation, and non-goals. |
| [comparison.md](comparison.md) | Buyer-facing "when to choose ai-knot vs Mem0 / Graphiti / Letta / LangMem". |
| [faq.md](faq.md) | Short answers for objections, evaluation questions, and benchmark skepticism. |
| [benchmarks.md](benchmarks.md) | Reproducible benchmark methodology, deterministic retrieval suite, LoCoMo, and LongMemEval. |
| [production-readiness.md](production-readiness.md) | Current readiness across storage, retrieval, integrations, observability, CI, and release. |

## Launch & distribution

| Guide | What it covers |
|---|---|
| [launch-post.md](launch-post.md) | Canonical launch announcement — seeds the GitHub Release, the pinned Discussion, and the Dev.to article. |
| [launch-plan.md](launch-plan.md) | Channel strategy, a dated 4-week calendar, and paste-ready copy for Show HN, r/LocalLLaMA, X, LinkedIn, and Product Hunt. |
| [gtm-readiness.md](gtm-readiness.md) | Launch-readiness audit, prioritized gap register, and remaining work in order. |

## Long-form and Pages

| Guide | What it covers |
|---|---|
| [whitepaper.md](whitepaper.md) | Research-style long-form argument for treating memory as a knowledge layer instead of a log. |
| [developer-article.md](developer-article.md) | Practical developer article for adding deterministic memory to an agent quickly. |
| [site/index.html](site/index.html) | GitHub Pages landing page. |
| [site/whitepaper.html](site/whitepaper.html) | GitHub Pages version of the whitepaper. |
| [site/developer-article.html](site/developer-article.html) | GitHub Pages version of the developer article. |
| [../scripts/render_site_articles.py](../scripts/render_site_articles.py) | Rebuilds the Pages article HTML from the Markdown sources. |
| [../scripts/render_whitepaper_pdf.py](../scripts/render_whitepaper_pdf.py) | Renders the whitepaper Markdown into a PDF artifact when needed. |

## Surface-specific case studies

| Guide | What it covers |
|---|---|
| [crewai-case-study.md](crewai-case-study.md) | CrewAI memory object, scoped agents, and proof flow. |
| [langgraph-case-study.md](langgraph-case-study.md) | LangGraph / LangChain tool-style memory surfaces and when to use them. |
| [llamaindex-case-study.md](llamaindex-case-study.md) | LlamaIndex `memory=...` seam and deterministic long-term memory underneath it. |
| [autogen-case-study.md](autogen-case-study.md) | AutoGen memory adapter path and usage shape. |
| [openai-agents-case-study.md](openai-agents-case-study.md) | OpenAI Agents SDK `RunConfig` seam for recalled memory. |
| [pydanticai-case-study.md](pydanticai-case-study.md) | Runtime `instructions=...` memory injection for PydanticAI. |
| [vercel-ai-sdk-case-study.md](vercel-ai-sdk-case-study.md) | TypeScript / Vercel AI SDK system-prompt memory path. |
| [openclaw-case-study.md](openclaw-case-study.md) | OpenClaw MCP setup and persistent-memory path. |
| [claude-mcp-case-study.md](claude-mcp-case-study.md) | Claude Desktop / Claude Code MCP setup and proof flow. |
| [http-sidecar-case-study.md](http-sidecar-case-study.md) | HTTP-first memory loop, sidecar path, and browser-inspector flow. |

## Maintainer docs

| Guide | What it covers |
|---|---|
| [RELEASE.md](RELEASE.md) | Release workflow, version sync, registry publishing, and pre-release checks. |
| [../server.json](../server.json) | MCP Registry manifest for the PyPI-backed `ai-knot` server. |
