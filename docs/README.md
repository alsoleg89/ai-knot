# ai-knot documentation

## Product docs

| Guide | What it covers |
|---|---|
| [usage.md](usage.md) | Full API reference: storage, learning, recall, MCP, integrations, multi-agent pool, bi-temporal recall, examples. |
| [../examples/notebook_walkthrough.ipynb](../examples/notebook_walkthrough.ipynb) | Rendered zero-network notebook walkthrough of the core `add` → `recall` loop, temporal recall, and next-step commands. |
| [site/index.html](site/index.html) | GitHub Pages-ready landing page for shareable launch links, outreach, and non-GitHub-first audiences. |
| [integrations.md](integrations.md) | Fast routing by ecosystem surface: CrewAI, AutoGen, OpenAI Agents SDK, OpenClaw, LangChain/LangGraph, MCP, TS, HTTP, plus the right install path for each. |
| [../skills/README.md](../skills/README.md) | Assistant-facing skill surface for coding tools that support the skills standard. |
| [deployment.md](deployment.md) | Install, storage backends, configuration (env vars), the MCP server, HTTP sidecar, browser inspector, observability, backup, security, scaling. |
| [troubleshooting.md](troubleshooting.md) | First-run, install, MCP, and public-release troubleshooting with `ai-knot doctor` and the public launch checker. |
| [benchmarks.md](benchmarks.md) | Reproducible benchmark methodology, deterministic retrieval suite, LoCoMo and LongMemEval results. |
| [competitor-bench-pack.md](competitor-bench-pack.md) | How to generate a publish-ready side-by-side benchmark scorecard against baseline, vector-store, and Mem0-style controls. |
| [production-readiness.md](production-readiness.md) | Feature-by-feature readiness status across storage, retrieval, governance, observability, CI, release, integrations, benchmarks. |
| [demo-script.md](demo-script.md) | Script and framing for the short terminal demo/GIF used in launch materials. |
| [launch-checklist.md](launch-checklist.md) | Maintainer-only preflight checklist: merge/publish/release/demo gates before public launch. |

## Positioning and launch

| Guide | What it covers |
|---|---|
| [positioning.md](positioning.md) | One-liner, ICP, pain solved, why now, differentiation, message map, non-goals. |
| [comparison.md](comparison.md) | Buyer-facing "when to choose ai-knot vs Mem0 / Graphiti / Letta / LangMem" guide. |
| [competitive-analysis.md](competitive-analysis.md) | OSS landscape review for Mem0, Graphiti/Zep, Letta, and LangMem, with adoption takeaways for ai-knot. |
| [readme-patterns.md](readme-patterns.md) | Narrow teardown of how leading memory projects structure README onboarding and expose integration surfaces. |
| [gap-analysis.md](gap-analysis.md) | Prioritized product and GTM gaps, what this repo update closes, and what remains maintainer-only. |
| [launch-post.md](launch-post.md) | Narrative launch post: market thesis, benchmark stance, and honest comparison framing. |
| [whitepaper.md](whitepaper.md) | Research-style long-form launch piece for GitHub Pages, a PDF, Substack, or a personal site. |
| [developer-article.md](developer-article.md) | Technical article for a developer audience: how to add deterministic memory to an agent quickly. |
| [crewai-case-study.md](crewai-case-study.md) | Surface-specific proof asset for the CrewAI channel: demo flow, post copy, and reply snippets. |
| [openclaw-case-study.md](openclaw-case-study.md) | Surface-specific proof asset for the OpenClaw / MCP app channel: config flow, demo path, and post copy. |
| [claude-mcp-case-study.md](claude-mcp-case-study.md) | Surface-specific proof asset for Claude Desktop / Claude Code via MCP: setup path, proof flow, and post copy. |
| [faq.md](faq.md) | FAQ and objections handling for launch threads, sales calls, and docs. |
| [announce.md](announce.md) | Channel-specific copy: Show HN, X/Twitter, LinkedIn, Reddit, GitHub release/discussion, Dev.to/Medium. |
| [launch-plan.md](launch-plan.md) | Channel strategy and the 4-week soft-launch → main-launch sequence. |
| [channel-playbook.md](channel-playbook.md) | Dated launch calendar, channel cards, outreach templates, and public reply snippets. |
| [submission-pack.md](submission-pack.md) | Ready-to-paste blurbs for awesome-lists, MCP directories, GitHub topics, and repo metadata. |
| [publish-ready-audit.md](publish-ready-audit.md) | Evidence-based audit of what is done, what is verified, and what still needs maintainer access. |

## Repository docs

See also, at the repository root: [ARCHITECTURE.md](../ARCHITECTURE.md),
[DECISIONS.md](../DECISIONS.md), [DEVELOPMENT.md](../DEVELOPMENT.md),
[CONTRIBUTING.md](../CONTRIBUTING.md), [CHANGELOG.md](../CHANGELOG.md),
and [docs/RELEASE.md](RELEASE.md).
