# Channel playbook

Updated: **July 1, 2026**

This is the execution layer for launch. Strategy lives in
[launch-plan.md](launch-plan.md); copy lives in [announce.md](announce.md); short
listing blurbs and repo metadata live in [submission-pack.md](submission-pack.md).
Use this file when the repo is ready to post. For the exact maintainer-only
pre-flight steps, see [launch-checklist.md](launch-checklist.md).

## Launch gates

Do not start the main launch until all three are true:

1. Public GitHub `main` matches this branch
2. npm is published at `0.11.0`
3. The README hero includes the short demo asset

## Calendar

Assumption: pre-flight finishes on **Wednesday, July 1, 2026**. If it slips,
move the whole sequence by exactly one week rather than launching half-ready.

| Date | Channel / action | Format | Angle | CTA | Goal |
|---|---|---|---|---|---|
| 2026-07-01 | Pre-flight | merge, publish, add demo asset | eliminate trust-killing friction | none | make public repo launch-safe |
| 2026-07-02 | GitHub release + pinned discussion | release notes + discussion | "reproducible agent memory is here" | star repo, run quickstart | soft-launch to existing GitHub traffic |
| 2026-07-03 | Direct outreach to 10-20 builders | short DM / email | "stress-test install + benchmark claims" | try one surface, send friction back | find breakage before public spike |
| 2026-07-06 | r/LocalLLaMA | text post | self-hosted, offline-friendly, no LLM on recall | repo + quickstart | reach local-first early adopters |
| 2026-07-07 | Docs cleanup day | repo/docs fixes | close soft-launch questions | none | remove friction before main launch |
| 2026-07-14 | Show HN | Show HN post | reproducible benchmark stance | repo + benchmark command | main launch, broad dev attention |
| 2026-07-14 | X thread | 6-post thread | benchmark credibility crisis | repo + thread replies | shareable thesis for technical audience |
| 2026-07-14 | LinkedIn | short post | self-hosted infra + trust angle | repo + article | reach infra/platform buyers |
| 2026-07-14 | Dev.to or Medium | technical article | "stop replaying the transcript" | README quickstart | durable searchable explainer |
| 2026-07-15 | Reply block | comments / issues / DMs | transparent benchmark defense | benchmark docs | convert attention into trust |
| 2026-07-21 | r/MachineLearning | methodology post | reproducible metrics over hype | benchmark docs | attract skeptics and researchers |
| 2026-07-21 | Awesome-list PRs | small PRs | category inclusion | repo link | long-tail discovery |
| 2026-07-28 | Follow-up post | X / GitHub discussion / Reddit comment | surface-specific proof: CrewAI | try the CrewAI demo or example | reactivate observers with concrete integration value |
| 2026-08-01 | Second follow-up post | X / GitHub discussion / Reddit comment | Python framework proof: PydanticAI | run the PydanticAI surface demo or integration example | reach Python agent builders evaluating lighter frameworks |
| 2026-08-04 | Third follow-up post | X / GitHub discussion / Reddit comment | TypeScript app proof: Vercel AI SDK | try the npm example or npm README path | reach TS app builders who will not start from Python |
| 2026-08-08 | Fourth follow-up post | X / GitHub discussion / Reddit comment | app/MCP proof: OpenClaw | run the OpenClaw demo or setup command | widen discovery beyond Python frameworks |
| 2026-08-11 | Fifth follow-up post | X / GitHub discussion / Reddit comment | Claude MCP proof | run the Claude setup demo or setup command | reach tool-first Claude users |

## Channel cards

| Channel | Primary audience | Message angle | Asset to link | Primary CTA |
|---|---|---|---|---|
| GitHub release | existing repo visitors | release credibility + what's new | README + CHANGELOG | run quickstart |
| GitHub discussion | contributors and evaluators | ask for install and benchmark feedback | benchmark docs | report friction |
| r/LocalLLaMA | self-hosted / local-first builders | offline-friendly deterministic memory | README | try SQLite or YAML path |
| Show HN | broad developer audience | reproducible benchmark stance | README + benchmarks | inspect the number, then star |
| X | builders, founders, AI infra crowd | benchmark story + concise proof points | repo | reply or share |
| LinkedIn | platform teams, engineering leaders | lower-risk infra, self-hosted, auditable | whitepaper or article | book attention, then click repo |
| Dev.to / Medium | developers who want depth | technical walkthrough | developer article | copy a snippet |
| r/MachineLearning | benchmark skeptics | methodology and evaluation discipline | benchmarks | discuss the harness |
| Awesome lists / MCP directories | discovery traffic | category placement | repo | passive installs later |

## Surface to emphasize by channel

- GitHub / Show HN: lead with the benchmark stance and deterministic recall.
- r/LocalLLaMA: lead with self-hosted, no-LLM-on-read-path, SQLite/YAML.
- LinkedIn: lead with auditability, storage control, and multi-agent governance.
- Follow-up posts: lead with one concrete surface, not the whole product.
- Best prepared follow-up surface today: `CrewAI` via `docs/crewai-case-study.md`.
- Second prepared follow-up surface: `PydanticAI` via `examples/pydanticai_surface_demo.py`, `examples/pydanticai_integration.py`, and `docs/announce.md`.
- Third prepared follow-up surface: `Vercel AI SDK` via `npm/examples/vercel-ai-sdk.ts` and `npm/README.md`.
- Fourth prepared follow-up surface: `OpenClaw` via `docs/openclaw-case-study.md`.
- Fifth prepared follow-up surface: `Claude MCP` via `docs/claude-mcp-case-study.md`.
- Next-best follow-up surfaces after that: `OpenAI Agents SDK`, then the HTTP sidecar.

## Direct outreach templates

### Builder / maintainer DM

> Built an OSS memory layer for agents called `ai-knot`. The wedge is deterministic,
> self-hosted recall plus benchmark numbers you can actually rerun. I am not asking
> for a signal boost yet; I want one honest pass on install friction or on whether the
> benchmark docs feel convincing. If you have 10 minutes, the fastest path is the
> README quickstart, the CLI loop (`ai-knot add/search/list/delete` or `ai-knot learn`),
> or the CrewAI / PydanticAI / OpenClaw / OpenAI Agents examples here: <repo link>

### Benchmark skeptic DM

> I know the agent-memory benchmark space is noisy, so I tried to make the repo
> falsifiable instead of impressive: named-reader QA numbers plus a deterministic
> retrieval suite with one command. If you are willing to poke holes in the harness,
> that would be more useful than a like. Repo: <repo link>

## Public reply snippets

### "How is this different from Mem0?"

> Different trade-off. Mem0 is broader and more platform-shaped. ai-knot is narrower:
> self-hosted, deterministic on the read path, storage-controlled, and benchmarked in
> a way you can rerun locally.

### "Deterministic sounds weaker than LLM memory."

> It is a trade-off, not magic. The point is that recall becomes cheap, auditable, and
> regression-testable. ai-knot keeps LLMs on the extraction side where they help more.

### "Why should I trust these benchmark claims?"

> Do not trust the claim. Re-run the deterministic suite in the repo, then inspect the
> named reader/judge setup for the QA numbers. The whole pitch is that the repo is
> easier to verify than to hand-wave.

### "What should I try first?"

> Start with the README quickstart or the CLI loop
> (`ai-knot add/search/list/delete`, or `ai-knot learn` for raw text) if you want
> the core product. Start with
> `examples/crewai_surface_demo.py` if you want the shortest CrewAI proof, or
> `examples/crewai_integration.py` if you are on CrewAI and want full wiring. Start with
> `python examples/pydanticai_surface_demo.py` if you want the shortest PydanticAI proof, or
> `examples/pydanticai_integration.py` if you want a real PydanticAI agent wiring path. Start with
> `python examples/openclaw_integration.py` if you want the shortest OpenClaw proof, or
> `ai-knot setup openclaw` if you want the app config path. Start with
> `python examples/claude_mcp_setup.py` if you want the shortest Claude/MCP proof, or
> `ai-knot setup claude` if you want the Claude config path. Start with
> `examples/openai_agents_integration.py` if you are on the OpenAI Agents SDK.

### "What do you want feedback on?"

> Broken install flows, unclear benchmark methodology, and which adapter or backend would
> pull this into your stack fastest.
>
> If you open an issue, use the dedicated templates for install bugs, integration
> requests, or benchmark questions so the report is actionable fast.

## Metrics to watch the first 72 hours

- broken-install issues or comments
- npm version-mismatch complaints
- which surface gets mentioned first: CrewAI, PydanticAI, Vercel AI SDK, OpenClaw, Claude MCP, OpenAI Agents, LangChain
- clicks or replies specifically on the CrewAI follow-up asset
- clicks or replies specifically on the PydanticAI follow-up asset
- clicks or replies specifically on the Vercel AI SDK follow-up asset
- clicks or replies specifically on the OpenClaw follow-up asset
- clicks or replies specifically on the Claude MCP follow-up asset
- benchmark questions vs general "cool project" comments
- stars-to-issue ratio: are people just starring, or actually trying it?
