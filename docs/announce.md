# Launch / announcement copy (ready to post)

Real numbers, copy-paste-ready. Pick the channel; the maintainer posts under their
own accounts. Positioning lives in [positioning.md](positioning.md), long-form
assets in [whitepaper.md](whitepaper.md) and [developer-article.md](developer-article.md),
and the sequence in [launch-plan.md](launch-plan.md).

---

## Show HN

**Title:**
> Show HN: ai-knot – deterministic agent memory with reproducible benchmarks

**Body:**
> ai-knot is a self-hosted memory layer for AI agents. It distills conversations
> into structured facts, retrieves only what the next turn needs, and forgets the
> rest — with **no LLM on the retrieval path** (BM25 + rank fusion + optional dense),
> so recall is deterministic, cheap, and auditable.
>
> I built it partly because LoCoMo/LongMemEval leaderboard numbers are a mess —
> Zep's 84% on LoCoMo became 58% on an independent re-run; Mem0's cited 91.6%
> reproduces at ~58–66%. So ai-knot reports two kinds of number: the LLM-judged QA
> accuracy *with the reader and judge named* (78.0% on LoCoMo cat1–4 with a gpt-4.1
> reader; 74–84% on every one of the 10 conversations), and a deterministic,
> one-command retrieval number that can't drift (ranking MRR 0.83 vs 0.18 for a
> naive log; LoCoMo evidence_recall@5 0.26 vs 0.15).
>
> It also has a real multi-agent layer: a shared pool with fan-in recall,
> evidence-before-belief publishing, per-agent visibility, and laundering-resistant
> trust — all deterministic, with an optional LLM seam for the semantic-conflict
> tail. Bi-temporal: every fact knows when it was learned and when its event
> happened, so `recall(now=…)` answers "what was true as of X".
>
> If you prefer a terminal proof over SDK snippets, the repo now exposes a short
> memory loop: `ai-knot add`, `ai-knot search`, `ai-knot list`, `ai-knot delete`,
> plus `ai-knot learn` when you want raw text distilled into facts.
>
> SQLite / Postgres / YAML behind one API, CrewAI + AutoGen + OpenAI Agents SDK
> + PydanticAI adapters, LangChain / LangGraph adapters, an MCP server for Claude
> Desktop/Code, and a TypeScript client plus Vercel AI SDK adapter for Node apps.
> MIT-licensed.
>
> Repo + reproducible benchmarks: https://github.com/alsoleg89/ai-knot

---

## GitHub release / discussion / pinned post

**Short release copy:**
> `ai-knot v0.11.0` turns agent memory into a reproducible, self-hosted knowledge layer:
> deterministic recall, no LLM on the retrieval path, SQLite/Postgres/YAML backends,
> bi-temporal `recall(now=...)`, an MCP server for Claude, CrewAI + AutoGen +
> OpenAI Agents SDK + PydanticAI adapters, a TypeScript client plus Vercel AI SDK adapter,
> and a multi-agent shared pool
> with trust, provenance, and visibility controls.
>
> The repo now ships both named-reader QA benchmarks and a deterministic retrieval suite
> you can rerun locally. Start with the 30-second quickstart in the README, or with
> the CLI loop: `ai-knot add`, `ai-knot search`, `ai-knot list`, `ai-knot delete`.
> If you want extraction from raw notes instead of manual facts, use
> `ai-knot learn` with `AI_KNOT_PROVIDER` / `OPENAI_API_KEY`.

**Pinned discussion opener:**
> If you are evaluating memory for agents, start with the README quickstart and the
> deterministic benchmark command in `docs/benchmarks.md`. Feedback that helps most:
> broken install flows, framework adapters you want next, and benchmark questions you
> think the docs still do not answer cleanly. If you open a GitHub issue, use the
> install / integration / benchmark templates so the report is actionable.

---

## Surface-specific follow-up: CrewAI

Use this as the first post-launch proof asset when someone asks for one concrete
integration instead of the full product tour.

**Title options:**

- CrewAI users: ai-knot adds deterministic long-term memory without changing `Crew(memory=...)`
- CrewAI memory, but self-hosted and deterministic on the read path

**Short post:**
> If you are already on CrewAI, `ai-knot` now plugs into the native memory surface:
> pass `AiKnotCrewAIMemory` into `Crew(memory=...)` and scoped views into
> `Agent(memory=memory.scope(...))`.
>
> The point is not "more context." The point is durable facts with deterministic
> recall: self-hosted storage, no LLM on the retrieval path, and per-agent scopes
> that map cleanly onto CrewAI roles.
>
> Fastest proof:
> - zero-network adapter demo: `python examples/crewai_surface_demo.py`
> - full Crew wiring example: `examples/crewai_integration.py`
>
> Repo: https://github.com/alsoleg89/ai-knot

**X / LinkedIn version:**
> CrewAI users: `ai-knot` now plugs into `Crew(memory=...)` and
> `Agent(memory=memory.scope(...))`.
>
> Deterministic long-term memory, self-hosted storage, no LLM on the read path.
> Zero-network proof in `examples/crewai_surface_demo.py`; full wiring in
> `examples/crewai_integration.py`.
>
> https://github.com/alsoleg89/ai-knot

**Reply snippet:**
> If you want the shortest possible CrewAI proof, start with
> `python examples/crewai_surface_demo.py`. It shows the exact memory object and
> scoped agent view without needing an API key. Then switch to
> `examples/crewai_integration.py` for a real Crew run.

---

## Surface-specific follow-up: PydanticAI

Use this when you want a Python-framework proof that feels close to the host
agent runtime, but lighter-weight than a full app or multi-agent story.

**Title options:**

- PydanticAI users: ai-knot now adds long-term memory through runtime `instructions=...`
- ai-knot adds deterministic, self-hosted memory to PydanticAI agents

**Short post:**
> If you're already using PydanticAI, `ai-knot` now plugs into the framework's
> runtime `instructions=...` surface through `AiKnotPydanticAIMemory`.
>
> The point is durable facts with query-aware recall: ai-knot appends only the
> relevant long-term memory block for the current prompt, while PydanticAI keeps
> its own short-term conversation flow.
>
> Fastest proof:
> - zero-network adapter demo: `python examples/pydanticai_surface_demo.py`
> - full wiring example: `examples/pydanticai_integration.py`
>
> Repo: https://github.com/alsoleg89/ai-knot

**X / LinkedIn version:**
> PydanticAI path for `ai-knot` is ready:
>
> `pip install "ai-knot[pydanticai]"`
>
> Use `AiKnotPydanticAIMemory` to append deterministic recalled facts through
> runtime `instructions=...`.
>
> Shortest proof: `python examples/pydanticai_surface_demo.py`
>
> https://github.com/alsoleg89/ai-knot

**Reply snippet:**
> If you want the shortest PydanticAI proof, start with
> `python examples/pydanticai_surface_demo.py`. It shows the exact runtime
> `instructions=...` payload ai-knot builds, with no API key or model call.

---

## Surface-specific follow-up: Vercel AI SDK

Use this when you want to lead with the mainstream TypeScript app path instead
of a Python framework or MCP client.

**Title options:**

- Vercel AI SDK users: ai-knot now fills your `system` prompt with recalled long-term memory
- ai-knot adds deterministic, self-hosted memory to Vercel AI SDK apps

**Short post:**
> If your app already runs on the Vercel AI SDK, `ai-knot` now has a native-feeling
> route for long-term memory:
>
> `npm install ai-knot ai @ai-sdk/openai`
>
> Then use `AiKnotAISDKMemory` to build the `system` string or `messages` array
> from ai-knot recall, while keeping model choice, streaming, and UI wiring in
> your existing AI SDK code.
>
> Fastest proof:
> - zero-network surface proof: `cd npm && npm run example:vercel-ai-sdk-surface`
> - repo-native example: `cd npm && OPENAI_API_KEY=... npm run example:vercel-ai-sdk`
> - npm package docs: `npm/README.md`
>
> Repo: https://github.com/alsoleg89/ai-knot

**X / LinkedIn version:**
> Vercel AI SDK path for `ai-knot` is ready:
>
> `npm install ai-knot ai @ai-sdk/openai`
>
> Use `AiKnotAISDKMemory` to build your `system` / `messages` surface from
> deterministic recalled facts, with no LLM on the retrieval path.
>
> Shortest proof: `cd npm && npm run example:vercel-ai-sdk-surface`
>
> https://github.com/alsoleg89/ai-knot

**Reply snippet:**
> If you want the shortest TypeScript proof, start with
> `cd npm && npm run example:vercel-ai-sdk-surface`. Then switch to
> `cd npm && OPENAI_API_KEY=... npm run example:vercel-ai-sdk` for the real
> `generateText()` path without hiding the model call behind framework magic.

---

## Surface-specific follow-up: OpenClaw

Use this when you want to lead with the app/MCP route instead of a Python
framework adapter.

**Title options:**

- OpenClaw users: add persistent memory through one MCP config block
- ai-knot gives OpenClaw deterministic, self-hosted memory over MCP

**Short post:**
> If you are using OpenClaw, the shortest path is now one command:
>
> `ai-knot setup openclaw --agent-id my_agent --storage sqlite`
>
> Paste the printed config into `~/.openclaw/openclaw.json` and you get
> persistent, self-hosted memory over MCP with no LLM on the retrieval path.
>
> There is also a zero-network proof in `examples/openclaw_integration.py` that
> shows both the MCP config flow and the Python-side compatibility adapter.
>
> Repo: https://github.com/alsoleg89/ai-knot

**X / LinkedIn version:**
> OpenClaw path for `ai-knot` is ready:
>
> `ai-knot setup openclaw --agent-id my_agent --storage sqlite`
>
> Paste the config into `~/.openclaw/openclaw.json` and you get self-hosted,
> deterministic memory over MCP. Zero-network proof:
> `examples/openclaw_integration.py`
>
> https://github.com/alsoleg89/ai-knot

**Reply snippet:**
> If you want the shortest OpenClaw proof, run
> `python examples/openclaw_integration.py`. It prints the MCP config you paste
> into OpenClaw and shows the Python adapter path without needing the app or an
> API key.

---

## Surface-specific follow-up: Claude MCP

Use this when you want to lead with the Claude Desktop / Claude Code tool path
instead of a framework adapter or app-specific route.

**Title options:**

- Claude Desktop users: give Claude persistent memory through one MCP server config
- ai-knot adds deterministic, self-hosted memory to Claude over MCP

**Short post:**
> If you are already using Claude Desktop or Claude Code, `ai-knot` now has a
> very short MCP setup path:
>
> `ai-knot setup claude --agent-id my_agent --storage sqlite`
>
> Paste the printed JSON into Claude's MCP config and Claude can call
> deterministic, self-hosted memory tools over stdio.
>
> There is also a zero-network proof in `examples/claude_mcp_setup.py` if you
> want to see the exact config block before wiring anything.
>
> Repo: https://github.com/alsoleg89/ai-knot

**X / LinkedIn version:**
> Claude Desktop / Claude Code path for `ai-knot` is ready:
>
> `ai-knot setup claude --agent-id my_agent --storage sqlite`
>
> Paste the config into Claude's MCP config and you get self-hosted,
> deterministic memory over MCP. Shortest proof:
> `python examples/claude_mcp_setup.py`
>
> https://github.com/alsoleg89/ai-knot

**Reply snippet:**
> If you want the shortest Claude proof, run
> `python examples/claude_mcp_setup.py`. It prints the exact MCP config block
> you paste into Claude Desktop / Claude Code, with no API key or local app run required.

---

## X / LinkedIn thread

1/ We re-ran the agent-memory benchmarks everyone quotes. The numbers don't
reproduce: Zep's 84% on LoCoMo → 58% on an independent re-run; Mem0's 91.6% →
~58–66%. An LLM-judged score swings 20 points with the reader, judge, and prompts.

2/ So we built ai-knot to report two numbers. One is the LLM-judged QA accuracy —
but *with the reader and judge named*: 78.0% on LoCoMo (cat1–4, gpt-4.1 reader),
holding 74–84% across all 10 conversations. Above Mem0's reproducible ~58–66%.

3/ The other number can't move: a deterministic, one-command retrieval suite — no
LLM, fixed seeds. Ranking MRR 0.83 vs 0.18 for a naive log; LoCoMo evidence_recall@5
0.26 vs 0.15. Re-run it, get the same number. That's the point.

4/ Under the hood: no LLM on the retrieval path. BM25 + rank fusion + optional dense.
Deterministic dedup, bi-temporal supersession, power-law forgetting. Cheap, auditable,
testable.

5/ For teams of agents: a shared memory pool with fan-in recall, evidence-gated
publishing, per-agent visibility, and laundering-resistant trust. Deterministic by
default; optional LLM seam for the semantic tail.

6/ Self-hosted, MIT, SQLite/Postgres/YAML, MCP server for Claude, CrewAI,
AutoGen, OpenAI Agents SDK, PydanticAI, TypeScript client + Vercel AI SDK adapter,
LangChain / LangGraph adapters.
The benchmark harness ships in the repo — so does the gate that keeps it honest.
→ https://github.com/alsoleg89/ai-knot

---

## Reddit: r/LocalLLaMA

**Title:**
> ai-knot: self-hosted agent memory with deterministic recall and reproducible benchmarks

**Body:**
> Built an OSS memory layer for agents that stores structured facts instead of replaying
> full chat logs into every prompt. It is self-hosted, retrieval does not call an LLM,
> and the default path is dependency-light: BM25 + rank fusion, SQLite/Postgres/YAML.
>
> The launch angle is reproducibility. The repo includes a deterministic retrieval suite
> plus named-reader LoCoMo / LongMemEval results, because current memory-benchmark claims
> vary wildly by harness and judge model.
>
> If you care about local-first or air-gapped agent stacks, the interesting surfaces are:
> MCP server for Claude, CrewAI, AutoGen, OpenAI Agents SDK, PydanticAI, TypeScript client,
> Vercel AI SDK adapter,
> HTTP sidecar + browser inspector, and a shared multi-agent pool with trust /
> provenance / visibility controls.
>
> Repo: https://github.com/alsoleg89/ai-knot

---

## Reddit: r/MachineLearning

**Title:**
> ai-knot: a reproducible benchmark stance for agent memory, plus a deterministic memory layer

**Body:**
> Sharing this less as "here is a new library" and more as "here is a benchmark position."
> Agent-memory leaderboards are currently hard to compare because the reader model, judge,
> prompts, and category filters move the headline number a lot.
>
> For ai-knot, we therefore ship two numbers:
> 1. named-reader QA accuracy on LoCoMo / LongMemEval;
> 2. a deterministic retrieval suite with no LLM in the loop.
>
> The system itself is a self-hosted memory layer for agents that keeps facts instead of
> transcripts and retrieves only what the next turn needs. Details and methodology are in
> `docs/benchmarks.md`.
>
> Repo: https://github.com/alsoleg89/ai-knot

---

## Dev.to / Medium

**Title options:**
- Stop replaying the whole transcript: deterministic memory for AI agents
- Agents do not need the whole chat log. They need the right 3 facts.
- Benchmark claims for agent memory are drifting. Build with the memory you can re-run.

**Standfirst:**
> Most agent stacks still treat memory as a log. ai-knot treats it as a knowledge base:
> extract facts, store them in a self-hosted backend, and retrieve only the few that matter
> for the next turn. The interesting part is not just the architecture but the benchmark
> stance: deterministic numbers first, named-reader QA numbers second.

---

## One-liner / tagline options

- Deterministic agent memory — with benchmark numbers you can actually reproduce.
- Agent memory without an LLM in the loop, and a multi-agent governance model with trust.
- The memory layer that reports the number a skeptic can re-run in 30 seconds.

---

## Where to list

- Show HN (Hacker News)
- r/LocalLLaMA, r/MachineLearning
- Awesome-Memory-for-Agents (TsinghuaC3I), NirDiamant/Agent_Memory_Techniques
- MCP server directories / awesome-mcp (it ships an MCP server)
- Dev.to / Medium (the long-form docs and developer article)

For the ready-to-paste short blurbs, repo description, and topic list, use
[submission-pack.md](submission-pack.md).
