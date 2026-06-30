# Launch / announcement copy (ready to post)

Real numbers, copy-paste-ready. Pick the channel; the maintainer posts under their
own accounts. Long-form rationale lives in [launch-post.md](launch-post.md); the
numbers are in [benchmarks.md](benchmarks.md).

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
> SQLite / Postgres / YAML behind one API, six LLM providers for extraction, an MCP
> server for Claude Desktop/Code, MIT-licensed. `pip install ai-knot`.
>
> Repo + reproducible benchmarks: https://github.com/alsoleg89/ai-knot

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

6/ Self-hosted, MIT, SQLite/Postgres/YAML, MCP server for Claude. The benchmark
harness ships in the repo — so does the gate that keeps it honest.
→ https://github.com/alsoleg89/ai-knot

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
- dev.to / Medium (the long-form [launch-post.md](launch-post.md))
