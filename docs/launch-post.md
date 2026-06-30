# ai-knot: treating agent memory as a knowledge base, not a log

*A draft launch / article piece. Technical, citable, and honest about where the
competition is strong.*

---

## The 400k-token problem

Most agent frameworks treat memory as a log. Every message, every tool call,
every system prompt gets appended and, eventually, replayed. It works until the
conversation is six months old and you are paying to inject 400k tokens into a
request that needs maybe three hundred of them. There is no obvious way to know
*which* three hundred without reading all 400k first — which is the thing you
were trying to avoid.

ai-knot starts from a different premise: a conversation is a stream of evidence,
and what you want to keep is the *distilled, structured knowledge*, not the
transcript. Extract facts, score them, store them in something you can query,
retrieve only what the next turn actually needs.

```
1000 messages (~400k tokens)
    ↓ extraction + verification
~12 facts (~300 tokens)
    ↓ BM25 + intent-weighted fusion
3–5 facts injected into the next prompt
```

## The claim, and how to check it

It is easy to assert that structured retrieval beats a log. It is harder to make
the claim auditable. So the headline number for ai-knot is deliberately the most
boring kind: deterministic, offline, reproducible in one command.

On an in-repo golden retrieval suite — same fixtures, same judge, the only
variable being the retrieval logic — a naive "keep everything, return the most
recent / most lexically similar" store puts the right fact first **10%** of the
time. ai-knot does it **70%** of the time, and never fails to surface the right
fact within the top five. Ranking quality (mean reciprocal rank) climbs from
**0.18 to 0.83**. With 200 distractor facts in the store, signal recall holds at
0.8–1.0 instead of degrading.

```bash
AI_KNOT_EMBED_URL="" python -m tests.eval.benchmark.runner \
  --mock-judge --skip-multi-agent --backends baseline,ai_knot_no_llm
```

No network, no LLM, no random seeds — re-run it and you get the same numbers.
That is the point.

This matters because the usual memory-benchmark numbers are in a credibility
crisis. Zep reported 84% on LoCoMo; an independent re-run put it at 58% once the
excluded adversarial category was removed and the prompts were held fixed. Mem0's
own materials cite 91.6%; independent reproductions land near 58–66%. An LLM-judged
score is a function of the reader model, the judge, the prompts, and the run
count — change any one and the headline swings twenty points.

So ai-knot reports both. On the LLM-judged side, with the reader and judge named:
**78.0% on LoCoMo** (cat1–4, gpt-4.1 reader / gpt-4o judge — adversarial category
excluded, the step Zep got wrong), accuracy holding at 74–84% on every one of the ten
conversations. That is above Mem0's reproducible ~58–66% and Zep's corrected 58%. On
**LongMemEval** it is near-perfect at information extraction (single-session 95–98%)
and declines false-premise questions 90% of the time — the confabulation failure mode
memory systems are supposed to prevent.

And on the side that cannot swing: a deterministic, one-command retrieval suite — `evidence_recall@5`
on LoCoMo of **0.26** vs 0.15 for a naive log, ranking MRR of **0.83** vs 0.18, no LLM,
fixed seeds, identical on every re-run. The QA number tells you how good the answers are
with a given reader; the deterministic number tells you how good the *memory* is, and it
is the one a skeptic can verify in thirty seconds. [benchmarks.md](benchmarks.md) carries
both, with every knob labeled — because in this field, an unlabeled number is how the
credibility crisis started. The reproducibility *is* the marketing.

## The deliberate constraint: no LLM on the hot path

The interesting design decision in ai-knot is what it *refuses* to do. Retrieval,
deduplication, conflict resolution, and bi-temporal supersession are all
deterministic — no model call on the read path. A fact that says "I switched to
Go" closes the earlier "I use Python" fact by slot address and event time, not by
asking a model whether they conflict.

This buys three things that matter in production:

1. **Reproducibility.** The same inputs produce the same outputs. You can write a
   regression test for retrieval behaviour and it will not flake.
2. **Cost and latency.** Recall is in-process BM25, not an embedding round-trip or
   a judge call. The dense channel is optional and degrades gracefully.
3. **Auditability.** When a fact is superseded you can point at *which* fact
   replaced it and *when* — a provenance pointer, not a model's opinion.

The trade-off is real and worth stating plainly: a purely deterministic pipeline
will not resolve every semantic conflict a frontier model would catch. ai-knot's
answer is a seam, not a dependency — an optional `SemanticConflictResolver` you
can wire to any model for the long tail, off by default.

## Where it goes beyond single-agent memory

The part that is hard to find elsewhere is the multi-agent layer. When several
agents share one store, ai-knot treats it as a shared memory pool with:

- **Fan-in recall** — query the pool, get facts from every agent, each tagged
  with a trust score.
- **Evidence-before-belief** — in a governed pool, a claim without a provenance
  pointer is withheld, not published.
- **Per-agent visibility** — keep a fact out of the global pool or share it
  selectively.
- **Laundering-resistant trust** — a known-malicious agent is discounted on wide
  recall, and the discount survives attempts to wash it through volume.

These are not slideware. They are enforced by a scored acceptance gate (scenarios
S8–S26: compare-and-swap correctness, fan-in recall, conflict resolution,
adversarial trust) that runs on every pull request.

## Honest positioning

ai-knot is not the only serious answer here, and a comparison that pretends
otherwise is not worth printing. The honest version:

- **Mem0** is a mature, widely-adopted product with a hosted offering and a large
  community. If you want a managed service and broad framework glue today, that
  is a real advantage ai-knot does not claim.
- **Zep** built a genuine temporal knowledge graph and is strong on relational,
  graph-shaped memory. Its bi-temporal reasoning leans on LLM-based contradiction
  detection — powerful, and a different point on the cost/determinism curve than
  ai-knot's deterministic supersession.
- **LangMem** is tightly integrated with LangGraph and is the path of least
  resistance if that is your stack.

ai-knot's bet is a narrow, defensible one: **deterministic, dependency-light,
self-hosted memory with a real multi-agent governance model** — bi-temporal
supersession, coherence, and trust *without* requiring an LLM in the loop, a
human-readable store you can commit to git, and numbers you can reproduce. If
your constraints are reproducibility, cost control, air-gapped or
regulated deployment, or coordinating a team of agents over shared state, that is
where it earns its place.

## Who it is for

- Teams paying too much to replay conversation history into every prompt.
- Anyone who needs memory behaviour to be *testable* — deterministic recall,
  regression-guarded.
- Multi-agent systems that need shared state with visibility, provenance, and
  trust, not just a common database.
- Deployments where "call a cloud LLM to decide what to remember" is a
  non-starter — cost, latency, privacy, or air-gap.

## Try it

```bash
pip install ai-knot          # Python
npm install ai-knot          # Node / TypeScript (needs Python 3.11+ in PATH)
```

```python
from ai_knot import KnowledgeBase
kb = KnowledgeBase(agent_id="my_agent")
kb.add("User is a senior backend developer who prefers Python")
print(kb.recall("what does the user do?"))
```

MIT-licensed, self-hosted, pluggable storage (SQLite / PostgreSQL / YAML), an MCP
server for Claude Desktop and Claude Code, and an HTTP sidecar for everything
else. The benchmark harness ships in the repo — so does the gate that keeps it
honest.

*Repository: https://github.com/alsoleg89/ai-knot*
