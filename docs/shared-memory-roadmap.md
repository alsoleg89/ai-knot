# Shared memory for always-on teams

Product goal: a lightweight shared-memory layer for long-lived teams of agents.

| Stage | Capability | Acceptance |
|---|---|---|
| Reliable return | Catch-up progress, retention recovery, checked saved context | SQLite/PostgreSQL restart, ACL, concurrent-write and bounded-context tests |
| Task-aware team context | Small task/role working sets; missing evidence and recent decisions | Long-lived traces, retrieval precision and total inference cost against ordinary recall |
| Reusable team experience | Grounded procedures and failure lessons with recorded outcomes | Equal-budget evaluation, retraction and independent-evidence checks |
| Operational adoption | Simple schedules/events recipes and diagnostics | Executable examples, recovery and reproducible load reports |

The product goal remains active. These four engineering stages are implemented and locally accepted; they establish mechanisms, not completion of the product goal. The [engineering acceptance record](shared-memory-completion.md) maps that first implementation batch to its evidence and limits. Keep ordinary KnowledgeBase use simple and always-on coordination optional. New mechanisms must beat the existing path at equal data and total inference budget before becoming defaults. No graph service or permanent model worker is required.

Track useful knowledge retained, invalid knowledge rejected, missed updates, redundant work, unnecessary model wakes, model calls/tokens per useful update, context bytes and p50/p95 latency separately. Green deterministic tests do not establish real-model answer quality.

See [primary-source research](research/2026-10-27-shared-memory-direction.md).

Current source progress: [reliable return](shared-memory-return.md) is committed,
and [explicit task working sets](task-context.md) add bounded requirement coverage,
applicability labels and soft role preference. The latter matches the strongest
existing scoped retrieval baseline on frozen evidence recall; role preference and
structured missing-evidence reports are its measured additions.
[Outcome-bound experience](team-experience.md) now binds explicitly recorded
lessons to action revisions and source observations, with idempotent writes and
automatic retraction. Host-authored procedural memory does not establish
automatic learning or verified external success. The [operational host recipe](always-on-team-cycle.md)
connects return/context/guarded work/experience. Final whole-program acceptance and
independent review of that implementation batch are recorded in the [engineering acceptance record](shared-memory-completion.md).

## Continuing product work

The [decision study](team-decision-evaluation.md) now records real GPT-4.1 mini actions, simulated receipts and all OpenAI embedding/reader tokens on both SQL backends. It found a same-ID vector-cache defect (fixed) and a remaining semantic selection gap, with a strong full-history control and explicit post-pilot protocol limits.

An [opt-in semantic acquisition path](semantic-context.md) now checks source/ACL/task/lineage/freshness before and after budgeted embeddings, with a zero-call complete-task snapshot and quiet event-driven host recipe. The [first independently authored case batch](semantic-context-evaluation.md) recovered more required evidence than lexical task search but did not beat a cheap scoped task bundle on reader correctness. It stays optional. The legacy MA verification now fails closed on missing/invalid selected evidence; raw reader-format failures and all paid work remain in the evaluation artifacts.

- Measure decision/task benefit for long-lived teams against strong existing memory and no-shared-memory controls, with the same observations and total inference budgets. Synthetic lexical coverage alone is insufficient.
- Diagnose the five retained advisory retrieval misses before choosing fixes; distinguish bad retrieval from ambiguous/incorrect labels. Keep explicit applicability and provenance intact.
- Test sustained accumulation, retrieval/context preparation cost and useful-update/idle work under realistic workloads. Set deployment-relevant targets before capacity claims.
- Turn demonstrated gaps into small product changes and verify them on both SQL backends. Maintain the source/ACL/revision/action guarantees already implemented.

Completing another batch, passing contract tests or writing an acceptance report is progress. Product readiness still requires evidence of useful memory in real always-on team decisions and a convenient adoption path; release/market outcomes remain separate work.
