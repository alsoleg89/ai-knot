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

A [model-free task snapshot](task-snapshot.md) now exposes whole-corpus scoped loading through Python, HTTP, MCP and TypeScript. It returns all admissible facts or explicit overflow, with no invented search requirement, embedding callback or partial-prefix completeness claim. This is an adoption improvement.

The [continued repository study](team-repository-evaluation.md) now includes real researcher/coder roles, their own carried code over three stages, fresh SQL-backed executor processes and isolated Docker tests. Task memory did not beat current documents plus the same researcher handoff: final contracts passed in two of three repositories versus three of three on both backends. All memory contexts used snapshots, with no embeddings. Identical evidence sets arrived in different orders, and handoffs differed across backend runs, so the small result does not identify a retrieval, ordering or database effect. Original failed runs/costs and the measured runtime are retained; later caller recovery and malformed-receipt repairs were tested without rescoring.

[Transaction-local read reuse](context-read-performance.md) now reduces repeated root/grant/freshness lookups in the four context read paths. A 64-fact/16-root snapshot fell from 3,330 to 99 backing reads while returning identical public results on both SQL backends. Local latency gains vary substantially; namespace scanning and admission CPU remain. This improves a measured operational cost without adding inference or claiming a task-quality gain.

The [quality diagnosis](research/2026-10-28-memory-quality-diagnosis.md) separates the five old advisory misses from actual source loss. It repairs a binding S26 false positive that credited generic notes as implementation coverage. Fresh ranked traces on both SQL backends confirm actual implementation coverage of all requested facets at the default scales; arbitrary exact-peer identity and row surplus remain separate metrics. S9's legacy presence probe is documented as no proof of CAS, and S19's variable-length keyword precision has no universal 0.60 ceiling. Numeric gates, fixtures and earlier paid results are preserved.

The [bounded planning smoke](research/2026-10-28-bounded-planning-smoke.md) now exercises an available-evidence/lexical-mismatch host path through wait, restart, a persisted proposal and scripted effect reconciliation. Both callback-enabled hosts matched on both SQL engines; the new helper did not beat the simple current-task-bundle control. Its measured addition is a tested adoption path with explicit bounds and source/authority checks, not a decision-accuracy claim.

[Task-source decoding](task-source-performance.md) now avoids constructing full Facts for records belonging to unrelated tasks or raw episodes before candidate admission. The same SQL, JSON/order semantics and source checks are retained, with complete cache reuse only when a same-snapshot count proves it. Sparse 10,000-record profiles decode 2 rather than 10,002 Facts on both SQL engines; raw database rows remain unchanged. The [growth research](research/2026-10-28-growing-task-memory.md) keeps this cost improvement separate from growing generic context and safe selection/compaction, which remain product work.

- Measure decision/task benefit for long-lived teams against strong existing memory and no-shared-memory controls, with the same observations and total inference budgets. Synthetic lexical coverage alone is insufficient.
- Control source order and researcher-output variation before interpreting acquisition differences; evaluate accumulating context and existing repository tasks rather than tuning the three retained synthetic cases.
- Use the diagnosed source/metric boundaries when choosing quality changes; inspect ranked-source evidence and query-specific constraints rather than tuning to ambiguous exact-peer labels or shrinking top-k blindly. Keep applicability and provenance intact.
- The [bounded planning-turn recipe](bounded-planning.md) now connects paraphrased current task evidence to one budgeted callback and a private persisted proposal, without promoting lexical coverage or authorizing effects. Continue beyond its small longitudinal adoption smoke: evaluate independently authored tasks, accumulating evidence and actual host integrations before quality or factory-scale claims.
- Test sustained accumulation, retrieval/context preparation cost and useful-update/idle work under realistic workloads. Set deployment-relevant targets before capacity claims.
- Turn demonstrated gaps into small product changes and verify them on both SQL backends. Maintain the source/ACL/revision/action guarantees already implemented.

Completing another batch, passing contract tests or writing an acceptance report is progress. Product readiness still requires evidence of useful memory in real always-on team decisions and a convenient adoption path; release/market outcomes remain separate work.
