# Shared memory for always-on teams

Product goal: a lightweight shared-memory layer for long-lived teams of agents.

| Stage | Capability | Acceptance |
|---|---|---|
| Reliable return | Catch-up progress, retention recovery, checked saved context | SQLite/PostgreSQL restart, ACL, concurrent-write and bounded-context tests |
| Task-aware team context | Small task/role working sets; missing evidence and recent decisions | Long-lived traces, retrieval precision and total inference cost against ordinary recall |
| Reusable team experience | Grounded procedures and failure lessons with recorded outcomes | Equal-budget evaluation, retraction and independent-evidence checks |
| Operational adoption | Simple schedules/events recipes and diagnostics | Executable examples, recovery and reproducible load reports |

All four stages are implemented and locally accepted in the source feature branch; package publication remains separate. The [completion audit](shared-memory-completion.md) maps the original program to current evidence and limits. Keep ordinary KnowledgeBase use simple and always-on coordination optional. New mechanisms must beat the existing path at equal data and total inference budget before becoming defaults. No graph service or permanent model worker is required.

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
independent review are recorded in the [completion audit](shared-memory-completion.md).
