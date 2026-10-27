# Shared memory for always-on teams

Product goal: a lightweight shared-memory layer for long-lived teams of agents.

| Stage | Capability | Acceptance |
|---|---|---|
| Reliable return | Catch-up progress, retention recovery, checked saved context | SQLite/PostgreSQL restart, ACL, concurrent-write and bounded-context tests |
| Task-aware team context | Small task/role working sets; missing evidence and recent decisions | Long-lived traces, retrieval precision and total inference cost against ordinary recall |
| Reusable team experience | Grounded procedures and failure lessons with recorded outcomes | Equal-budget evaluation, retraction and independent-evidence checks |
| Operational adoption | Simple schedules/events recipes and diagnostics | Executable examples, recovery and reproducible load reports |

This is a development program, not a claim that all stages have shipped. Prioritize reproduced gaps and task benefit; keep ordinary KnowledgeBase use simple and always-on coordination optional. New mechanisms must beat the existing path at equal data and total inference budget before becoming defaults. No graph service or permanent model worker is required.

Track useful knowledge retained, invalid knowledge rejected, missed updates, redundant work, unnecessary model wakes, model calls/tokens per useful update, context bytes and p50/p95 latency separately. Green deterministic tests do not establish real-model answer quality.

See [primary-source research](research/2026-10-27-shared-memory-direction.md).
