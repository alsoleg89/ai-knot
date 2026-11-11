# Bounded planning over available task evidence

2026-10-28; base `abe8be8`. One longitudinal adoption smoke, repeated across three hosts and two SQL engines. This is a follow-up to the [source-quality diagnosis](2026-10-28-memory-quality-diagnosis.md), not a new memory-accuracy benchmark.

## Gap and change

The existing host keeps `inspect_turn` at `missing_context` when requirements and observations differ lexically. Its optional semantic acquisition returns candidates without authorizing a planning callback. A small task snapshot can already expose all current evidence; what was missing was a complete bounded host path from that evidence to a private, persisted proposal.

The [planning recipe](../bounded-planning.md) adds that path without changing stable memory APIs, source admission, existing inspection, storage schema or dependencies. One explicitly budgeted callback can propose a step or wait. It preserves original requirements, checks current authority/evidence before and after I/O, suppresses immediate repeated input, and leaves domain validation and connector effects to the host. SQL boundary tests cover races and withdrawal separately from this uncontended smoke.

## Frozen method

One constructed task has two shared researcher/connector observations: the designated cluster is Boreal; controller sign-off is initially outstanding and later completed for Boreal. Requirements are “activation destination” and “approval receipt”, so both lexical matches stay missing. After an initial decision, five idle checks have no trigger; the virtual clock advances 61 seconds, the connector updates sign-off and the operator opens an independent SQL facade. A repeated event tests suppression. A guarded scripted effect then loses its response; a separate scripted connector receipt reconciles it before completion. No actual external effect is executed, and this is not a long-running agent process.

Three hosts receive the same exogenous updates, public response contract and application/action policy:

- The unchanged strict lexical host admits a callback only through `turn.wake_model`.
- A simple current-task-bundle host explicitly supplies the same whole task snapshot to its callback, then checkpoints its proposal and latest-input marker. It shares the source/output/domain policy for comparability, but omits the helper's cross-await race recovery; no such race is injected here.
- The new bounded planner uses the public example helper.

Bundle/helper callback payloads, system instruction and output caps are identical and checked byte-semantically per backend. Every arm is offered at most two GPT‑4.1 mini requests, 20,000 total tokens and 512 output tokens per request, with the same 30-second cooperative outer callback timeout. The strict gate consumes zero requests; equal offered caps do not imply equal consumed inference. Calls have no automatic retries. Embeddings are unused because this small corpus fits its model-free snapshot.

Source, driver, prompt, scoring and budgets were frozen before live requests. The raw archive retains the exact driver/runtime, earlier offline driver/report before timestamp-only metadata additions, final offline report, complete live transcripts and metering, failed local checks and successful validation logs. Ledger/report is atomically persisted before schema cleanup; cleanup failures have a separate sanitized field. The same task/prompt was inspected during development, so this is not held-out evaluation.

## Observed result

| SQL | Host | Requests | Tokens | Proposed sequence | Scripted effects | Final state |
|---|---|---:|---:|---|---:|---|
| SQLite | strict lexical | 0 | 0 | gate closed twice | 0 | READY |
| SQLite | current task bundle | 2 | 465 | wait → plan | 1 | COMPLETED |
| SQLite | bounded planner | 2 | 459 | wait → plan | 1 | COMPLETED |
| PostgreSQL | strict lexical | 0 | 0 | gate closed twice | 0 | READY |
| PostgreSQL | current task bundle | 2 | 466 | wait → plan | 1 | COMPLETED |
| PostgreSQL | bounded planner | 2 | 460 | wait → plan | 1 | COMPLETED |

All eight provider requests succeeded: **1,850 actual reported tokens**, zero embedding requests. All four callback-enabled runs made no extra callback for five idle checks, the repeated event or the unknown-effect reconciliation path. Each saved plan cited both current observations and passed the same structured domain gate; completion followed the scripted known receipt. Outputs differ slightly despite identical paired inputs; the few-token difference is not a cost improvement claim.

This shows the new host path can progress through this available-evidence/lexical-mismatch scenario. It **matches the simple task-bundle control**, so the smoke shows no memory-specific accuracy advantage. The strict gate's inactivity identifies an admission/adoption gap, not inferiority at equal consumed model work. Two SQL repetitions are implementation checks, not independent samples. Virtual time, tiny evidence, scripted observations/effects and a previously inspected prompt cannot establish performance, factory-scale coordination, real connector recovery or general answer quality.

The meaningful next evidence is an actual host integration and independently authored accumulating tasks with strong bundle/history controls. Preserve the exclusive lease-controller, selected-evidence action guards and point-in-time snapshot limits described in the guide while doing so.

Machine-readable [results](../performance/bounded-planning-smoke.json), [raw evidence](../performance/bounded-planning-smoke-raw.json.gz), [local validation](../superpowers/validation/2026-10-28-bounded-planning-turn.json) and [independent review](../superpowers/reviews/2026-10-28-bounded-planning-turn.md) accompany this note. The broad product goal remains active.
