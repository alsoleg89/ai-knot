# Continued team work: measured outcomes

Date: 2026-11-11. Three synthetic repositories, three successive contracts each,
six memory policies. This is an exploratory case series with disclosed protocol
repairs and exact-prefix continuations, not an untouched preregistered trial or
a production/general memory-accuracy claim.

## Results

All **54 planned outcomes** are accounted for: **39 stages** ran in fresh executor
processes; **15 consumer stages** failed because the common researcher summary was
oversized. PostgreSQL replay matched the complete outcome/state ledger with
**zero new provider requests**.

| Context policy | Passed stages / 9 | All stages passed / 3 cases | Final stage passed / 3 | Allocated tokens + reservations |
| --- | ---: | ---: | ---: | ---: |
| Current source documents | 6 | 2 | 2 | 43,099 |
| Current documents + common handoff | 6 | 2 | 2 | 59,044 |
| Raw observed team history | 3 | 0 | 0 | 51,725 |
| Lexical selection | 5 | 1 | 2 | 38,495 |
| Semantic selection | 4 | 0 | 2 | 53,923 |
| Semantic + all handoff-cited roots required | 0 | 0 | 0 | 12,903 |

Each consuming standalone arm is allocated the entire common formation cost.
The table duplicates shared costs; it is not an account bill. Lower cost after
refusing context or formation is not an efficiency win.

The first two cases had valid common formation. Across their six stage contracts,
results were respectively **6, 6, 3, 5, 4, 0**. The third summary was 1284 characters
against the unchanged 1200-character cap. Its five consumers received explicit
formation failures; their coders did not run. The direct control did run with its
own failed implementations and receipts, but passed none of that case's contracts.
This distinguishes shared formation failure from coder failure with complete sources.

The cited-root policy refused all six stages with usable formation because the
mandatory bundle exceeded the frozen context limits. Its remaining three outcomes
were formation failures. This is correct fail-closed behavior for `required_refs`,
and a poor retention policy under these budgets.

**Product decision:** do not promote semantic selection or automatic retention of
every summary citation to the default on this evidence. Keep a fitting current
[task snapshot](task-snapshot.md) as a simple explicit host option; reserve
[required references](required-context.md) for a small known obligation set.
Raw history does not resolve current evidence. A grounded summary still needs a
valid output contract and a host fallback policy. Three authored cases establish
no universal winner.

## Method and safety checks

The [foundation corpus](team-work-foundation.md) is unchanged at SHA256
`0b11a369992ee24052b69c014a8dcc4bf8e1e4bb2cd6b4a7ec40df5d997d782e`.
Observers apply chronological updates/withdrawal without clearing executor basis
or code. The common researcher sees only public current observations before coder
arms. Every executor stage reopens SQL state, carries its own failed code/receipts,
and tests candidates only in the existing pinned Docker sandbox. No reference
implementation rescues a live arm.

Models: `gpt-4.1-mini` (returned alias `gpt-4.1-mini-2025-04-14`) and
`text-embedding-3-small`, temperature 0. Per case/arm: 12 requests/60000 tokens,
including formation, embeddings, failures and repairs. Coder output: 8192 tokens,
at most two proposals/tests per stage. Formation: three requests/18000 tokens per
case, 768 output tokens/1200 summary characters. A missing coder HTTP response may
retry once inside those budgets; coder waiting is bounded at 90 seconds.

Full documents/history have a 24000-character public JSON ceiling; bounded policies
have eight facts/2000 whole-content characters, with JSON overhead reported
separately. Mean admitted content: direct 2775 characters, documents + handoff 3411,
history 4618, lexical 1941, semantic 1914. Equal offered total ceilings do not mean
equal consumed compute or equal context ceilings.

The primary contrast was cited roots versus ordinary semantic selection; full
current documents + the same handoff was the strong reference. Roots come from
public researcher citations, not hidden obligations. Rendering is chronological
for every arm. Captured task guards fence checkpoint/action admission.

All 39 source sets stayed unchanged during worker execution. **26 consecutive
worker boundaries** preserved prior code and basis before ordinary resume. Every
passing receipt matched its authored test count. PostgreSQL matched context/guard/
code hashes, refs, resume/state transitions, allocations and receipts, excluding
timing and opaque identifiers. This is backend parity, not another model sample.
The same-interpreter grader is not adversarial attestation.

## Failures and expenditure

The [prospective protocol and amendments](superpowers/plans/2026-10-29-team-cycle-execution.md)
preserve all intermediate failures. Version 1 stopped on a 3072-token unfinished
answer; its old meter did not expose truncated text. Output authority was then
raised equally. Version 2 lost an HTTP response after about 32 seconds. Version 3
reused its exact prefix and obtained a recorded length response repeating one
comment **443 times**. Length responses became invalid proposals within the two
attempts; arm budget failures became isolated outcomes. Version 4 reached the
oversized third summary. Version 5 reused that response, failed its consumers and
executed only the remaining independent control. No summary was clipped or
regenerated, and completed answers were not resampled after version 2.

The final continued trace contains **59 logical requests**: 53 completions/six
embedding calls, **189874 reported tokens plus a 17703-token conservative reserve**
for the missing response. Across the whole exercise, including superseded version
1: **87 new API attempts, 255126 reported tokens and the same 17703-token reserve**.
The overlapping continuation ledgers must not be added together. PostgreSQL sent
zero new requests; its logical allocations reproduce the original charges.

Unhandled provider/replay/grader failure still stops globally. Unknown worker
billing reserves its remaining authority. Actual reported overspend cannot be
disguised as a zero-charge arm budget refusal. Model failures remain in the planned
denominator. Inspecting the partial answer diagnosed generation degeneration;
task scores were not used to choose protocol repairs. The fixed order, partial
blinding, shared formation and amended pilot prohibit significance/generalization
claims.

## Local validation and reproduction

- Python 3.12, PostgreSQL required: **2870 passed, 9 skipped; 90.13% coverage**.
- Ruff, format and strict source mypy pass. TypeScript: **63 passed**, typecheck/build pass.
- Recovery acceptance: **12/12 SQLite**, **12/12 PostgreSQL**.
- Prefix-only checks used fresh processes/Docker and a forbidden live factory:
  recorded requests were consumed exactly before refusing the first missing
  inference, with zero API calls. Input/schema/limit drift has no paid fallback.
- [Ponytail review](superpowers/reviews/2026-11-11-team-cycle-ponytail.md) was performed
  by the implementing agent; no independent model review is claimed. **No Astra.**

The [validation record](superpowers/validation/2026-11-11-team-cycle.json) and
[raw evidence](performance/team-cycle-execution-raw.json.gz) retain reports,
outputs/vectors, five exact runtime versions, freezes, partial attempts, scripts
and failed local logs. Initial local PostgreSQL setup hit a full Docker disk and
a failed small temporary instance; final gates used an owned 2 GiB temporary
database with bounded WAL. These are infrastructure failures, not task scores.

From this exact checkout, with `PYTHONPATH=src`, run
`python -m tests.eval.team_work.experiment --mode offline --backend sqlite --output PATH --work DIRECTORY`.
Create the output parent first; existing reports are refused. Offline mode uses
reference code solely as a wiring oracle. The pinned Docker image must be cached.

For paid-free reproduction, extract `reports["v5-live-sqlite.json"]` from the archive
and run the same module with
`--mode replay --backend postgres --replay RECORDING --output NEW_REPORT --work NEW_DIRECTORY`.
Set `AI_KNOT_TEST_PG_DSN` to a disposable database; the runner owns a temporary
schema. Exact runtime hashes must match. Replay uses no credential/live fallback.
Live mode uses an explicit macOS Keychain service for this local research tool;
the production library does not acquire that requirement.

No production API, schema, dependency, version or release-preparation file changed
in this batch. Original release-checkout status/hashes are audited separately.
The broader product goal remains active; production repositories, long-horizon
capacity and concurrent factories remain unverified by this pilot.
