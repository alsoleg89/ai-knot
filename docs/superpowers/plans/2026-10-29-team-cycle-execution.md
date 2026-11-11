# Continued team coding execution plan and prospective protocol

> Native execution, existing worktree; autonomous work authorized. Ponytail source review before paid execution, performed by the implementing agent; no independent model review is claimed. No Astra. Original release checkout protected. Foundation corpus v3 is immutable at SHA256 `0b11a369992ee24052b69c014a8dcc4bf8e1e4bb2cd6b4a7ec40df5d997d782e`.

**Goal:** Execute the prepared three-case/three-stage corpus with a common researcher, fresh durable-state coder workers, equal offered model budgets and exact PostgreSQL replay. This is an exploratory paired synthetic case series, not product-goal completion.

## Frozen policy before paid work

- One paid SQLite run per protocol version, no quality-target rerun. Version 1 was interrupted by output truncation; the technical amendment below was fixed before version 2. Three cases × six arms × three stages; exactly one common researcher output per case/stage formed before any coder arm. No coder feedback/hidden tests/reference code in formation inputs. Preserve every failure, charge and planned denominator. All case bytes and public/private repairs remain unchanged.
- Arms in fixed order: `direct_sources`, `current_docs_handoff`, `raw_team_history`, `lexical`, `semantic`, `cited_sources`. Primary comparison: `cited_sources` vs `semantic`; full current docs + same handoff is the strong reference. Secondary lexical/history/direct comparisons remain separate. Fixed order and tiny sample prohibit causal/significance/general-reliability claims.
- All arms receive the same observations and public goal/requirements/brief. Direct sources do not consume the common handoff and avoid formation cost. Each consuming standalone arm is charged the complete common formation calls/tokens; actual account spend counts shared formation once.
- Same model `gpt-4.1-mini`, temperature0 via existing MeteredOpenAI; embedding `text-embedding-3-small`. Per case/arm ceiling12 requests/60000 tokens inclusive of formation/embeddings/repairs. Researcher output768 tokens; coder output8192 in version 2 (3072 in version 1); at most two proposals/tests per stage. Shared formation ceiling3 calls/18000 tokens per case, summary <=1200 characters by explicit prompt/local validation. Malformed formation is retained and stops later paid work.
- Full reference/history public JSON context ceiling24000 characters, no truncation. Bounded arms use production count8/whole-fact-content2000-character limits; public JSON envelope/metadata is reported separately and still fits global24000. All prompts/code/feedback/schema count toward token budgets. Equal ceilings are not equal consumed compute.
- Cited-source policy pins exactly the current source refs cited by the common summary, not hidden obligations or the summary itself. Summary remains an ordinary optional candidate. It is a public host retention policy, not proof those citations discover all prerequisites. An oversized mandatory set refuses explicitly, without clipping or repair. No post-result bounds tuning.
- All rendered entries use the same deterministic chronological observation order with namespaced-ref ties, preserving historical version chronology and equal model-visible order when sets match. No arm name/hidden case fields or refs appear in provider prompts. Own carried code/feedback may diverge after treatment.
- Source publisher applies chronological deltas and preserves executor intent/basis/code. Ordinary resume may persist BLOCKED after staleness. No caller clears stale basis merely to run acquisition. A valid newly reviewed proposal and captured task guard replace basis at checkpoint, then a fresh lease/action admission rechecks guard before sandbox execution. Invalid/wait/refused acquisition leaves saved code/basis intact.
- Each coder stage runs in a fresh Python worker. Candidate code executes only in existing resource-limited Docker sandbox. Only bounded test receipts enter feedback. Keep each arm's own failed code and receipts; no reference rescue. Primary outcome: all three stage contracts passed /3 cases; final-stage and nine stage outcomes/attempts/costs separate.
- Provider/budget failure, unknown worker billing or infrastructure loss stops further paid work globally; remaining planned pairs/stages are explicit not-run and summaries incomplete with quality interpretation suppressed. Local malformed proposals and ordinary assertion failures may use only the second prespecified stage attempt.
- All provider inputs/raw outputs/vectors and conservative pre-I/O reservations are durable. No credential values in jobs/argv/env/reports/containers. Workers read explicit Keychain service only in live mode. All SQL transactions closed before provider I/O. Replay has no key/provider and must consume exact ordered messages/schema/output limits/texts; drift fails closed. Logical original charges control remaining replay budget; actual replay provider spend0.
- PostgreSQL replay is backend parity, not a second quality sample. Compare code/candidate/context hashes, refs/guards, source-update/resume transitions and receipt counts/status (exclude timing/opaque IDs). No live fallback.

## Implementation tasks

- [x] Durable RecordingClient/ReplayClient with fake-provider RED→GREEN tests, exact matching and cancellation/charge/refusal checks.
- [x] Stage runner with real SQL-backed transitions, mandatory citation retention, contextual ordering and guarded proposals/actions; test invalid/wait/no-context preservation, reopen and late source updates.
- [x] Common formation and fresh-worker coordinator with complete scheduled ledger, bounded Keychain-only live path, reference-stub offline wiring and exact replay.
- [x] Ponytail source/method review; full local PostgreSQL/static gates and offline plumbing proof. Fix observed defects before paid calls without fixture edits. No independent model review was run.
- [x] Freeze exact runtime/protocol; execute the amended continued ledger, retain all costs/failures, and complete PostgreSQL exact replay. Document negative results, limitations and protected-release audit; prepare the local commit. Broad goal remains active.

## Technical amendment before version 2, 2026-11-11

Version 1 stopped globally at case 2 / direct sources / stage 0. OpenAI returned
HTTP 200 with 3072 completion tokens and an unfinished response. There were 28
provider calls and 65252 charged tokens, 18 completed stages, one failed stage and
35 unrun stages. Its aggregate quality stays null. No task scores or candidate
code were inspected to choose this amendment; diagnosis used only request status,
HTTP status, token usage and the error type. This is exploratory protocol repair,
not an untouched confirmatory preregistration.

The raw truncated text was unavailable because the old metered client rejected
the finish reason before exposing text. Preserve that limitation, the original
partial run, hashes, exact version-1 runtime and this original plan. Version 2
records completion text separately before response validation and raises only
the common coder output authority to 8192. Default meter authority remains128.
Models, prompts, data, selection, context limits, case/arm total budgets, stage
attempt limits and stopping rules are unchanged. Revalidate locally, freeze again,
then run version 2 once from fresh storage. Report both versions' total expenditure.
Do not combine their quality outcomes or retry version 2 to obtain a higher score.

## Transport continuation amendment before version 3, 2026-11-11

Version 2 stopped after 22 completed stages, at case 2 / current docs + handoff /
stage 1. The coder request had no HTTP response or usable usage. Retain the full
17703-token conservative reservation, all 32 attempted calls and89411 accounted
tokens; quality stays null. Do not pay to regenerate its completed answers.

Version 3 is an exact-prefix continuation, not a fresh quality rerun. All recorded
version-2 requests, including its failed transport attempt, must match ordered
messages, schemas, output limits and embedding texts before any new transport is
opened. Mismatch never falls back to live inference. Original charges reduce the
remaining 60000-token/12-request authority. One coder completion with no HTTP response
may retry once with identical inputs, still inside those budgets. HTTP errors,
truncation, pending/cancelled/unknown-worker records, embedding failures and budget
refusals do not qualify. Formation retains its original3-call/no-retry ceiling.

Only failed or previously unrun requests use live transport. Successful source
responses are not resampled. Retain both interrupted attempts and the continuation
lineage; do not double-count reused version-2 requests as new account spend. An
offline prefix-only wiring proof must reach exactly the missing-response boundary
with zero provider calls before continuation. Revalidate locally, freeze once more,
then continue and replay the resulting complete trace on PostgreSQL. Record/replay
must deterministically consume the failed request and its reservation too.

This is a disclosed exploratory transport-policy amendment. It does not preserve
the untouched original preregistration. No case bytes, model prompts, selection
parameters, model names or offered total budgets change; no task scores were
inspected to choose the continuation policy.

## Model-failure classification amendment before version 4, 2026-11-11

Version 3 reused all 32 version-2 calls and sent one new request. That request
returned HTTP 200, valid usage,8192 completion tokens and `finish_reason=length`;
its 40250-character partial output repeated one comment 443 times. Diagnosing the
partial response revealed generation degeneration, not a need for more output
authority. Version 3 remains incomplete:22 completed stages, one failed,31 unrun;
99973 logical tokens/reservations, including the89411-token reused prefix, and
10562 newly reported tokens. Preserve its full response and original interpretation.

Version 4 continues this exact version-3 prefix. It does not regenerate completed
answers or change prompts, data, selection, model,8192 output ceiling, context
limits or60000-token/12-request total ceilings. The response deadline is90 seconds
for coders (default meter/formation remains30); the worker deadline is420 seconds.
These transport bounds were set before the version-3 paid request.

A valid-model, valid-usage HTTP 200 length response is now an invalid model proposal:
retain it, reject it even if partial JSON parses, and charge it against the existing
two-proposal stage allowance. There is no automatic length retry outside that
allowance. A per-arm budget refusal ends only that arm, with every remaining stage
explicitly budget-exhausted and counted as unsuccessful in the planned denominator.
Other arms continue. Unknown-worker loss, unhandled transport, authentication,
malformed provider usage/model and sandbox infrastructure failure still stop the
study. One identical-input coder transport retry remains bounded and recorded.

Exact replay explicitly admits recorded length responses and zero-charge budget
refusals, in addition to the recorded missing-response attempt. Default replay
eligibility stays strict. These are deterministic recorded outcomes, never live
fallbacks. An offline prefix proof must replay all completed answers and consume
the recorded partial response before refusing the first missing inference, with
zero new provider calls. Test classification and arm isolation before freezing
and continuing. Quality is descriptive for this amended, continued exploration;
do not present it as the original untouched single-run protocol.

## Formation failure amendment before version 5, 2026-11-11

Version 4 completed both first cases (36 stages) and stopped at the third common
formation. Its HTTP 200/stopped response contained a 1284-character summary, 84 above
the unchanged 1200 limit; all 14 citations were unique. Preserve the entire invalid
response, 179963 logical tokens/reservations and 53 logical calls. Only 20 new calls
were sent in version4; the remaining 33 calls were the exact version-3 prefix.

Version 5 continues that prefix, including the invalid formation response, without
regenerating it. A returned proposal that violates the public summary contract
now fails the consuming case/arms explicitly (`formation_failed`), preserving its
formation charges and all planned denominators. The direct-source control does
not consume the summary and still executes. No clipping, regenerated summary or
reference rescue is allowed. Unhandled provider/replay/infrastructure exceptions
still stop globally. Reported overspend is not a zero-charge budget refusal.

The offline prefix proof must consume all 36 completed stages and the invalid
formation with zero new provider calls, then refuse the first missing direct-source
inference. Freeze after local tests; pay only for that remaining control work and
replay the complete planned outcome ledger on PostgreSQL. Do not claim 54 successful
worker executions: formation/budget failures are explicit planned outcomes. This
is another disclosed model-failure isolation amendment; prompts, data, model,
selection, output/context ceilings and offered total budgets remain unchanged.

## Completed outcome

Version 5 accounts for all 54 planned outcomes: 39 executed worker stages and 15
explicit formation failures. PostgreSQL matches the complete semantic projection
with zero new provider calls. The final local gate passed 2870 Python tests with
PostgreSQL required, 9 skips and 90.13% coverage; TypeScript passed 63 tests,
typecheck and build. The original release checkout's 30 protected file hashes and
Git status remain unchanged. Temporary owned PostgreSQL containers were removed.
The evaluation report separates reported usage, unknown reservations, overlapping
prefixes, model failures and completed task contracts. This closes the requested
evaluation cycle, not the broader product goal.
