# Continued team execution: Ponytail source review

Date: 2026-11-11. Scope: the uncommitted `tests/eval/team_work/{provider,runner,experiment}.py`
and their tests, connected storage, public prompt projection, frozen corpus loader,
existing metered OpenAI transport and Docker sandbox. Reviewed by the implementing
agent using the Ponytail review procedure; this is not an independent model review.
No Astra was invoked.

Verdict: ship the bounded research runner after the recorded local gates. No
unresolved source-review finding remains. Paid outcomes are not part of this
pre-execution verdict.

## What this change does

Runs three successive coding stages with durable agent state and six context policies.
It records the model and embedding boundary, keeps each arm's own failed work, and
can replay successful provider responses against PostgreSQL without credentials.
Expected load is one bounded, sequential research run with fresh stage processes;
this harness is not a production service or a concurrent agent factory.

## Findings and resolutions before paid execution

1. **Lost worker charge could disappear if its partial report was corrupt.** The
   exception path parsed that report before recording the conservative reservation.
   A malformed JSON file reproduced a total of 3 instead of 60003 accounted tokens.
   The reservation now persists first; unreadable partial diagnostics cannot erase
   it. The new lost/interrupt/corrupt regression keeps the remaining schedule unrun
   and suppresses quality claims.
2. **Raw history depended on a successful proposal.** Observations were saved only
   in the validated proposal checkpoint, so two invalid answers could hide that
   stage from the next history reader. The worker now returns its observed history
   in the durable research receipt. The coordinator carries it separately from
   executable code and evidence basis. A real SQL test checks old and new versions
   reach the next reader while invalid proposals leave saved code/basis intact.
3. **Handoff roots had an unlocked validation/read gap.** The prepared runner could
   bind a summary to a root revision it had not summarized. Validation, root reads
   and publication now share the existing namespace write transaction. The SQL
   regression checks that the unseen revision is never blessed.
4. **A stale historical citation prevented the allowed second attempt.** It now
   receives bounded invalid-proposal feedback, just like malformed JSON, only if
   the captured source guard still matches. Authorization and execution failures
   are outside that local parsing handler.

The last two findings were reproduced and fixed during the interrupted preparation;
their regressions were rerun in this continuation.

## Version 1 failure and the next pre-execution review

The first paid attempt stopped on an unfinished 3072-token completion, with quality
suppressed. This was not a passing end-to-end trial. Its old metered transport also
discarded the truncated text before the recorder could retain it. The amended
protocol and raw version-1 runtime are retained alongside its partial report.

The explicit coder output authority is now 8192 for every arm, still within the
unchanged 60000-token case/arm budget. The shared test-only meter keeps its default 128
authority and has a separately copied `raw_completions` research property, populated
before validation. The public metadata ledger still contains no source/output text,
HTTP headers or error bodies. A mocked HTTP length response tests retention, actual
charging, and rejection above 8192. An out-of-order response test found an indexing
error in this new recorder; a local per-call index fixes it. These are fixes to
observed technical failures, not selection/fixture/prompt changes to improve scores.

Version 2 is another exploratory attempt after a disclosed protocol amendment,
not an untouched single-run preregistered comparison. Do not pool its quality with
version 1, omit version-1 expenditure, or claim an independent reviewer.

## Continuation and model-failure review

Version 2 then lost an HTTP response; version 3 reused its exact 32-request prefix
and obtained a recorded length response from one new request. Inspection of that
partial text found 443 repetitions of one comment. Raising token limits again or
resampling successful cases would hide the model's failure.

Version 4 keeps the same inputs, output/total budgets and all recorded responses.
The runner rejects a length response as an invalid proposal, even if the partial
JSON happens to parse; the existing two-proposal limit applies. A budget refusal
stops only the corresponding arm and explicitly fails its remaining planned
stages. Tests cover both rules, charge preservation and continued execution of the
other arms. These are amendments to an exploratory protocol, not unchanged
preregistered results.

Exact prefix consumption happens before a paid client is opened. Message/schema/
limit drift cannot fall back to a new answer. Missing-response requests, truncated
outputs and zero-charge budget refusals can be replayed only through explicit
study options; default replay remains strict. The actual prefix wiring proof uses
fresh worker processes and the Docker grader, with the new provider factory
replaced by a failing stub: all completed stages are replayed, then the first
missing inference is refused, with zero API calls. Failed requests and unknown
billing reservations are preserved. A missing-response coder request may retry
once; HTTP errors and repeated transport failures do not loop.

The observed ~32-second missing response motivated an explicit 90-second coder
deadline before version 3; the meter's default and formation deadline remain
30 seconds. The worker deadline is 420 seconds. Token and call authority remain
unchanged. These are waiting bounds, not measured speed claims.

The final formation check found a second isolation error: a returned 1284-character
summary violated the 1200-character contract and prevented even the independent
direct-source control from running. `InvalidHandoff` now distinguishes rejected
model data from provider, callback or replay failures. Its consuming arms get
explicit case failures; the direct control still runs. No summary is clipped,
regenerated or replaced by reference code. A regression checks all 15 failed consumer
stage outcomes, the three direct-control invocations and retained formation cost.
A separate regression ensures actual reported overspend still stops globally;
it cannot be disguised as a zero-charge arm-budget refusal.

## Final owner verification

2870 Python tests passed with PostgreSQL required (9 skips, 90.13% coverage), as
did static checks, 63 TypeScript tests/build/typecheck and both 12-scenario recovery
gates. Seven final documentation/content checks passed. The completed experiment
accounts for 39 worker stages and 15 formation failures; PostgreSQL replay matches
its complete semantic projection with zero new API calls. Source/guard/code/receipt
and 26 consecutive-worker carry checks pass. Release-checkout status and 30 file
hashes are unchanged. No production source, version or dependency changed.

Verdict: ship this bounded research runner and its candid results. Do not turn
the tested semantic or all-cited-root policies into a new default on this pilot.

## Deliberate limits

- One small exploratory synthetic case series, fixed arm order, partly unblinded
  fixture development. No significance, production reliability or general accuracy
  claim follows from it.
- Citation retention protects known sources; it does not infer missing obligations.
- Full documents/history and bounded selection have different context ceilings;
  every arm has the same offered call/token ceiling, not identical consumed compute.
- No SQL transaction remains open during provider calls. Guarded checkpoints and
  action admission fence the controlled stage changes; this does not attest an
  arbitrary long-running external effect after admission.
- The existing same-interpreter unittest grader is not tamper-proof attestation.
  Candidate code runs only in the existing restricted, credential-free container.
- An unknown worker reserves its entire remaining authority. Actual call counts
  are explicitly incomplete in that case; partial known receipts remain diagnostics.
- Replay uses exact ordered inputs and original logical charges. It is backend
  parity evidence, not another model sample or a promise of future model output.

## Verification

Final commands, outcomes and immutable experiment hashes are recorded in the
accompanying validation artifact and evaluation report. No release metadata,
dependencies, production schema or production API changed in this batch.
