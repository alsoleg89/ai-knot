What this change does: The program adds durable catch-up and snapshot recovery, checked task/role working sets, and procedures bound to recorded action outcomes. The final example composes those APIs into an event/timer inspection cycle with a durable host inbox, without starting a model worker or performing an external effect.

Review scope: fresh source review from `3618bfb573bedf48c3356e8000847a1ac02d839a` through `fcbcad5`, `f877440`, and HEAD `1b21ed455f2cac4f0639111ee1e9d67c6f3158d2`, plus the uncommitted team-cycle example and tests. I read the four relevant design/implementation plans, the roadmap, the public guides, connected Python/SQL/HTTP/MCP/TypeScript code, callers of the changed assembler, and focused regression tests. Prior review verdicts were not used as evidence. I made no checkout, index, HEAD, production, or test-file changes; my reproduction and this report are outside the checkout. The parent was editing the team-cycle tests/docs during review.

Assumed load: independent agents/processes sharing the supported SQLite or PostgreSQL store, with small bounded model contexts and the local workloads described in the repository. I did not extrapolate local latency to production capacity.

**Must fix — Important**

1. **A manual block becomes a model wake when its evidence changes** (`examples/always_on_team.py:177-178`; related state semantics in `src/ai_knot/always_on/_state.py:claim` and `resume`).
   - **What this is:** `inspect_turn` decides whether the host should spend a model call. A host can explicitly checkpoint an intent as BLOCKED while retaining its evidence, for example while waiting for manual approval.
   - **Problem:** After that evidence changes, `resume` correctly retains `host_blocked` and adds `changed`, but the recipe's unconditional `packet.stale_basis` branch returns `status='replan'` and `wake_model=True`. This reproduces on both SQL backends. Claim-before-inference does not repair the decision: claiming a BLOCKED intent is intentionally allowed for recovery.
   - **Fix:** Preserve `host_blocked` as a quiet blocked status before the stale-basis replanning branch. Add the combined manual-block plus changed-basis regression; the existing independent tests for these states do not cover their interaction.
   - **If we skip it:** A host following the recipe pays for automatic replanning on work deliberately held for manual intervention. The example itself does not run an external effect, and action-time guards remain separate, so this is not a claim that inspection directly executes an unsafe action.

Public minimal reproduction: `/tmp/ai-knot-team-cycle/astra-repros/test_program_review.py`. It uses only `KnowledgeBase`, `AlwaysOnMemory`, the example's public `Inbox`/`inspect_turn`, and a public fact replacement. Steps: add a source; capture it as basis; create/claim an intent; checkpoint BLOCKED with that basis; subscribe; correct the source; inspect. Actual packet reasons are `('blocked', 'changed', 'host_blocked')`; actual observation is `replan` with `wake_model=True`. Expected observation is quiet and blocked.

No Critical or Minor findings were confirmed. No deletion/refactor finding met the concrete-case bar.

Checked integration evidence:

- Delivery progress: `poll` distinguishes scanned frontier from namespace head, keeps filtered empty pages acknowledgeable, and leaves debounced backlog without a token. A pending delivery bridges retention only through its retained frontier; missing later history forces recovery. Snapshot capture and writers share namespace locks; replay checks revisions/ACL/active time/source lineage, and overflow rolls back before replacing a pending token. The focused SQL tests exercise these cases and post-snapshot writes.
- Host cache: selected context and the latest delivery receipt share one SQLite transaction before ack; a failed commit cannot advance delivery. A crash after commit before ack replays safely, and receiving context does not suppress unfinished READY work. The final reselect after `resume` avoids returning text made stale by settling work.
- Timer behavior and action boundaries: inspection rereads saved refs and current selection even without a journal event. Direct expiry/freshness and verification-deadline paths remain quiet; unknown effects are routed to reconciliation and live leases/deadlines/terminal states suppress model wakes. Inspection takes no lease and grants no action permit. The one exception is finding 1.
- Task selection: task labels are exact eligibility constraints, role is a score multiplier, and coverage comes from content-bearing original-query overlap in emitted text. ACL, lineage, freshness, current-slot/support checks and whole-fact budgets run before admission. Private/shared ID collisions remain namespaced; the old assembler caller retains defaults. Greedy lexical coverage is not semantic entailment or a guarantee of the optimal selection.
- Experience: outcome recording validates the current action revision and supplied sources, requires shared maintain authority, keeps action payloads out of the generated metadata, and stores exact roots/intermediate dependencies. The binding table supplies destination locks; `save_action` invalidates registered outputs atomically with changed action state. Reopen without a pool still discovers shared destinations. Ordinary pool publication rejects directly marked lessons before slot mutation; common SQL writes prevent stale marked snapshots/imports from restoring support. Current workspace ownership rejects terminal aliases. Retained dependencies make copied lessons fail after source/outcome retraction, while independent-evidence counting remains based on observations.
- Compatibility/migration: schema-7-to-8 initialization is additive; current action rows survive and ordinary unmarked fact writes avoid the new binding query. The focused suite covers schema migration, unaffected action paths, publication, snapshot restore, current aliases, source/ACL/freshness changes, rollback, and recording/reconciliation concurrency.
- Measurement claims: the frozen comparison includes the strong scoped `Integrate` baseline and both selectors at the same output budget. That baseline already matches selector recall. The role-presence result is fixture metadata preference, and model/embedding calls are blocked in the comparison. The reports do not establish paid-model accuracy, external success, automatic learning, or production capacity.

Commands/results actually run in this review:

1. Independent public repro, with the supplied local PostgreSQL DSN and PostgreSQL required:

   ```sh
   PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH=/Users/alsoleg/.codex/worktrees/shared-memory-return/ai-knot/src \
   AI_KNOT_TEST_PG_DSN="$AI_KNOT_TEST_PG_DSN" AI_KNOT_REQUIRE_POSTGRES=1 \
   /tmp/ai-knot-pr-ci/py312/bin/python -m pytest \
   /tmp/ai-knot-team-cycle/astra-repros/test_program_review.py \
   --no-cov -q -p no:cacheprovider
   ```

   Result: **2 failed in 0.34s** (SQLite and PostgreSQL), both expected-contract assertions for finding 1. Each PostgreSQL case used its own disposable schema. The repro directory's `conftest.py` is a copy of the repository SQL fixture, outside the checkout.

2. Focused existing regression suite, with the same environment:

   ```sh
   /tmp/ai-knot-pr-ci/py312/bin/python -m pytest \
   tests/always_on/test_memory_return.py \
   tests/always_on/test_context_review.py \
   tests/always_on/test_task_context.py \
   tests/always_on/test_team_experience.py \
   tests/always_on/test_team_cycle.py \
   tests/test_multi_agent_assembly.py \
   --no-cov -q -p no:cacheprovider
   ```

   Result: **183 passed, 2 failed in 36.30s**. The parent added `test_manual_block_survives_source_changes` before this run collected tests; these two new SQLite/PostgreSQL cases were the only failures and reproduce finding 1. I did not author or alter that repository test.

3. `git diff --check`: exit 0 at the checked source state.

The failing example's SHA-256 was `c540b1f83d84aadf456bc4764e48f8b67cac03cbf8ed07c99217e36ce14e275b`. Source SHA-256 values at review: context `6739dcb2c3afaca6d38c58c48308d7993c17776971c3318682ee07f4359e5051`; experience `58629db0908ca0d3b8710e1e9e247ebd0dbd6a2b64594fb2caffe38586bfc9e0`; memory `85e9838ff60cd70d4f3528f1a63693d2dcf77b6f805de9181adbf8f8c0f78769`.

Declined to judge:

- Paid-model answer accuracy, generalization, or end-task benefit: no real model was exercised by this review or the deterministic fixtures.
- Whether host-supplied lesson prose follows from a receipt, and whether a remote operation actually succeeded: those are explicitly outside the recorded-outcome contract.
- Production-scale throughput, high availability, or service limits: bounded local tests cannot establish them.
- Final full-suite/static/TypeScript/load acceptance, final docs/completion audit, protected-original-byte verification, or release readiness: the parent owns those concurrent checks. I reviewed transport changes and test source but did not independently rerun those full gates.
- Historical observation lifetime as semantic truth: exact archived source revisions are deliberately retained by the existing lineage contract; I did not treat that documented behavior as a new expiry defect.

Verdict: fix 1 first (**Revise**). The broader changed paths produced no other confirmed material finding in this review.

Not checked: real external effects or model quality, production capacity, and the parent's final full acceptance/documentation/protected-checkout audit.

Additional benchmark review requested by the parent: I compared the current `/tmp/ai-knot-team-cycle/trace-{sqlite,postgres}/results.json` with the corresponding committed compressed raw reports using Python JSON/gzip loading. Corpus-manifest hashes match on both backends. SQLite has zero changed selected-ID lists, metrics, or controls. PostgreSQL has exactly two changed selected-ID lists: `release-reviewer-mismatch` at `peer_update` and `handoff`, both in `ordinary_integrate_recall`; `release-operator` becomes `release-reviewer`. Selected text order, all per-row metrics, gold, and controls are identical in those rows; there are zero metric/control changes across all rows. The selectors and strongest scoped baseline retain their selected identities. This is non-material tied-source variation in the older ordinary baseline, not a demonstrated safety/contract regression or a gain for the selector. The documentation should say equivalent measured results rather than claim all arms reproduce identical source identities across runs. No algorithm tuning is justified by these two observations.

## Implementation owner follow-up

The Important finding was reproduced before repair in `test_manual_block_survives_source_changes` on both SQL backends. The host recipe now preserves `host_blocked` before automatic stale-basis replanning. All37recipe SQL/CLIcases pass, followed by1993fullPythonpasses with PostgreSQL required, two expected skips,89.38%coverage. The final TypeScript59tests/build/typecheck and Pythonstatic/recovery/MA/load gates pass; the [validation record](../validation/2026-10-28-shared-memory-program.json) carries scope and source hashes. The equal-budget strong receipt-control test was added afterward as acceptance evidence and passes on bothSQL; it adds no production behavior. The policy-fencing test budget correction is documented in the [completion audit](../../shared-memory-completion.md).

These final repairs/checks are the owner's verification, not a second independent Astra Ship verdict. No confirmed material finding remains deferred. Declined-to-judge items and tied-source variance are explicitly bounded in the completion audit.
