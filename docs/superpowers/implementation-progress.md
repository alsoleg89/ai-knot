# P1–P3 production implementation ledger

Started 2026-10-27, base `69e2adf`, branch `codex/always-on-p1-p3`.

User authorizes complete P1–P3 implementation, three implementation agents, real PostgreSQL tests, then independent Astra and Ponytail review.

## Binding decisions

- Canonical specification: `specs/2026-10-26-always-on-memory-design.md`.
- Existing checkout reused on a feature branch; prior untracked research and plans retained.
- Three agents have disjoint production-file ownership. P1 SQL seam and subscription mixin, P2 state facade/models, P3 maintenance/transports. Root owns acceptance runner, shared fixtures and CI. Dependency contracts coordinated before integration.
- No release/version bump, push or deployment in this task.
- PostgreSQL 16.15 runs in isolated Docker container `ai-knot-always-on-pg`, loopback port 55439. Tests use per-test schemas.

## Baseline

`.venv/bin/python -m pytest -n auto --timeout=20 -m 'not slow' --no-cov`: **1239 passed, 10 skipped**, 8.10s, one pre-existing OpenClaw warning.

## Implementation

- P1: in progress, agent `/root/implement_p1`.
- P2: in progress, agent `/root/implement_p2`.
- P3.1–P3.4: in progress, agent `/root/implement_p3`.
- P3.5: in progress, root.
- Final reviews: pending implementation and integration verification.

## Integrated gates before final review fixes

- Whole project with live PostgreSQL and coverage: **1391 passed, 8 skipped, coverage 87.64%** (18.68s). Earlier integration failures fixed: concurrent PostgreSQL schema CREATE race; old mock transaction fixture; disabled MCP construction mocks. One older context-rot assertion failed once in an earlier concurrent run; isolated baseline/current and twelve fixed hash seeds each all passed; next full gate passed unchanged retrieval thresholds.
- Ruff entire src/tests: clean, 246 files formatted. Strict mypy entire project: clean, 71 files.
- npm typecheck, ESM/CJS build: passed. Vitest: **56 passed**.
- Retrieval gates: **2 passed**. Multi-agent benchmark: **11/11 binding thresholds PASS**, advisory metrics unchanged.
- Recovery acceptance runs: SQLite/PostgreSQL, real process exits at after_poll/checkpoint/begin/external/finish. No-proof unknown remains blocked. Named snapshots preserve action/delivery history.
- Replay **100,000 events**, SQLite: passed. Matrices 1k/10k facts, 1/10/100 subscriptions collected for both backends; final reports regenerated after implementation integration.
- AO-disabled recall before/after comparison, five repetitions: 1k median p95 0.030486s→0.031420s (+3.06%); 10k 0.309027s→0.303314s (-1.85%). Same local machine, offline BM25 mode; not a general throughput claim.

## Production review (in progress)

Independent reviewers: `/root/production_astra_review` (gpt-6-astra) and `/root/production_ponytail_review` (Ponytail review skill).

Accepted findings being fixed with red/green SQLite+PostgreSQL regressions:

1. Pruning while consolidation awaits erased its output-slot conflict evidence. Guard retained frontier before trusting journal absence.
2. Old-source verifier response could confirm reconfigured policy. Capture/compare internal monotonic policy revision under commit lock, including identical-reset ABA; preserve cancellation propagation.
3. Consolidation accepted already inactive semantic output and closed source episodes. Reject invalid output temporal state before mutation.
4. due_verifications(limit=1) did O(policies×facts) full namespace decoding before maintenance budget checks. Replace with bounded SQL selection and targeted fact/slot reads, durable invalidation for changed policy basis.

P3 owns findings 1–3, P2 owns internal policy revision/storage and finding 4. Reviews require targeted recheck and final complete gate after fixes.

## Complete

All P1–P3 production tasks and reviewer remediation complete. Final gates: **1453 passed, 2 intentional skips, 87.80% coverage**; **18 slow performance tests**, **56 npm tests**; Ruff/format/mypy and retrieval/MA gates green. Both independent reviewers closed their findings. See `implementation-final.md` for exact evidence, supported scope, JSON measurements and operational boundaries. Current source is left in the feature branch working tree; no release, push or deployment.

## GitHub delivery

The user subsequently requested a GitHub push and documentation refresh. Public guides, npm instructions, example index and changelog now describe the unreleased source feature; local Markdown links use portable repository paths. Delivery target: `origin/codex/always-on-p1-p3`; package release and deployment remain separate. Earlier no-push notes describe the implementation phase before this request.
