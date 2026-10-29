# Required context implementation plan

> Native execution in the existing worktree. The user authorized autonomous implementation and opted out of approval loops. No Astra without a separate explicit request.

**Goal:** Preserve host-known mandatory exact observations during bounded task selection, without claiming automatic prerequisite discovery.

**Design:** Add opt-in `required_refs: tuple[EvidenceRef, ...] = ()` to lexical selection and semantic acquisition. Validate at most128 typed refs, reject conflicting revisions, deduplicate exact repeats in input order. Reserve mandatory whole facts within the existing count/character budgets, then rank optional facts into the remainder. Mandatory source admission/revisions/task applicability must pass; unavailable or stale refs raise sanitized StaleEvidence before inference. Mandatory overflow raises SnapshotTooLarge before inference. After embedding, mandatory withdrawal returns changed with empty accepted text and retained call reservations. This is one AND set, not automatic dependency closure or OR-group search. Existing source checks, guard capture and action authority remain necessary.

**Alternatives:** Threshold/MMR tuning does not discover prerequisites, as the frozen-vector diagnosis shows. Automatically including all generic facts creates unbounded mandatory context. Host-known exact refs give a small explicit contract with no new service, dependency, schema or default inference.

## Constraints

- Original checkout, all30 protected files and deferred release prep remain unchanged; no push/release.
- No gold-derived required refs in a paid comparison. Deterministic fixtures may declare their host policy directly.
- Empty required_refs preserves old selection/order/budgets. Required context supplies no new authority or semantic-completeness guarantee.
- Exact owner/shared IDs remain distinct. Task mismatch, raw episodes, inactive/unsupported/stale/hidden sources cannot be forced into context.

## Tasks

- [x] Write failing SQLite/PostgreSQL cases for mandatory weak/negative relevance, deduplication, count/character equality and overflow, stale/foreign/task-inapplicable refs and zero provider calls on refusal.
- [x] Implement shared required-context validation/projection in context.py; reserve budget in select_context and SemanticContext.acquire, preserving mandatory-first order and post-I/O checks. Keep the old path unchanged when empty.
- [x] Mirror lexical required_refs through strict HTTP/MCP and TypeScript; add boundary tests. Connect opt-in refs to the planning helper with input fingerprint/dedup and no model call after refusal.
- [x] Run full local pytest with PostgreSQL, static checks, TypeScript, focused Python compatibility and independent non-Astra review. Record failures, limits, protected-checkout audit and local commit.

## Review focus

Mandatory ref failure must not return a partial mandatory set or disclose hidden text. Required refs with no query terms must be included but must not create lexical requirement coverage. Post-I/O failure retains paid reservations; invalid/overflow mandatory inputs cause no provider I/O. Required refs changing with an unchanged source universe must change a host planning fingerprint. Preserve the unselected-update guard and normal action/lease controls.

## Acceptance

Full accepted source:2795 tests passed,8 skipped,90.13% coverage with PostgreSQL required;298 focused tests each on Python3.11/3.13;63 TypeScript tests/typecheck/build;Ruff and strict mypy passed. Independent non-Astra review found cache pruning regression; owner verified RED→GREEN and the full suite. Owner also verified exact-character-budget diagnostics RED→GREEN. Initial PostgreSQL connection failures and all earlier logs are retained. See [validation](../validation/2026-10-29-required-context.json). No paid quality gain or completed product goal is claimed.
