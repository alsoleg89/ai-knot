# Semantic working context and fail-closed verification

> Execute autonomously with TDD. Independent gate repair, design review and sealed case authoring are delegated under dispatching-parallel-agents; parent integrates the memory feature and runs SQL/live acceptance. No release or original-checkout changes.

**Goal:** let a ready host acquire source-checked semantic candidates when literal task requirements miss relevant memory, without treating vector similarity as task completion.

**Evidence:** [decision study](../../team-decision-evaluation.md), committed at cf5c210, found a paraphrase gap under the same source corpus. A separate legacy MA run displayed a smaller passing denominator after a selected scenario failed. Both need concrete fixes before broader product claims.

## Constraints

- Synchronous select_context and ordinary defaults remain model-free. No worker, graph service, new dependency or schema/version change.
- Semantic work is explicit and provider-cost-bounded. A caller-owned metered embedding port must be usable; no automatic idle/reflection work.
- Admit own/shared candidate sources by task, read ACL, support, active time, lineage and freshness before provider I/O. No inaccessible source text in requests.
- Release the SQL snapshot before awaiting a provider. Recheck exact captured revisions, access, task applicability and freshness afterward; stale results cannot be replaced silently with new unembedded text.
- Distinguish candidate retrieval/currentness from lexical requirement coverage, semantic entailment, task readiness and action authority. Keep begin_action checks.
- Bound candidate count and input before paid work; never silently treat a truncated candidate universe as complete. Return whole facts under output limits.
- Namespace-qualified identities prevent equal private/shared IDs from colliding.
- Keep default use simple; choose the smallest interface after inspecting existing result types/provider configuration. Shared/vector-cache infrastructure requires cost evidence, not speculation.

## Work and evidence

Concrete contract chosen before opening the independent fixtures: a Python-only `SemanticContext(memory, embed=explicit_async_callback)` helper, with `acquire(requirements, budget=SemanticBudget(...), task_key, role, max_facts, max_chars)`. The callback owns one fixed embedding space and an authoritative aggregate request/token meter. The per-acquisition budget defaults to zero provider calls, bounds admitted candidates and serialized UTF-8 input, and reserves one invocation before awaiting it. It never retries or infers server spending authority from a transport caller's limits.

If the complete admissible task corpus fits the output limits, return a `snapshot` without embeddings. Otherwise one batch embeds uncached queries/texts. A helper-local cache retains only currently admissible unchanged content and the latest query tuple; concurrent calls serialize. Exact refs are rechecked after I/O. The existing assembler handles the count/whole-fact character limits during ranking; unselected positive-score candidates are reported as omitted. `SemanticCandidates.review.complete` concerns admission/omission of candidates, never entailment or task completion. No new `complete` flag or automatic inference worker is added.

Before fixture access, independent source review found and the owner reproduced/fixed count-omission reporting and an oversized-winner selection bug. Focused SQLite validation: 29 passed, 22 PostgreSQL skips. Source/test SHA-256 freeze is retained in the acceptance artifacts; later changes must be disclosed rather than silently counted as untouched held-out evaluation.

- [x] Design review chose the lean interface; the concrete contract was recorded before opening independent cases.
- [x] RED/GREEN semantic candidate retrieval, provider failure/limits, namespace collision, source and ACL/task/freshness changes during I/O on SQLite/PostgreSQL.
- [x] Gate repair preserves deliberate partial runs but fails on selected-scenario errors, missing/empty/non-finite measurements and empty verification; reports retain sanitized failure evidence.
- [x] Independent author froze new EN/RU multi-domain cases without retrieving or calling models. Parent first opened them after implementation source hashes were frozen.
- [x] Ran both SQL backends with requested OpenAI models and jointly metered embeddings/readers, compared strong scoped controls on identical sources/ceilings, and retained all negatives without tuning gold. Both CLI runs correctly exit 1 for an invalid reader citation; the semantic arm did not beat the task bundle.
- [x] Integrated documented opt-in host use and fresh Astra/Ponytail review. Full local CI/PostgreSQL passed; complete structural MA gates ran separately. Existing HTTP/MCP tests passed in full CI; no inference-capable transport endpoint was added.
- [x] All 30 original protected files and HEAD/status verified unchanged. Reviewed local commit is the integration step; no release changes, push or merge. The broad product goal remains active after this engineering batch.

Known interpretation limits: the earlier study has three inspected fixtures and five proposals. This work must not turn that into a held-out result retroactively. New data are frozen independently before parent implementation evaluation. Structural/privacy tests may use deterministic vectors; only live results measure model behavior.

Final evidence: [API contract](../../semantic-context.md), [single-pass results and limits](../../semantic-context-evaluation.md), [acceptance](../validation/2026-10-28-semantic-context.json), and [independent review](../reviews/2026-10-28-semantic-context-astra-ponytail.md). This batch adds a bounded optional path and trustworthy verification; it does not close product readiness.
