# Team decision outcomes implementation plan

> **For agentic workers:** use executing-plans for parent integration and dispatching-parallel-agents for the three independent modules below. The user has authorized autonomous implementation and local verification; no further plan approval is needed.

**Goal:** measure actual simulated work across pauses, source corrections and uncertain effects with the requested OpenAI models and a strong existing-memory control; fix demonstrated runtime defects without expanding default product complexity.

**Architecture:** a small deterministic task world supplies exogenous observations; real SQL-backed memory selects context; a metered GPT-4.1 mini reader proposes actions. The simulator applies a separate explicit admission policy and records effects, receipts, rejected proposals and completed work. Report reader errors separately from harm prevented by host guards. This benchmark cannot establish production readiness or close the broad product goal.

**Stack:** existing Python/httpx/SQL backends; no new dependency, service, product worker or release metadata.

**Spec:** the active product objective and [OpenAI evaluation plan](2026-10-28-openai-quality-eval.md).

## Global constraints and interfaces

- Keep original checkout HEAD/status and all 30 protected files unchanged. Feature work stays in this managed worktree; no push, merge or release.
- Fixed models `gpt-4.1-mini`, `text-embedding-3-small`; record returned identities, all formation/query/reader calls and actual provider usage. No key in files, argv, reports, subagent prompts or model input. Parent runtime may read the user-authorized OS credential record.
- Equal per-scenario/per-arm ceilings: 20 requests and 20,000 total input+output+embedding tokens. Conservatively reserve UTF-8 input bytes plus fixed framing and requested output, then settle actual usage. Unknown billing on failures stays reserved. Never turn provider/parsing failures into favorable quality scores.
- Five arms: full raw-observation history without shared retrieval; strong task-scoped existing pool recall; matched flat-query lexical recall; task working set; task-scoped async flat recall with OpenAI embeddings. Same exogenous world/schedule, source corpus, host authority, context ceiling 12,000 characters and reader output ceiling 128 tokens. Endogenous tool results may diverge after different choices; disclose that causal distinction.
- Three frozen scenarios: English, Russian, and an English paraphrased-requirement stress variant. Five steps: initially pending prerequisite receipt; peer confirmation; return after an unknown operation result; corrected destination with a pending receipt; new destination confirmation. First destination execution returns unknown but happens once; reconciliation finds its receipt. Second destination returns success. Wrong/duplicate/unsupported executions are rejected by an explicit common host policy and counted as bad proposals.
- Agent payload: goal, task key, requirements, selected source entries with exact revision references, and common host operation receipts. No hidden current target, oracle action, gold label, world state or outcome counter enters the prompt. Response is JSON `{action,target,basis}`; action is execute/verify/reconcile/hold and basis is indices into supplied entries.
- Score final confirmed destinations, physical effect counts, unsafe/duplicate/stale proposals, wasted actions, objective next-action matches, model/provider failures, context cost and latency separately. Do not infer model safety from simulator rejection.

## Task 1: metered OpenAI port

Owner files: `tests/eval/team_decisions/openai.py`, `tests/eval/test_team_decision_openai.py`.

- [x] RED/GREEN offline tests for hard budget, concurrent reservations, actual usage/model recording, malformed/truncated/model-substituted/provider-failed results and secret-free errors/reports.
- [x] Implement `MeteredOpenAI(api_key, *, max_calls=20, max_tokens=20000, transport=None)` async context manager; `.complete(messages, *, max_output_tokens=128, response_schema=None) -> str`; `.embed(texts, *, base_url="https://api.openai.com", model="text-embedding-3-small", api_key=None, timeout=30) -> list[list[float]]` compatible with the pool's embedder port; `.calls` and `.spent_tokens` expose public accounting only. Raise sanitized `EvaluationError`/`BudgetExceeded` rather than silently falling back. Official endpoint only; no key discovery in this module.

## Task 2: revision-correct dense cache

Owner files: pool/retriever implementation and `tests/test_pool_embedding_revisions.py` only.

- [x] Reproduce same-ID content update on a long-lived pool versus a reopened pool, and a source update during embedding I/O, with deterministic vectors.
- [x] Fix once in shared retrieval. Preserve incremental work for unchanged texts, lexical fallback, defaults and explicit provider configuration. Reclaim deleted/inactive vectors if the same design needs it. Do not hide the race by removing correctness assertions or introduce a new background/index service.
- [x] Run focused shared-pool/embedder regressions; hand back concrete evidence and source hashes. Full SQL acceptance and independent review belong to parent.

## Task 3: frozen world and mechanical outcomes

Owner files: `tests/eval/team_decisions/world.py`, `tests/eval/test_team_decision_world.py`.

- [x] RED/GREEN a deterministic world with the schedule above, three scenario definitions, repeat/unknown/correction branches, clear oracle and no LLM/self-grading dependency. Freeze manifest/hash before retrieval and reader execution.
- [x] Export `SCENARIOS`; `World(scenario)` with `.advance(step)` for steps0..4, `.facts()` current `Fact` objects, `.history` changed `Fact` snapshots, `.progress()` JSON host receipts, `.expected_decision() -> tuple[str,str]`, `.apply(action,target,*,basis_valid: bool) -> dict`, `.summary() -> dict`. Scenario fields: `id`, `task_key`, `goal`, `requirements`, `targets`.
- [x] Facts have stable IDs, current versions, fixed timestamps and task labels; text and versions change only on source changes. Host progress includes observed operations/receipts, never hidden source readiness/current target or oracle. Execution requires correct current target, confirmed receipt, valid evidence and no unresolved/duplicate effect. First target returns unknown; reconciliation proves success; second target returns success.

## Task 4: SQL/reader runner and acceptance

Parent files: package `__init__.py`, `runner.py`, `tests/always_on/test_team_decisions.py`, guide/report.

- [x] Integrate independent worlds/stores/ledgers per arm. Reuse existing SQL fixture isolation and current evidence review. Keep hybrid document/query calls inside its ledger; parallelize independent arms, never share mutable counters or source data.
- [x] Test perfect scripted proposals and deliberately unsafe/stale proposals on SQLite and PostgreSQL; test report/control integrity with mocked HTTP. These are contract checks, not live quality results.
- [x] Freeze manifest, run live requested models on both SQL backends (or a justified selection-equivalence proof plus live SQL run), inspect mistakes without tuning gold, and identify the next product improvement from evidence.
- [x] Fresh Astra/Ponytail review; fix material findings; full local CI with PostgreSQL and relevant static checks for product changes. Preserve prior reports at their actual source states, and document measured scope/remaining work.

Review focus: a budget failure must stop spending; a stale vector must not affect the current text; wrong environment must not become authority; unknown effects must not be repeated; safe host rejection must not inflate model accuracy. Tests above own each case.

Ruling before the first live run: Astra reproduced that an `Integrate` query exits via lexical facet assembly and discards dense signals even after embedding HTTP succeeds. Preserve that strong existing baseline; use the same flat requirements-join query for both `scoped_flat_pool` and `hybrid_scoped_pool`, and assert actual dense ranking is exercised. This adds the matched lexical control, keeps all ceilings unchanged and prevents query formulation from masquerading as an embedding gain. The frozen world/gold is unchanged. The protocol hash is revised before any live result; no product facet redesign is justified by this harness issue.

## Accepted engineering batch and continuing work

[Validation](../validation/2026-10-28-team-decisions.json): 2106 local Python passes with PostgreSQL required, 89.74% coverage, 86 focused passes on each of Python3.11/3.13, complete19-scenario/11-binding MA acceptance on both SQL backends, and independent Astra/Ponytail Ship after both material findings were fixed. The cache uses a single explicit as-of time across embedding/read snapshots.

The [live study](../../team-decision-evaluation.md) preserves all preliminary costs and failures. Protocol1 had an ambiguous target type; protocol2 exposed implicit citation-index conformance noise. Protocol3 uses explicit zero-based handles and strict Structured Outputs uniformly. Its final hash is f3ddea2a84a999d1e850f3f7de5c9918ea0f920430f48b169e736cff80f15d80; the world/gold remained unchanged. These are post-pilot exploratory repairs on inspected fixtures, not memory-gain evidence. Exact measured runtime sources and both raw SQL reports are retained separately from the final canonical-redaction diagnostic fix.

A local connection interruption under overlapping CI/MA loads made the legacy MA CLI print9/9without S26. That result was rejected; unchanged sequential checks supplied every19scenario and11binding threshold. Hardening that runner to reject selected-scenario exceptions remains the next verification fix, rather than treating a smaller denominator as acceptance.

The product goal remains active: held-out decision tasks, bounded semantic candidate acquisition for missing context, and sustained-worker/corpus costs remain required next work. Neither these checks nor this five-proposal simulation establishes production capacity, broad accuracy or adoption.

