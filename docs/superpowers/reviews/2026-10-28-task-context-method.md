# Independent task-context method audit

2026-10-28. Scope: fixture/gold construction, six-arm runner, harness tests, current documentation and the parent-reproduced SQLite/PostgreSQL reports. This is not another production-selector review. No benchmark, model call, installation or production mutation was performed; arithmetic was independently recomputed from retained selected IDs, content and gold.

**Verdict: publish the current narrow selection claims. No material methodological blocker found.** The strongest existing scoped Integrate baseline matches the new selectors on available-evidence recall and output size. The observed role result is metadata/source preference, not better task performance.

## Verified

- Both reports contain 384 records: six arms × 16 traces × four checkpoints. Independent manifest serialization hashes to `283a0b281c29383a97813538185369da6e44cc1401440580c61405bd05ae04d0` for both backends. A frozen hash establishes retained-data integrity; it is not by itself proof of when preregistration occurred.
- Recomputed available-group coverage, selected characters, forbidden selections and role choices match every stored metric. The strongest scoped baseline, task-only and task-role each select 112/112 available groups and 11,200 content characters. There are 192 requirement positions per arm, of which 80 intentionally have no applicable current gold evidence. This is not 112 completed tasks or 100% model accuracy.
- Preferred-role presence is 24/32 matching checkpoints for the strongest control/task-only and 32/32 for task-role. The corresponding purity is 24/48 = 50% versus 32/48 = 66.7%. Presence and purity use different denominators; documentation now states both correctly. Operator/reviewer alternatives deliberately have identical valid text, so this fixture demonstrates recipient-source preference only.
- SQLite/PostgreSQL selected records and quality metrics are identical. All recorded corpus/grant/usage/trust immutability controls pass. Each backend has 96 handoff rows with fresh PIDs, saved refs and source data reloaded; the following checkpoint reviews saved refs against corrected/revoked evidence.
- Task applicability is provided through the same frozen qualifier data, including to the stronger old-path projection. Role and task inputs are not derived from gold selections. The weaker semicolon query remains as a transparent sensitivity control; adding the stronger conjunction baseline without changing sources appropriately prevents a false retrieval-improvement claim.
- Gold groups, acceptable/forbidden IDs and `safe_next_action` are declared by fixture logic, not generated from selector output. The next-action text is not scored, and no real action outcome is measured. Failed/unknown lesson projections are explicitly `supported=False`; their exclusion does not demonstrate inference of external success. Current documentation makes this limit explicit.
- All arms share the three-fact/600-content-character ceiling and invoke no selection inference in these runs. This is a returned-content budget, not an equal end-to-end model-prompt/CPU budget. No model-quality or comparative latency claim is supported. Fixed arm order can warm database caches, so latency remains descriptive.
- The lexical-report oracle shares the production tokenizer but independently checks returned text overlap; frozen ID groups provide an additional domain label check. Neither is semantic entailment. Six arms/backends and repeated roles/checkpoints are not independent replications of 384 tasks.

## Diagnostic limitation

Current `results.json` artifacts contain no DSN descriptor/key or PostgreSQL connection/credential patterns. Transient handoff jobs are created mode `0600`, under the CLI's temporary directory, and removed in `finally`; reports omit the storage descriptor.

**Should fix before publishing failure logs:** `tests/eval/task_context/runner.py:460–462` appends the last 4,000 raw stderr characters to an exception note. If worker stderr contains a DSN or `password=...`, that value reaches the parent diagnostic unchanged. The current successful reports are unaffected, and the audit found no actual credential exposure in them. Redact the descriptor DSN/credentials before attaching stderr, or keep failure logs private and sanitize them before export. The existing diagnostic test verifies preservation, not redaction.

## Claim boundary

Supported: explicit applicability/coverage interface, role-label preference in this frozen corpus, whole-fact output bounds, source retraction checks and process-reopen controls. Not supported: better underlying retrieval than the strongest existing scoped control, outcome understanding, useful role-specific instruction quality, end-task success, real-model accuracy or production capacity. Keep the selector opt-in and retain all six arms when publishing results.

## Owner follow-up after this audit

The failure-diagnostic concern was corrected after the independent review. Worker stderr now redacts the descriptor DSN, raw and URL-encoded password/SSL-password values before attaching a bounded exception note; an unparseable descriptor suppresses the diagnostic. A new regression failed before the helper and passed afterward. The full trace-harness tests pass (11), and the final required-PostgreSQL suite passes (1,889 tests). This follow-up is the implementer's verification, not an additional independent reviewer claim. Successful retained reports were unaffected.
