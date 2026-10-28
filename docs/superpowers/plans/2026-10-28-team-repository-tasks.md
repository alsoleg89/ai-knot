# Continued repository work with team memory

The previous turn was progress: `0dc1197` added the source-checked model-free snapshot and passed local PostgreSQL/client acceptance. The broad goal still needs actual task-outcome evidence, not more lexical coverage counts.

## Frozen design before fixture access

An independent author seals three small standard-library Python repository tasks, each with an initial stage and two source updates. Each stage has public documents/brief, hidden unittest cases and an independently validated reference. Parent does not open fixture content/gold before the generic runtime and protocol are frozen. Identifiers derive from paths rather than editorial relevance; no fixture/protocol tuning after live results.

Compare four explicit acquisition policies: current direct sources; all team source/note history; current sources plus the same latest researcher handoff; current task memory (`task_snapshot`, then explicit `SemanticContext` if too large). Current-source-plus-handoff is the strong cheap control. Each arm retains its own generated code into the next stage; no reference reset. Common researcher output sees only public inputs, is formed once, and its full cost is allocated to every arm that uses it. Direct sources need no formation. Actual unique account spend is separately deduplicated.

Use requested `gpt-4.1-mini` and `text-embedding-3-small`. Per-case/arm ceiling: 12 total provider calls and 60,000 total reported/reserved tokens, including allocated formation. At most two patch/test attempts per stage, 2,048 coder output tokens, 768 researcher output tokens, 24,000 source-context Unicode characters, and one semantic batch per stage. No hidden history truncation, model retry or source fallback. Snapshot versus semantic use is measured, never forced by editing sealed data.

The coder emits a whole replacement for `solution.py` and visible one-based evidence citations. Current-ref checks are common to every arm and do not consult hidden gold groups. Test success is the primary task score; provenance validity, incomplete/unsafe proposals, cost and context size are separate measures. Hidden expected values/test source never enter model prompts; feedback is bounded test counts/names only.

Candidate code runs only in a fresh hardened cached Python 3.12 Docker image: no network, read-only filesystem, non-root, dropped capabilities, no-new-privileges, CPU/memory/PID/wall limits, own read-only temporary fixture mount, no keys/user files/daemon mount. Zero tests, abnormal exit, malformed receipts and timeouts cannot pass. This is bounded local code execution, not a production external-effect or adversarial-code containment guarantee.

Researcher and executor are real separate model roles. Actual executor stages run in fresh processes over reopened SQL state; no hidden model conversation is carried across restarts. All arms share the same intent/action guards and idempotent candidate-write/test policy; the knowledge acquisition policy differs. A worker lost without a final ledger invalidates that arm and conservatively consumes its remaining budget, with no automatic retry.

## Work and acceptance

- [x] TDD extend the existing evaluation meter with an explicit per-client output ceiling, preserving the 128-token default and total-budget reservations.
- [x] Generic source/formation/acquisition/coder orchestration with fake transport tests; guard failures, parse failures, budget exhaustion and cleanup failures retain charges and actual code state.
- [x] Independent safe container port and mocked/real Docker probes. Parent never executes candidate model code on the host.
- [x] Independent fixtures validate references, initial failures and meaningful sequential updates. Initial runtime/source/protocol freeze preceded primary fixture access; the public task-scope adapter correction and partial contributor exposure are explicitly disclosed.
- [x] Astra/Ponytail method and source review before paid calls. Same data/authority/context/action ceilings; strongest cheap control retained.
- [x] One bounded live run per SQL backend, all failures/costs retained; no after-result fixture tuning or model reruns. Local deterministic tests including PostgreSQL and relevant regressions pass before acceptance.
- [x] Preserve original checkout/deferred release state; reviewed local commits only. Product goal remains active regardless of a small authored experiment's outcome.

## Observed outcome and limits

See the [evaluation report](../../team-repository-evaluation.md). There was no task-memory advantage over the cheap current-documents-plus-handoff control. All 18 memory acquisitions used a six-fact snapshot; no semantic benefit was measured. Original live account spend was 113 calls/251,676 tokens. SQLite's contradictory-reference caller abort remains a failed run with its original cost; the later recovery repair and Python 3.11 malformed-receipt repair were not rescored. The original measured runtime is archived separately from repaired current sources.

These are three synthetic single-file tasks, not nine independent repositories. Controls had unspecified source order, researcher outputs varied across backends, every arm retained common SQL coordination state, and stage events explicitly reset intent basis. The work is engineering/evaluation progress, not proof of production usefulness, automatic retained-memory invalidation or completion of the broader product goal.
