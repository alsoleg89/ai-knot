# Preparing a continued team-work evaluation

This foundation prepares a new task-outcome study. It has **no paid researcher/coder loop or quality result yet**. It does not replace or rescore the [earlier repository experiment](team-repository-evaluation.md).

## Implemented controls

The study-only code in `tests/eval/team_work` strictly loads hash-frozen typed JSON, projects chronological source changes and renders current evidence in canonical namespaced-ref order. A publisher changes only its own observer records in its case slot prefix. It preserves other actors' memory, derived observations and executor state. Unchanged source IDs, versions and times survive current-stage replay.

Real SQLite/PostgreSQL tests save code, a failed receipt and evidence basis, update a source, then reopen from a fresh Python process. Publication leaves the intent intact; ordinary `resume` detects the stale basis and persists BLOCKED while retaining code, receipt and refs. The repeated process return preserves that block. This tests durable return/invalidation, not a new model completing a task after restart.

Equal current source sets now produce byte-identical rendered evidence and coder messages, including citation indices and observation metadata. Source timestamps use deterministic microseconds to break ties and precede the reader clock. Historical multi-version contexts will need their own chronological rendering policy; current-set canonical ordering alone is not a history protocol. Publishers are applied chronologically, with replay of the current stage. Arbitrary out-of-order snapshot delivery is not implemented or claimed.

## New authored corpus

Three new synthetic single-file maintenance tasks cover feature policies, retry dispatch and artifact auditing. Each has three stages with private stdlib unittest contracts and reference solutions: nine contracts, 45 test methods. Every transition changes public rules; final stages also withdraw an obsolete source. There are 141 document occurrences across snapshots and at most 14 applicable observations in a stage. The largest full source context is 4,657 rendered characters, below the 24,000-character reference ceiling. This supplies selection pressure at an eight-fact cap, not evidence that a particular necessary subset fits a smaller character allowance.

The author used a fresh context without old cases/results. The parent froze 91 generic source/test hashes before opening the new corpus; all remained unchanged afterward. An independent reviewer then read public sources, references and contracts. The parent subsequently inspected mutation-target lines and public metadata, so this is documented partial blinding, not a double-blind or contamination-free benchmark.

All original 24 Docker controls matched: nine current references passed, nine initial stubs and six preceding-stage references failed. Review nevertheless found two missing checks in artifact contracts: Unicode case folding and an empty manifest at a zero budget. Three identical mutants passed original tests; private-contract repairs make all three fail by assertion with no execution errors. Public conditions, initial code and reference implementations are unchanged.

| Retained corpus | SHA256 | Change |
|---|---|---|
| Original | `8557254849082076acc9f36948804718f8b2df0d76403a2738f74aacc2f36265` | First authored bytes |
| v2 | `bf4b1330f623be62607fbcd18a92368d1b63ea5f00223932ae06710c5c523210` | Stage 3 Unicode assertions |
| v3, current | `0b11a369992ee24052b69c014a8dcc4bf8e1e4bb2cd6b4a7ec40df5d997d782e` | Stage 2/3 zero-budget assertions; v2 retained |

All 24 controls match on v3 too. Together with both three-mutant checks, this is 54 Docker executions. Positive/negative controls exercise selected behaviors; they do not prove that every valid implementation passes or every incorrect implementation fails. Each candidate/reference runs only in the existing pinned, network-disabled, resource-limited Docker sandbox. The same-interpreter grader is not adversarial attestation.

## Verification and reproduction

Local CI suite (`-m not slow`) with PostgreSQL required: 2,817 passed, eight skipped, 90.13% coverage. Ruff and strict source mypy pass. Source/public-boundary, fresh-process return and receipt-gate tests add 22 checks. No production source, version, dependency, schema or release preparation changed.

Run the offline Docker controls explicitly, using the cached image recorded by the sandbox:

```sh
PYTHONPATH=src python -m tests.eval.team_work.controls \
  --cases tests/eval/team_work/cases.json \
  --sha256 0b11a369992ee24052b69c014a8dcc4bf8e1e4bb2cd6b4a7ec40df5d997d782e \
  --output /tmp/team-work-controls-new.json
```

Existing output files are refused. There is no model/provider/credential argument. The [retained evidence](performance/team-work-foundation-raw.json.gz) includes all fixture revisions, source hashes, control receipts, mutation proof, failures and author/reviewer provenance. The [validation record](superpowers/validation/2026-10-29-team-work-foundation.json) and [review with owner resolutions](superpowers/reviews/2026-10-29-team-work-foundation.md) distinguish independent review from owner-run tests. Initial tuple-fixture/test-expectation errors, a local PostgreSQL connection closure, import-path failure and lint corrections are retained rather than reported as task outcomes. No Astra or paid API calls were used.

## Still required for a quality study

Freeze the actual arm matrix, selection/pinning policy, offered budgets, common researcher outputs, repetitions and stopping rule before model results. Implement fresh researcher/coder workers carrying each arm's own code, failures and action history; source producers must not clear executor basis. Use public formation/citations rather than hidden tests to obtain any [required_refs](required-context.md). Count all formation, embeddings, repairs and failures. Capture/recheck task guards at decision/action boundaries. Replay exact recorded requests on PostgreSQL without paid calls or fallback, and fail on divergence.

The [method research](research/2026-10-29-team-work-evaluation-method.md) defines these boundaries. Paid task-quality comparison, production repositories, concurrent factory behavior and long-horizon capacity remain unverified. The broader product goal remains active.
