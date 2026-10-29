What this change does: A host can reserve exact mandatory task observations before optional lexical or semantic ranking. The same whole-fact budgets and current-source checks apply, and the planning helper includes mandatory refs in input identity and saved basis.

## Should fix

1. **Required-only reads retain vectors for withdrawn sources** (`src/ai_knot/always_on/semantic.py:171–186`).
   - **What this is:** The helper is documented to retain only currently admitted unchanged document vectors. Every previous successful snapshot path prunes its cache against the admitted snapshot.
   - **Problem:** The new required-only early return occurs before that pruning, and required-context refusal also occurs before it. Warm the helper with source A and B, remove/revoke/change A, then acquire B as mandatory with max_facts=1. The returned text is correctly limited to B, but the no-longer-admitted A entry remains in `_documents`. Repeated mandatory-only acquisitions can retain it indefinitely. This is a cache-retention contract regression; I did not find stale text disclosure or an authority bypass.
   - **Fix:** Move the existing fingerprint/cache-pruning block immediately after candidate-limit refusal, before `required_context(...)` and the required-only return. It needs no new cache layer. A useful regression observes provider inputs: warm A, observe its deletion during mandatory-only B acquisition, restore the identical A record, then acquire normally; A should be re-embedded rather than resurrecting an entry retained across the observed deletion.
   - **If we skip it:** An observed withdrawal/deletion no longer clears the source's derived embedding/fingerprint at this successful acquisition boundary, contrary to the existing retention guarantee.

## Checked and boundaries

Read the plan, source diff since `696a49d`, connected task admission/source views, strict transport codec/router maps, TS adapter, coverage assembler, and current new core/host/transport tests. Assumed the existing bounded 128-candidate helper in one owner/event loop and cooperating SQL writers. No graph/default inference/service/schema or release change is introduced.

Mandatory membership uses exact namespaced refs; source failure returns no required partial set. Optional embeddings exclude pinned content and preserve post-I/O validation/reservations. Lexical coverage of mandatory facts still requires actual token relevance. Empty required_refs follows the old paths and leaves the helper input fingerprint unchanged. Explicit required changes affect fingerprints, and mandatory refs join saved basis even when the model cites another selected fact. HTTP/MCP get strict typed nested refs through the existing reflected signature; the TS optional input mirrors this.

The helper's catch-and-zero handling for StaleEvidence/SnapshotTooLarge relies on the native selector's contract that those exceptions are pre-I/O required-input refusals. Callback exceptions themselves are caught inside the selector, so no ordinary callback-raised exception of those types reaches that host catch. Arbitrary subclass overrides of acquire can violate the native contract; this is host-supplied trusted extension behavior, not a demonstrated native bug.

A required set is not automatic discovery of all task conditions, an execution permit, or a safety proof. Full-task guards and exclusive controller/lease assumptions remain necessary. I found no further concrete correctness issue in the reviewed revision.

Verdict: fix 1 before final acceptance.
Not checked: no tests, SQL, Docker, API, Keychain, live-model quality or performance runs; owner verification remains required. No Astra was used. Source was changing only through the owner's ongoing work; hashes below identify this review snapshot, not later fixes.

## Reviewed source hashes

- `src/ai_knot/always_on/context.py`: `0069a5c43387afd647b9b3287461713f54ca9bad86c3da2bfb523fb0cf6c788d`
- `src/ai_knot/always_on/semantic.py`: `4803da7f629b919b21e345aeb4066068c4768591357bac594349b8e392422d29`
- `src/ai_knot/always_on/memory.py`: `4c21f817136b35d622d1339eec4e7c4d66d81ccd8b2bb065f7c9389626a0a145`
- `examples/always_on_team.py`: `527f70ab9c884f8e19ae8a70f8376cdf34684330f0cbeae48202bb79a1513142`
- `npm/src/always_on.ts`: `f88d59366e7ae1dcda77b3c55083154ceb0d48be8515fc9f2e0c507a7a473536`
- `tests/always_on/test_required_context.py`: `be4da594e7f3758c3efeb694e25ca814d36f01ea4c56026b88b85728f7e14e87`
- `tests/always_on/test_selected_planning.py`: `2f0e6c60a2e4211970338af2ed205456bf1b61e5403dd5fe82a1c8b7a6f59f7b`
- `tests/always_on/test_required_context_surface.py`: `13d16a835f6796db70205fed828f60fc4add3a6da6b12b9635fb8b7c39db85a0`

## Owner resolution and verification

Finding1 fixed:existing fingerprint/cache pruning moved before required-context validation and required-only return. Observable regression `test_required_only_read_evicts_withdrawn_vectors_before_later_restoration` failed on both SQL engines then passed. Source review predates this fix. Owner separately fixed the exact-character-budget `budget_limited` diagnostic with a four-case RED→GREEN regression. Final full suite:2795passed8skipped90.13%;focused3.11/3.13:298each. No second independent review or Astra acceptance is claimed.
