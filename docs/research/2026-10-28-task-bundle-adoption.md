# Queryless task context as an adoption seam

Retrieved **2026-10-28**. Local inspection is against `813daa5fe332359eb98461336a96b84cc825c356`, before the proposed task-bundle API. Documentation URLs are mutable; retrieval dates are not publication dates. This is a source/design note, not a new accuracy or performance result.

## Recommendation

A small, explicit, model-free task-bundle read fills a real public-API gap. The chosen contract returns all current readable task facts with their exact evidence refs when they fit, or raises the existing `SnapshotTooLarge` error. It does not infer requirements, create persistent memory blocks, mutate delivery cursors, or add a new background process. Ranked lexical/semantic acquisition remains available when a task corpus does not fit or the host has a specific retrieval question.

This recommendation is an inference from the local API gaps and the scoped-context patterns below. It is **not** evidence that canonical truncation is a better retrieval algorithm. The [previous single-pass study](../semantic-context-evaluation.md) reports 15/16 grounded answers for its scoped bundle and 14/16 for semantic acquisition, but relevant facts were favored by low-ID editorial placement. Use that result to justify testing a cheap control, not to promise gains on naturally ordered, accumulating memory.

## What current primary sources support

| Primary source | Observed mechanism | What transfers to AI Knot |
|---|---|---|
| [Letta V1 memory blocks](https://docs.letta.com/v1-sdk/memory/memory-blocks) | Attached blocks are inserted into the prompt without retrieval; blocks have content limits and may be read-only. Full-value replacement has last-write-wins behavior. The documentation navigation marks this API as legacy. | A deliberately small context component can be loaded without lexical search. Do not copy whole-block overwrite semantics into grounded fact storage. |
| [Current Letta shared-memory concept](https://docs.letta.com/concepts/shared-memory) | Current shared memory uses organization-owned Git repositories. Attachment exposes repository paths/top-level files; agents read files and synchronize commits. The page explicitly recommends migrating older shared-block users to repositories. | Separate knowledge availability from the material actually loaded into model context. An attached repository is not automatically a complete current prompt or a durable delivery protocol. |
| [Letta SDK repository management](https://docs.letta.com/agent-sdk/repositories) | Session resource links expire when the session closes; persistent agent attachments require explicit detachment. File reads can name a ref, and updates can use a content-hash precondition. | Explicit scope/lifetime and version identity are useful. AI Knot already has exact evidence revisions; a new task-bundle read does not need a repository, attachment registry, or copied state. |
| [LangGraph memory overview](https://docs.langchain.com/oss/python/concepts/memory) | Thread checkpoints and cross-thread knowledge stores have different scopes. Long-term memory uses application-defined namespaces. The guide distinguishes semantic knowledge from semantic search. | Reading shared task knowledge need not invoke embeddings or become another thread checkpoint. A task label is an applicability selector, not automatically a permission boundary. |
| [Deep Agents memory](https://docs.langchain.com/oss/python/deepagents/memory) | Memory files can load at startup or on demand; backend namespaces determine agent/user/organization scope. Shared policy memory is typically read-only. The guide also warns about concurrent file writes. | Provide an inspectable, bounded read that a host can insert explicitly. Preserve owner/source ACLs and fact-level currentness instead of adopting a new file-write workflow. |

These pages describe mechanisms, not comparative accuracy of AI Knot or proof of safe multi-agent decisions. In particular, the current Letta repository model should not be described as automatically loading a shared memory block into every prompt.

## Existing AI Knot paths and their actual limits

| Path at the inspected commit | Already useful | Why it is not the same public operation |
|---|---|---|
| [`AlwaysOnMemory.select_context`](../../src/ai_knot/always_on/context.py) | Read-only task applicability, current evidence, whole-fact budgets, lexical coverage and role preference. | Requires one to five content-bearing requirements and selects by word relevance. Supplying invented keywords to obtain all task facts confuses acquisition with coverage. |
| [`review_context`](../../src/ai_knot/always_on/memory.py) | Revalidates exact saved refs, with source/currentness issues and character omissions. | The host must already know those refs. It is validation rather than discovery of the task's current corpus. |
| [`SemanticContext.acquire`](../../src/ai_knot/always_on/semantic.py) | Its `snapshot` fast path already returns a complete fitting admissible task corpus with zero provider calls. | Construction still requires an async embedding callback and a query. Overflow enters a different acquisition contract: paid ranking, `budget_exhausted`, or `input_limit`. A host should not invent a callback or query to request a local task bundle. |
| [`KnowledgeBase.list_facts`](../../src/ai_knot/knowledge.py), [`SharedMemoryPool.list_shared_facts`](../../src/ai_knot/pool.py) | Raw inspection and administration. | The first reads the owner's raw stored facts; the second explicitly includes closed/invalidated shared facts and takes no requesting owner. Neither provides an admitted task-context snapshot. |
| [`resync`](../../src/ai_knot/always_on/memory.py), [`WatchSelector`](../../src/ai_knot/always_on/subscriptions.py) | Bounded complete subscription recovery with a journal head and durable pending receipt. | Resync changes inflight delivery state, requires a subscription, and selects by fact/slot IDs rather than task labels. A context read must not accidentally acknowledge or replace delivery progress. |
| [Evaluation-only `_scoped_bundle`](../../tests/eval/semantic_context_runner.py) | Demonstrates a small model-free task/control implementation with existing admission. | Private helper, shared-only fixture scope, and canonical prefix selection. It is evidence for a reusable seam, not production API compatibility. |

## Smallest useful contract

The owner chose one synchronous facade method rather than a new memory object, registry, result model or partial-result flag:

```python
memory.task_snapshot(*, task_key: str, max_facts=8, max_chars=12000)
# -> tuple[ContextFact, ...], or existing SnapshotTooLarge
```

The exact task key is a required named argument; an explicitly empty string requests generic candidates only. Existing `ContextFact` already carries the exact namespaced revision, content and slot identity. The API adds no `complete` field that callers could mistake for task satisfaction.

The essential contract is:

- One read-only SQL snapshot over the configured owner's private memory and, when durably configured, its shared pool. No inference, embeddings, usage-counter writes, lease changes or delivery acknowledgements.
- The existing task rule: exact task label or unlabelled generic candidate; an empty requested label includes only generic candidates. Do not turn recipient-role hints or task metadata into read authority.
- Existing support, active-time, slot-currentness, lineage/access, freshness and outcome-dependency checks apply before a fact can enter the response. Reuse the central basis checks rather than constructing a second security policy.
- Whole facts fit both count and content-character ceilings. Any **admitted** fact that would overflow either ceiling causes `SnapshotTooLarge`, with no partial facts or omitted-source identities returned. Denied/invalid sources do not expose IDs, counts or content through overflow diagnostics.
- If the entire admissible corpus fits, all of it is returned in canonical namespaced order. This order supplies deterministic serialization, not relevance. The operation never silently substitutes a low-ID prefix for the complete task corpus.
- An empty tuple is a valid read and does not establish missing requirements or successful work. This read does not return a journal frontier or prove catch-up. A fact may change immediately afterward; the host keeps saved-ref review and guarded action admission.

A bounded-partial result was considered: it could help a host inspect a larger corpus, but would need another result type/flag and invite treating an arbitrary prefix as sufficient evidence. The selected all-or-nothing contract is smaller and removes that ambiguity. Its cost is explicit: callers with larger task corpora must narrow applicability, increase a justified output budget within existing limits, or use the separate ranked selectors. The prior 15/16 canonical-prefix control is therefore not an accuracy result for this new all-or-nothing operation.

## Implementation seams and acceptance

Extract only the common source-admission seam needed by `SemanticContext` and the new local read. Preserve the semantic helper's existing candidate/input order and eligibility; do not change ranking, cache or paid-work behavior as part of that extraction. Avoid rewriting the lexical selector merely to deduplicate nearby code: it has an optimized original-content index and its own coverage contract. Preserve exact namespaced identities and central `_validate_basis` behavior, including slot/freshness/outcome checks. The new result must not gain transport callers' identity or spending authority.

For convenient adoption, mirror the one operation as HTTP `POST /v1/always-on/evidence/snapshot`, MCP `memory_context_snapshot`, and TypeScript `taskSnapshot`. Reuse the existing [HTTP route table](../../src/ai_knot/server/always_on.py), [shared strict invocation/MCP boundary](../../src/ai_knot/_always_on_tools.py), [codec](../../src/ai_knot/always_on/codec.py) and [TypeScript client](../../npm/src/always_on.ts), including the existing overflow error. These are ordinary read-only transport additions, not a reason to expose semantic provider invocation on the server. Host recipes should preserve task requirements, catch-up state, claims and action guards separately.

Test both SQL backends with queryless paraphrases, same-ID private/shared records, scope denial/revocation, changed roots and recorded outcomes, stale freshness, task mismatch, raw episodes, source ambiguity and expired/future facts. Exact-fit cases must return all admissible facts; one-overflow and oversized-first cases must raise without a partial result. Block all provider HTTP and verify durable state/counters remain unchanged. Reuse existing transport identity/auth/type checks and verify HTTP/MCP/TypeScript field parity.

For the next quality comparison, freeze cases before seeing outputs and permute irrelevant identifier/source ordering independently of semantic relevance. Include naturally accumulating task/generic facts and reader distraction. Preserve all embedding/reader costs and abstention failures. Until that evidence exists, the claim is a simpler, source-checked acquisition interface; a general accuracy improvement remains unproven.

No product code, dependencies, release files, credentials or provider configuration were changed for this note. No new model or load experiment was run.
