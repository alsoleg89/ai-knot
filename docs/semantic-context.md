# Source-checked semantic candidates

`SemanticContext` is an opt-in async Python helper for a host whose task requirements miss paraphrased or cross-language observations. It uses the same SQLite/PostgreSQL evidence checks as always-on memory. Ordinary `select_context` remains synchronous and model-free.

```python
from ai_knot.always_on import SemanticBudget, SemanticContext

# The host supplies a fixed embedding model and an aggregate request/token meter.
# This callback makes at most one request per invocation, without retries.
selector = SemanticContext(memory, embed=host_embed_once)

candidates = await selector.acquire(
    ("Which preparation is required before opening this assembly?",),
    task_key="repair42",
    budget=SemanticBudget(max_candidates=128, max_input_bytes=65536,
                          max_provider_calls=1, timeout=5.0),
    max_facts=3, max_chars=600,
)
refs = tuple(item.ref for item in candidates.review.accepted)
```

Reuse the helper for one logical owner and fixed embedding space within one event loop. The callback must return one finite, nonzero vector per input in input order and keep its dimension fixed. The helper normalizes vectors and uses cosine similarity with the existing assembler; an optional matching `for_role` qualifier increases rank by 25%. Scores never establish entailment, source independence or action authority.

Before any provider call, the helper admits only supported, active, non-episodic facts applicable to the task (`task_key` matches or is empty). Shared facts require a durable pool and readable visibility grants. Derived facts retain the currentness/access requirements of their original evidence. Freshness policies and recorded action-outcome bindings still apply. Private and shared facts with equal IDs remain distinct.

If the entire admissible corpus fits the output count and whole-fact character limits, it returns that corpus as `snapshot` with zero provider calls. A cheap task bundle is sufficient in this case; vector ranking adds no selection benefit. Otherwise a single batch embeds uncached requirements and unchanged-source cache misses. All SQL snapshots are released before awaiting the callback. Exact source revisions, access, lineage, applicability and freshness are checked again afterward; changed text is never silently substituted for the text that was embedded.

| Status | Meaning |
|---|---|
| `snapshot` | The whole admissible corpus fits; no embeddings were needed. This may be an empty corpus. |
| `ok` | Semantic selection completed over the admitted snapshot. |
| `input_limit` | The candidate or serialized UTF-8 provider-input ceiling was exceeded; no partial prefix was embedded. |
| `budget_exhausted` | A cold acquisition needs a callback, but no invocation was authorized. |
| `unavailable` | Callback timeout, failure or invalid vectors; no candidates returned and the invocation reservation remains spent. |
| `changed` | Some captured evidence became inadmissible during provider I/O; the review retains only still-current candidates and reports issues. |

`SemanticBudget()` authorizes zero provider calls. Warm acquisitions can use cached vectors with this default; cold acquisitions return `budget_exhausted`. `provider_calls_reserved` and `input_bytes_reserved` report this acquisition's reservations, including failed/unknown calls. Input bytes cover the compact JSON array of uncached query/document texts, not HTTP overhead or tokens. A caller-owned meter must enforce aggregate calls, tokens and account spending across acquisitions and agents. This helper's limits do not grant spending rights to HTTP/MCP clients, and it adds no inference-capable server endpoint.

The helper retains only currently admitted unchanged document vectors and the most recent requirements tuple. It serializes concurrent acquisitions to avoid duplicate cold calls. Caches are local to the helper, so separate processes or a restart can pay again. The 128-candidate cap bounds paid work; source admission still scans retained namespaces and validates evidence. This is a bounded working-set path, not a large-corpus index or fleet capacity guarantee.

`review.accepted` contains whole facts and exact refs. `review.issues` reports rejected captured evidence; access failures redact its identity. `review.omitted` lists positive-score candidates excluded by output/rank selection. Its `complete` flag means there were no admission issues or such omissions. An empty review can be complete; neither this field nor `status="ok"` certifies that task requirements are satisfied. Preserve the original requirements and lexical coverage separately. Recheck chosen refs before use and retain `claim`, checkpointed basis and `begin_action` for effects.

For an event/timer host, the [always-on team recipe](always-on-team-cycle.md#opt-in-to-semantic-candidates) invokes this helper only for missing lexical context after an explicit relevant event or bounded retry. Idle checks never trigger paid work automatically.

The [evaluation and local profile](semantic-context-evaluation.md) records the independent cases, equal context/cost ceilings, actual OpenAI calls and measured SQL overhead. Package versions and release files are unchanged; this API currently requires the matching source checkout.
