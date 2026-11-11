# Preserve host-known required observations

`required_refs` preserves an explicit AND set of observations known to a trusted host, such as an approval and a restriction. It does not discover prerequisites or make that list complete.

```python
required = memory.capture_evidence((
    ("__shared__", "approval"), ("__shared__", "restriction"),
))
guard = memory.task_guard(task_key="rollout")
context = memory.select_context(
    ("rollout readiness",), task_key="rollout",
    required_refs=required, max_facts=8, max_chars=2000,
)
# Assess original policy; preserve required refs in the saved decision basis.
# Explicitly reuse guard for checkpoint and action admission.
```

Mandatory whole facts occupy the existing count/character budget first, in caller order; exact duplicates count once. Relevance fills the remainder. Mandatory content without query matches does not manufacture lexical coverage. Equal private/shared IDs remain separate. No text is truncated and no dependency graph or provenance closure is inferred.

Unknown, stale, inaccessible, unsupported, expired, superseded, raw-episode or wrong-task mandatory sources raise sanitized `StaleEvidence` without a partial group. Mandatory output overflow raises `SnapshotTooLarge`. Untyped refs and lists longer than128 are invalid; conflicting revisions fail as stale evidence. These reads grant no new authority.

## Semantic selection

Pass the same `required_refs=required` to `SemanticContext.acquire`. Mandatory text need not be embedded for ranking. When it fills the output allowance, acquisition costs zero provider calls; otherwise existing snapshot/ranking rules apply. Optional ranking still requires its explicit budget and can refuse without returning context.

The candidate-universe limit remains: overflow returns `input_limit` before mandatory checks, without admitting/embedding a prefix. Within that universe, mandatory-source and mandatory-output failures raise before inference. After provider I/O, mandatory withdrawal returns `changed` with **no accepted context**, preserving spent reservations and existing issue redaction. Cache pruning also runs on required-only reads and required-input refusal. The callback's meter owns aggregate spend.

## Planning and clients

The [planning helper](bounded-planning.md) accepts optional refs. They enter its input fingerprint only when supplied; changing them can trigger a new plan despite an unchanged source-set guard. Unchanged refs/input suppress repeat acquisition. Required-source refusal blocks the planner; known native pre-I/O refusals reserve zero calls, while unknown cancellation retains its conservative reservation. Required refs enter checkpoint basis even if the model cites fewer observations. The returned guard still needs explicit action-time reuse.

A fitting whole-task helper snapshot keeps its existing order; semantic overflow selection puts required observations first. The callback retains its evidence-index interface without refs, lease tokens or work IDs.

HTTP `POST /v1/always-on/evidence/select`, MCP `memory_context_select` and TypeScript `alwaysOn.selectContext` accept an optional array `required_refs` of strict `{namespace,fact_id,revision}` objects. Semantic acquisition remains a Python callback without a new inference-capable endpoint. Empty refs preserve existing behavior.

This preserves **known** obligations. The host still determines applicability, sufficient evidence, correct decisions and connector policy; guards still detect unselected updates. No paid accuracy improvement is claimed. See [diagnosis](research/2026-10-29-selection-loss-diagnosis.md) and [primary-source research](research/2026-10-29-budgeted-evidence-selection.md). Versions, schema, dependencies and deferred release preparation are unchanged.
