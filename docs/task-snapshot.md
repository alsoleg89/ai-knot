# Load a task's current evidence without a query

Use `AlwaysOnMemory.task_snapshot` when a small task corpus should be loaded directly. It needs no search words, embedding callback, model or background worker. The operation returns all admissible task facts as whole text and exact namespaced evidence refs, or raises `SnapshotTooLarge` if they do not all fit.

```python
from ai_knot.always_on import SnapshotTooLarge

try:
    facts = memory.task_snapshot(task_key="release42", max_facts=8, max_chars=12000)
except SnapshotTooLarge:
    # A separate, explicit selection policy handles a larger working set.
    selected = memory.select_context(
        ("rollback checkpoint", "certificate rotation"),
        task_key="release42", max_facts=3, max_chars=600,
    )
    facts = selected.facts
    # Keep selected.missing_requirements; these facts may be insufficient.

checked = memory.review_context(tuple(item.ref for item in facts))
```

`facts` is a tuple of existing `ContextFact` values in namespace/ID order. That order is deterministic serialization, not relevance ranking. A fitting empty corpus returns `()`. No completeness/coverage field is added. The snapshot tells the host what current evidence is available, not whether a requirement follows from it, the work is ready, or an action is permitted. Keep the original task requirements and guarded action path. `review_context(())` can be complete too; that is not evidence for a decision.

## Scope and bounds

The task key is a required named argument. Facts with the exact `task_key` qualifier or no task qualifier are applicable. Explicit `task_key=""` selects only generic/unlabelled facts. A task label is not an access grant; recipient-role hints do not filter this whole-corpus read.

One read-only SQL transaction checks the configured owner's private memory and its durably configured shared pool. All admitted facts must be supported, active, non-episodic, readable and current for their slot/source lineage and freshness policies. Grounded lessons retain current recorded-outcome requirements. A copy does not bypass its root permissions. Rejected or inaccessible facts do not count toward overflow, and the error discloses no omitted IDs or counts.

Both output limits are exact: `max_facts` 1..128 (default 8), `max_chars` 1..65536 (default 12000). Characters are Unicode code points of fact content, excluding refs/formatting, not bytes or model tokens. Labels use the existing exact-string limit of 256 characters without surrounding whitespace. Any admitted fact exceeding the remaining count or character budget causes an error with no partial payload. Raising the budget is explicit; the method never clips text, returns an arbitrary prefix, invokes embeddings or falls back to another selector automatically.

This read does not update usage counters, intents, leases, subscriptions or delivery receipts. It has no journal frontier and cannot prove catch-up. Facts can change immediately after the transaction, so re-review saved refs before use and retain `claim`, checkpointed basis and `begin_action`. Already delivered text cannot be erased after a later source withdrawal.

The bounds limit the output, not namespace scanning or inherited-source SQL checks. Large retained corpora still need explicit applicability/selection; no large-corpus or concurrent-fleet performance claim follows.

Repeated fact/root/permission/policy reads are now reused within this operation's SQL snapshot. Every new operation reads fresh state; no cache survives a transaction or substitutes for action-time checks. The [local profile](context-read-performance.md) records lower SQL work, unchanged returned results and the remaining full-scan/CPU costs.

## Rebuild context from non-derived facts

If the host rejects a generated handoff or wants to reconstruct context directly
from current observations, it can request:

```python
facts = memory.task_snapshot(
    task_key="release42", include_derived=False, max_facts=32, max_chars=6000,
)
```

The default `include_derived=True` retains the existing whole-corpus behavior.
False omits facts recognized by the existing derivation/projection metadata,
including grounded summaries and recorded lessons. The filter runs before output
bounds and inherited-lineage work. Every returned fact still passes currentness,
ACL, freshness, supported/active, ambiguity and task checks. If the non-derived set
does not fit, the same all-or-error `SnapshotTooLarge` contract applies. No text is
clipped, no model is called, and no stored memory is deleted or reclassified.

Non-derived is a provenance category, not proof of truth or human authorship.
Useful learned procedures can be omitted. The host keeps its original requirements
and domain checks; this is not an automatic retry or permission to execute. A
separately captured `task_guard` still covers the full admitted universe, including
derived facts. A filtered read never weakens that guard.

HTTP/MCP accept `"include_derived": false`; TypeScript accepts
`taskSnapshot({task_key: "release42", include_derived: false})`. The flag must be a
real boolean: strings, integers and null are rejected by the shared decoder. The
runnable example below also demonstrates a six-fact mixed snapshot reducing to
one current observation under this explicit policy, without claiming that one
observation contains the task's complete procedure. See the
[design evidence](research/2026-11-11-source-context.md).

## The same read through each client

| Surface | Operation |
|---|---|
| Python | `memory.task_snapshot(task_key="release42")` |
| HTTP | `POST /v1/always-on/evidence/snapshot` with `{"task_key":"release42"}` |
| MCP | `memory_context_snapshot` with the same fields |
| TypeScript | `await client.alwaysOn.taskSnapshot({task_key: "release42"})` |

HTTP/MCP return an array of existing context-fact objects; TypeScript returns `ContextFact[]`. All transports accept the same optional count/character limits and `include_derived` boolean, and retain configured-owner identity, authentication and strict input decoding. Missing `task_key`, unknown fields, bool/string count coercions and owner overrides are rejected. Overflow uses existing `SnapshotTooLarge` (HTTP 413 / structured MCP error / `AlwaysOnError`), with no context array.

This is an additive source-engine/client API; package versions and release files remain unchanged. See the [host flow](always-on-team-cycle.md#load-a-small-task-corpus), [task selection](task-context.md), and [semantic candidates](semantic-context.md) when a fitting whole task corpus is not the desired context.

The runnable model-free recipe exercises both whole snapshot and explicit overflow handling:

```sh
PYTHONPATH=src python examples/task_context.py
```

The [adoption research](research/2026-10-28-task-bundle-adoption.md) records the API gap and primary sources. The earlier [15/16 canonical-prefix control](semantic-context-evaluation.md) is not an accuracy result for this all-or-error operation. This change makes source-checked acquisition simpler; improved real-world decisions still need naturally ordered longitudinal evaluation at equal aggregate budgets.

Local acceptance: 2,335 Python tests passed with PostgreSQL required (two expected skips, 89.92% coverage), plus 61 TypeScript tests, typecheck and ESM/CJS builds. Exact prior/current semantic acquisition matched in 36 deterministic SQL cases. See [validation](superpowers/validation/2026-10-28-task-snapshot.json) and [Astra/Ponytail review](superpowers/reviews/2026-10-28-task-snapshot-astra-ponytail.md). These establish the API/source-admission contract, not model quality or fleet capacity.

The 2026-11-11 non-derived-view addendum passed 2,889 Python tests with PostgreSQL
required (9 skips, 90.16% coverage), 63 TypeScript tests, typing/formatting and both
builds. It adds no paid study or new quality score. See the
[independent reviews](superpowers/reviews/2026-11-11-source-context-independent.md)
and [validation](superpowers/validation/2026-11-11-source-context.json). No Astra was
used for this addendum; the older review above is historical.
