"""S4 — Deduplication.

Tests two things:
  A) True-positive dedup: 50 paraphrases of one rule → backend should collapse
     them to a small number of unique facts (high dedup_ratio).
  B) False-positive guard: 20 genuinely distinct rules → backend should NOT
     deduplicate (high retention_ratio).

Metrics (all deterministic, no judge calls):
  dedup_ratio      — 1 - (unique_facts / 50). Higher is better for ai-knot.
  retention_ratio  — distinct_rules_found / 20. Should be near 1.0 for all.

Counting strategy:
  - Prefer count_stored() if the backend supports it (exact count of stored facts).
  - Fallback: retrieve with large top_k and count unique texts (proxy, less accurate
    for backends where retrieve returns ranked subset not full store).
"""

from __future__ import annotations

from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import DEDUP
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s4_deduplication"


async def _count_unique_stored(backend: MemoryBackend, query: str, n_inserted: int) -> int:
    """Count unique facts stored — uses count_stored() if available, else retrieve proxy."""
    exact = await backend.count_stored()
    if exact is not None:
        return exact
    # Fallback: retrieve with oversized top_k to capture all stored facts
    r = await backend.retrieve(query, top_k=n_inserted + 10)
    return len(set(t.strip().lower() for t in r.texts))


async def run(backend: MemoryBackend, judge: BaseJudge) -> ScenarioResult:
    # --- Sub-test A: true-positive dedup ---
    await backend.reset()
    last_insert_a: InsertResult | None = None
    for para in DEDUP.paraphrases:
        last_insert_a = await backend.insert(para)

    unique_stored = await _count_unique_stored(
        backend, DEDUP.canonical_rule, len(DEDUP.paraphrases)
    )
    dedup_ratio = 1.0 - unique_stored / max(len(DEDUP.paraphrases), 1)
    dedup_ratio = max(0.0, min(1.0, dedup_ratio))

    # --- Sub-test B: false-positive guard ---
    await backend.reset()
    for rule in DEDUP.distinct_rules:
        await backend.insert(rule)

    # Use retrieve as proxy here — we want to check recallability, not just storage
    r_b = await backend.retrieve("software engineering rules", top_k=len(DEDUP.distinct_rules) + 10)
    found = 0
    for rule in DEDUP.distinct_rules:
        rule_lower = rule.lower()
        if any(rule_lower in t.lower() or t.lower() in rule_lower for t in r_b.texts):
            found += 1
    retention_ratio = found / max(len(DEDUP.distinct_rules), 1)

    notes = (
        f"paraphrases_inserted={len(DEDUP.paraphrases)}, "
        f"unique_after_dedup={unique_stored}, "
        f"dedup_ratio={dedup_ratio:.2%}, "
        f"distinct_rules_retained={found}/{len(DEDUP.distinct_rules)}, "
        f"retention_ratio={retention_ratio:.2%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "dedup_ratio": [dedup_ratio, dedup_ratio, dedup_ratio],
            "retention_ratio": [retention_ratio, retention_ratio, retention_ratio],
        },
        insert_result=last_insert_a,
        retrieval_result=r_b,
        notes=notes,
    )
