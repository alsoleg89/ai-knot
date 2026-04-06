"""S4 — Memory Compression F1.

Community metric: MemGPT compression ratio, mem0 dedup benchmark.
Pain point (#2): "memory grows unboundedly" / "agent sees 50 copies of the same fact."

Extends the existing deduplication test with Compression F1:
  F1 = 2 * (dedup_ratio * retention_ratio) / (dedup_ratio + retention_ratio)

Sub-test A: 50 paraphrases of one rule → should collapse (high dedup_ratio).
Sub-test B: 20 genuinely distinct rules → should NOT merge (high retention_ratio).

Metrics (all deterministic, no judge):
  dedup_ratio      — 1 - unique_stored / n_paraphrases (higher = better for smart backends)
  retention_ratio  — distinct_rules_found / n_distinct (higher = better for all)
  compression_f1   — harmonic mean of dedup_ratio and retention_ratio
"""

from __future__ import annotations

from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import BUNDLE_EN, DeduplicationFixture, LanguageBundle
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s4_compression_f1"


async def _count_stored(backend: MemoryBackend, query: str, n_inserted: int) -> int:
    exact = await backend.count_stored()
    if exact is not None:
        return exact
    r = await backend.retrieve(query, top_k=n_inserted + 10)
    return len({t.strip().lower() for t in r.texts})


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    bundle: LanguageBundle = BUNDLE_EN,
) -> ScenarioResult:
    """Measure dedup_ratio, retention_ratio, and Compression F1."""
    dedup: DeduplicationFixture = bundle.dedup

    # --- Sub-test A: true-positive dedup ---
    await backend.reset()
    last_insert_a: InsertResult | None = None
    for para in dedup.paraphrases:
        last_insert_a = await backend.insert(para)

    unique_stored = await _count_stored(backend, dedup.canonical_rule, len(dedup.paraphrases))
    dedup_ratio = max(0.0, min(1.0, 1.0 - unique_stored / max(len(dedup.paraphrases), 1)))

    # --- Sub-test B: false-positive guard ---
    await backend.reset()
    for rule in dedup.distinct_rules:
        await backend.insert(rule)

    r_b = await backend.retrieve("software engineering rules", top_k=len(dedup.distinct_rules) + 10)
    found = sum(
        1
        for rule in dedup.distinct_rules
        if any(rule.lower() in t.lower() or t.lower() in rule.lower() for t in r_b.texts)
    )
    retention_ratio = found / max(len(dedup.distinct_rules), 1)

    # Compression F1
    if dedup_ratio + retention_ratio > 0:
        compression_f1 = 2 * dedup_ratio * retention_ratio / (dedup_ratio + retention_ratio)
    else:
        compression_f1 = 0.0

    notes = (
        f"lang={bundle.language}, "
        f"paraphrases={len(dedup.paraphrases)}, "
        f"unique_after_dedup={unique_stored}, "
        f"dedup_ratio={dedup_ratio:.2%}, "
        f"retained={found}/{len(dedup.distinct_rules)}, "
        f"retention_ratio={retention_ratio:.2%}, "
        f"compression_f1={compression_f1:.3f}"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "dedup_ratio": [dedup_ratio, dedup_ratio, dedup_ratio],
            "retention_ratio": [retention_ratio, retention_ratio, retention_ratio],
            "compression_f1": [compression_f1, compression_f1, compression_f1],
        },
        insert_result=last_insert_a,
        retrieval_result=r_b,
        notes=notes,
    )
    result_obj.language = bundle.language
    return result_obj
