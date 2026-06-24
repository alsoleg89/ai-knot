"""S9 — Scale Sensitivity.

Tests how retrieval quality and latency degrade as corpus size grows.
Inserts 5 signal facts + N noise facts (N = 0, 50, 200, 500, 1000),
measures MRR and p95 latency at each scale point.

Metrics (all deterministic, no judge):
  mrr_at_N        — lexical MRR at each corpus size N
  p95_ms_at_N     — 95th-percentile retrieval latency at each N
  mrr_degradation — fractional MRR loss from N=0 to N=1000
"""

from __future__ import annotations

import time

from tests.eval.benchmark._eval_utils import hit_rank_lexical, mrr, percentile
from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import EN_NOISE_TOLERANCE, LOAD_FACTS
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s9_scale"
TOP_K = 5
_ATC_THRESHOLD = 0.5
_SCALE_POINTS = [0, 50, 200, 500, 1000]


def _make_noise(n: int) -> list[str]:
    """Return *n* noise facts, duplicating LOAD_FACTS with mutations if needed."""
    if n <= len(LOAD_FACTS):
        return LOAD_FACTS[:n]
    result = list(LOAD_FACTS)
    copy_idx = 1
    while len(result) < n:
        for fact in LOAD_FACTS:
            if len(result) >= n:
                break
            result.append(f"{fact} [variant {copy_idx}]")
        copy_idx += 1
    return result


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
) -> ScenarioResult:
    """Measure MRR and p95 latency at increasing corpus sizes."""
    signal_facts = EN_NOISE_TOLERANCE.signal_facts[:5]
    signal_queries = EN_NOISE_TOLERANCE.signal_queries[:5]

    mrr_values: dict[int, float] = {}
    p95_values: dict[int, float] = {}
    last_insert: InsertResult | None = None

    for n in _SCALE_POINTS:
        await backend.reset()

        # Insert noise first, then signal (signal is more recent)
        noise = _make_noise(n)
        for fact in noise:
            last_insert = await backend.insert(fact)
        for fact in signal_facts:
            last_insert = await backend.insert(fact)

        # Query each signal query, collect ranks and latencies
        ranks: list[int | None] = []
        latencies_ms: list[float] = []

        for i, query in enumerate(signal_queries):
            await backend.reset_session()
            t0 = time.perf_counter()
            result = await backend.retrieve(query, top_k=TOP_K)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)

            rank = hit_rank_lexical(signal_facts[i], result.texts, threshold=_ATC_THRESHOLD)
            ranks.append(rank)

        mrr_values[n] = mrr(ranks)
        p95_values[n] = percentile(latencies_ms, 95)

    mrr_at_0 = mrr_values[0]
    mrr_at_1000 = mrr_values[1000]
    degradation = (mrr_at_0 - mrr_at_1000) / max(mrr_at_0, 0.01)

    judge_scores: dict[str, list[float]] = {}
    for n in _SCALE_POINTS:
        m = mrr_values[n]
        p = p95_values[n]
        judge_scores[f"mrr_at_{n}"] = [m, m, m]
        judge_scores[f"p95_ms_at_{n}"] = [p, p, p]
    judge_scores["mrr_degradation"] = [degradation, degradation, degradation]

    notes = (
        f"scale_points={_SCALE_POINTS}, "
        f"signal={len(signal_facts)}, "
        f"mrr@0={mrr_at_0:.3f}, mrr@1000={mrr_at_1000:.3f}, "
        f"degradation={degradation:.2f}, "
        f"p95@0={p95_values[0]:.0f}ms, p95@1000={p95_values[1000]:.0f}ms"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=judge_scores,
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
