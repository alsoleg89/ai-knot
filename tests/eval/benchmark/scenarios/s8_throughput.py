"""S8 — Latency & Throughput at Scale.

Community metric: BEIR latency benchmarks, MemoryOS throughput.
Pain point: "memory system is too slow for real-time agents."

Inserts 200 semantically distinct facts (LOAD_FACTS), runs 10 warmup queries,
then measures p50/p95/p99 latency and throughput over 20 concurrent queries.

Metrics (all deterministic, no judge):
  p50_ms      — median retrieve latency in milliseconds
  p95_ms      — 95th-percentile latency
  p99_ms      — 99th-percentile latency
  throughput  — queries per second (20 queries / wall time)
"""

from __future__ import annotations

import asyncio
import time

from tests.eval.benchmark._eval_utils import percentile as _percentile
from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import LOAD_FACTS, LOAD_QUERIES
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s8_throughput"
N_WARMUP = 5
N_MEASURED = 20
CONCURRENCY = 5  # parallel retrieve tasks per batch


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    quick: bool = False,
) -> ScenarioResult:
    """Insert 200 facts, warmup, measure latency and throughput."""
    await backend.reset()

    last_insert: InsertResult | None = None
    for fact in LOAD_FACTS:
        last_insert = await backend.insert(fact)

    queries = (LOAD_QUERIES * 10)[: N_WARMUP + N_MEASURED]
    warmup_queries = queries[:N_WARMUP]
    measure_queries = queries[N_WARMUP : N_WARMUP + N_MEASURED]

    # Warmup (ignored in metrics)
    for q in warmup_queries:
        await backend.retrieve(q, top_k=5)

    # Measured: run CONCURRENCY queries at a time
    latencies_ms: list[float] = []
    t_start = time.perf_counter()

    async def _timed_retrieve(q: str) -> float:
        t0 = time.perf_counter()
        await backend.retrieve(q, top_k=5)
        return (time.perf_counter() - t0) * 1000

    for batch_start in range(0, len(measure_queries), CONCURRENCY):
        batch = measure_queries[batch_start : batch_start + CONCURRENCY]
        batch_times = await asyncio.gather(*[_timed_retrieve(q) for q in batch])
        latencies_ms.extend(batch_times)

    wall_time = time.perf_counter() - t_start
    throughput = len(latencies_ms) / max(wall_time, 0.001)

    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    p99 = _percentile(latencies_ms, 99)

    notes = (
        f"facts={len(LOAD_FACTS)}, "
        f"measured_queries={N_MEASURED}, "
        f"concurrency={CONCURRENCY}, "
        f"p50={p50:.0f}ms, p95={p95:.0f}ms, p99={p99:.0f}ms, "
        f"throughput={throughput:.1f}qps"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "p50_ms": [p50, p50, p50],
            "p95_ms": [p95, p95, p95],
            "p99_ms": [p99, p99, p99],
            "throughput": [throughput, throughput, throughput],
        },
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
    return result_obj
