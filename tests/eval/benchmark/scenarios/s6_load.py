"""S6 — Load & Reliability.

Tests backend behaviour under load: latency at scale, concurrent reads,
and graceful handling of edge cases.

Sub-tests:
  0. Empty KB: retrieve from empty backend → no crash
  1. Long input: insert 10k-char string → no crash
  2. Latency at scale: insert 200 facts → measure p50/p95/p99 on 20 queries
  3. Concurrent reads: 50 simultaneous retrieval tasks (asyncio.gather)
     M5 Pro: 14 perf cores + Metal can handle 50 concurrent easily
  4. Mixed read/write: 10 alternating insert+retrieve pairs

Metrics:
  relevance        — judge score on a representative query (1-5)
  p95_latency_ms   — 95th-percentile retrieval latency in ms (deterministic)
  error_rate       — fraction of sub-tests that raised an exception (deterministic)

Pass --quick to runner for CI (20 concurrent tasks instead of 50).
"""

from __future__ import annotations

import asyncio
import time

from tests.eval.benchmark.base import MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import LOAD_FACTS, LOAD_QUERIES, PROFILE
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s6_load"
TOP_K = 5
CONCURRENT_TASKS_DEFAULT = 50  # M5 Pro: 14 perf cores + Metal
CONCURRENT_TASKS_QUICK = 20  # CI / low-resource environments


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    top_k: int = TOP_K,
    quick: bool = False,
) -> ScenarioResult:
    concurrent_tasks = CONCURRENT_TASKS_QUICK if quick else CONCURRENT_TASKS_DEFAULT
    errors = 0
    latencies: list[float] = []

    # --- Sub-test 0: empty KB ---
    await backend.reset()
    try:
        await backend.retrieve("anything", top_k=top_k)
    except Exception:
        errors += 1

    # --- Sub-test 1: long input ---
    await backend.reset()
    long_text = "word " * 2000  # ~10k chars
    try:
        await backend.insert(long_text)
    except Exception:
        errors += 1

    # --- Sub-test 2: latency at scale ---
    await backend.reset()
    for fact in LOAD_FACTS:
        await backend.insert(fact)

    # Warmup: 3 queries to prime caches / indexes (important on M5 APFS)
    for _ in range(3):
        await backend.retrieve(LOAD_QUERIES[0], top_k=top_k)

    # Measure 20 queries (2 passes over LOAD_QUERIES)
    for query in LOAD_QUERIES * 2:
        t0 = time.perf_counter()
        try:
            await backend.retrieve(query, top_k=top_k)
        except Exception:
            errors += 1
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)

    # --- Sub-test 3: concurrent reads ---
    # Warmup before concurrency test to ensure WAL is primed
    for _ in range(3):
        await backend.retrieve(LOAD_QUERIES[0], top_k=top_k)

    concurrent_errors = 0
    # Cycle through queries for variety; use default-arg capture to avoid closure bug
    query_list = (LOAD_QUERIES * 5)[:concurrent_tasks]

    async def _safe_retrieve(q: str) -> None:
        nonlocal concurrent_errors
        try:
            await asyncio.wait_for(backend.retrieve(q, top_k=top_k), timeout=10.0)
        except Exception:
            concurrent_errors += 1

    await asyncio.gather(*[_safe_retrieve(q) for q in query_list])
    errors += concurrent_errors

    # --- Sub-test 4: mixed read/write ---
    for i in range(10):
        try:
            await backend.insert(LOAD_FACTS[i % len(LOAD_FACTS)])
            await backend.retrieve(LOAD_QUERIES[i % len(LOAD_QUERIES)], top_k=top_k)
        except Exception:
            errors += 1

    # Judge one representative query
    rep_query = PROFILE.queries[0]
    rep_result = await backend.retrieve(rep_query, top_k=top_k)
    judge_scores_raw = await judge.score_all_async(rep_query, rep_result.texts)

    total_ops = 20 + concurrent_tasks + 10
    error_rate = errors / max(total_ops, 1)

    notes = (
        f"facts_loaded={len(LOAD_FACTS)}, concurrent={concurrent_tasks}, "
        f"p50={p50:.1f}ms, p95={p95:.1f}ms, p99={p99:.1f}ms, "
        f"errors={errors}/{total_ops}, error_rate={error_rate:.1%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "relevance": judge_scores_raw.get("relevance", [3.0, 3.0, 3.0]),
            "p95_latency_ms": [p95, p95, p95],
            "error_rate": [error_rate, error_rate, error_rate],
        },
        insert_result=None,
        retrieval_result=rep_result,
        notes=notes,
    )
