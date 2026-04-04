"""S2 — Avoid Repeats.

Tests whether the backend surfaces relevant content while helping avoid
recommending already-published topics. Measures recall of known-published
titles (so the agent can warn "this topic was covered before") and novelty.

Metrics:
  relevance    — are the retrieved titles relevant to the query? (judge, 1-5)
  recall       — fraction of expected titles found in top-k (deterministic)
  novelty      — 1 - (overlap with first retrieval) / top_k (deterministic)
"""

from __future__ import annotations

import statistics

from tests.eval.benchmark.base import MemoryBackend, RetrievalResult, ScenarioResult
from tests.eval.benchmark.fixtures import (
    AVOID_REPEATS_EXPECTED_SEEN,
    AVOID_REPEATS_QUERIES,
    PUBLISHED_TITLES,
)
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s2_avoid_repeats"
TOP_K = 8


def _recall(retrieved: list[str], expected: list[str]) -> float:
    """Soft recall: substring match (title in retrieved text or vice versa)."""
    if not expected:
        return 0.0
    hits = sum(
        1
        for exp in expected
        if any(exp.lower() in r.lower() or r.lower() in exp.lower() for r in retrieved)
    )
    return hits / len(expected)


async def run(backend: MemoryBackend, judge: BaseJudge, *, top_k: int = TOP_K) -> ScenarioResult:
    await backend.reset()

    for title in PUBLISHED_TITLES:
        await backend.insert(title)

    judge_runs: dict[str, list[float]] = {"relevance": [], "recall": [], "novelty": []}
    last_result: RetrievalResult | None = None

    for query in AVOID_REPEATS_QUERIES:
        # First retrieval
        r1 = await backend.retrieve(query, top_k=top_k)
        seen_set = set(r1.texts)

        # Second retrieval (same query — novelty check)
        r2 = await backend.retrieve(query, top_k=top_k)
        overlap = len(seen_set & set(r2.texts))
        novelty = max(0.0, 1.0 - overlap / max(top_k, 1))
        judge_runs["novelty"].append(round(novelty, 4))

        expected = AVOID_REPEATS_EXPECTED_SEEN.get(query, [])
        judge_runs["recall"].append(_recall(r1.texts, expected))

        scores = await judge.score_all_async(query, r1.texts)
        med = statistics.median(scores.get("relevance", [3.0]))
        judge_runs["relevance"].append(med)
        last_result = r1

    notes = (
        f"avg_recall={statistics.mean(judge_runs['recall']):.2f}, "
        f"avg_novelty={statistics.mean(judge_runs['novelty']):.2f}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=judge_runs,
        insert_result=None,
        retrieval_result=last_result,
        notes=notes,
    )
