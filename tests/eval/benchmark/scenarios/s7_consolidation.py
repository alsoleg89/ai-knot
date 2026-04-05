"""S7 — Temporal Consolidation.

Simulates a user whose facts evolve over time: 5 topics × 5 temporal versions
= 25 inserts, interleaved (v1 of all topics, then v2, …, v5).

Ideal backend tracks the *latest* state per topic and discards or down-ranks
stale versions. Extraction-based backends (ai_knot, mem0) should consolidate
evolving facts; verbatim backends (qdrant, baseline) store all 25 versions.

Metrics:
  consolidation_ratio  — 1 - count_stored / 25 (deterministic; higher = better
                         compression; qdrant/baseline expected ~0.0, ai_knot
                         and mem0 expected > 0.5)
  latest_recall        — fraction of queries where the v5 keyword appears in
                         the top-k results (deterministic; 0.0–1.0)
  faithfulness         — judge score: retrieved snippets should not mix old and
                         new contradictory facts (1-5; lower if stale facts leak)
"""

from __future__ import annotations

import asyncio
import statistics

from tests.eval.benchmark.base import MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import CONSOLIDATION
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s7_consolidation"
TOP_K = 5


async def _estimate_stored_count(backend: MemoryBackend) -> int:
    """Return number of stored facts.

    Prefers count_stored() (exact).  Fallback: union of unique texts retrieved
    across all topic queries — an upper-bound proxy for verbatim backends.
    """
    exact = await backend.count_stored()
    if exact is not None:
        return exact
    # Retrieve with top_k > n_versions so all versions of a topic can surface.
    oversized_k = CONSOLIDATION.n_versions + 5
    all_texts: set[str] = set()
    for q in CONSOLIDATION.queries:
        r = await backend.retrieve(q, top_k=oversized_k)
        all_texts.update(t.strip().lower() for t in r.texts)
    return len(all_texts)


def _latest_recall(retrieved_texts: list[str], keywords: list[str]) -> float:
    """1.0 if any keyword appears in any retrieved text, else 0.0."""
    joined = " ".join(retrieved_texts).lower()
    return 1.0 if any(kw in joined for kw in keywords) else 0.0


async def run(backend: MemoryBackend, judge: BaseJudge, *, top_k: int = TOP_K) -> ScenarioResult:
    await backend.reset()

    last_insert = None
    for fact in CONSOLIDATION.facts:
        last_insert = await backend.insert(fact)

    # --- Consolidation ratio (deterministic) ---
    stored_count = await _estimate_stored_count(backend)
    n_total = CONSOLIDATION.n_topics * CONSOLIDATION.n_versions  # 25
    consolidation_ratio = max(0.0, 1.0 - stored_count / n_total)

    # --- Per-query: latest_recall + judge faithfulness ---
    recall_scores: list[float] = []
    faithfulness_runs: list[float] = []

    for query in CONSOLIDATION.queries:
        keywords = CONSOLIDATION.latest_keywords[query]
        result = await backend.retrieve(query, top_k=top_k)

        # _latest_recall is synchronous; wrap so asyncio.gather can pair it
        # with the async judge call and overlap scheduling.
        async def _recall_coro(r: list[str] = result.texts, kw: list[str] = keywords) -> float:
            return _latest_recall(r, kw)

        recall_score, judge_scores = await asyncio.gather(
            _recall_coro(),
            judge.score_all_async(query, result.texts),
        )
        recall_scores.append(recall_score)
        faithfulness_runs.extend(judge_scores.get("faithfulness", [3.0]))

    avg_latest_recall = statistics.mean(recall_scores) if recall_scores else 0.0
    med_faithfulness = statistics.median(faithfulness_runs) if faithfulness_runs else 3.0

    notes = (
        f"facts_inserted={n_total}, "
        f"estimated_stored={stored_count}, "
        f"consolidation_ratio={consolidation_ratio:.2%}, "
        f"latest_recall={avg_latest_recall:.2%}, "
        f"faithfulness_median={med_faithfulness:.2f}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "consolidation_ratio": [consolidation_ratio] * 3,
            "latest_recall": [avg_latest_recall] * 3,
            "faithfulness": faithfulness_runs[:3]
            if len(faithfulness_runs) >= 3
            else faithfulness_runs + [3.0] * (3 - len(faithfulness_runs)),
        },
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
