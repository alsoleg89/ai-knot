"""S9 — Multi-Agent Shared Pool: Publish & Recall.

Verifies that facts published to the shared pool by one agent can be
retrieved by a different agent with no private knowledge.

Flow:
  1. Agent A inserts N facts into its private namespace.
  2. Agent A publishes all facts to the shared pool.
  3. Agent B (empty private namespace) queries the shared pool.
  4. Measure how many expected facts Agent B can find.

Metrics (deterministic + optional judge):
  pool_recall        — fraction of expected facts found by Agent B via pool queries
  publish_count      — number of facts published (informational)
  relevance          — judge score on pool retrieval quality (1–5)
"""

from __future__ import annotations

import asyncio
import statistics

from tests.eval.benchmark.base import MultiAgentMemoryBackend, RetrievalResult, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s9_ma_pool_publish"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
    top_k: int = 3,
) -> ScenarioResult:
    await backend.reset()

    for fact in fixture.pool_facts:
        await backend.insert_for_agent("agent_a", fact)

    publish_count = await backend.publish_to_pool("agent_a")

    # Retrieve + judge all queries concurrently.
    async def _query(query: str, keyword: str) -> tuple[bool, RetrievalResult, list[float]]:
        r = await backend.pool_retrieve("agent_b", query, top_k=top_k)
        hit = any(keyword.lower() in t.lower() for t in r.texts)
        scores_dict = await judge.score_all_async(query, r.texts)
        return hit, r, scores_dict.get("relevance", [3.0])

    results = await asyncio.gather(*[_query(q, kw) for q, kw in fixture.pool_queries])

    found = sum(1 for hit, _, _ in results if hit)
    last_retrieval = results[-1][1] if results else None
    all_judge_scores = [s for _, _, scores in results for s in scores]

    pool_recall = found / max(len(fixture.pool_queries), 1)
    med_relevance = statistics.median(all_judge_scores) if all_judge_scores else 3.0

    notes = (
        f"published={publish_count}, "
        f"pool_recall={pool_recall:.2%} ({found}/{len(fixture.pool_queries)}), "
        f"relevance_median={med_relevance:.2f}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "pool_recall": [pool_recall],
            "relevance": all_judge_scores[:3]
            if len(all_judge_scores) >= 3
            else all_judge_scores + [3.0] * (3 - len(all_judge_scores)),
        },
        insert_result=None,
        retrieval_result=last_retrieval,
        notes=notes,
    )
