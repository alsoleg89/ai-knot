"""S1 — Profile Retrieval.

Measures token reduction and retrieval quality when retrieving facts from
a user profile. Tests the core value proposition: inject only relevant
facts rather than the whole profile into the LLM context.

Metrics:
  relevance    — does the retrieved text answer the query? (judge, 1-5)
  completeness — does it cover all needed aspects? (judge, 1-5)
  token_reduction — (raw_tokens - retrieved_tokens) / raw_tokens (deterministic)
"""

from __future__ import annotations

import statistics

from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import PROFILE
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s1_profile_retrieval"
TOP_K = 5


async def run(backend: MemoryBackend, judge: BaseJudge, *, top_k: int = TOP_K) -> ScenarioResult:
    """Insert profile facts, retrieve for each query, score with judge."""
    await backend.reset()

    # Insert all profile facts
    last_insert: InsertResult | None = None
    for fact in PROFILE.raw_facts:
        last_insert = await backend.insert(fact)

    raw_token_count = sum(len(f.split()) for f in PROFILE.raw_facts)

    judge_runs: dict[str, list[float]] = {
        "relevance": [],
        "completeness": [],
        "token_reduction": [],
    }

    for query in PROFILE.queries:
        result = await backend.retrieve(query, top_k=top_k)
        retrieved_tokens = sum(len(t.split()) for t in result.texts)
        reduction = max(0.0, (raw_token_count - retrieved_tokens) / max(raw_token_count, 1))
        judge_runs["token_reduction"].append(round(reduction, 4))

        scores = await judge.score_all_async(query, result.texts)
        for metric in ("relevance", "completeness"):
            med = statistics.median(scores.get(metric, [3.0]))
            judge_runs[metric].append(med)

    notes = (
        f"raw_tokens={raw_token_count}, "
        f"avg_token_reduction={statistics.mean(judge_runs['token_reduction']):.1%}, "
        f"facts_stored={last_insert.facts_stored if last_insert else 0}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=judge_runs,
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
