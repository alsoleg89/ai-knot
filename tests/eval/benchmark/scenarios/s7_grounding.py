"""S7 — Grounding Rate (Hallucination Resistance).

Community metric: HaluMem hallucination detection, RAGAS faithfulness.
Pain point (#3): "LLM extraction adds details that were never stated."

After inserting facts, retrieves all stored content via broad queries.
For each retrieved text, checks max ATC overlap against all original inserted facts.
Verbatim backends score ~1.0; extraction backends may score lower if LLM paraphrases
or hallucinates. Low grounding (<0.3) = hallucination risk.

Metrics (all deterministic, no judge):
  mean_grounding      — avg max-ATC across all retrieved texts (higher = more faithful)
  hallucination_rate  — fraction of retrieved texts with max-ATC < 0.3
"""

from __future__ import annotations

import statistics

from tests.eval.benchmark._eval_utils import best_atc_against
from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import EN_RETRIEVAL_ACCURACY, RetrievalAccuracyFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s7_grounding"
TOP_K = 5
_HALLUCINATION_THRESHOLD = 0.3

# Broad queries that together cover most profile domains
_GROUNDING_QUERIES = [
    "What tools and technologies does Alex use?",
    "What are Alex's engineering policies and rules?",
    "What are Alex's team structure and work arrangements?",
    "What databases and data platforms does Alex use?",
    "What are Alex's workflow and deployment practices?",
]


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    fixture: RetrievalAccuracyFixture = EN_RETRIEVAL_ACCURACY,
) -> ScenarioResult:
    """Insert facts, retrieve broadly, check grounding against original text."""
    await backend.reset()

    last_insert: InsertResult | None = None
    for fact in fixture.facts:
        last_insert = await backend.insert(fact)

    all_retrieved: list[str] = []
    for query in _GROUNDING_QUERIES:
        await backend.reset_session()
        result = await backend.retrieve(query, top_k=TOP_K)
        all_retrieved.extend(result.texts)

    # Deduplicate retrieved texts (same fact retrieved by multiple queries)
    seen: set[str] = set()
    unique_retrieved: list[str] = []
    for text in all_retrieved:
        key = text.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_retrieved.append(text)

    if not unique_retrieved:
        notes = "no texts retrieved"
        result_obj = ScenarioResult(
            scenario_id=SCENARIO_ID,
            backend_name=backend.name,
            judge_scores={
                "mean_grounding": [0.0, 0.0, 0.0],
                "hallucination_rate": [1.0, 1.0, 1.0],
            },
            insert_result=last_insert,
            retrieval_result=None,
            notes=notes,
        )
        return result_obj

    grounding_scores: list[float] = []
    hallucinations = 0

    for retrieved in unique_retrieved:
        score = best_atc_against(retrieved, fixture.facts)
        grounding_scores.append(score)
        if score < _HALLUCINATION_THRESHOLD:
            hallucinations += 1

    mean_grounding = statistics.mean(grounding_scores)
    hallucination_rate = hallucinations / max(len(unique_retrieved), 1)

    notes = (
        f"facts_inserted={len(fixture.facts)}, "
        f"unique_retrieved={len(unique_retrieved)}, "
        f"mean_grounding={mean_grounding:.3f}, "
        f"hallucination_rate={hallucination_rate:.2f}"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "mean_grounding": [mean_grounding, mean_grounding, mean_grounding],
            "hallucination_rate": [hallucination_rate, hallucination_rate, hallucination_rate],
        },
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
    return result_obj
