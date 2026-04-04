"""S3 — Feedback Learning.

Tests whether the backend correctly stores and retrieves feedback rules
so that an agent can avoid past mistakes and apply learned preferences.

Metrics:
  completeness — does the backend surface the relevant rules? (judge, 1-5)
  rule_coverage — keyword-match fraction of expected rules recalled (deterministic)
"""

from __future__ import annotations

import statistics

from tests.eval.benchmark.base import MemoryBackend, RetrievalResult, ScenarioResult
from tests.eval.benchmark.fixtures import FEEDBACK_HISTORY, FEEDBACK_QUERIES
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s3_feedback_learning"
TOP_K = 5

# Keywords expected in the retrieved facts for each query
_EXPECTED_KEYWORDS: dict[str, list[str]] = {
    FEEDBACK_QUERIES[0]: ["long", "question", "emoji", "words", "300"],
    FEEDBACK_QUERIES[1]: ["code", "fenced", "block", "inline", "backtick"],
    FEEDBACK_QUERIES[2]: ["cta", "call", "action", "end", "summary"],
}


def _rule_coverage(retrieved: list[str], keywords: list[str]) -> float:
    """Fraction of expected keywords found anywhere in retrieved texts."""
    if not keywords:
        return 1.0
    joined = " ".join(t.lower() for t in retrieved)
    hits = sum(1 for kw in keywords if kw.lower() in joined)
    return hits / len(keywords)


async def run(backend: MemoryBackend, judge: BaseJudge, *, top_k: int = TOP_K) -> ScenarioResult:
    await backend.reset()

    # Insert feedback as "Query: ... \nFeedback: ..." text blocks
    for q, fb in FEEDBACK_HISTORY:
        await backend.insert(f"Query: {q}\nFeedback: {fb}")

    judge_runs: dict[str, list[float]] = {"completeness": [], "rule_coverage": []}
    last_result: RetrievalResult | None = None

    for query in FEEDBACK_QUERIES:
        result = await backend.retrieve(query, top_k=top_k)
        last_result = result

        keywords = _EXPECTED_KEYWORDS.get(query, [])
        judge_runs["rule_coverage"].append(_rule_coverage(result.texts, keywords))

        scores = await judge.score_all_async(query, result.texts)
        med = statistics.median(scores.get("completeness", [3.0]))
        judge_runs["completeness"].append(med)

    notes = (
        f"avg_rule_coverage={statistics.mean(judge_runs['rule_coverage']):.2f}, "
        f"feedback_items={len(FEEDBACK_HISTORY)}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=judge_runs,
        insert_result=None,
        retrieval_result=last_result,
        notes=notes,
    )
