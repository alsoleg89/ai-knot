"""S3 — Feedback Learning.

Tests whether the backend correctly stores and retrieves feedback rules
so that an agent can avoid past mistakes and apply learned preferences.

Metrics:
  completeness      — does the backend surface the relevant rules? (judge, 1-5)
  rule_coverage     — keyword-match fraction of expected rules recalled (deterministic)
  semantic_coverage — cosine-similarity coverage (fair for dense backends)
"""

from __future__ import annotations

import statistics

from tests.eval.benchmark.backends.qdrant_emulator import _cosine, embed_batch
from tests.eval.benchmark.base import MemoryBackend, RetrievalResult, ScenarioResult
from tests.eval.benchmark.fixtures import (
    FEEDBACK_EXPECTED_RULES,
    FEEDBACK_HISTORY,
    FEEDBACK_QUERIES,
)
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s3_feedback_learning"
TOP_K = 5

# Keywords expected in the retrieved facts for each query
_EXPECTED_KEYWORDS: dict[str, list[str]] = {
    FEEDBACK_QUERIES[0]: ["long", "question", "emoji", "words", "300"],
    FEEDBACK_QUERIES[1]: ["code", "fenced", "block", "inline", "backtick"],
    FEEDBACK_QUERIES[2]: ["cta", "call", "action", "end", "summary"],
}

# 0.72: qwen2.5:7b cosine of paraphrases of the same rule clusters ~0.75–0.90;
# unrelated texts fall below 0.65.
_SEMANTIC_THRESHOLD = 0.72


def _rule_coverage(retrieved: list[str], keywords: list[str]) -> float:
    """Fraction of expected keywords found anywhere in retrieved texts."""
    if not keywords:
        return 1.0
    joined = " ".join(t.lower() for t in retrieved)
    hits = sum(1 for kw in keywords if kw.lower() in joined)
    return hits / len(keywords)


async def _semantic_coverage(
    retrieved: list[str], expected_rules: list[str], threshold: float = _SEMANTIC_THRESHOLD
) -> float:
    """Fraction of expected rules semantically covered by any retrieved text.

    Falls back to keyword _rule_coverage when embeddings are unavailable.
    """
    if not expected_rules or not retrieved:
        return 0.0
    try:
        all_texts = retrieved + expected_rules
        all_embs = await embed_batch(all_texts)
        r_embs = all_embs[: len(retrieved)]
        e_embs = all_embs[len(retrieved) :]
        hits = sum(
            1
            for e_emb in e_embs
            if any(_cosine(e_emb, r_emb) >= threshold for r_emb in r_embs)
        )
        return hits / len(expected_rules)
    except Exception:
        # Degrade to keyword coverage so Ollama downtime doesn't bias rankings
        keywords = [w for rule in expected_rules for w in rule.lower().split()]
        return _rule_coverage(retrieved, keywords)


async def run(backend: MemoryBackend, judge: BaseJudge, *, top_k: int = TOP_K) -> ScenarioResult:
    await backend.reset()

    # Insert feedback as "Query: ... \nFeedback: ..." text blocks
    for q, fb in FEEDBACK_HISTORY:
        await backend.insert(f"Query: {q}\nFeedback: {fb}")

    judge_runs: dict[str, list[float]] = {
        "completeness": [],
        "rule_coverage": [],
        "semantic_coverage": [],
    }
    last_result: RetrievalResult | None = None

    for query in FEEDBACK_QUERIES:
        result = await backend.retrieve(query, top_k=top_k)
        last_result = result

        keywords = _EXPECTED_KEYWORDS.get(query, [])
        judge_runs["rule_coverage"].append(_rule_coverage(result.texts, keywords))

        expected_rules = FEEDBACK_EXPECTED_RULES.get(query, [])
        judge_runs["semantic_coverage"].append(
            await _semantic_coverage(result.texts, expected_rules)
        )

        scores = await judge.score_all_async(query, result.texts)
        med = statistics.median(scores.get("completeness", [3.0]))
        judge_runs["completeness"].append(med)

    notes = (
        f"avg_rule_coverage={statistics.mean(judge_runs['rule_coverage']):.2f}, "
        f"avg_semantic_coverage={statistics.mean(judge_runs['semantic_coverage']):.2f}, "
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
