"""S2 — Avoid Repeats.

Tests whether the backend surfaces relevant content while helping avoid
recommending already-published topics. Measures recall of known-published
titles (so the agent can warn "this topic was covered before") and novelty.

Metrics:
  relevance       — are the retrieved titles relevant to the query? (judge, 1-5)
  recall          — fraction of expected titles found in top-k (lexical substring)
  semantic_recall — same, but via cosine similarity (fair for dense backends)
  novelty         — 1 - (overlap with first retrieval) / top_k (deterministic)
"""

from __future__ import annotations

import asyncio
import statistics

from tests.eval.benchmark.backends.qdrant_emulator import _cosine, embed_batch
from tests.eval.benchmark.base import MemoryBackend, RetrievalResult, ScenarioResult
from tests.eval.benchmark.fixtures import (
    AVOID_REPEATS_EXPECTED_SEEN,
    AVOID_REPEATS_QUERIES,
    PUBLISHED_TITLES,
)
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s2_avoid_repeats"
TOP_K = 8
# 0.72: empirically, qwen2.5:7b cosine of paraphrases of the same title
# clusters around 0.75–0.90; unrelated titles fall below 0.65.
_SEMANTIC_THRESHOLD = 0.72


def _recall(retrieved: list[str], expected: list[str]) -> float:
    """Lexical recall: substring match (title in retrieved text or vice versa)."""
    if not expected:
        return 0.0
    hits = sum(
        1
        for exp in expected
        if any(exp.lower() in r.lower() or r.lower() in exp.lower() for r in retrieved)
    )
    return hits / len(expected)


async def _semantic_recall(
    retrieved: list[str], expected: list[str], threshold: float = _SEMANTIC_THRESHOLD
) -> float:
    """Semantic recall: cosine similarity via Ollama embeddings.

    Fair metric for dense backends that return paraphrases rather than
    exact substrings. Falls back to lexical _recall() if embeddings fail
    (Ollama down) so the metric doesn't unfairly penalise any backend.
    """
    if not expected or not retrieved:
        return 0.0
    try:
        all_texts = retrieved + expected
        all_embs = await embed_batch(all_texts)
        r_embs = all_embs[: len(retrieved)]
        e_embs = all_embs[len(retrieved) :]
        hits = sum(
            1 for e_emb in e_embs if any(_cosine(e_emb, r_emb) >= threshold for r_emb in r_embs)
        )
        return hits / len(expected)
    except Exception:
        return _recall(retrieved, expected)


async def run(backend: MemoryBackend, judge: BaseJudge, *, top_k: int = TOP_K) -> ScenarioResult:
    await backend.reset()

    for title in PUBLISHED_TITLES:
        await backend.insert(title)

    judge_runs: dict[str, list[float]] = {
        "relevance": [],
        "recall": [],
        "semantic_recall": [],
        "novelty": [],
    }
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
        # semantic_recall (embed semaphore) and judge (LLM, no semaphore) use independent
        # resource pools — run them concurrently to overlap embed + LLM inference time.
        sem_score, scores = await asyncio.gather(
            _semantic_recall(r1.texts, expected),
            judge.score_all_async(query, r1.texts),
        )
        judge_runs["semantic_recall"].append(sem_score)
        med = statistics.median(scores.get("relevance", [3.0]))
        judge_runs["relevance"].append(med)
        last_result = r1

    notes = (
        f"avg_recall={statistics.mean(judge_runs['recall']):.2f}, "
        f"avg_semantic_recall={statistics.mean(judge_runs['semantic_recall']):.2f}, "
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
