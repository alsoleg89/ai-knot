"""S7 — Temporal Consolidation.

Simulates a user whose facts evolve over time: 5 topics × 5 temporal versions
= 25 inserts, interleaved (v1 of all topics, then v2, …, v5).

Ideal backend tracks the *latest* state per topic and discards or down-ranks
stale versions. Extraction-based backends (ai_knot, mem0) should consolidate
evolving facts; verbatim backends (qdrant, baseline) store all 25 versions.

Metrics:
  consolidation_ratio      — 1 - count_stored / 25 (deterministic; higher = better
                             compression; qdrant/baseline expected ~0, ai_knot
                             and mem0 expected > 0.4)
  semantic_latest_recall   — max cosine similarity between any retrieved text and
                             the v5 (latest) ground-truth fact per query, averaged
                             across all 5 topics (0.0–1.0; higher = latest state
                             is surfaced despite compression)
  faithfulness             — judge score: retrieved snippets should not mix
                             contradictory old and new facts (1–5)
"""

from __future__ import annotations

import asyncio
import statistics

import httpx

from tests.eval.benchmark.backends.qdrant_emulator import _cosine, embed_batch
from tests.eval.benchmark.base import MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import BUNDLE_EN, LanguageBundle
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s7_consolidation"
TOP_K = 5
SEMANTIC_THRESHOLD = 0.72  # same as S2/S3


async def _semantic_latest_recall(retrieved_texts: list[str], latest_fact: str) -> float:
    """Max cosine similarity between any retrieved text and the v5 ground-truth fact.

    Returns the raw max score (0–1).  A score > SEMANTIC_THRESHOLD means the
    latest state is semantically present in the retrieval results.
    If Ollama is unavailable, returns 0.0 (graceful degradation).
    """
    if not retrieved_texts:
        return 0.0
    try:
        embs = await embed_batch(retrieved_texts + [latest_fact])
    except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout):
        return 0.0

    retrieved_embs = embs[:-1]
    latest_emb = embs[-1]

    if not any(latest_emb):
        return 0.0

    scores = [_cosine(re, latest_emb) for re in retrieved_embs if any(re)]
    return max(scores) if scores else 0.0


async def _estimate_stored_count(backend: MemoryBackend, n_total: int, queries: list[str]) -> int:
    """Return number of stored facts.

    Prefers count_stored() (exact).  Fallback: union of unique texts retrieved
    across all topic queries with top_k=n_total so every stored fact can surface
    in at least one query — fixes the undercount bug from using top_k=n_versions.
    """
    exact = await backend.count_stored()
    if exact is not None:
        return exact
    oversized_k = n_total + 5
    all_texts: set[str] = set()
    for q in queries:
        r = await backend.retrieve(q, top_k=oversized_k)
        all_texts.update(t.strip().lower() for t in r.texts)
    return len(all_texts)


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    bundle: LanguageBundle = BUNDLE_EN,
    top_k: int = TOP_K,
) -> ScenarioResult:
    await backend.reset()

    consolidation = bundle.consolidation
    last_insert = None
    for fact in consolidation.facts:
        last_insert = await backend.insert(fact)

    n_total = consolidation.n_topics * consolidation.n_versions  # 25
    n_topics = consolidation.n_topics  # 5

    # --- Consolidation ratio (deterministic) ---
    # Ideal: one fact per topic (n_topics). Worst: all n_total facts stored.
    # ratio = how much of the max possible compression was achieved:
    #   (n_total - stored) / (n_total - n_topics)
    # 0.0 = no compression, 1.0 = perfect (one fact per topic), clamped [0, 1].
    stored_count = await _estimate_stored_count(backend, n_total, consolidation.queries)
    _max_compression = max(1, n_total - n_topics)
    consolidation_ratio = min(1.0, max(0.0, (n_total - stored_count) / _max_compression))

    # --- Per-query: semantic_latest_recall + judge faithfulness ---
    sem_scores: list[float] = []
    faithfulness_runs: list[float] = []

    for query, latest_fact in zip(consolidation.queries, consolidation.latest_facts, strict=True):
        result = await backend.retrieve(query, top_k=top_k)

        # semantic_latest_recall (embed) and judge (LLM) use independent
        # resource pools — run concurrently to overlap embed + LLM inference.
        sem_score, judge_scores = await asyncio.gather(
            _semantic_latest_recall(result.texts, latest_fact),
            judge.score_all_async(query, result.texts),
        )
        sem_scores.append(sem_score)
        faithfulness_runs.extend(judge_scores.get("faithfulness", [3.0]))

    avg_sem_recall = statistics.mean(sem_scores) if sem_scores else 0.0
    med_faithfulness = statistics.median(faithfulness_runs) if faithfulness_runs else 3.0

    notes = (
        f"lang={bundle.language}, "
        f"facts_inserted={n_total}, "
        f"estimated_stored={stored_count}, "
        f"consolidation_ratio={consolidation_ratio:.2%}, "
        f"semantic_latest_recall={avg_sem_recall:.2f}, "
        f"faithfulness_median={med_faithfulness:.2f}"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "consolidation_ratio": [consolidation_ratio] * 3,
            "semantic_latest_recall": [avg_sem_recall] * 3,
            "faithfulness": faithfulness_runs[:3]
            if len(faithfulness_runs) >= 3
            else faithfulness_runs + [3.0] * (3 - len(faithfulness_runs)),
        },
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
    result_obj.language = bundle.language
    return result_obj
