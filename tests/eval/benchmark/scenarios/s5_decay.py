"""S5 — Decay (ai-knot specific).

Tests Ebbinghaus forgetting curve: facts that have not been accessed recently
should score lower in retrieval after simulated time passage.

Only ai-knot supports tick_decay() — other backends use the default no-op,
so this scenario measures the delta between pre- and post-decay relevance.

Metrics:
  relevance        — judge score on post-decay retrieval (1-5)
  retention_delta  — post_score - pre_score (negative = decay worked, deterministic)
"""

from __future__ import annotations

import statistics

from tests.eval.benchmark.base import MemoryBackend, RetrievalResult, ScenarioResult
from tests.eval.benchmark.fixtures import BUNDLE_EN, LanguageBundle
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s5_decay"
TOP_K = 5
DECAY_HOURS = 336.0  # 2 weeks


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    bundle: LanguageBundle = BUNDLE_EN,
    top_k: int = TOP_K,
) -> ScenarioResult:
    await backend.reset()

    for fact in bundle.profile.raw_facts:
        await backend.insert(fact)

    query = bundle.profile.queries[0]

    # Pre-decay retrieval
    pre_result: RetrievalResult = await backend.retrieve(query, top_k=top_k)
    pre_scores_raw = await judge.score_all_async(query, pre_result.texts)
    pre_relevance = statistics.median(pre_scores_raw.get("relevance", [3.0]))

    # Simulate time passage
    await backend.tick_decay(hours=DECAY_HOURS)

    # Post-decay retrieval
    post_result: RetrievalResult = await backend.retrieve(query, top_k=top_k)
    post_scores_raw = await judge.score_all_async(query, post_result.texts)
    post_relevance = statistics.median(post_scores_raw.get("relevance", [3.0]))

    retention_delta = post_relevance - pre_relevance

    notes = (
        f"lang={bundle.language}, "
        f"pre_relevance={pre_relevance:.2f}, "
        f"post_relevance={post_relevance:.2f}, "
        f"retention_delta={retention_delta:+.2f}, "
        f"decay_hours={DECAY_HOURS}, "
        f"decay_supported={'yes' if hasattr(backend, 'tick_decay') else 'no'}"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "relevance": post_scores_raw.get("relevance", [post_relevance] * 3),
            "retention_delta": [retention_delta, retention_delta, retention_delta],
        },
        insert_result=None,
        retrieval_result=post_result,
        notes=notes,
    )
    result_obj.language = bundle.language
    return result_obj
