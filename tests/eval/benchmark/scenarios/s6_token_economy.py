"""S6 — Context Economy (Token Efficiency).

Community metric: MemoryOS context budget, token compression from LongMemEval.
Pain point (#4): "memory fills the entire context window."

Inserts a user profile (15 facts), retrieves top-5 for each query.
Measures how many tokens the backend injects vs the full profile.
Quality_per_token = P@3 (lexical) / token_ratio — higher means more signal per token.

Metrics (all deterministic, no judge):
  token_compression  — 1 - retrieved_tokens / raw_tokens (higher = leaner context)
  p_at_3_lexical     — fraction of queries with relevant fact in top-3 (ATC-based)
  quality_per_token  — p@3 / (1 - token_compression) — signal density
"""

from __future__ import annotations

from tests.eval.benchmark._eval_utils import hit_rank_lexical
from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import EN_RETRIEVAL_ACCURACY, RetrievalAccuracyFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s6_token_economy"
TOP_K = 5


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    fixture: RetrievalAccuracyFixture = EN_RETRIEVAL_ACCURACY,
) -> ScenarioResult:
    """Insert profile, measure token compression vs retrieval quality."""
    await backend.reset()

    last_insert: InsertResult | None = None
    for fact in fixture.facts:
        last_insert = await backend.insert(fact)

    raw_token_count = sum(len(f.split()) for f in fixture.facts)

    total_retrieved_tokens = 0
    hits_at3 = 0

    for query in fixture.queries:
        relevant_text = fixture.relevant_fact_per_query[query]
        await backend.reset_session()
        result = await backend.retrieve(query, top_k=TOP_K)

        total_retrieved_tokens += sum(len(t.split()) for t in result.texts)
        rank = hit_rank_lexical(relevant_text, result.texts[:3])
        if rank is not None:
            hits_at3 += 1

    n_queries = len(fixture.queries)
    avg_retrieved = total_retrieved_tokens / max(n_queries, 1)
    token_compression = max(0.0, 1.0 - avg_retrieved / max(raw_token_count, 1))
    p_at_3 = hits_at3 / max(n_queries, 1)
    token_ratio = 1.0 - token_compression  # fraction of raw tokens injected
    quality_per_token = p_at_3 / max(token_ratio, 0.01)

    notes = (
        f"raw_tokens={raw_token_count}, "
        f"avg_retrieved_tokens={avg_retrieved:.0f}, "
        f"token_compression={token_compression:.1%}, "
        f"p@3_lexical={p_at_3:.2f}, "
        f"quality_per_token={quality_per_token:.3f}"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "token_compression": [token_compression, token_compression, token_compression],
            "p_at_3_lexical": [p_at_3, p_at_3, p_at_3],
            "quality_per_token": [quality_per_token, quality_per_token, quality_per_token],
        },
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
    return result_obj
