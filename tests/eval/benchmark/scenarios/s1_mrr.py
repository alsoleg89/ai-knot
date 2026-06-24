"""S1 — MRR & Precision@k.

Community metric: Mean Reciprocal Rank (TREC, BEIR, MIRACL benchmarks).
Pain point: "my agent can't find the right fact" — retrieval quality.

Each query has one ground-truth relevant fact (RetrievalAccuracyFixture).
A retrieved text "hits" if ATC token containment >= 0.5 (deterministic, no judge).
When Ollama is available, also computes semantic_mrr via cosine >= 0.65.

Metrics:
  lexical_mrr  — MRR using ATC token containment (always available)
  semantic_mrr — MRR using cosine similarity (Ollama required; falls back to lexical)
  p_at_1       — Precision@1
  p_at_3       — Precision@3
  p_at_5       — Precision@5
"""

from __future__ import annotations

from tests.eval.benchmark._eval_utils import (
    hit_rank_lexical,
    hit_rank_semantic,
    maybe_embed_batch,
    mrr,
    precision_at_k,
)
from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import EN_RETRIEVAL_ACCURACY, RetrievalAccuracyFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s1_mrr"
TOP_K = 5


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    fixture: RetrievalAccuracyFixture = EN_RETRIEVAL_ACCURACY,
) -> ScenarioResult:
    """Insert profile facts, evaluate MRR & P@k for each query."""
    await backend.reset()

    last_insert: InsertResult | None = None
    for fact in fixture.facts:
        last_insert = await backend.insert(fact)

    queries = fixture.queries
    relevant_texts = [fixture.relevant_fact_per_query[q] for q in queries]

    # --- Embed for semantic evaluation (optional) ---
    all_texts = queries + relevant_texts
    embs = await maybe_embed_batch(all_texts)
    if embs:
        q_embs = embs[: len(queries)]
        rel_embs = embs[len(queries) :]
    else:
        q_embs = None
        rel_embs = None

    ranks_lexical: list[int | None] = []
    ranks_semantic: list[int | None] = []

    for i, (query, rel_text) in enumerate(zip(queries, relevant_texts, strict=False)):
        # Reset session novelty state so each query evaluates independently.
        # MRR measures per-query retrieval quality, not cross-query novelty.
        await backend.reset_session()
        result = await backend.retrieve(query, top_k=TOP_K)

        # Lexical hit (always available)
        rank_lex = hit_rank_lexical(rel_text, result.texts)
        ranks_lexical.append(rank_lex)

        # Semantic hit (requires Ollama)
        if q_embs and rel_embs:
            retrieved_embs_list = await maybe_embed_batch(result.texts)
            if retrieved_embs_list:
                rank_sem = hit_rank_semantic(rel_embs[i], retrieved_embs_list)
                ranks_semantic.append(rank_sem)
            else:
                ranks_semantic.append(rank_lex)
        else:
            ranks_semantic.append(rank_lex)

    lexical_mrr = mrr(ranks_lexical)
    semantic_mrr = mrr(ranks_semantic)
    # Use lexical ranks for P@k (always available, deterministic).
    p1 = precision_at_k(ranks_lexical, 1)
    p3 = precision_at_k(ranks_lexical, 3)
    p5 = precision_at_k(ranks_lexical, 5)

    notes = (
        f"n_queries={len(queries)}, "
        f"lexical_mrr={lexical_mrr:.3f}, semantic_mrr={semantic_mrr:.3f}, "
        f"p@1={p1:.2f}, p@3={p3:.2f}, p@5={p5:.2f}, "
        f"embed={'yes' if q_embs else 'no (ATC fallback)'}"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "lexical_mrr": [lexical_mrr, lexical_mrr, lexical_mrr],
            "semantic_mrr": [semantic_mrr, semantic_mrr, semantic_mrr],
            "p_at_1": [p1, p1, p1],
            "p_at_3": [p3, p3, p3],
            "p_at_5": [p5, p5, p5],
        },
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
    return result_obj
