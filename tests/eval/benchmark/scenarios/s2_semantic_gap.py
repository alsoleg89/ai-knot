"""S2 — Semantic Gap (Paraphrase Recall).

Community metric: BEIR paraphrase robustness, MIRACL cross-lingual gap.
Pain point: "BM25 doesn't find a fact if the query is phrased differently."

Inserts verbatim facts, queries with paraphrases (different wording, same meaning).
Measures gap between semantic recall (embed-based) and lexical recall (ATC-based).

Metrics:
  lexical_recall_at3   — fraction of paraphrase queries with lexical hit in top-3
  semantic_recall_at3  — fraction of paraphrase queries with semantic hit in top-3
  semantic_gap         — semantic_recall - lexical_recall (>0 = dense helps)
"""

from __future__ import annotations

from ai_knot.embedder import cosine as _cosine
from tests.eval.benchmark._eval_utils import atc_score, maybe_embed_batch
from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import EN_PARAPHRASE, ParaphraseFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s2_semantic_gap"
TOP_K = 3
_SEMANTIC_THRESHOLD = 0.65
_LEXICAL_THRESHOLD = 0.45


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    fixture: ParaphraseFixture = EN_PARAPHRASE,
) -> ScenarioResult:
    """Insert verbatim facts, query with paraphrases, measure lexical vs semantic recall."""
    await backend.reset()

    last_insert: InsertResult | None = None
    for fact in fixture.verbatim_facts:
        last_insert = await backend.insert(fact)

    # Embed verbatim facts for semantic comparison
    fact_embs = await maybe_embed_batch(fixture.verbatim_facts)

    lexical_hits = 0
    semantic_hits = 0
    n = len(fixture.paraphrase_queries)

    for i, (paraphrase, verbatim) in enumerate(
        zip(fixture.paraphrase_queries, fixture.verbatim_facts, strict=False)
    ):
        await backend.reset_session()
        result = await backend.retrieve(paraphrase, top_k=TOP_K)

        # Lexical hit: any retrieved text has significant token overlap with verbatim fact
        lex_hit = any(
            max(atc_score(verbatim, r), atc_score(r, verbatim)) >= _LEXICAL_THRESHOLD
            for r in result.texts
        )
        if lex_hit:
            lexical_hits += 1

        # Semantic hit: any retrieved text embeds close to verbatim fact
        sem_hit = lex_hit  # fallback to lexical
        if fact_embs and result.texts:
            retrieved_embs = await maybe_embed_batch(result.texts)
            if retrieved_embs:
                best_cos = max(_cosine(fact_embs[i], re) for re in retrieved_embs)
                sem_hit = best_cos >= _SEMANTIC_THRESHOLD
        if sem_hit:
            semantic_hits += 1

    lexical_recall = lexical_hits / max(n, 1)
    semantic_recall = semantic_hits / max(n, 1)
    gap = semantic_recall - lexical_recall

    notes = (
        f"n_pairs={n}, "
        f"lexical_recall@3={lexical_recall:.2f}, "
        f"semantic_recall@3={semantic_recall:.2f}, "
        f"gap={gap:+.2f}"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "lexical_recall_at3": [lexical_recall, lexical_recall, lexical_recall],
            "semantic_recall_at3": [semantic_recall, semantic_recall, semantic_recall],
            "semantic_gap": [gap, gap, gap],
        },
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
    return result_obj
