"""S3 — Staleness Resistance.

Community metric: LongMemEval temporal faithfulness, MemGPT staleness.
Pain point (#1): "system returns an outdated fact instead of the current one."

Inserts 5 topics × 5 versions (interleaved, same as S7 old). After inserting all 25,
queries each topic. Freshness@1 = top-1 is semantically closer to v5 than to v1-v3.

Metrics (all deterministic, no judge):
  freshness_at1       — fraction of queries where top-1 is the latest version
  staleness_rate      — fraction where top-1 is an older version
  consolidation_ratio — 1 - stored_count / n_inserted (memory compression)
"""

from __future__ import annotations

from ai_knot.embedder import cosine as _cosine
from tests.eval.benchmark._eval_utils import atc_score, maybe_embed_batch
from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import BUNDLE_EN, ConsolidationFixture, LanguageBundle
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s3_staleness"
TOP_K = 3
_ATC_FRESH_THRESHOLD = 0.5


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    bundle: LanguageBundle = BUNDLE_EN,
) -> ScenarioResult:
    """Insert temporal facts, measure staleness resistance."""
    await backend.reset()
    fixture: ConsolidationFixture = bundle.consolidation

    last_insert: InsertResult | None = None
    for fact in fixture.facts:
        last_insert = await backend.insert(fact)

    n_inserted = len(fixture.facts)
    stored = await backend.count_stored()
    estimated_stored = stored if stored is not None else n_inserted
    consolidation_ratio = max(0.0, 1.0 - estimated_stored / max(n_inserted, 1))

    # Embed latest facts for semantic freshness check
    latest_embs = await maybe_embed_batch(fixture.latest_facts)

    # Identify old facts per topic (all versions except v5)
    n_topics = fixture.n_topics
    n_versions = fixture.n_versions
    old_facts_by_topic: list[list[str]] = [[] for _ in range(n_topics)]
    for v in range(n_versions - 1):  # versions 0 .. n_versions-2
        for t in range(n_topics):
            old_facts_by_topic[t].append(fixture.facts[v * n_topics + t])

    old_embs_by_topic: list[list[list[float]] | None] = []
    if latest_embs:
        for t in range(n_topics):
            old_e = await maybe_embed_batch(old_facts_by_topic[t])
            old_embs_by_topic.append(old_e)
    else:
        old_embs_by_topic = [None] * n_topics

    freshness_count = 0
    staleness_count = 0

    for i, (query, latest_text) in enumerate(
        zip(fixture.queries, fixture.latest_facts, strict=False)
    ):
        await backend.reset_session()
        result = await backend.retrieve(query, top_k=TOP_K)
        if not result.texts:
            staleness_count += 1
            continue

        top1 = result.texts[0]

        # Lexical freshness: top-1 has high token overlap with latest fact
        lex_fresh = (
            atc_score(latest_text, top1) >= _ATC_FRESH_THRESHOLD
            or atc_score(top1, latest_text) >= _ATC_FRESH_THRESHOLD
        )

        sem_fresh = lex_fresh  # fallback

        if latest_embs and old_embs_by_topic[i]:
            top1_embs = await maybe_embed_batch([top1])
            if top1_embs:
                top1_emb = top1_embs[0]
                cos_latest = _cosine(top1_emb, latest_embs[i])
                cos_old_max = max(
                    (_cosine(top1_emb, oe) for oe in old_embs_by_topic[i]),  # type: ignore[union-attr]
                    default=0.0,
                )
                sem_fresh = cos_latest > cos_old_max

        is_fresh = sem_fresh if latest_embs else lex_fresh
        if is_fresh:
            freshness_count += 1
        else:
            staleness_count += 1

    n_q = len(fixture.queries)
    freshness_at1 = freshness_count / max(n_q, 1)
    staleness_rate = staleness_count / max(n_q, 1)

    notes = (
        f"lang={bundle.language}, "
        f"facts_inserted={n_inserted}, "
        f"estimated_stored={estimated_stored}, "
        f"consolidation_ratio={consolidation_ratio:.1%}, "
        f"freshness@1={freshness_at1:.2f}, "
        f"staleness_rate={staleness_rate:.2f}"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "freshness_at1": [freshness_at1, freshness_at1, freshness_at1],
            "staleness_rate": [staleness_rate, staleness_rate, staleness_rate],
            "consolidation_ratio": [consolidation_ratio, consolidation_ratio, consolidation_ratio],
        },
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
    result_obj.language = bundle.language
    return result_obj
