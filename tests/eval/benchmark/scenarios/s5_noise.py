"""S5 — Noise Tolerance.

Community metric: MemoryOS signal-to-noise ratio, HaluMem contamination.
Pain point (#2): "noise in context degrades LLM answer quality."

Inserts 200 noise facts + 5 signal facts. Measures whether signal facts
surface in top-3 for their corresponding queries.

Metrics (all deterministic, no judge):
  signal_recall_at3  — fraction of signal queries with signal hit in top-3
  contamination_at3  — fraction of retrieved items across all queries that are noise
  snr                — signal_recall / max(contamination, 0.01)
"""

from __future__ import annotations

from tests.eval.benchmark._eval_utils import atc_score
from tests.eval.benchmark.base import InsertResult, MemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import EN_NOISE_TOLERANCE, NoiseToleranceFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s5_noise"
TOP_K = 3
_SIGNAL_ATC_THRESHOLD = 0.45


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    fixture: NoiseToleranceFixture = EN_NOISE_TOLERANCE,
) -> ScenarioResult:
    """Insert noise + signal, measure signal recall and contamination."""
    await backend.reset()

    # Insert noise first (so it's "older"), then signal (more recent)
    last_insert: InsertResult | None = None
    for fact in fixture.noise_facts:
        last_insert = await backend.insert(fact)
    for fact in fixture.signal_facts:
        last_insert = await backend.insert(fact)

    signal_hits = 0
    total_retrieved = 0
    noise_retrieved = 0

    for i, (_, query) in enumerate(zip(fixture.signal_facts, fixture.signal_queries, strict=False)):
        await backend.reset_session()
        result = await backend.retrieve(query, top_k=TOP_K)
        texts = result.texts[:TOP_K]
        total_retrieved += len(texts)

        # For each retrieved text compute ATC once against all signal facts to
        # distinguish signal hits (current query's fact) from noise (no match).
        hit = False
        for r in texts:
            scores = [max(atc_score(sf, r), atc_score(r, sf)) for sf in fixture.signal_facts]
            is_signal = max(scores, default=0.0) >= _SIGNAL_ATC_THRESHOLD
            if not is_signal:
                noise_retrieved += 1
            elif scores[i] >= _SIGNAL_ATC_THRESHOLD:
                hit = True
        if hit:
            signal_hits += 1

    n = len(fixture.signal_queries)
    signal_recall = signal_hits / max(n, 1)
    contamination = noise_retrieved / max(total_retrieved, 1)
    snr = signal_recall / max(contamination, 0.01)

    notes = (
        f"n_signal={len(fixture.signal_facts)}, "
        f"n_noise={len(fixture.noise_facts)}, "
        f"signal_recall@3={signal_recall:.2f}, "
        f"contamination@3={contamination:.2f}, "
        f"snr={snr:.2f}"
    )

    result_obj = ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "signal_recall_at3": [signal_recall, signal_recall, signal_recall],
            "contamination_at3": [contamination, contamination, contamination],
            "snr": [snr, snr, snr],
        },
        insert_result=last_insert,
        retrieval_result=None,
        notes=notes,
    )
    return result_obj
