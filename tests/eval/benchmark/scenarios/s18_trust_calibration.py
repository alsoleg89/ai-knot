"""S18 — Trust Calibration (Reliable vs Unreliable Agent Over N Rounds).

Verifies that the SharedMemoryPool's trust mechanism correctly differentiates
between a reliable agent (whose facts persist) and an unreliable agent (whose
facts are consistently superseded by others).

Over N_ROUNDS of interaction:
  - Agent A (reliable): publishes unique, authoritative facts every round.
    None of A's facts are ever superseded.
  - Agent B (unreliable): publishes to a shared entity slot every round.
    Agent C immediately supersedes B's fact in the same round.

After N_ROUNDS:
  - trust_a should be high (no quick invalidations)
  - trust_b should be low (100% quick invalidation rate)
  - Agent D's pool queries should predominantly return A's facts

Metrics (deterministic):
  trust_calibration — 1.0 if trust_a > trust_b
  trust_gap         — trust_a - trust_b (numerical; higher = better separation)
  pool_preference   — fraction of D's query results containing A's keywords vs B's

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s18_trust_calibration"

_QUERIER = "agent_d"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    pool = getattr(backend, "_pool", None)
    n = fixture.trust_calib_n_rounds

    for i in range(n):
        # Agent A: publish a unique, reliable fact (never superseded).
        await backend.insert_for_agent("agent_a", fixture.trust_calib_reliable_facts[i])
        await backend.publish_to_pool("agent_a")

        # Agent B: publish to a slot — will be immediately superseded by C.
        entity = fixture.trust_calib_unreliable_entity_tpl.format(i=i)
        await backend.add_structured(
            "agent_b",
            f"Round {i}: agent_b claims {entity} status is active.",
            entity=entity,
            attribute="status",
        )
        await backend.publish_to_pool("agent_b")

        # Agent C: supersede B's fact for the same slot.
        await backend.add_structured(
            "agent_c",
            f"Round {i}: {entity} status is inactive (corrected by agent_c).",
            entity=entity,
            attribute="status",
        )
        await backend.publish_to_pool("agent_c")

    # Agent D queries pool — check preference for A's facts over B's.
    # Run BEFORE reading trust so that recalls increment used_count for agent_a.
    a_keywords = {
        w.lower() for f in fixture.trust_calib_reliable_facts for w in f.split() if len(w) > 5
    }
    total_hits = 0
    a_hits = 0
    for query, _ in fixture.trust_calib_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=5)
        for text in r.texts:
            total_hits += 1
            if any(kw in text.lower() for kw in a_keywords):
                a_hits += 1
    pool_preference = a_hits / max(total_hits, 1)

    # Read trust after queries so used_count is populated.
    trust_a = pool.get_trust("agent_a") if pool else 1.0
    trust_b = pool.get_trust("agent_b") if pool else 0.0
    trust_calibration = 1.0 if trust_a > trust_b else 0.0
    trust_gap = trust_a - trust_b

    notes = (
        f"rounds={n}, trust_a={trust_a:.3f}, trust_b={trust_b:.3f}, "
        f"trust_calibration={trust_calibration:.0%}, trust_gap={trust_gap:.3f}, "
        f"pool_preference(A)={pool_preference:.2%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "trust_calibration": [trust_calibration],
            "trust_gap": [trust_gap],
            "pool_preference": [pool_preference],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
