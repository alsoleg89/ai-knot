"""S9 — Competing Documentation Sources (Pool with Conflicts + Supersession).

Verifies that the shared pool handles conflicting facts from multiple agents
and that CAS supersession surfaces the most current information.

Three agents publish documentation with deliberate conflicts:
  - Agent A publishes 4 facts (some outdated).
  - Agent B publishes 4 facts (includes corrections to A's outdated claims).
  - Agent C publishes 4 facts (includes deprecation notices).
  - Agent A then self-corrects one fact via add_structured (CAS supersession).

Agent D queries the pool — must find the NEWEST/correct answer for each topic.

Metrics:
  conflict_resolution      — fraction of conflicting queries returning the newer answer
  supersession_propagation — 1.0 if the CAS-updated slot surfaces the corrected value
  precision_at_3           — fraction of top-3 results per query that are relevant

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s9_ma_pool_publish"

_QUERIER = "agent_d"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    # Phase 1: Three agents publish their documentation (with conflicts).
    for fact in fixture.competing_facts_a:
        await backend.insert_for_agent("agent_a", fact)
    await backend.publish_to_pool("agent_a")

    for fact in fixture.competing_facts_b:
        await backend.insert_for_agent("agent_b", fact)
    await backend.publish_to_pool("agent_b")

    for fact in fixture.competing_facts_c:
        await backend.insert_for_agent("agent_c", fact)
    await backend.publish_to_pool("agent_c")

    # Phase 2: Agent A self-corrects one fact via CAS supersession.
    await backend.add_structured(
        "agent_a",
        fixture.competing_slot_v2,
        entity=fixture.competing_slot_entity,
        attribute=fixture.competing_slot_attribute,
    )
    await backend.publish_to_pool("agent_a")

    # Phase 3: Agent D queries — must find correct/newest answers.
    conflict_hits = 0
    precision_total = 0
    precision_relevant = 0

    for query, correct_kw, wrong_kw in fixture.competing_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=3)
        texts_lower = [t.lower() for t in r.texts]

        correct_found = any(correct_kw.lower() in t for t in texts_lower)
        wrong_found = any(wrong_kw.lower() in t for t in texts_lower) if wrong_kw else False

        if correct_found and not wrong_found:
            conflict_hits += 1

        # Precision: how many of top-3 are relevant to the query topic?
        for t in texts_lower:
            precision_total += 1
            if correct_kw.lower() in t:
                precision_relevant += 1

    n_queries = len(fixture.competing_queries)
    conflict_resolution = conflict_hits / n_queries

    # Check supersession: the CAS-updated slot must surface v2 keyword.
    r_slot = await backend.pool_retrieve(_QUERIER, fixture.competing_slot_v2[:40], top_k=3)
    supersession_propagation = 1.0 if any("3 minutes" in t.lower() for t in r_slot.texts) else 0.0

    precision_at_3 = precision_relevant / max(precision_total, 1)

    notes = (
        f"agents=3+supersession, facts=12+1, queries={n_queries}, "
        f"conflict_resolution={conflict_resolution:.0%}, "
        f"supersession_propagation={supersession_propagation:.0%}, "
        f"precision_at_3={precision_at_3:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "conflict_resolution": [conflict_resolution],
            "supersession_propagation": [supersession_propagation],
            "precision_at_3": [precision_at_3],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
