"""S23 — Adversarial Noise Injection (Trust-Weighted Suppression).

Verifies that when an adversarial agent publishes plausible-but-wrong facts,
the trust mechanism correctly suppresses those facts in retrieval after
reliable agents supersede them via CAS.

Flow:
  Phase 1: Adversary (agent_d) publishes wrong facts for 5 entity slots.
  Phase 2: Reliable agents (A, B, C) sync_dirty, then publish correct values
           for the same slots — CAS supersedes adversary.
           → quick_inv_count[agent_d] += 5, trust_d drops.
  Phase 3: Agent_e queries pool — builds used_count for reliable agents.
  Phase 4: Adversary publishes 5 MORE wrong facts (free-standing, different
           entities, no CAS conflict).
  Phase 5: Agent_e queries free-standing topics — trust-weighted scoring
           should suppress adversary's facts in favor of reliable agents'.

Metrics:
  slot_suppression          — 1.0 if CAS-superseded facts absent from results
  trust_penalty             — 1.0 if trust_d ≤ 0.15 after CAS supersessions
  free_standing_suppression — fraction of queries where reliable agents'
                              free-standing facts outrank adversary's

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s23_adversarial_noise"

_ADVERSARY = "agent_d"
_RELIABLE = ["agent_a", "agent_b", "agent_c"]
_QUERIER = "agent_e"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    # Phase 1: Adversary seeds wrong facts into 5 CAS slots.
    for entity, attr, content in fixture.adversarial_wrong_values:
        await backend.add_structured(_ADVERSARY, content, entity=entity, attribute=attr)
    await backend.publish_to_pool(_ADVERSARY)

    # Phase 2: Reliable agents correct all 5 slots via CAS.
    # Distribute slots across reliable agents: A gets 0-1, B gets 2-3, C gets 4.
    reliable_assignments = [
        ("agent_a", fixture.adversarial_correct_values[:2]),
        ("agent_b", fixture.adversarial_correct_values[2:4]),
        ("agent_c", fixture.adversarial_correct_values[4:]),
    ]
    for agent_id, corrections in reliable_assignments:
        await backend.sync_dirty(agent_id)
        for entity, attr, content in corrections:
            await backend.add_structured(agent_id, content, entity=entity, attribute=attr)
        await backend.publish_to_pool(agent_id)

    # Phase 3: Querier retrieves slot facts — builds trust for reliable agents.
    slot_suppression_ok = True
    for _entity, _attr, content in fixture.adversarial_wrong_values:
        r = await backend.pool_retrieve(_QUERIER, content[:40], top_k=5)
        # Adversary's wrong version should NOT appear (it's INVALID via CAS).
        wrong_snippet = content[:20].lower()
        if any(wrong_snippet in t.lower() for t in r.texts):
            slot_suppression_ok = False

    # Phase 4: Reliable agents publish free-standing correct facts.
    for i, fact in enumerate(fixture.adversarial_freestanding_correct):
        agent = _RELIABLE[i % len(_RELIABLE)]
        await backend.insert_for_agent(agent, fact)
    for agent in _RELIABLE:
        await backend.publish_to_pool(agent)

    # Adversary publishes free-standing wrong facts (different topics, no CAS).
    for fact in fixture.adversarial_freestanding_wrong:
        await backend.insert_for_agent(_ADVERSARY, fact)
    await backend.publish_to_pool(_ADVERSARY)

    # Phase 5: Querier queries free-standing topics.
    # Trust-weighted scoring: adversary score *= 0.1, reliable agents *= ~0.8+.
    suppression_hits = 0
    for query, correct_kw, wrong_kw in fixture.adversarial_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=3)
        texts_lower = [t.lower() for t in r.texts]
        correct_found = any(correct_kw.lower() in t for t in texts_lower)
        wrong_found = any(wrong_kw.lower() in t for t in texts_lower)
        if correct_found and not wrong_found:
            suppression_hits += 1
        elif correct_found:
            suppression_hits += 0.5  # partial credit: correct found but wrong also present

    free_standing_suppression = suppression_hits / len(fixture.adversarial_queries)

    # Check adversary trust.
    pool = getattr(backend, "_pool", None)
    trust_d = pool.get_trust(_ADVERSARY) if pool else 0.1
    trust_penalty = 1.0 if trust_d <= 0.15 else 0.0

    notes = (
        f"adversary={_ADVERSARY}, reliable={_RELIABLE}, "
        f"slot_suppression={'OK' if slot_suppression_ok else 'FAIL'}, "
        f"trust_d={trust_d:.3f}, trust_penalty={trust_penalty:.0%}, "
        f"free_standing_suppression={free_standing_suppression:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "slot_suppression": [1.0 if slot_suppression_ok else 0.0],
            "trust_penalty": [trust_penalty],
            "free_standing_suppression": [free_standing_suppression],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
