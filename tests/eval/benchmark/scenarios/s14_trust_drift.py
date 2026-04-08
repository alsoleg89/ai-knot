"""S14 — Trust Drift (Auto-Trust Convergence After High Invalidation Rate).

Verifies that the SharedMemoryPool's auto-trust mechanism correctly penalises
an agent whose published facts are rapidly superseded by multiple other agents,
and that trust begins to recover once that agent publishes correct information.

Trust formula (Marsh 1994):
    trust = min(1, used / published) × (1 − quick_invalidation_rate)

Flow:
  1. Agent A publishes N_FACTS structured facts (entity+attribute slots) to the pool.
  2. Agents B, C, D each insert corrected values for a subset of Agent A's slots
     concurrently (asyncio.gather), then publish sequentially — each triggers
     quick-invalidation penalties for Agent A.
  3. With all N_FACTS superseded, Agent A's quick_inv_rate → 1.0 and trust drops
     to the floor (≤ TRUST_LOW_THRESHOLD).
  4. Agent A publishes corrected versions of all slots (self-correction).
     Trust should begin to recover above the floor.

Metrics (deterministic):
  trust_floor_reached — 1.0 if Agent A's trust ≤ TRUST_LOW_THRESHOLD after step 3
  trust_recovery      — 1.0 if Agent A's trust after step 4 > trust after step 3

Only runs against MultiAgentMemoryBackend (AiKnotMultiAgentBackend exposes _pool).
"""

from __future__ import annotations

import asyncio

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s14_trust_drift"

_TRUST_LOW_THRESHOLD = 0.25

# 9 slots split evenly across 3 superseding agents (3 each).
_SLOTS = [
    ("Mia Torres", "annual_salary"),
    ("Mia Torres", "job_title"),
    ("Mia Torres", "office_location"),
    ("Liam Chen", "annual_salary"),
    ("Liam Chen", "job_title"),
    ("Liam Chen", "office_location"),
    ("Liam Chen", "team_name"),
    ("Liam Chen", "start_date"),
    ("Liam Chen", "tech_lead"),
]

_AGENT_A_VALUES = [
    "Mia Torres's annual salary is $88,000.",
    "Mia Torres's job title is Junior SRE.",
    "Mia Torres works in the Dublin office.",
    "Liam Chen's annual salary is $105,000.",
    "Liam Chen's job title is Senior Engineer.",
    "Liam Chen works in the Munich office.",
    "Liam Chen is on the Observability team.",
    "Liam Chen's start date was April 2020.",
    "Liam Chen is not a tech lead.",
]

# Agent B supersedes slots 0-2.
_AGENT_B_VALUES = [
    "Mia Torres's annual salary is $120,000 after promotion.",
    "Mia Torres's job title is Senior SRE after promotion.",
    "Mia Torres works in the Toronto office after relocation.",
]

# Agent C supersedes slots 3-5.
_AGENT_C_VALUES = [
    "Liam Chen's annual salary is $145,000 after promotion.",
    "Liam Chen's job title is Principal Engineer after promotion.",
    "Liam Chen works in the Seoul office after relocation.",
]

# Agent D supersedes slots 6-8.
_AGENT_D_VALUES = [
    "Liam Chen is now on the Platform Reliability team after reorg.",
    "Liam Chen joined the company in June 2019 (corrected record).",
    "Liam Chen is the tech lead for the Platform Reliability team.",
]

_N_FACTS = len(_SLOTS)


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    # Agent A publishes all stale values.
    await asyncio.gather(
        *[
            backend.add_structured("agent_a", val, entity=e, attribute=a)
            for (e, a), val in zip(_SLOTS, _AGENT_A_VALUES, strict=True)
        ]
    )
    await backend.publish_to_pool("agent_a")

    # Agents B, C, D insert corrections concurrently (they've processed their turns).
    await asyncio.gather(
        *[
            backend.add_structured("agent_b", val, entity=e, attribute=a)
            for (e, a), val in zip(_SLOTS[:3], _AGENT_B_VALUES, strict=True)
        ],
        *[
            backend.add_structured("agent_c", val, entity=e, attribute=a)
            for (e, a), val in zip(_SLOTS[3:6], _AGENT_C_VALUES, strict=True)
        ],
        *[
            backend.add_structured("agent_d", val, entity=e, attribute=a)
            for (e, a), val in zip(_SLOTS[6:], _AGENT_D_VALUES, strict=True)
        ],
    )

    # Publish sequentially: each supersedes Agent A's facts within the quick-inv window.
    await backend.publish_to_pool("agent_b")
    await backend.publish_to_pool("agent_c")
    await backend.publish_to_pool("agent_d")

    pool = getattr(backend, "_pool", None)
    if pool is None:
        return ScenarioResult(
            scenario_id=SCENARIO_ID,
            backend_name=backend.name,
            judge_scores={},
            insert_result=None,
            retrieval_result=None,
            notes="backend does not expose _pool.get_trust(); skipped",
        )

    trust_a_floor = pool.get_trust("agent_a")
    trust_floor_reached = 1.0 if trust_a_floor <= _TRUST_LOW_THRESHOLD else 0.0

    # Step 4: Agent A self-corrects — publishes verified values for all 9 slots.
    await asyncio.gather(
        *[
            backend.add_structured("agent_a", val, entity=e, attribute=a)
            for (e, a), val in zip(_SLOTS, fixture.agent_a_corrected_values, strict=True)
        ]
    )
    await backend.publish_to_pool("agent_a")

    # Step 5: Post-correction retrieval — agent_e queries the pool so that
    # agent_a's corrected facts accumulate used_count hits.  Without these
    # queries, trust stays at the floor because the formula requires
    # used_count > 0 for quality > 0.
    _RECOVERY_QUERIES = [
        "What is Mia Torres's salary?",
        "What is Mia Torres's job title?",
        "Where does Mia Torres work?",
        "What is Liam Chen's salary?",
        "What is Liam Chen's title?",
        "Where does Liam Chen work?",
        "What team is Liam Chen on?",
        "When did Liam Chen start?",
        "Is Liam Chen a tech lead?",
    ]
    for q in _RECOVERY_QUERIES:
        await backend.pool_retrieve("agent_e", q, top_k=3)

    trust_a_recovered = pool.get_trust("agent_a")
    trust_recovery = 1.0 if trust_a_recovered > trust_a_floor else 0.0

    notes = (
        f"n_facts={_N_FACTS}, superseding_agents=3 (b+c+d), "
        f"agent_a_trust={trust_a_floor:.3f} (expected≤{_TRUST_LOW_THRESHOLD}), "
        f"trust_floor_reached={trust_floor_reached:.0%}, "
        f"trust_after_recovery={trust_a_recovered:.3f}, "
        f"trust_recovery={trust_recovery:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "trust_floor_reached": [trust_floor_reached],
            "trust_recovery": [trust_recovery],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
