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
    ("Alex Chen", "annual_salary"),
    ("Alex Chen", "job_title"),
    ("Alex Chen", "office_location"),
    ("Jordan Lee", "annual_salary"),
    ("Jordan Lee", "job_title"),
    ("Jordan Lee", "office_location"),
    ("Jordan Lee", "team_name"),
    ("Jordan Lee", "start_date"),
    ("Jordan Lee", "tech_lead"),
]

_AGENT_A_VALUES = [
    "Alex Chen's annual salary is $95,000.",
    "Alex Chen's job title is Junior Engineer.",
    "Alex Chen works in the London office.",
    "Jordan Lee's annual salary is $110,000.",
    "Jordan Lee's job title is Senior Engineer.",
    "Jordan Lee works in the Berlin office.",
    "Jordan Lee is on the Data team.",
    "Jordan Lee's start date was January 2021.",
    "Jordan Lee is not a tech lead.",
]

# Agent B supersedes slots 0-2.
_AGENT_B_VALUES = [
    "Alex Chen's annual salary is $115,000 after promotion.",
    "Alex Chen's job title is Senior Engineer after promotion.",
    "Alex Chen works in the New York office after relocation.",
]

# Agent C supersedes slots 3-5.
_AGENT_C_VALUES = [
    "Jordan Lee's annual salary is $140,000 after promotion.",
    "Jordan Lee's job title is Staff Engineer after promotion.",
    "Jordan Lee works in the Singapore office after relocation.",
]

# Agent D supersedes slots 6-8.
_AGENT_D_VALUES = [
    "Jordan Lee is now on the ML Platform team after reorg.",
    "Jordan Lee joined the company in March 2020 (corrected record).",
    "Jordan Lee is the tech lead for the ML Platform team.",
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
        "What is Alex Chen's salary?",
        "What is Alex Chen's job title?",
        "Where does Alex Chen work?",
        "What is Jordan Lee's salary?",
        "What is Jordan Lee's title?",
        "Where does Jordan Lee work?",
        "What team is Jordan Lee on?",
        "When did Jordan Lee start?",
        "Is Jordan Lee a tech lead?",
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
