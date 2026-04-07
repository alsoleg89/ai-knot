"""S14 — Trust Drift (Auto-Trust Convergence After High Invalidation Rate).

Verifies that the SharedMemoryPool's auto-trust mechanism correctly penalises
an agent whose published facts are repeatedly superseded by others soon after
publishing.  According to the Marsh (1994) trust formula used by ai-knot:

    trust = min(1, used / published) × (1 − quick_invalidation_rate)

When another agent quickly supersedes N facts from Agent A, Agent A's
quick_inv_rate → 1.0 and trust → 0.1 (floor).

Flow:
  1. Agent A publishes N_FACTS structured facts (entity+attribute) to the pool.
  2. Agent B immediately supersedes each of Agent A's facts with an updated value.
     (These supersessions happen within _QUICK_INV_WINDOW_S, triggering penalties.)
  3. Measure Agent A's trust score from the pool.

Metrics (deterministic):
  trust_floor_reached — 1.0 if Agent A's trust ≤ TRUST_LOW_THRESHOLD, else 0.0

Only runs against MultiAgentMemoryBackend (specifically AiKnotMultiAgentBackend,
which exposes `_pool` with `get_trust()`).
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s14_trust_drift"

_N_FACTS = 8
_TRUST_LOW_THRESHOLD = 0.25  # trust must drop to or below this after max invalidation

# Pairs of (entity, attribute) for slot-addressed CAS.
_SLOTS = [
    ("Alex Chen", "annual_salary"),
    ("Alex Chen", "job_title"),
    ("Alex Chen", "office_location"),
    ("Alex Chen", "team_name"),
    ("Jordan Lee", "annual_salary"),
    ("Jordan Lee", "job_title"),
    ("Jordan Lee", "office_location"),
    ("Jordan Lee", "team_name"),
]

# Agent A's "stale" values — will be immediately superseded by Agent B.
_AGENT_A_VALUES = [
    "Alex Chen's annual salary is $95,000.",
    "Alex Chen's job title is Junior Engineer.",
    "Alex Chen works in the London office.",
    "Alex Chen is on the Platform team.",
    "Jordan Lee's annual salary is $110,000.",
    "Jordan Lee's job title is Senior Engineer.",
    "Jordan Lee works in the Berlin office.",
    "Jordan Lee is on the Data team.",
]

# Agent B's corrections — published immediately after Agent A.
_AGENT_B_VALUES = [
    "Alex Chen's annual salary is $115,000 after promotion.",
    "Alex Chen's job title is Senior Engineer after promotion.",
    "Alex Chen works in the New York office after relocation.",
    "Alex Chen is on the Infrastructure team after reorg.",
    "Jordan Lee's annual salary is $140,000 after promotion.",
    "Jordan Lee's job title is Staff Engineer after promotion.",
    "Jordan Lee works in the Singapore office after relocation.",
    "Jordan Lee is on the ML Platform team after reorg.",
]


async def run(backend: MultiAgentMemoryBackend, judge: BaseJudge) -> ScenarioResult:
    await backend.reset()

    # Agent A publishes stale values.
    for i, (entity, attribute) in enumerate(_SLOTS[:_N_FACTS]):
        await backend.add_structured(
            "agent_a",
            _AGENT_A_VALUES[i],
            entity=entity,
            attribute=attribute,
        )
    await backend.publish_to_pool("agent_a")

    # Agent B immediately supersedes all of Agent A's facts (same slots).
    for i, (entity, attribute) in enumerate(_SLOTS[:_N_FACTS]):
        await backend.add_structured(
            "agent_b",
            _AGENT_B_VALUES[i],
            entity=entity,
            attribute=attribute,
        )
    await backend.publish_to_pool("agent_b")

    # Retrieve Agent A's trust from the pool (AiKnotMultiAgentBackend exposes _pool).
    pool = getattr(backend, "_pool", None)
    if pool is None:
        # Trust API not available — record skipped, no scores emitted (shows as — in report).
        return ScenarioResult(
            scenario_id=SCENARIO_ID,
            backend_name=backend.name,
            judge_scores={},
            insert_result=None,
            retrieval_result=None,
            notes="backend does not expose _pool.get_trust(); skipped",
        )

    trust_a = pool.get_trust("agent_a")
    trust_floor_reached = 1.0 if trust_a <= _TRUST_LOW_THRESHOLD else 0.0

    notes = (
        f"n_facts={_N_FACTS}, "
        f"agent_a_trust={trust_a:.3f} (expected≤{_TRUST_LOW_THRESHOLD}), "
        f"trust_floor_reached={trust_floor_reached:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={"trust_floor_reached": [trust_floor_reached]},
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
