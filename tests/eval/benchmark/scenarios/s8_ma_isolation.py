"""S8 — Multi-Agent Private Namespace Isolation.

Verifies that each agent's private knowledge base is invisible to other agents.
Two agents learn facts from entirely different domains:
  Agent A: DevOps / infrastructure
  Agent B: Python / coding style

Sub-tests:
  A) Self-recall: each agent should retrieve its own domain facts reliably.
  B) Cross-contamination: each agent should NOT retrieve the other agent's facts.

Metrics (deterministic, no judge calls):
  self_recall        — fraction of self-domain queries that return ≥1 relevant result
  isolation_score    — 1 - cross_contamination_rate
                       1.0 = perfect isolation, 0.0 = no isolation

Only runs against MultiAgentMemoryBackend (skipped for single-agent backends).
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s8_ma_isolation"


def _keyword_set(facts: list[str]) -> set[str]:
    """Build a set of significant keywords (length > 4) from a list of fact strings."""
    return {w.lower() for fact in facts for w in fact.split() if len(w) > 4}


def _has_domain_hit(texts: list[str], domain_keywords: set[str]) -> bool:
    """Return True if any retrieved text shares a keyword with the precomputed set."""
    return any({w.lower().strip(".,;:") for w in text.split()} & domain_keywords for text in texts)


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
    top_k: int = 3,
) -> ScenarioResult:
    await backend.reset()

    for fact in fixture.agent_a_facts:
        await backend.insert_for_agent("agent_a", fact)
    for fact in fixture.agent_b_facts:
        await backend.insert_for_agent("agent_b", fact)

    # Precompute keyword sets once — reused across all query loops.
    kw_a = _keyword_set(fixture.agent_a_facts)
    kw_b = _keyword_set(fixture.agent_b_facts)

    self_hits = 0
    self_total = len(fixture.agent_a_queries) + len(fixture.agent_b_queries)

    for query in fixture.agent_a_queries:
        r = await backend.retrieve_for_agent("agent_a", query, top_k=top_k)
        if _has_domain_hit(r.texts, kw_a):
            self_hits += 1

    for query in fixture.agent_b_queries:
        r = await backend.retrieve_for_agent("agent_b", query, top_k=top_k)
        if _has_domain_hit(r.texts, kw_b):
            self_hits += 1

    self_recall = self_hits / max(self_total, 1)

    # Agent A queries B's domain and vice versa — should return no hits.
    cross_hits = 0
    cross_total = len(fixture.agent_b_queries) + len(fixture.agent_a_queries)

    for query in fixture.agent_b_queries:
        r = await backend.retrieve_for_agent("agent_a", query, top_k=top_k)
        if _has_domain_hit(r.texts, kw_b):
            cross_hits += 1

    for query in fixture.agent_a_queries:
        r = await backend.retrieve_for_agent("agent_b", query, top_k=top_k)
        if _has_domain_hit(r.texts, kw_a):
            cross_hits += 1

    isolation_score = max(0.0, min(1.0, 1.0 - cross_hits / max(cross_total, 1)))

    notes = (
        f"self_recall={self_recall:.2%} ({self_hits}/{self_total}), "
        f"cross_hits={cross_hits}/{cross_total}, "
        f"isolation_score={isolation_score:.2%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "self_recall": [self_recall],
            "isolation_score": [isolation_score],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
