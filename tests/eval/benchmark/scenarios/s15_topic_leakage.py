"""S15 — Topic Channel Cross-Contamination (Leakage Test).

Verifies that per-channel pool retrieval returns facts only from the
requested channel — zero cross-channel leakage.

Unlike S12 (which tests publish gating), S15 focuses on retrieval isolation:
after both channels are fully populated, queries scoped to one channel must
NOT surface results from the other.

Flow:
  1. Agent A inserts DEVOPS_FACTS and publishes them as "devops" channel.
  2. Agent B inserts FRONTEND_FACTS and publishes them as "frontend" channel.
  3. Agent C (no private knowledge) queries each channel:
     a. Devops query via pool_retrieve_for_channel(channel="devops")
        → must return devops facts, must NOT contain frontend keywords.
     b. Frontend query via pool_retrieve_for_channel(channel="frontend")
        → must return frontend facts, must NOT contain devops keywords.
  4. Compute leakage: fraction of channel queries where the wrong-domain
     keywords appear in results.

Metrics (deterministic):
  channel_isolation  — 1.0 - cross_channel_leakage_rate
                        1.0 = perfect isolation, 0.0 = total leakage
  devops_recall      — 1.0 if devops query returned ≥1 devops-domain result
  frontend_recall    — 1.0 if frontend query returned ≥1 frontend-domain result

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark._eval_utils import has_domain_hit as _has_domain_hit
from tests.eval.benchmark.base import MultiAgentMemoryBackend, RetrievalResult, ScenarioResult
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s15_topic_leakage"

_DEVOPS_FACTS = [
    "Kubernetes clusters are managed via Helm charts and ArgoCD.",
    "Terraform provisions all cloud infrastructure on AWS.",
    "Prometheus and Grafana handle observability and alerting.",
    "Istio service mesh enforces mTLS between all microservices.",
    "GitHub Actions pipelines run on self-hosted ARM runners.",
]

_FRONTEND_FACTS = [
    "React 18 with TypeScript is the standard for all web UIs.",
    "Vite is the build tool and Tailwind CSS handles styling.",
    "Playwright is used for end-to-end browser testing.",
    "Next.js 14 powers all customer-facing server-rendered pages.",
    "Storybook documents every shared component in the design system.",
]

# Keywords unique to each domain — used to detect leakage.
_DEVOPS_KEYWORDS = {"kubernetes", "terraform", "prometheus", "istio", "argocd"}
_FRONTEND_KEYWORDS = {"react", "vite", "tailwind", "playwright", "next.js", "storybook"}

_DEVOPS_QUERIES = [
    "How are Kubernetes deployments managed?",
    "What tool provisions cloud infrastructure?",
]

_FRONTEND_QUERIES = [
    "What framework is used for web UI components?",
    "How is end-to-end browser testing done?",
]


async def _query_channel(
    backend: MultiAgentMemoryBackend,
    queries: list[str],
    channel: str,
    own_kw: set[str],
    leak_kw: set[str],
) -> tuple[int, int]:
    """Return (hits, leaks) across all queries for one channel."""
    hits = leaks = 0
    for query in queries:
        r: RetrievalResult = await backend.pool_retrieve_for_channel(
            "agent_c", query, top_k=5, topic_channel=channel
        )
        if _has_domain_hit(r.texts, own_kw):
            hits += 1
        if _has_domain_hit(r.texts, leak_kw):
            leaks += 1
    return hits, leaks


async def run(backend: MultiAgentMemoryBackend, judge: BaseJudge) -> ScenarioResult:
    await backend.reset()

    for fact in _DEVOPS_FACTS:
        await backend.insert_for_agent_with_meta(
            "agent_a", fact, topic_channel="devops", importance=0.8
        )
    await backend.publish_to_pool("agent_a")

    for fact in _FRONTEND_FACTS:
        await backend.insert_for_agent_with_meta(
            "agent_b", fact, topic_channel="frontend", importance=0.8
        )
    await backend.publish_to_pool("agent_b")

    devops_hits, devops_leaks = await _query_channel(
        backend, _DEVOPS_QUERIES, "devops", _DEVOPS_KEYWORDS, _FRONTEND_KEYWORDS
    )
    frontend_hits, frontend_leaks = await _query_channel(
        backend, _FRONTEND_QUERIES, "frontend", _FRONTEND_KEYWORDS, _DEVOPS_KEYWORDS
    )

    total_queries = len(_DEVOPS_QUERIES) + len(_FRONTEND_QUERIES)
    total_leaks = devops_leaks + frontend_leaks
    channel_isolation = max(0.0, 1.0 - total_leaks / max(total_queries, 1))
    devops_recall = devops_hits / max(len(_DEVOPS_QUERIES), 1)
    frontend_recall = frontend_hits / max(len(_FRONTEND_QUERIES), 1)

    notes = (
        f"devops_hits={devops_hits}/{len(_DEVOPS_QUERIES)}, "
        f"frontend_hits={frontend_hits}/{len(_FRONTEND_QUERIES)}, "
        f"leaks={total_leaks}/{total_queries}, "
        f"channel_isolation={channel_isolation:.2%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "channel_isolation": [channel_isolation],
            "devops_recall": [devops_recall],
            "frontend_recall": [frontend_recall],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
