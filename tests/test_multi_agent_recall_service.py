"""Tests for SharedPoolRecallService — experimental V3 pipeline guard."""

from __future__ import annotations

import inspect

from ai_knot.multi_agent import recall_service
from ai_knot.multi_agent.models import ExplorationMode, QueryAnalysis, RetrievalIntent
from ai_knot.multi_agent.recall_service import SharedPoolRecallService
from ai_knot.types import Fact


def _facts() -> list[Fact]:
    out: list[Fact] = []
    for aid, content in [
        ("a1", "The crypto module does lattice reduction for encrypted computation."),
        ("a2", "The graph module does hypergraph bisection for data partitioning."),
        ("a3", "The firmware module does RTOS ceilings for microcontroller scheduling."),
    ]:
        f = Fact(content=content)
        f.origin_agent_id = aid
        out.append(f)
    return out


class TestRecallV3Experimental:
    """recall_v3 is an experimental, non-default entry point: it must still
    function (smoke) but must not be wired into the production recall path.
    """

    def test_recall_v3_smoke(self) -> None:
        svc = SharedPoolRecallService()
        query = "integrate lattice reduction and hypergraph bisection"
        analysis = QueryAnalysis(
            raw_query=query,
            intent=RetrievalIntent.ASSEMBLY,
            exploration_mode=ExplorationMode.BALANCED,
        )
        results = svc.recall_v3(
            query,
            analysis=analysis,
            requesting_agent_id="querier",
            active_facts=_facts(),
            top_k=3,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_recall_v3_not_in_production_path(self) -> None:
        # Guard against silently re-wiring the experimental pipeline: the
        # production recall() path must not invoke recall_v3.
        src = inspect.getsource(SharedPoolRecallService.recall)
        assert "recall_v3" not in src
        # And it really is unused elsewhere in the module's production methods.
        retrieve_src = inspect.getsource(recall_service.SharedPoolRecallService._retrieve_per_facet)
        assert "recall_v3" not in retrieve_src
