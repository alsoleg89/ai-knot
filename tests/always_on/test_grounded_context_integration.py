"""Contextual retrieval must not revive evidence lost through shared ancestors."""

import pytest

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AccessDenied, AlwaysOnMemory, StaleEvidence, WorkspaceMemory
from ai_knot.integrations.openai_agents import AiKnotAgentsMemory
from ai_knot.types import Fact

QUERY = (
    "Integrate lattice reduction tuning, hypergraph partitioning settings, "
    "and RTOS scheduling rules."
)


@pytest.mark.parametrize("followup", ["Continue.", "Продолжай."])
@pytest.mark.parametrize("change", ["revoke", "delete"])
def test_context_and_coverage_respect_revoked_ancestor(sql_storage, clock, followup, change):
    contents = (
        "Lattice reduction uses a BKZ block size of 40.",
        "Hypergraph partitioning uses a cut imbalance of 3 percent.",
        "RTOS scheduling uses priority ceilings for shared resources.",
    )
    sources = [
        Fact(
            text,
            origin_agent_id=f"publisher-{i}",
            source_snippets=[text],
            created_at=clock(),
            last_accessed=clock(),
            valid_from=clock(),
            visibility_scope="team:crypto" if i == 0 else "global",
        )
        for i, text in enumerate(contents)
    ]
    sql_storage.save("__shared__", sources)
    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock)
    workspace = WorkspaceMemory(sql_storage, "context-team")
    workspace.grant("writer", "maintain")
    for name in ("writer", "reader"):
        pool.grant_read(name, "team:crypto")
    writer = AlwaysOnMemory(
        KnowledgeBase("writer", storage=sql_storage, embed_url=""),
        pool=pool,
        workspace=workspace,
        clock=clock,
    )
    reader_kb = KnowledgeBase("reader", storage=sql_storage, embed_url="")
    reader = AlwaysOnMemory(reader_kb, pool=pool, clock=clock)
    shared = writer.derive_fact(
        "Lattice reduction deployment checklist uses block size 40.",
        namespace="__shared__",
        basis=writer.capture_evidence((("__shared__", sources[0].id),)),
    )
    private = reader.derive_fact(
        "Lattice reduction deployment checklist uses block size 40.",
        basis=reader.capture_evidence((("__shared__", shared.id),)),
    )
    memory = AiKnotAgentsMemory(reader_kb)
    query = memory.extract_query(
        [
            {"role": "user", "content": "Review lattice reduction deployment checklist."},
            {"role": "assistant", "content": "I will check the evidence."},
            {"role": "user", "content": followup},
        ]
    )
    assert "lattice" in query.lower()
    assert private.id in {fact.id for fact in reader_kb.recall_facts(query)}
    assert pool.recall(QUERY, "reader", top_k=4, now=clock())
    assert pool._last_recall_meta.coverage == 1.0
    basis = reader.capture_evidence((("reader", private.id),))
    intent = reader.create_intent(client_key="context-action", goal="Check deployment")
    lease = reader.claim(intent.id, worker_id="worker")
    action = reader.prepare_action(
        intent.id, lease_token=lease.token, action_key="check", request={}, basis=basis
    )

    if change == "revoke":
        pool.revoke_read("reader", "team:crypto")
    else:
        sql_storage.delete("__shared__", sources[0].id)

    assert private.id not in {fact.id for fact in reader_kb.recall_facts(query)}
    remaining = pool.recall(QUERY, "reader", top_k=4, now=clock())
    assert {fact.id for fact, _ in remaining} == {fact.id for fact in sources[1:]}
    assert pool._last_recall_meta.coverage == pytest.approx(2 / 3)
    assert pool._last_recall_meta.uncovered_facets == frozenset({"f0"})
    assert pool.last_recall_risk >= 1 / 3 - 1e-9
    with pytest.raises((AccessDenied, StaleEvidence)):
        reader.begin_action(action.id, lease_token=lease.token)
