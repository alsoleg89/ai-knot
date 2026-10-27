"""Public pool coverage follows eligible evidence, on both SQL backends."""

from datetime import UTC, datetime, timedelta

import pytest

from ai_knot.pool import SharedMemoryPool
from ai_knot.storage.base import StorageBackend
from ai_knot.types import Fact, MemoryType

NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)
QUERY = (
    "Integrate lattice reduction tuning, hypergraph partitioning settings, "
    "and RTOS scheduling rules."
)
CONTENTS = (
    "Lattice reduction uses a BKZ block size of 40.",
    "Hypergraph partitioning uses a cut imbalance of 3 percent.",
    "RTOS scheduling uses priority ceilings for shared resources.",
)


def _fact(content: str, agent: str) -> Fact:
    return Fact(
        content=content,
        origin_agent_id=agent,
        source_snippets=[content],
        valid_from=NOW - timedelta(days=1),
        created_at=NOW - timedelta(days=1),
    )


@pytest.mark.parametrize(
    ("available", "top_k", "expected_coverage", "returned"),
    [(3, 1, 1 / 3, 1), (2, 3, 2 / 3, 2), (0, 3, 0.0, 0), (3, 3, 1.0, 3)],
)
def test_pool_preserves_real_facet_coverage(
    sql_storage: StorageBackend, available: int, top_k: int, expected_coverage: float, returned: int
) -> None:
    facts = [_fact(text, f"a{i}") for i, text in enumerate(CONTENTS[:available])]
    # Keep a visible but irrelevant row even for the no-evidence case; positive
    # rank-only RRF must not manufacture facet relevance.
    facts.append(_fact("The cafeteria serves seasonal vegetarian lunches.", "food"))
    sql_storage.save("__shared__", facts)
    pool = SharedMemoryPool(sql_storage)

    results = pool.recall(QUERY, "reader", top_k=top_k, now=NOW)

    meta = pool._last_recall_meta
    assert meta is not None
    assert meta.coverage == pytest.approx(expected_coverage)
    assert len(results) == meta.returned == returned
    assert len(meta.covered_facets) == round(expected_coverage * 3)
    assert meta.covered_facets | meta.uncovered_facets == frozenset({"f0", "f1", "f2"})
    assert not meta.covered_facets & meta.uncovered_facets
    assert meta.low_coverage is (expected_coverage < 0.5)
    assert meta.should_abstain is (expected_coverage < 0.5)
    assert meta.unsupported_answer_risk == pytest.approx(1 - expected_coverage)
    assert all(fact.origin_agent_id != "food" for fact, _ in results)


@pytest.mark.parametrize("excluded", ["unsupported", "episodic", "expired", "future"])
def test_ineligible_fact_cannot_fill_missing_facet(
    sql_storage: StorageBackend, excluded: str
) -> None:
    facts = [_fact(text, f"a{i}") for i, text in enumerate(CONTENTS)]
    last = facts[-1]
    if excluded == "unsupported":
        last.supported = False
    elif excluded == "episodic":
        last.type = MemoryType.EPISODIC
    elif excluded == "expired":
        last.valid_until = NOW
    else:
        last.valid_from = NOW + timedelta(seconds=1)
    sql_storage.save("__shared__", facts)
    pool = SharedMemoryPool(sql_storage)

    results = pool.recall(QUERY, "reader", top_k=3, now=NOW)

    assert {fact.id for fact, _ in results} == {fact.id for fact in facts[:2]}
    assert pool._last_recall_meta is not None
    assert pool._last_recall_meta.coverage == pytest.approx(2 / 3)
    assert pool._last_recall_meta.uncovered_facets == frozenset({"f2"})


def test_pool_temporal_window_honors_now_and_future_expiry(sql_storage: StorageBackend) -> None:
    facts = [_fact(text, f"a{i}") for i, text in enumerate(CONTENTS)]
    facts[-1].valid_until = NOW + timedelta(minutes=1)
    sql_storage.save("__shared__", facts)
    pool = SharedMemoryPool(sql_storage)

    assert len(pool.recall(QUERY, "reader", top_k=3, now=NOW)) == 3
    assert pool._last_recall_meta is not None
    assert pool._last_recall_meta.coverage == 1.0
    assert len(pool.recall(QUERY, "reader", top_k=3, now=NOW + timedelta(minutes=1))) == 2
    assert pool._last_recall_meta.coverage == pytest.approx(2 / 3)


def test_facet_recall_credits_each_publisher_once(sql_storage: StorageBackend) -> None:
    sql_storage.save("__shared__", [_fact(text, "publisher") for text in CONTENTS])
    pool = SharedMemoryPool(sql_storage)
    assert len(pool.recall(QUERY, "reader", top_k=3, now=NOW)) == 3
    assert pool._used_count["publisher"] == 1


def test_no_visible_facts_preserves_all_uncovered_facets(sql_storage: StorageBackend) -> None:
    hidden = _fact(CONTENTS[0], "owner")
    hidden.visibility_scope = "team:secret"
    sql_storage.save("__shared__", [hidden])
    pool = SharedMemoryPool(sql_storage)
    assert pool.recall(QUERY, "reader", now=NOW) == []
    assert pool._last_recall_meta is not None
    assert pool._last_recall_meta.coverage == 0.0
    assert pool._last_recall_meta.covered_facets == frozenset()
    assert pool._last_recall_meta.uncovered_facets == frozenset({"f0", "f1", "f2"})


def test_selected_fact_can_cover_two_facets_via_evidence_fields(
    sql_storage: StorageBackend,
) -> None:
    fact = _fact("Operational configuration approved by the team.", "publisher")
    fact.tags = ["orchard nutrients"]
    fact.source_snippets = ["Warehouse pallets follow the shortest routing schedule."]
    sql_storage.save("__shared__", [fact])
    pool = SharedMemoryPool(sql_storage)

    results = pool.recall(
        "Integrate orchard nutrient dosing, warehouse pallet routing, "
        "and satellite orbit telemetry.",
        "reader",
        top_k=1,
        now=NOW,
    )

    assert [returned.id for returned, _ in results] == [fact.id]
    assert pool._last_recall_meta is not None
    assert pool._last_recall_meta.coverage == pytest.approx(2 / 3)
    assert pool._last_recall_meta.covered_facets == frozenset({"f0", "f1"})
    assert pool._last_recall_meta.uncovered_facets == frozenset({"f2"})
