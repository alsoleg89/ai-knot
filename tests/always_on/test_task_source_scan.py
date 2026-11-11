"""Task-scoped admission must not deserialize unrelated retained documents."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import pytest

from ai_knot import KnowledgeBase
from ai_knot.always_on import AlwaysOnMemory, SemanticBudget, SemanticContext
from ai_knot.always_on.storage import SQLStatement
from ai_knot.types import Fact, MemoryType


@pytest.mark.anyio
@pytest.mark.parametrize("task_key", ["rollout", "", "部署'\"[null]\\"])
@pytest.mark.parametrize("operation", ["snapshot", "semantic"])
async def test_only_current_task_and_generic_candidates_are_decoded(
    sql_storage, clock, task_key, operation
):
    current = Fact("A current prerequisite", id="current", qualifiers={"task_key": task_key})
    generic = Fact("A generic prerequisite", id="generic")
    irrelevant = [
        Fact(f"Other task {i}", id=f"other-{i:03d}", qualifiers={"task_key": "other"})
        for i in range(150)
    ]
    episodes = [replace(current, id=f"episode-{i}", type=MemoryType.EPISODIC) for i in range(20)]
    sql_storage.save("reader", [current, generic, *irrelevant, *episodes])
    memory = AlwaysOnMemory(KnowledgeBase("reader", storage=sql_storage, embed_url=""), clock=clock)
    decoded = []
    decoder = sql_storage._fact_from_row

    def count(row):
        fact = decoder(row)
        decoded.append(fact.id)
        return fact

    async def forbidden(_):
        raise AssertionError("The whole task corpus fits without embeddings")

    with patch.object(type(sql_storage), "_fact_from_row", staticmethod(count)):
        if operation == "snapshot":
            facts = memory.task_snapshot(task_key=task_key)
        else:
            result = await SemanticContext(memory, embed=forbidden).acquire(
                ("prerequisite",), task_key=task_key, budget=SemanticBudget()
            )
            assert result.status == "snapshot" and result.provider_calls_reserved == 0
            facts = result.review.accepted
    assert [f.ref.fact_id for f in facts] == ["current", "generic"]
    assert decoded == ["current", "generic"]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "{}",
        '{"task_key":"other","task_key":"rollout"}',
        '{"task_key":"rollout","task_key":"other"}',
        '{"task_key":"other","\\u0074ask_key":"rollout"}',
        '{"task_key":null}',
        '{"task_key":false}',
        '{"task_key":123}',
        '{"task_key":["rollout"]}',
        '{"task_key":"rollout","unrelated":{"task_key":"other"}}',
        '{"task_key":"rollout","comment":"\\u0000"}',
    ],
)
def test_sql_prefilter_preserves_python_legacy_qualifier_semantics(sql_storage, clock, raw):
    sql_storage.save("reader", [Fact("Current prerequisite", id="item")])
    with sql_storage.always_on_transaction(("reader",), write=True) as session:
        session.execute(
            SQLStatement(
                "UPDATE facts SET qualifiers=? WHERE agent_id=? AND id=?",
                'UPDATE "ai-knot_facts" SET qualifiers=%s WHERE agent_id=%s AND id=%s',
            ),
            (raw, "reader", "item"),
        )
    memory = AlwaysOnMemory(KnowledgeBase("reader", storage=sql_storage, embed_url=""), clock=clock)
    qualifiers = json.loads(raw) if raw else {}
    expected = ("item",) if qualifiers.get("task_key", "") in ("", "rollout") else ()
    assert tuple(f.ref.fact_id for f in memory.task_snapshot(task_key="rollout")) == expected


@pytest.mark.parametrize("root_type", [MemoryType.SEMANTIC, MemoryType.EPISODIC])
def test_other_task_or_episodic_root_still_validates_a_current_projection(
    sql_storage, clock, root_type
):
    from ai_knot.always_on.lineage import derivation_qualifiers

    root = Fact(
        "Retained source receipt", id="root", type=root_type, qualifiers={"task_key": "source"}
    )
    lesson = Fact(
        "Current recovery prerequisite",
        id="lesson",
        qualifiers={**derivation_qualifiers("reader", [root]), "task_key": "rollout"},
    )
    sql_storage.save("reader", [root, lesson])
    memory = AlwaysOnMemory(KnowledgeBase("reader", storage=sql_storage, embed_url=""), clock=clock)
    facts = memory.task_snapshot(task_key="rollout")
    assert [f.ref.fact_id for f in facts] == ["lesson"]
    assert [r.fact_id for r in memory.independent_evidence(tuple(f.ref for f in facts))] == ["root"]


def test_task_subset_never_becomes_a_complete_slot_or_namespace_cache(sql_storage, clock):
    from ai_knot.always_on.reads import ReadCache

    old = Fact(
        "Old route",
        id="old",
        slot_key="route::target",
        version=1,
        qualifiers={"task_key": "rollout"},
    )
    new = replace(old, id="new", content="New route", version=2, qualifiers={"task_key": "other"})
    sql_storage.save("reader", [old, new])
    with sql_storage.always_on_transaction(("reader",), write=False) as session:
        cache = ReadCache(session)
        assert [f.id for f in cache.load_task_facts("reader", task_key="rollout")] == ["old"]
        assert [f.id for f in cache.load_facts("reader", fact_id="new")] == ["new"]
        assert {f.id for f in cache.load_facts("reader", slot_key="route::target")} == {
            "old",
            "new",
        }
        assert {f.id for f in cache.load_facts("reader")} == {"old", "new"}
    memory = AlwaysOnMemory(KnowledgeBase("reader", storage=sql_storage, embed_url=""), clock=clock)
    assert memory.task_snapshot(task_key="rollout") == ()


def test_full_task_result_preserves_complete_slot_cache(sql_storage, clock, monkeypatch):
    from ai_knot.always_on.reads import ReadCache

    facts = [Fact(f"Route {i}", id=f"route-{i}", slot_key=f"route::{i}") for i in range(3)]
    sql_storage.save("reader", facts)
    with sql_storage.always_on_transaction(("reader",), write=False) as session:
        cache = ReadCache(session)
        assert len(tuple(cache.load_task_facts("reader", task_key="rollout"))) == 3

        def forbidden(*args, **kwargs):
            raise AssertionError("The complete namespace is already present")

        monkeypatch.setattr(session, "load_facts", forbidden)
        for fact in facts:
            assert cache.load_facts("reader", fact_id=fact.id) == [fact]
            assert cache.load_facts("reader", slot_key=fact.slot_key) == [fact]
        assert len(cache.load_facts("reader")) == 3


def test_concurrent_deletion_cannot_falsely_prove_a_task_subset_complete(
    sql_storage, clock, monkeypatch
):
    from ai_knot.always_on.reads import ReadCache

    old = Fact(
        "Old route",
        id="old",
        slot_key="route::target",
        version=1,
        qualifiers={"task_key": "rollout"},
    )
    peer = replace(old, id="peer", version=2, qualifiers={"task_key": "other"})
    sql_storage.save("reader", [old, peer])
    with sql_storage.always_on_transaction(("reader",), write=False) as session:
        load = session.load_task_facts

        def during(namespace, *, task_key):
            facts = load(namespace, task_key=task_key)
            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(sql_storage.atomic_update, "reader", lambda _: [old]).result(
                    timeout=5
                )
            return facts

        monkeypatch.setattr(session, "load_task_facts", during)
        cache = ReadCache(session)
        assert [f.id for f in cache.load_task_facts("reader", task_key="rollout")] == ["old"]
        assert {f.id for f in cache.load_facts("reader", slot_key="route::target")} == {
            "old",
            "peer",
        }
    assert [f.id for f in sql_storage.load("reader")] == ["old"]
