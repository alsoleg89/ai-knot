"""Semantic candidates remain source checked; no semantic-completeness claim."""

import asyncio
from dataclasses import replace

import pytest

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import (
    AlwaysOnMemory,
    SemanticBudget,
    SemanticContext,
    VerificationOutcome,
    VerificationResult,
)
from ai_knot.types import Fact, MemoryType


def setup(sql_storage, clock):
    facts = [
        Fact("Backup receipt verified", id="backup", created_at=clock(), valid_from=clock()),
        Fact("Coffee supplies renewed", id="coffee", created_at=clock(), valid_from=clock()),
        Fact("Office heating repaired", id="office", created_at=clock(), valid_from=clock()),
    ]
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    kb.replace_facts(facts)
    memory = AlwaysOnMemory(kb, clock=clock)
    return kb, memory, facts


def vectors(texts):
    return [
        [1.0, 0.0] if ("restore readiness" in text or "Backup receipt" in text) else [0.0, 1.0]
        for text in texts
    ]


@pytest.mark.anyio
async def test_complete_task_bundle_costs_no_provider_call(sql_storage, clock):
    _, memory, facts = setup(sql_storage, clock)

    async def forbidden(texts):
        raise AssertionError("Whole task bundle should not need embeddings")

    selector = SemanticContext(memory, embed=forbidden)
    result = await selector.acquire(
        ("restore readiness",), budget=SemanticBudget(), max_facts=3, max_chars=600
    )
    assert result.status == "snapshot"
    assert {f.ref.fact_id for f in result.review.accepted} == {f.id for f in facts}
    assert result.provider_calls_reserved == result.input_bytes_reserved == 0
    assert not hasattr(result, "complete")


@pytest.mark.anyio
async def test_semantic_ranking_reuses_vectors_and_refreshes_only_changed_text(sql_storage, clock):
    kb, memory, facts = setup(sql_storage, clock)
    calls = []

    async def embed(texts):
        calls.append(list(texts))
        return vectors(texts)

    selector = SemanticContext(memory, embed=embed)
    blocked = await selector.acquire(("restore readiness",), budget=SemanticBudget(), max_facts=1)
    assert blocked.status == "budget_exhausted" and not calls
    result = await selector.acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    assert result.status == "ok" and result.review.accepted[0].ref.fact_id == "backup"
    assert len(calls) == 1 and len(calls[0]) == 4
    warm = await selector.acquire(("restore readiness",), budget=SemanticBudget(), max_facts=1)
    assert warm.status == "ok" and warm.provider_calls_reserved == 0 and len(calls) == 1
    kb.replace_facts(
        [replace(facts[0], content="Backup receipt verified revision two"), *facts[1:]]
    )
    changed = await selector.acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    assert changed.review.accepted[0].content.endswith("revision two")
    assert calls[1] == ["Backup receipt verified revision two"]


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["content", "task", "type", "freshness"])
async def test_changes_during_provider_io_cannot_recapture_new_evidence(sql_storage, clock, change):
    kb, memory, facts = setup(sql_storage, clock)
    if change == "freshness":
        ref = memory.capture_evidence((("owner", "backup"),))[0]
        memory.set_freshness(ref, source_ref="remote", recheck_after_seconds=1)
        memory.record_verification(
            ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"checked": True}, "v1")
        )

    async def embed(texts):
        if change == "freshness":
            clock.advance(seconds=2)
        else:
            attrs = (
                {"content": "Receipt withdrawn"}
                if change == "content"
                else {"qualifiers": {"task_key": "another"}}
                if change == "task"
                else {"type": MemoryType.EPISODIC}
            )
            kb.replace_facts([replace(facts[0], **attrs), *facts[1:]])
        return vectors(texts)

    result = await SemanticContext(memory, embed=embed).acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    assert result.status == "changed"
    assert "backup" not in {f.ref.fact_id for f in result.review.accepted}
    assert result.provider_calls_reserved == 1


@pytest.mark.anyio
async def test_acl_task_support_and_namespace_gates_apply_before_and_after_io(sql_storage, clock):
    kb, _, facts = setup(sql_storage, clock)
    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock, embed_url="")
    shared = replace(
        facts[0],
        id="same",
        visibility_scope="ops",
        origin_agent_id="peer",
        qualifiers={"task_key": "release"},
    )
    denied = replace(shared, id="denied", content="Denied secret", visibility_scope="closed")
    wrong = replace(
        shared, id="wrong", content="Other task secret", qualifiers={"task_key": "other"}
    )
    unsupported = replace(shared, id="unsupported", content="Unsupported secret", supported=False)
    kb.replace_facts([replace(facts[1], id="same"), facts[2]])
    sql_storage.save("__shared__", [shared, denied, wrong, unsupported])
    pool.grant_read("owner", "ops")
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    seen = []

    async def embed(texts):
        seen.extend(texts)
        pool.revoke_read("owner", "ops")
        return vectors(texts)

    result = await SemanticContext(memory, embed=embed).acquire(
        ("restore readiness",),
        budget=SemanticBudget(max_provider_calls=1),
        task_key="release",
        max_facts=1,
    )
    assert not any("secret" in text for text in seen)
    assert result.status == "changed"
    assert not result.review.accepted
    assert any(
        issue.reason == "access_changed" and issue.ref is None for issue in result.review.issues
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "budget",
    [
        SemanticBudget(max_candidates=2, max_provider_calls=1),
        SemanticBudget(max_input_bytes=10, max_provider_calls=1),
    ],
)
async def test_input_ceiling_never_embeds_a_silent_prefix(sql_storage, clock, budget):
    _, memory, _ = setup(sql_storage, clock)

    async def forbidden(texts):
        raise AssertionError("Over-limit input must not reach provider")

    result = await SemanticContext(memory, embed=forbidden).acquire(
        ("restore readiness",), budget=budget, max_facts=1
    )
    assert result.status == "input_limit"
    assert not result.review.accepted and result.provider_calls_reserved == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response", [[], [[0, 0]] * 4, [[float("nan"), 0]] * 4, [[1, 0], [1], [1, 0], [1, 0]]]
)
async def test_malformed_vectors_are_not_semantic_success(sql_storage, clock, response):
    _, memory, _ = setup(sql_storage, clock)

    async def embed(texts):
        return response

    result = await SemanticContext(memory, embed=embed).acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    assert result.status == "unavailable" and not result.review.accepted
    assert result.provider_calls_reserved == 1 and result.input_bytes_reserved > 0


@pytest.mark.anyio
async def test_timeout_keeps_request_reservation_and_returns_no_candidates(sql_storage, clock):
    _, memory, _ = setup(sql_storage, clock)

    async def embed(texts):
        await asyncio.sleep(10)
        return vectors(texts)

    result = await SemanticContext(memory, embed=embed).acquire(
        ("restore readiness",),
        budget=SemanticBudget(max_provider_calls=1, timeout=0.001),
        max_facts=1,
    )
    assert result.status == "unavailable" and result.provider_calls_reserved == 1
    assert not result.review.accepted


@pytest.mark.anyio
async def test_multiple_requirements_share_best_candidate_score(sql_storage, clock):
    kb, memory, facts = setup(sql_storage, clock)
    kb.replace_facts([replace(f, content=f.id) for f in facts])
    mapping = {
        "first": [1, 0],
        "second": [0, 1],
        "backup": [1, 0.01],
        "coffee": [0.1, 1],
        "office": [0.5, 0.5],
    }

    async def embed(texts):
        return [mapping[t] for t in texts]

    result = await SemanticContext(memory, embed=embed).acquire(
        ("first", "second"), budget=SemanticBudget(max_provider_calls=1), max_facts=2
    )
    assert {f.ref.fact_id for f in result.review.accepted} == {"backup", "coffee"}


@pytest.mark.anyio
async def test_concurrent_acquisitions_reuse_cache_without_mutating_memory(sql_storage, clock):
    kb, memory, _ = setup(sql_storage, clock)
    intent = memory.create_intent(client_key="untouched", goal="Observe only")
    before = sql_storage.load("owner")
    calls = []

    async def embed(texts):
        calls.append(texts)
        await asyncio.sleep(0)
        return vectors(texts)

    selector = SemanticContext(memory, embed=embed)
    results = await asyncio.gather(
        *[
            selector.acquire(
                ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
            )
            for _ in range(4)
        ]
    )
    assert len(calls) == 1
    assert sum(r.provider_calls_reserved for r in results) == 1
    assert all(r.review.accepted[0].ref.fact_id == "backup" for r in results)
    assert sql_storage.load("owner") == before and memory.get_intent(intent.id) == intent


@pytest.mark.anyio
async def test_same_id_in_private_and_shared_namespaces_remains_distinct(sql_storage, clock):
    kb, _, facts = setup(sql_storage, clock)
    shared = replace(facts[0], id="same", origin_agent_id="peer", visibility_scope="ops")
    kb.replace_facts([replace(shared, content="Private receipt"), facts[1], facts[2]])
    sql_storage.save("__shared__", [shared])
    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock, embed_url="")
    pool.grant_read("owner", "ops")

    async def embed(texts):
        return [
            [1, 0] if t in ("restore readiness", "Private receipt", shared.content) else [0, 1]
            for t in texts
        ]

    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    result = await SemanticContext(memory, embed=embed).acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=2
    )
    assert {(f.ref.namespace, f.ref.fact_id) for f in result.review.accepted} == {
        ("owner", "same"),
        ("__shared__", "same"),
    }


@pytest.mark.anyio
async def test_whole_fact_budget_skips_oversized_candidate(sql_storage, clock):
    kb, memory, facts = setup(sql_storage, clock)
    kb.replace_facts([replace(facts[0], content="x" * 100), facts[1], facts[2]])

    async def embed(texts):
        return [[1, 0] for _ in texts]

    result = await SemanticContext(memory, embed=embed).acquire(
        ("restore readiness",),
        budget=SemanticBudget(max_provider_calls=1),
        max_facts=2,
        max_chars=60,
    )
    assert {f.ref.fact_id for f in result.review.accepted} == {"coffee", "office"}
    assert [ref.fact_id for ref in result.review.omitted] == ["backup"]
    assert not result.review.complete
    assert sum(len(f.content) for f in result.review.accepted) <= 60


@pytest.mark.anyio
async def test_withdrawn_root_excludes_derived_text_before_and_after_io(sql_storage, clock):
    kb, memory, facts = setup(sql_storage, clock)
    basis = memory.capture_evidence((("owner", "backup"),))
    derived = memory.derive_fact("Confidential derived readiness", basis=basis)
    seen = []

    async def embed(texts):
        seen.extend(texts)
        kb.replace_facts(
            [
                replace(f, content="Receipt revoked") if f.id == "backup" else f
                for f in sql_storage.load("owner")
            ]
        )
        return [[1, 0] for _ in texts]

    selector = SemanticContext(memory, embed=embed)
    result = await selector.acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    assert result.status == "changed" and derived.content in seen
    assert derived.id not in {f.ref.fact_id for f in result.review.accepted}
    seen.clear()
    await selector.acquire(
        ("a new query",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    assert derived.content not in seen


@pytest.mark.anyio
async def test_invalid_provider_result_does_not_poison_retry_cache(sql_storage, clock):
    _, memory, _ = setup(sql_storage, clock)
    calls = []

    async def embed(texts):
        calls.append(list(texts))
        return [] if len(calls) == 1 else vectors(texts)

    selector = SemanticContext(memory, embed=embed)
    first = await selector.acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    second = await selector.acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    assert first.status == "unavailable" and second.status == "ok"
    assert calls[0] == calls[1] and second.provider_calls_reserved == 1


@pytest.mark.anyio
async def test_count_budget_records_whole_fact_omissions(sql_storage, clock):
    _, memory, _ = setup(sql_storage, clock)

    async def embed(texts):
        return [[1, 0] for _ in texts]

    result = await SemanticContext(memory, embed=embed).acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    assert len(result.review.accepted) == 1 and len(result.review.omitted) == 2
    assert not result.review.complete


@pytest.mark.anyio
async def test_oversized_winner_does_not_hide_fitting_lower_score(sql_storage, clock):
    kb, memory, facts = setup(sql_storage, clock)
    kb.replace_facts([replace(facts[0], content="x" * 20), replace(facts[1], content="b")])

    async def embed(texts):
        return [[0.1, 1] if text == "b" else [1, 0] for text in texts]

    result = await SemanticContext(memory, embed=embed).acquire(
        ("restore readiness",),
        budget=SemanticBudget(max_provider_calls=1),
        max_facts=1,
        max_chars=1,
    )
    assert [f.content for f in result.review.accepted] == ["b"]
    assert [ref.fact_id for ref in result.review.omitted] == ["backup"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_candidates": True},
        {"max_candidates": 129},
        {"max_input_bytes": 0},
        {"max_provider_calls": -1},
        {"max_provider_calls": 0.5},
        {"timeout": float("inf")},
        {"timeout": True},
    ],
)
def test_budget_rejects_invalid_authority(kwargs):
    with pytest.raises(ValueError):
        SemanticBudget(**kwargs)
