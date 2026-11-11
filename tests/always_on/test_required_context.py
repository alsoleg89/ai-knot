import pytest
from tests.always_on.test_semantic_context import setup

from ai_knot.always_on import SemanticBudget, SemanticContext, SnapshotTooLarge, StaleEvidence


@pytest.mark.parametrize("query,limited", [("backup", True), ("quartz", False)])
def test_required_exact_char_budget_reports_omitted_relevant_optional_fact(
    sql_storage, clock, query, limited
):
    _, memory, facts = setup(sql_storage, clock)
    refs = memory.capture_evidence((("owner", "coffee"),))
    result = memory.select_context(
        (query,), required_refs=refs, max_facts=2, max_chars=len(facts[1].content)
    )
    assert [f.ref.fact_id for f in result.facts] == ["coffee"]
    assert result.budget_limited is limited


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["lexical", "semantic"])
async def test_required_fact_survives_zero_relevance_without_inventing_coverage(
    sql_storage, clock, mode
):
    _, memory, _ = setup(sql_storage, clock)
    required = memory.capture_evidence((("owner", "coffee"),))

    async def forbidden(texts):
        raise AssertionError("Required facts fill the count budget; no embedding needed")

    if mode == "lexical":
        result = memory.select_context(("backup receipt",), max_facts=1, required_refs=required)
        facts = result.facts
        assert result.covered_requirements == ()
        assert result.missing_requirements == ("backup receipt",)
    else:
        result = await SemanticContext(memory, embed=forbidden).acquire(
            ("backup receipt",), budget=SemanticBudget(), max_facts=1, required_refs=required
        )
        facts = result.review.accepted
        assert result.provider_calls_reserved == 0
    assert [f.ref.fact_id for f in facts] == ["coffee"]


@pytest.mark.anyio
async def test_required_negative_cosine_reserves_budget_before_optional_ranking(sql_storage, clock):
    _, memory, _ = setup(sql_storage, clock)
    required = memory.capture_evidence((("owner", "coffee"),))
    inputs = []

    async def embed(texts):
        inputs.extend(texts)
        return [[1.0, 0.0] if "backup" in t.lower() else [-1.0, 0.0] for t in texts]

    result = await SemanticContext(memory, embed=embed).acquire(
        ("backup receipt",),
        budget=SemanticBudget(max_provider_calls=1),
        max_facts=2,
        required_refs=required,
    )
    assert [f.ref.fact_id for f in result.review.accepted] == ["coffee", "backup"]
    assert result.provider_calls_reserved == 1
    assert "Coffee supplies renewed" not in inputs


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["lexical", "semantic"])
async def test_required_group_has_whole_fact_budget_and_deduplicates(sql_storage, clock, mode):
    _, memory, facts = setup(sql_storage, clock)
    refs = memory.capture_evidence((("owner", "backup"), ("owner", "coffee")))
    limit = len(facts[0].content) + len(facts[1].content)

    async def forbidden(texts):
        raise AssertionError("No provider allowed")

    async def select(count, chars):
        if mode == "lexical":
            return memory.select_context(
                ("backup",), required_refs=(*refs, refs[0]), max_facts=count, max_chars=chars
            ).facts
        return (
            await SemanticContext(memory, embed=forbidden).acquire(
                ("backup",),
                budget=SemanticBudget(),
                required_refs=(*refs, refs[0]),
                max_facts=count,
                max_chars=chars,
            )
        ).review.accepted

    assert [f.ref.fact_id for f in await select(2, limit)] == ["backup", "coffee"]
    with pytest.raises(SnapshotTooLarge):
        await select(1, limit)
    with pytest.raises(SnapshotTooLarge):
        await select(2, limit - 1)


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["lexical", "semantic"])
async def test_required_stale_ref_rejects_even_when_whole_snapshot_fits(sql_storage, clock, mode):
    from dataclasses import replace

    kb, memory, facts = setup(sql_storage, clock)
    refs = memory.capture_evidence((("owner", "coffee"),))
    kb.replace_facts([facts[0], replace(facts[1], content="Withdrawn supplies"), facts[2]])

    async def forbidden(texts):
        raise AssertionError("Stale mandatory input must be rejected before I/O")

    with pytest.raises(StaleEvidence):
        if mode == "lexical":
            memory.select_context(("backup",), required_refs=refs)
        else:
            await SemanticContext(memory, embed=forbidden).acquire(
                ("backup",), budget=SemanticBudget(max_provider_calls=1), required_refs=refs
            )


@pytest.mark.anyio
async def test_mandatory_withdrawal_during_embedding_returns_no_partial_group(sql_storage, clock):
    kb, memory, facts = setup(sql_storage, clock)
    refs = memory.capture_evidence((("owner", "coffee"),))
    calls = []

    async def embed(texts):
        calls.append(texts)
        kb.replace_facts([facts[0], facts[2]])
        return [[1.0, 0.0] for _ in texts]

    result = await SemanticContext(memory, embed=embed).acquire(
        ("backup",),
        budget=SemanticBudget(max_provider_calls=1),
        max_facts=2,
        required_refs=refs,
    )
    assert result.status == "changed" and result.review.accepted == ()
    assert result.provider_calls_reserved == len(calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["lexical", "semantic"])
@pytest.mark.parametrize("change", ["task", "episode", "unsupported", "foreign"])
async def test_mandatory_refs_cannot_override_task_or_admission(sql_storage, clock, mode, change):
    from dataclasses import replace

    from ai_knot.always_on import EvidenceRef
    from ai_knot.types import MemoryType

    kb, memory, facts = setup(sql_storage, clock)
    if change == "foreign":
        refs = (EvidenceRef("foreign", "backup", "unknown"),)
    else:
        attrs = (
            {"qualifiers": {"task_key": "another"}}
            if change == "task"
            else {"type": MemoryType.EPISODIC}
            if change == "episode"
            else {}
        )
        kb.replace_facts([replace(facts[0], **attrs), *facts[1:]])
        refs = memory.capture_evidence((("owner", "backup"),))
        if change == "unsupported":
            kb.replace_facts([replace(facts[0], supported=False), *facts[1:]])

    async def forbidden(texts):
        raise AssertionError("Invalid mandatory context reached provider")

    with pytest.raises(StaleEvidence, match="Required task context unavailable"):
        if mode == "lexical":
            memory.select_context(("backup",), task_key="rollout", required_refs=refs)
        else:
            await SemanticContext(memory, embed=forbidden).acquire(
                ("backup",),
                budget=SemanticBudget(max_provider_calls=1),
                task_key="rollout",
                required_refs=refs,
            )


@pytest.mark.anyio
@pytest.mark.parametrize("bad", [None, True, "backup", ("backup",)])
async def test_required_refs_reject_untyped_values(sql_storage, clock, bad):
    _, memory, _ = setup(sql_storage, clock)

    async def forbidden(texts):
        raise AssertionError("Invalid input reached provider")

    with pytest.raises(ValueError):
        memory.select_context(("backup",), required_refs=bad)
    with pytest.raises(ValueError):
        await SemanticContext(memory, embed=forbidden).acquire(
            ("backup",), budget=SemanticBudget(), required_refs=bad
        )


@pytest.mark.anyio
async def test_required_same_ids_keep_namespaces_and_revocation_rejects_before_io(
    sql_storage, clock
):
    from ai_knot import KnowledgeBase, SharedMemoryPool
    from ai_knot.always_on import AlwaysOnMemory
    from ai_knot.types import Fact

    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    kb.replace_facts([Fact("Private condition", id="same"), Fact("Backup receipt", id="backup")])
    sql_storage.save("__shared__", [Fact("Shared condition", id="same", visibility_scope="ops")])
    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock, embed_url="")
    pool.grant_read("owner", "ops")
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    refs = tuple(reversed(memory.capture_evidence((("owner", "same"), ("__shared__", "same")))))

    async def forbidden(texts):
        raise AssertionError("No embedding allowed")

    selected = memory.select_context(("backup",), required_refs=refs, max_facts=2).facts
    assert [(f.ref.namespace, f.content) for f in selected] == [
        ("owner", "Private condition"),
        ("__shared__", "Shared condition"),
    ]
    result = await SemanticContext(memory, embed=forbidden).acquire(
        ("backup",), budget=SemanticBudget(), required_refs=refs, max_facts=2
    )
    assert result.review.accepted == selected
    pool.revoke_read("owner", "ops")
    with pytest.raises(StaleEvidence):
        memory.select_context(("backup",), required_refs=refs)
    with pytest.raises(StaleEvidence):
        await SemanticContext(memory, embed=forbidden).acquire(
            ("backup",), budget=SemanticBudget(max_provider_calls=1), required_refs=refs
        )


@pytest.mark.anyio
async def test_required_freshness_expiring_during_io_discards_whole_selection(sql_storage, clock):
    from ai_knot.always_on import VerificationOutcome, VerificationResult

    _, memory, _ = setup(sql_storage, clock)
    refs = memory.capture_evidence((("owner", "coffee"),))
    memory.set_freshness(refs[0], source_ref="connector", recheck_after_seconds=1)
    memory.record_verification(
        refs[0], result=VerificationResult(VerificationOutcome.CONFIRMED, {"checked": True}, "one")
    )

    async def embed(texts):
        clock.advance(seconds=2)
        return [[1.0, 0.0] for _ in texts]

    result = await SemanticContext(memory, embed=embed).acquire(
        ("backup",), budget=SemanticBudget(max_provider_calls=1), required_refs=refs, max_facts=2
    )
    assert result.status == "changed" and result.review.accepted == ()
    assert result.provider_calls_reserved == 1


@pytest.mark.anyio
async def test_required_only_read_evicts_withdrawn_vectors_before_later_restoration(
    sql_storage, clock
):
    from tests.always_on.test_semantic_context import vectors

    kb, memory, facts = setup(sql_storage, clock)
    calls = []

    async def embed(texts):
        calls.append(list(texts))
        return vectors(texts)

    selector = SemanticContext(memory, embed=embed)
    await selector.acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    refs = memory.capture_evidence((("owner", "backup"),))
    kb.replace_facts([facts[0], facts[2]])
    required_only = await selector.acquire(
        ("restore readiness",), budget=SemanticBudget(), max_facts=1, required_refs=refs
    )
    assert required_only.provider_calls_reserved == 0
    kb.replace_facts(facts)
    restored = await selector.acquire(
        ("restore readiness",), budget=SemanticBudget(max_provider_calls=1), max_facts=1
    )
    assert restored.provider_calls_reserved == 1
    assert calls[-1] == ["Coffee supplies renewed"]
