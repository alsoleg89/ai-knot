"""Generated aggregates and legacy summaries share guarded evidence semantics."""

import asyncio
from datetime import timedelta

import pytest

from ai_knot import KnowledgeBase
from ai_knot.always_on import AlwaysOnMemory, RevisionConflict, StaleEvidence
from ai_knot.always_on.changes import evidence_revision
from ai_knot.always_on.lineage import evidence_dependencies, evidence_roots
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact, MemoryType


def structured(value, attribute):
    return Fact(
        f"Deployment {attribute}: {value}",
        entity="deployment",
        attribute=attribute,
        value_text=value,
    )


@pytest.mark.parametrize("change", ["correct", "delete"])
def test_automatic_aggregate_retraction_blocks_prepared_evidence(sql_storage, clock, change):
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    memory = AlwaysOnMemory(kb, clock=clock)
    source, _other, aggregate = kb.add_resolved(
        [structured("42", "build"), structured("green", "status")]
    )
    refs = memory.capture_evidence((("owner", aggregate.id),))
    assert {ref.fact_id for ref in memory.independent_evidence(refs)} == {source.id, _other.id}
    if change == "correct":
        kb.add_resolved([structured("43", "build")])
    else:
        kb.forget(source.id)
    with pytest.raises(StaleEvidence):
        memory.independent_evidence(refs)
    assert aggregate.id not in {f.id for f in kb.recall_facts("Deployment build 42", top_k=30)}


@pytest.mark.parametrize("kind", ["aggregate", "summary"])
def test_identifiable_unrooted_generated_evidence_fails_closed(sql_storage, clock, kind):
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    generated = Fact(
        "Old unsupported generated evidence",
        verification_source="aggregate" if kind == "aggregate" else "",
        qualifiers={} if kind == "aggregate" else {"consolidation_ids": '["old"]'},
    )
    kb.replace_facts([generated])
    memory = AlwaysOnMemory(kb, clock=clock)
    with pytest.raises(StaleEvidence):
        memory.capture_evidence((("owner", generated.id),))
    manual = kb.add("Trusted manual evidence")
    assert memory.capture_evidence((("owner", manual.id),))


@pytest.mark.parametrize("kind", ["aggregate", "legacy"])
@pytest.mark.parametrize("change", ["revoke", "delete_intermediate", "correct_root"])
def test_automatic_derivatives_keep_shared_roots_and_intermediate_dependencies(
    sql_storage, clock, kind, change
):
    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock)
    pool.grant_read("owner", "team:ops")
    root = Fact("Deployment build 42", visibility_scope="team:ops", origin_agent_id="writer")
    sql_storage.save("__shared__", [root])
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    intermediate = memory.derive_fact(
        "Deployment checked build 42", basis=memory.capture_evidence((("__shared__", root.id),))
    )
    facts = kb.list_facts()
    source = facts[0]
    if kind == "aggregate":
        source.entity, source.attribute, source.slot_key = (
            "deployment",
            "build",
            "deployment::build",
        )
        kb.replace_facts(facts)
        kb.add_resolved([structured("green", "status")])
        output = next(f for f in kb.list_facts() if f.attribute == "_aggregate")
    else:
        source.type = MemoryType.EPISODIC
        source.created_at = source.valid_from = clock() - timedelta(days=2)
        kb.replace_facts(facts)
        kb._default_provider = "offline"
        kb._extract_phase = lambda turns: [Fact("Deployment checked summary 42")]
        assert asyncio.run(kb.consolidate_episodic()) == 1
        output = next(f for f in kb.list_facts() if f.id != source.id)
    assert root.id in {r.fact_id for r in evidence_roots("owner", output)}
    dep = next(r for r in evidence_dependencies(output) if r.fact_id == intermediate.id)
    assert dep.revision == evidence_revision(kb.get(intermediate.id))
    refs = memory.capture_evidence((("owner", output.id),))
    if change == "revoke":
        pool.revoke_read("owner", "team:ops")
    elif change == "delete_intermediate":
        kb.forget(intermediate.id)
    else:
        root.content = "Deployment build failed"
        sql_storage.save("__shared__", [root])
    assert output.id not in {f.id for f in kb.recall_facts("Deployment checked summary", top_k=30)}
    from ai_knot.always_on import AccessDenied

    with pytest.raises((StaleEvidence, AccessDenied)):
        memory.independent_evidence(refs)


def test_legacy_commit_rechecks_underlying_roots_after_provider(sql_storage, clock, monkeypatch):
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    memory = AlwaysOnMemory(kb, clock=clock)
    root = kb.add("Deployment build 42")
    episode = memory.derive_fact(
        "Deployment episode", basis=memory.capture_evidence((("owner", root.id),))
    )
    facts = kb.list_facts()
    item = next(f for f in facts if f.id == episode.id)
    item.type = MemoryType.EPISODIC
    item.created_at = item.valid_from = clock() - timedelta(days=2)
    kb.replace_facts(facts)
    kb._default_provider = "offline"

    def extract(turns):
        kb.forget(root.id)
        return [Fact("Deployment summary")]

    monkeypatch.setattr(kb, "_extract_phase", extract)
    with pytest.raises(RevisionConflict):
        asyncio.run(kb.consolidate_episodic())
    assert kb.get(episode.id).is_active(clock())
    assert len(kb.list_facts()) == 1


def test_legacy_archive_preserves_freshness_attestation_and_deadline(
    sql_storage, clock, monkeypatch
):
    from ai_knot.always_on import VerificationOutcome, VerificationResult

    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    memory = AlwaysOnMemory(kb, clock=clock)
    source = Fact(
        "Deployment old episode",
        type=MemoryType.EPISODIC,
        created_at=clock() - timedelta(days=2),
        valid_from=clock() - timedelta(days=2),
    )
    kb.replace_facts([source])
    (ref,) = memory.capture_evidence((("owner", source.id),))
    memory.set_freshness(ref, source_ref="deployment", recheck_after_seconds=60)
    original = memory.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"build": 42}, "v1")
    )
    clock.advance(seconds=20)
    kb._default_provider = "offline"
    monkeypatch.setattr(kb, "_extract_phase", lambda turns: [Fact("Deployment summary")])
    assert asyncio.run(kb.consolidate_episodic()) == 1
    summary = next(f for f in kb.list_facts() if f.type == MemoryType.SEMANTIC)
    basis = memory.capture_evidence((("owner", summary.id),))
    intent = memory.create_intent(client_key="legacy-archive", goal="Use summary")
    lease = memory.claim(intent.id, worker_id="host", ttl_seconds=300)
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="use", request={}, basis=basis
    )
    with memory._transaction(write=False) as session:
        policy = memory._policy(session, "owner", source)
    assert policy.checked_at == original.checked_at
    assert policy.verified_until == original.verified_until
    clock.advance(seconds=41)
    with pytest.raises(StaleEvidence):
        memory.begin_action(action.id, lease_token=lease.token)


@pytest.mark.parametrize("count", [127, 128])
def test_full_capacity_derivation_does_not_block_optional_entity_aggregation(
    sql_storage, clock, count
):
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    memory = AlwaysOnMemory(kb, clock=clock)
    sources = [Fact(f"Observation {i}") for i in range(count)]
    kb.replace_facts(sources)
    basis = memory.capture_evidence(tuple(("owner", f.id) for f in sources))
    derived = memory.derive_fact("Deployment region Europe", basis=basis)
    facts = kb.list_facts()
    next(f for f in facts if f.id == derived.id).entity = "deployment"
    next(f for f in facts if f.id == derived.id).attribute = "region"
    kb.replace_facts(facts)
    inserted = kb.add_resolved([structured("green", "status")])
    assert any(f.attribute == "status" for f in inserted)
    assert not any(f.attribute == "_aggregate" for f in inserted)
    assert (
        len(memory.independent_evidence(memory.capture_evidence((("owner", derived.id),)))) == count
    )


def test_legacy_summary_capacity_does_not_block_pg_primary_write(kb, clock):
    import json

    class Provider:
        name = "review-offline"
        default_model = "fake"

        def __init__(self, attribute, value):
            self.output = [
                {
                    "content": f"Deployment {attribute} is {value}",
                    "entity": "deployment",
                    "attribute": attribute,
                    "value": value,
                }
            ]

        def call(self, *args, **kwargs):
            return json.dumps(self.output)

    for attribute, value in (("region", "Europe"), ("status", "green")):
        kb.replace_facts(
            kb.list_facts()
            + [
                Fact(
                    f"Deployment {attribute} is {value}. Observation {i}",
                    type=MemoryType.EPISODIC,
                    created_at=clock() - timedelta(days=2),
                    valid_from=clock() - timedelta(days=2),
                )
                for i in range(65)
            ]
        )
        kb._default_provider = Provider(attribute, value)
        assert asyncio.run(kb.consolidate_episodic()) == 1
    (accepted,) = kb.add_resolved([Fact("Unrelated accepted claim")])
    assert kb.get(accepted.id).is_active(clock())
    assert not [f for f in kb.list_facts() if f.attribute == "_aggregate" and f.is_active(clock())]


def test_required_modern_consolidation_over_capacity_still_rolls_back(sql_storage, clock):
    import json

    from ai_knot.always_on import ConsolidationProposal
    from ai_knot.always_on.maintenance import commit_consolidation

    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    memory = AlwaysOnMemory(kb, clock=clock)
    sources = [Fact(f"Observation {i}") for i in range(130)]
    kb.replace_facts(sources)
    episodes = []
    for group in (sources[:65], sources[65:]):
        summary = memory.derive_fact(
            "Deployment event", basis=memory.capture_evidence(tuple(("owner", f.id) for f in group))
        )
        episodes.append(summary)
    facts = kb.list_facts()
    for fact in facts:
        if fact.id in {f.id for f in episodes}:
            fact.type = MemoryType.EPISODIC
            fact.created_at = fact.valid_from = clock() - timedelta(days=2)
    kb.replace_facts(facts)
    refs = memory.capture_evidence(tuple(("owner", f.id) for f in episodes))
    output = Fact(
        "Required summary", qualifiers={"consolidation_ids": json.dumps([f.id for f in episodes])}
    )
    with pytest.raises(ValueError, match="1..128"):
        commit_consolidation(
            memory, ConsolidationProposal((output,), tuple(f.id for f in episodes), refs), refs
        )
    assert all(kb.get(f.id).is_active(clock()) for f in episodes)
    assert output.id not in {f.id for f in kb.list_facts()}


@pytest.mark.parametrize("future", [False, True])
def test_timed_aggregate_cannot_be_captured_outside_direct_input_window(kb, clock, future):
    memory = AlwaysOnMemory(kb, clock=clock)
    now = clock()
    deadline = now + timedelta(hours=1)
    build = structured("42", "build")
    build.valid_from = now + timedelta(days=1) if future else now
    build.valid_until = None if future else deadline
    inserted = kb.add_resolved([build, structured("Europe", "region")])
    if future:
        assert not any(f.attribute == "_aggregate" for f in inserted)
    else:
        output = next(f for f in inserted if f.attribute == "_aggregate")
        assert memory.capture_evidence((("owner", output.id),))
        clock.advance(hours=1)
        assert output.id not in {
            f.id for f in kb.recall_facts("Deployment build 42", now=clock(), top_k=30)
        }
        assert kb.recall_by_tag("aggregate", now=clock()) == []
        with pytest.raises(StaleEvidence):
            memory.capture_evidence((("owner", output.id),))


def test_publishing_timed_aggregate_keeps_its_expiry(kb, sql_storage, clock, monkeypatch):
    from datetime import datetime

    class PublicationClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock()

    monkeypatch.setattr("ai_knot.pool.datetime", PublicationClock)
    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock)
    pool.register("owner")
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    build = structured("42", "build")
    build.valid_until = clock() + timedelta(hours=1)
    output = next(
        f
        for f in kb.add_resolved([build, structured("Europe", "region")])
        if f.attribute == "_aggregate"
    )
    (published,) = pool.publish("owner", [output.id], kb=kb)
    assert memory.capture_evidence((("__shared__", published.id),))
    assert published.id in {
        f.id for f, _ in pool.recall("Deployment 42 Europe", "owner", top_k=30, now=clock())
    }
    clock.advance(hours=1)
    assert published.id not in {
        f.id for f, _ in pool.recall("Deployment 42 Europe", "owner", top_k=30, now=clock())
    }
    with pytest.raises(StaleEvidence):
        memory.capture_evidence((("__shared__", published.id),))
