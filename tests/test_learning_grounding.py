"""Automatic extraction and projections retain the evidence they actually accepted."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from ai_knot import KnowledgeBase
from ai_knot.always_on.changes import evidence_revision
from ai_knot.always_on.lineage import evidence_dependencies, evidence_roots
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import ConversationTurn, Fact, MemoryOp, MemoryType


class JsonProvider:
    name = "grounding-test"
    default_model = "offline"

    def __init__(self, output):
        self.output = output

    def call(self, system_prompt, user_content, model, *, timeout=None):
        return json.dumps(self.output)


def entry(value, attribute="city", **kwargs):
    return {
        "content": f"Alice {attribute} is {value}",
        "entity": "alice",
        "attribute": attribute,
        "value": value,
        **kwargs,
    }


def learn(kb, source, *entries):
    return kb.learn([ConversationTurn("user", source)], provider=JsonProvider(entries))


def seed(kb):
    learn(
        kb, "Alice city is Moscow. Alice job is engineer", entry("Moscow"), entry("engineer", "job")
    )
    return next(f for f in kb.list_facts() if f.attribute == "city")


def aggregate(kb):
    return next(f for f in kb.list_facts() if f.attribute == "_aggregate" and f.is_active())


@pytest.fixture(params=["yaml", "sqlite"])
def kb(request, tmp_path):
    storage = (
        YAMLStorage(str(tmp_path))
        if request.param == "yaml"
        else SQLiteStorage(str(tmp_path / "memory.db"))
    )
    return KnowledgeBase("owner", storage=storage, embed_url="")


@pytest.mark.parametrize(
    "op,value", [("ADD", "Atlantis"), ("UPDATE", "Atlantis"), ("ADD", "Moscow")]
)
def test_rejected_extraction_cannot_replace_or_reinforce_accepted_slot(kb, op, value):
    original = seed(kb)
    facts = kb.list_facts()
    next(f for f in facts if f.id == original.id).state_confidence = 0.4
    kb.replace_facts(facts)
    before = evidence_revision(kb.get(original.id))
    learn(kb, "Let's discuss gardening", entry(value, op=op))
    assert kb.get(original.id).is_active()
    assert evidence_revision(kb.get(original.id)) == before
    assert "Atlantis" not in kb.recall("Alice Atlantis", top_k=30)
    assert all(
        "Atlantis" not in f.content for f in kb.list_facts() if f.supported and f.is_active()
    )


def test_low_overlap_extracted_delete_and_host_manual_contract(kb):
    original = seed(kb)
    learn(kb, "Forget that", entry("", op="DELETE"))
    assert not kb.get(original.id).is_active()
    assert not [f for f in kb.list_facts() if f.attribute == "_aggregate" and f.is_active()]
    # A trusted host may deliberately store an unsupported declaration via the manual API.
    host = Fact(
        "Host city Atlantis",
        entity="host",
        attribute="city",
        value_text="Atlantis",
        supported=False,
    )
    assert kb.add_resolved([host])[0].id == host.id
    assert kb.get(host.id).supported is False
    assert kb.add("Manual independent assertion").supported


@pytest.mark.parametrize("change", ["correct", "delete", "forget"])
def test_aggregate_is_revision_bound_and_expires_when_source_group_shrinks(kb, change):
    original = seed(kb)
    old = aggregate(kb)
    roots = evidence_roots("owner", old)
    assert {r.fact_id for r in roots} == {
        f.id for f in kb.list_facts() if f.attribute in ("city", "job")
    }
    if change == "correct":
        learn(kb, "Alice city is Kazan", entry("Kazan"))
        assert "Kazan" in aggregate(kb).content
        assert not kb.get(old.id).is_active()
    elif change == "delete":
        kb.add_resolved([Fact("Forget city", slot_key="alice::city", op=MemoryOp.DELETE)])
        assert not kb.get(old.id).is_active()
    else:
        kb.forget(original.id)
        assert not kb.get(old.id).is_active()
    assert old.id not in {f.id for f in kb.recall_facts("Alice Moscow", top_k=30)}


def test_old_generated_rows_fail_closed_without_inventing_revisions(kb):
    source = kb.add("Manual city Moscow")
    old = Fact("Old automatic Atlantis", verification_source="aggregate", tags=["old"])
    summary = Fact(
        "Old episodic Atlantis",
        qualifiers={"consolidation_ids": json.dumps([source.id])},
        tags=["old"],
    )
    kb.replace_facts([source, old, summary])
    assert not kb.recall_facts("Atlantis", top_k=30)
    assert kb.recall_by_tag("old") == []
    assert source.id in {f.id for f in kb.recall_facts("Manual city Moscow")}
    assert kb.get(old.id).qualifiers == {}


@pytest.mark.parametrize("change", ["content", "delete"])
def test_legacy_summary_archives_exact_sources_and_retracts(kb, monkeypatch, change):
    ago = datetime.now(UTC) - timedelta(days=2)
    source = Fact("Deployment build 42", type=MemoryType.EPISODIC, created_at=ago, valid_from=ago)
    kb.replace_facts([source])
    kb._default_provider = JsonProvider([{"content": "Deployment build 42 summary"}])
    assert asyncio.run(kb.consolidate_episodic()) == 1
    summary = next(f for f in kb.list_facts() if f.type == MemoryType.SEMANTIC)
    (root,) = evidence_roots("owner", summary)
    assert root.fact_id == source.id
    assert root.revision == evidence_revision(kb.get(source.id))
    assert summary.id in {f.id for f in kb.recall_facts("Deployment summary")}
    if change == "delete":
        kb.forget(source.id)
    else:
        facts = kb.list_facts()
        next(f for f in facts if f.id == source.id).content = "Deployment failed"
        kb.replace_facts(facts)
    assert summary.id not in {f.id for f in kb.recall_facts("Deployment summary")}


def test_aggregate_rebuilds_same_text_after_source_revision_changes(kb):
    source = seed(kb)
    old = aggregate(kb)
    learn(kb, "Alice city is Moscow", entry("Moscow"))
    rebuilt = aggregate(kb)
    assert rebuilt.id != old.id
    assert rebuilt.content == old.content
    refs = evidence_dependencies(rebuilt)
    assert next(r for r in refs if r.fact_id == source.id).revision == evidence_revision(
        kb.get(source.id)
    )


def test_aggregate_accepts_summary_grounded_in_archived_episode(kb):
    ago = datetime.now(UTC) - timedelta(days=2)
    source = Fact("Alice city Moscow", type=MemoryType.EPISODIC, created_at=ago, valid_from=ago)
    kb.replace_facts([source])
    kb._default_provider = JsonProvider([entry("Moscow")])
    assert asyncio.run(kb.consolidate_episodic()) == 1
    kb.add_resolved(
        [Fact("Alice job engineer", entity="alice", attribute="job", value_text="engineer")]
    )
    assert "Moscow" in aggregate(kb).content


def test_unsupported_manual_source_cannot_become_supported_aggregate(kb):
    kb.add_resolved(
        [
            Fact("Alice city Atlantis", entity="alice", attribute="city", supported=False),
            Fact("Alice job engineer", entity="alice", attribute="job"),
        ]
    )
    assert not [f for f in kb.list_facts() if f.attribute == "_aggregate" and f.is_active()]


def _legacy_entity_summary(kb, count, attribute, value):
    ago = datetime.now(UTC) - timedelta(days=2)
    kb.replace_facts(
        kb.list_facts()
        + [
            Fact(
                f"Alice {attribute} is {value}. Observation {i}",
                type=MemoryType.EPISODIC,
                created_at=ago,
                valid_from=ago,
            )
            for i in range(count)
        ]
    )
    kb._default_provider = JsonProvider([entry(value, attribute)])
    assert asyncio.run(kb.consolidate_episodic()) == 1


@pytest.mark.parametrize("operation", ["add", "delete", "forget"])
def test_optional_aggregate_capacity_preserves_primary_mutation(kb, operation):
    _legacy_entity_summary(kb, 65, "region", "Europe")
    kb.add_resolved(
        [Fact("Alice status booting", entity="alice", attribute="status", value_text="booting")]
    )
    old = aggregate(kb)
    target = kb.add_resolved([Fact("Unrelated current target", slot_key="other::target")])[0]
    _legacy_entity_summary(kb, 65, "status", "green")
    if operation == "add":
        inserted = kb.add_resolved([Fact("Unrelated accepted claim")])
        assert inserted[0].content == "Unrelated accepted claim"
        assert kb.get(inserted[0].id).is_active()
    elif operation == "delete":
        kb.add_resolved([Fact("Delete target", slot_key="other::target", op=MemoryOp.DELETE)])
        assert not kb.get(target.id).is_active()
    else:
        kb.forget(target.id)
        with pytest.raises(KeyError):
            kb.get(target.id)
    assert not kb.get(old.id).is_active()
    assert not [f for f in kb.list_facts() if f.attribute == "_aggregate" and f.is_active()]
    assert {
        f.value_text
        for f in kb.list_facts()
        if f.attribute in ("region", "status") and f.is_active()
    } == {"Europe", "green"}


@pytest.mark.parametrize("count,available", [(63, True), (64, False)])
def test_optional_aggregate_checks_dependency_bound_without_truncation(kb, count, available):
    for attribute, value in (("region", "Europe"), ("status", "green")):
        _legacy_entity_summary(kb, count, attribute, value)
    kb.add_resolved([Fact("Unrelated accepted claim")])
    outputs = [f for f in kb.list_facts() if f.attribute == "_aggregate" and f.is_active()]
    assert bool(outputs) is available
    if available:
        assert len(evidence_roots("owner", outputs[0])) == 126
        assert len(evidence_dependencies(outputs[0])) == 128


@pytest.mark.parametrize("timing", ["future", "expired"])
def test_aggregate_excludes_inactive_incoming_inputs_and_tag_reads(kb, timing):
    now = datetime.now(UTC)
    timed = Fact(
        "Alice private future value",
        entity="alice",
        attribute="timed",
        tags=["timed"],
        valid_from=now + timedelta(days=1) if timing == "future" else now - timedelta(days=2),
        valid_until=None if timing == "future" else now - timedelta(days=1),
    )
    kb.add_resolved(
        [
            timed,
            Fact("Alice region Europe", entity="alice", attribute="region"),
            Fact("Alice status green", entity="alice", attribute="status"),
        ]
    )
    assert "private future value" not in aggregate(kb).content
    assert timed.id not in {
        f.id for f in kb.recall_facts("Alice private future value", now=now, top_k=30)
    }
    assert kb.recall_by_tag("timed", include_unsupported=True) == []
    assert kb.get(timed.id).id == timed.id
    assert timed.id in {f.id for f in kb.list_facts()}


def test_timed_aggregate_expires_at_earliest_direct_input_deadline(kb):
    now = datetime.now(UTC)
    deadline = now + timedelta(hours=1)
    kb.add_resolved(
        [
            Fact("Alice build 42", entity="alice", attribute="build", valid_until=deadline),
            Fact(
                "Alice region Europe",
                entity="alice",
                attribute="region",
                valid_until=deadline + timedelta(hours=1),
            ),
        ]
    )
    output = aggregate(kb)
    assert output.id not in {
        f.id for f in kb.recall_facts("Alice build 42", now=deadline, top_k=30)
    }
    assert kb.recall_by_tag("aggregate", now=deadline) == []
    assert output.id in {
        f.id for f in kb.recall_by_tag("aggregate", now=deadline - timedelta(microseconds=1))
    }


def test_manual_input_revision_survives_storage_roundtrip_for_aggregate(kb):
    observed = datetime.now(UTC) - timedelta(hours=1)
    first = Fact(
        "Alice region Europe",
        entity="alice",
        attribute="region",
        slot_key="alice::region",
        value_text="Europe",
        event_time=observed,
        valid_from=observed,
        valid_until=observed + timedelta(days=2),
    )
    before = evidence_revision(first)
    kb.add_resolved(
        [first, Fact("Alice status green", entity="alice", attribute="status", value_text="green")]
    )
    assert evidence_revision(kb.get(first.id)) == before
    assert aggregate(kb).id in {f.id for f in kb.recall_by_tag("aggregate")}
