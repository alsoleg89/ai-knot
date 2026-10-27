"""Task applicability, recipient role and current evidence share one snapshot."""

from dataclasses import replace

import pytest

from ai_knot.always_on import VerificationOutcome, VerificationResult
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact, MemoryType


def test_task_scope_is_hard_but_role_is_only_a_preference(ao_factory, sql_storage):
    ao = ao_factory()
    facts = [
        Fact(
            id="operator",
            content="Rollback checkpoint operator",
            qualifiers={"task_key": "release42", "for_role": "operator"},
        ),
        Fact(
            id="reviewer",
            content="Rollback checkpoint reviewer",
            qualifiers={"task_key": "release42", "for_role": "reviewer"},
        ),
        Fact(
            id="wrong-task",
            content="Rollback checkpoint operator",
            qualifiers={"task_key": "release41", "for_role": "operator"},
        ),
        Fact(
            id="role-only", content="General operator duties", qualifiers={"for_role": "operator"}
        ),
        Fact(id="prerequisite", content="Backup restoration test is required"),
    ]
    sql_storage.save("owner", facts)
    for role in ("operator", "reviewer"):
        view = ao.select_context(
            ("rollback checkpoint",), task_key="release42", role=role, max_facts=1
        )
        assert view.complete
        assert [item.ref.fact_id for item in view.facts] == [role]
    view = ao.select_context(
        ("backup restoration",), task_key="release42", role="operator", max_facts=1
    )
    assert [item.ref.fact_id for item in view.facts] == ["prerequisite"]
    assert not ao.select_context(("rollback checkpoint",), role="operator").complete


def test_role_only_overlap_cannot_manufacture_requirement_coverage(ao_factory, sql_storage):
    ao = ao_factory()
    sql_storage.save(
        "owner", [Fact(id="role", content="Operator on call", qualifiers={"for_role": "operator"})]
    )
    view = ao.select_context(("restore certificates",), role="operator")
    assert not view.facts and not view.complete
    assert view.covered_requirements == ()
    assert view.missing_requirements == ("restore certificates",)


@pytest.mark.parametrize("surface", ["stopwords", "tags", "canonical", "snippets"])
def test_coverage_requires_content_bearing_words_in_emitted_text(ao_factory, sql_storage, surface):
    ao = ao_factory()
    fact = Fact(id="unrelated", content="Lunch starts at noon")
    requirement = "certificate expiry"
    if surface == "stopwords":
        fact.content = "How to rotate passwords"
        requirement = "How to restore certificates"
    elif surface == "tags":
        fact.tags = ["certificate", "expiry"]
    elif surface == "canonical":
        fact.canonical_surface = "certificate expiry"
    else:
        fact.source_snippets = ["certificate expiry"]
    sql_storage.save("owner", [fact])
    view = ao.select_context((requirement,))
    assert not view.complete and not view.facts
    assert view.missing_requirements == (requirement,)


def test_stopword_hits_do_not_hide_ubiquitous_original_content(ao_factory, sql_storage):
    ao = ao_factory()
    facts = [Fact(id=f"valid-{i}", content=f"Recover certificates daily {i}") for i in range(10)]
    facts.append(Fact(id="unrelated", content="How to rotate passwords"))
    sql_storage.save("owner", facts)
    view = ao.select_context(("How to recover certificates",), max_facts=1)
    assert view.complete and view.facts[0].ref.fact_id.startswith("valid-")


def test_whole_fact_budget_reports_only_selected_coverage(ao_factory, sql_storage):
    ao = ao_factory()
    short = Fact(id="short", content="Rollback checkpoint ready")
    other = Fact(id="other", content="Backup restoration verified")
    verbose = Fact(id="verbose", content="Rollback checkpoint " * 40)
    sql_storage.save("owner", [verbose, other, short])
    requirements = ("rollback checkpoint", "backup restoration", "certificate expiry")
    ceiling = len(short.content) + len(other.content)
    view = ao.select_context(requirements, max_chars=ceiling)
    assert {item.ref.fact_id for item in view.facts} == {"short", "other"}
    assert view.covered_requirements == requirements[:2]
    assert view.missing_requirements == requirements[2:] and not view.complete
    assert view.budget_limited
    assert sum(len(item.content) for item in view.facts) <= ceiling


def test_private_and_shared_same_id_remain_distinct_evidence(ao_factory, sql_storage):
    pool = SharedMemoryPool(sql_storage, persist_stats=True)
    ao = ao_factory(pool=pool)
    sql_storage.save("owner", [Fact(id="same", content="Restore backups")])
    sql_storage.save("__shared__", [Fact(id="same", content="Rotate certificates")])
    view = ao.select_context(("restore backups", "rotate certificates"))
    assert view.complete
    assert {(item.ref.namespace, item.ref.fact_id) for item in view.facts} == {
        ("owner", "same"),
        ("__shared__", "same"),
    }
    assert ao.review_context(tuple(item.ref for item in view.facts)).complete


@pytest.mark.parametrize("mutation", ["correct", "delete", "revoke"])
def test_source_retraction_removes_task_procedure_after_reopening(
    ao_factory, sql_storage, mutation
):
    pool = SharedMemoryPool(sql_storage, persist_stats=True)
    pool.grant_read("owner", "team")
    ao = ao_factory(pool=pool)
    source = Fact(
        id="source", content="Observed tool result", visibility_scope="team", origin_agent_id="peer"
    )
    sql_storage.save("__shared__", [source])
    procedure = ao.derive_fact(
        "Restore procedure uses checkpoint",
        basis=ao.capture_evidence((("__shared__", "source"),)),
        memory_type=MemoryType.PROCEDURAL,
        task_key="release42",
        for_role="operator",
    )
    view = ao.select_context(("restore procedure",), task_key="release42", role="operator")
    assert view.complete and view.facts[0].ref.fact_id == procedure.id
    if mutation == "correct":
        sql_storage.save("__shared__", [replace(source, content="Retracted tool result")])
    elif mutation == "delete":
        sql_storage.delete("__shared__", source.id)
    else:
        pool.revoke_read("owner", "team")
    fresh = ao_factory(pool=pool).select_context(
        ("restore procedure",), task_key="release42", role="operator"
    )
    assert not fresh.facts and not fresh.complete
    assert "source" not in repr(fresh)


def test_freshness_conflicts_and_raw_episodes_do_not_fill_a_requirement(ao_factory, sql_storage):
    ao = ao_factory()
    a = Fact(id="a", content="Rollback checkpoint", slot_key="release::region", value_text="EU")
    b = Fact(id="b", content="Rollback checkpoint", slot_key="release::region", value_text="US")
    episode = Fact(id="episode", content="Rollback checkpoint", type=MemoryType.EPISODIC)
    sql_storage.save("owner", [a, b, episode])
    assert not ao.select_context(("rollback checkpoint",)).facts
    sql_storage.save("owner", [a])
    ref = ao.capture_evidence((("owner", "a"),))[0]
    ao.set_freshness(ref, source_ref="checkpoint API", recheck_after_seconds=30)
    assert not ao.select_context(("rollback checkpoint",)).complete
    ao.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"checkpoint": True}, "v1")
    )
    assert ao.select_context(("rollback checkpoint",)).complete


def test_common_original_tokens_are_relevant_without_role_terms(ao_factory, sql_storage):
    ao = ao_factory()
    sql_storage.save(
        "owner", [Fact(id=f"f{i}", content=f"Rollback checkpoint {i}") for i in range(8)]
    )
    assert ao.select_context(("rollback checkpoint",), max_facts=1).complete


def test_full_context_budget_does_not_revalidate_every_matching_candidate(
    ao_factory, sql_storage, monkeypatch
):
    ao = ao_factory()
    sql_storage.save(
        "owner", [Fact(id=f"f{i:04}", content="Rollback checkpoint") for i in range(200)]
    )
    original = ao._validate_basis
    checks = 0

    def counted(*args, **kwargs):
        nonlocal checks
        checks += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ao, "_validate_basis", counted)
    view = ao.select_context(("rollback checkpoint",), max_chars=19, max_facts=8)
    assert view.complete and len(view.facts) == 1 and view.budget_limited
    assert checks <= 2  # One selected fact and one usable budget omission suffice.


def test_typed_derivation_metadata_is_atomic_and_defaults_stay_compatible(
    ao_factory, kb, sql_storage
):
    ao = ao_factory()
    source = kb.add("Recorded successful restoration")
    basis = ao.capture_evidence((("owner", source.id),))
    generic = ao.derive_fact("Generic observation", basis=basis)
    assert generic.type == MemoryType.SEMANTIC and "task_key" not in generic.qualifiers
    lesson = ao.derive_fact(
        "Check backups before restoration",
        basis=basis,
        memory_type=MemoryType.PROCEDURAL,
        task_key="release42",
        for_role="operator",
    )
    stored = next(f for f in sql_storage.load("owner") if f.id == lesson.id)
    assert stored.type == MemoryType.PROCEDURAL
    assert (
        stored.qualifiers["task_key"] == "release42" and stored.qualifiers["for_role"] == "operator"
    )
    assert ao.independent_evidence(ao.capture_evidence((("owner", lesson.id),))) == basis
    before = len(sql_storage.load("owner"))
    for key, value in (
        ("task_key", " spaced"),
        ("for_role", "x" * 257),
        ("memory_type", "unknown"),
    ):
        with pytest.raises(ValueError):
            ao.derive_fact("Invalid", basis=basis, **{key: value})
    assert len(sql_storage.load("owner")) == before


def test_applicability_change_invalidates_saved_context_revision(ao_factory, sql_storage):
    ao = ao_factory()
    fact = Fact(
        id="procedure",
        content="Rollback checkpoint",
        qualifiers={"task_key": "release42", "for_role": "operator"},
    )
    sql_storage.save("owner", [fact])
    view = ao.select_context(("rollback checkpoint",), task_key="release42", role="operator")
    assert view.complete
    sql_storage.save(
        "owner", [replace(fact, qualifiers={"task_key": "release41", "for_role": "reviewer"})]
    )
    assert not ao.review_context(tuple(item.ref for item in view.facts)).accepted
    assert not ao.select_context(("rollback checkpoint",), task_key="release42").complete


def test_selection_validation_and_read_only_state(ao_factory, kb, sql_storage):
    ao = ao_factory()
    fact = kb.add("Rollback checkpoint")
    intent = ao.create_intent(client_key="work", goal="Review release")
    for requirements in ("cpu", (), ("",), ("and",), ("same", "same"), ("q",) * 6, ("x" * 4097,)):
        with pytest.raises(ValueError):
            ao.select_context(requirements)
    for key, value in (("max_facts", True), ("max_facts", 129), ("max_chars", 0), ("role", " x")):
        with pytest.raises(ValueError):
            ao.select_context(("rollback checkpoint",), **{key: value})
    assert ao.select_context(("rollback checkpoint",)).complete
    assert ao.get_intent(intent.id) == intent
    assert next(f for f in sql_storage.load("owner") if f.id == fact.id).access_count == 0
