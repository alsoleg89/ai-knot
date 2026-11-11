"""Admit saved context without restoring stale text or mutating durable work."""

from dataclasses import replace

import pytest

from ai_knot.always_on import EvidenceRef, StaleEvidence, VerificationOutcome, VerificationResult
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact


def test_review_keeps_valid_text_and_rejects_changed_context_without_mutation(
    ao_factory, sql_storage
):
    ao = ao_factory()
    original = [Fact(id="stable", content="Keep this"), Fact(id="changed", content="Old region")]
    sql_storage.save("owner", original)
    refs = ao.capture_evidence((("owner", "stable"), ("owner", "changed")))
    intent = ao.create_intent(client_key="task", goal="Do work")
    sql_storage.save("owner", [original[0], replace(original[1], content="New region")])
    review = ao.review_context(refs)
    assert [(item.ref.fact_id, item.content) for item in review.accepted] == [
        ("stable", "Keep this")
    ]
    assert [(issue.reason, issue.ref.fact_id) for issue in review.issues] == [
        ("changed", "changed")
    ]
    assert not review.complete and not review.omitted
    assert ao.get_intent(intent.id) == intent
    assert sql_storage.load("owner")[0].access_count == 0


def test_context_budget_preserves_input_priority_whole_text_and_dedup(ao_factory, sql_storage):
    ao = ao_factory()
    sql_storage.save("owner", [Fact(id="a", content="AAAAA"), Fact(id="z", content="ZZZ")])
    a = ao.capture_evidence((("owner", "a"),))[0]
    z = ao.capture_evidence((("owner", "z"),))[0]
    review = ao.review_context((z, a, z), max_chars=5)
    assert [item.content for item in review.accepted] == ["ZZZ"]
    assert review.omitted == (a,) and not review.complete
    assert ao.review_context((z, a), max_chars=8).complete
    assert ao.review_context(()).complete
    for value in (0, True, 65537):
        with pytest.raises(ValueError):
            ao.review_context((z,), max_chars=value)
    with pytest.raises(ValueError):
        ao.review_context((z,) * 129)
    with pytest.raises(StaleEvidence):
        ao.review_context((a, replace(a, revision="another")))


def test_review_redacts_revoked_derived_and_foreign_content(ao_factory, sql_storage):
    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True)
    pool.grant_read("owner", "team")
    ao = ao_factory(pool=pool)
    source = Fact(
        id="source", content="Sensitive region", visibility_scope="team", origin_agent_id="peer"
    )
    sql_storage.save("__shared__", [source])
    basis = ao.capture_evidence((("__shared__", source.id),))
    summary = ao.derive_fact("Deploy in sensitive region", basis=basis)
    derived = ao.capture_evidence((("owner", summary.id),))
    assert len(ao.review_context((*basis, *derived)).accepted) == 2
    pool.revoke_read("owner", "team")
    review = ao_factory(pool=pool).review_context(
        (*basis, *derived, EvidenceRef("foreign", "secret", "rev"))
    )
    assert not review.accepted and not review.complete
    assert all(issue.reason == "access_changed" and issue.ref is None for issue in review.issues)
    assert "Sensitive" not in repr(review) and "secret" not in repr(review)


def test_review_checks_freshness_time_support_and_current_slots(ao_factory, sql_storage, clock):
    ao = ao_factory()
    fact = Fact(id="region", content="EU", slot_key="deploy::region", value_text="EU")
    sql_storage.save("owner", [fact])
    refs = ao.capture_evidence((("owner", fact.id),))
    ao.set_freshness(refs[0], source_ref="region API", recheck_after_seconds=30)
    due = ao.review_context(refs)
    assert not due.accepted and due.issues[0].reason == "verification_due"
    ao.record_verification(
        refs[0], result=VerificationResult(VerificationOutcome.CONFIRMED, {"region": "EU"}, "v1")
    )
    assert ao.review_context(refs).complete
    clock.advance(seconds=30)
    assert not ao.review_context(refs).accepted
    sql_storage.save("owner", [replace(fact, valid_until=clock())])
    assert not ao.review_context(refs).accepted
