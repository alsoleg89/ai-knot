"""Public state contracts run against both durable SQL backends."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from threading import Barrier

import pytest

from ai_knot import KnowledgeBase
from ai_knot.always_on import (
    AccessDenied,
    AlwaysOnMemory,
    IdempotencyConflict,
    IntentState,
    LeaseLost,
    RevisionConflict,
    StaleEvidence,
    VerificationOutcome,
    VerificationResult,
    WaitForChange,
    WaitForSlot,
    WaitUntil,
)
from ai_knot.types import Fact


def fact(store, clock, **kwargs):
    item = Fact(
        "build uses v1", created_at=clock(), last_accessed=clock(), valid_from=clock(), **kwargs
    )
    store.save("owner", [item])
    return item


def checkpoint(ao, intent, lease, state=IntentState.READY, **kwargs):
    return ao.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=ao.get_intent(intent.id).revision,
        state=state,
        progress=kwargs.pop("progress", {}),
        **kwargs,
    )


def based(ao, item):
    refs = ao.capture_evidence((("owner", item.id),))
    intent = ao.create_intent(client_key="based", goal="deploy")
    lease = ao.claim(intent.id, worker_id="w")
    checkpoint(ao, intent, lease, basis=refs)
    lease = ao.claim(intent.id, worker_id="w")
    return intent, lease, refs


def test_creation_idempotency_and_owner_isolation(sql_storage, ao_factory, clock):
    ao = ao_factory()
    intent = ao.create_intent(client_key="x", goal="ship", deadline=clock() + timedelta(seconds=50))
    assert (
        ao_factory().create_intent(
            client_key="x", goal="ship", deadline=clock() + timedelta(seconds=50)
        )
        == intent
    )
    with pytest.raises(IdempotencyConflict):
        ao.create_intent(client_key="x", goal="other")
    foreign = AlwaysOnMemory(KnowledgeBase("foreign", storage=sql_storage), clock=clock)
    assert foreign.list_intents() == []
    with pytest.raises(AccessDenied):
        foreign.get_intent(intent.id)
    with pytest.raises(AccessDenied):
        ao.get_intent("missing")


def test_atomic_claim_and_generation(sql_storage, ao_factory, clock):
    ao = ao_factory()
    intent = ao.create_intent(client_key="race", goal="ship")
    barrier = Barrier(2)

    def claim():
        other = ao_factory()
        barrier.wait()
        try:
            return other.claim(intent.id, worker_id="w")
        except RevisionConflict:
            return None

    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda _: claim(), range(2)))
    assert sum(result is not None for result in results) == 1
    first = next(result for result in results if result)
    clock.advance(60)
    second = ao_factory().claim(intent.id, worker_id="replacement")
    assert second.generation == first.generation + 1
    with pytest.raises(LeaseLost):
        ao.renew(intent.id, lease_token=first.token)
    with pytest.raises(LeaseLost):
        checkpoint(ao, intent, first)


def test_renewal_cas_deadline_and_terminal(ao_factory, clock):
    ao = ao_factory()
    intent = ao.create_intent(
        client_key="deadline", goal="ship", deadline=clock() + timedelta(seconds=100)
    )
    lease = ao.claim(intent.id, worker_id="w", ttl_seconds=120)
    revision = ao.get_intent(intent.id).revision
    clock.advance(10)
    renewed = ao.renew(intent.id, lease_token=lease.token, ttl_seconds=120)
    assert renewed.token == lease.token
    assert ao.get_intent(intent.id).revision == revision
    with pytest.raises(RevisionConflict):
        ao.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=1,
            state=IntentState.READY,
            progress={},
        )
    clock.advance(90)
    assert "deadline_exceeded" in ao.resume(intent.id).reasons
    with pytest.raises(RevisionConflict):
        checkpoint(ao, intent, lease)
    checkpoint(ao, intent, lease, state=IntentState.CANCELLED)
    with pytest.raises(RevisionConflict):
        ao.claim(intent.id, worker_id="w")


@pytest.mark.parametrize("target", list(IntentState))
def test_started_checkpoint_never_loses_lease(ao_factory, target):
    ao = ao_factory()
    intent = ao.create_intent(client_key="started", goal="ship")
    lease = ao.claim(intent.id, worker_id="w")
    action = ao.prepare_action(intent.id, lease_token=lease.token, action_key="deploy", request={})
    ao.begin_action(action.id, lease_token=lease.token)
    with pytest.raises(RevisionConflict):
        checkpoint(
            ao,
            intent,
            lease,
            state=target,
            **(
                {"waiting_for": WaitUntil(lease.expires_at)}
                if target == IntentState.WAITING
                else {}
            ),
        )
    assert ao.get_intent(intent.id).lease == lease


def test_unknown_blocked_checkpoint_and_cancel(ao_factory, clock):
    ao = ao_factory()
    intent = ao.create_intent(client_key="unknown", goal="ship")
    lease = ao.claim(intent.id, worker_id="w")
    action = ao.prepare_action(intent.id, lease_token=lease.token, action_key="deploy", request={})
    ao.begin_action(action.id, lease_token=lease.token)
    clock.advance(60)
    ao.resume(intent.id)
    lease = ao.claim(intent.id, worker_id="recover")
    for target in (IntentState.READY, IntentState.WAITING, IntentState.COMPLETED):
        with pytest.raises(RevisionConflict):
            checkpoint(
                ao,
                intent,
                lease,
                state=target,
                **(
                    {"waiting_for": WaitUntil(clock() + timedelta(seconds=1))}
                    if target == IntentState.WAITING
                    else {}
                ),
            )
    blocked = checkpoint(ao, intent, lease, state=IntentState.BLOCKED)
    assert blocked.blocked_reasons == ("unknown_action",)
    lease = ao.claim(intent.id, worker_id="recover")
    checkpoint(ao, intent, lease, state=IntentState.CANCELLED)
    assert ao.resume(intent.id).open_actions[0].state == "unknown"


def test_changed_evidence_blocks_inherited_action_and_persists_reason(
    sql_storage, ao_factory, clock
):
    ao = ao_factory()
    item = fact(sql_storage, clock)
    intent, lease, refs = based(ao, item)
    action = ao.prepare_action(intent.id, lease_token=lease.token, action_key="deploy", request={})
    assert action.basis == refs
    sql_storage.save("owner", [replace(item, content="build uses v2")])
    with pytest.raises(StaleEvidence):
        ao.begin_action(action.id, lease_token=lease.token)
    assert ao_factory().get_intent(intent.id).state == IntentState.BLOCKED
    assert "changed" in ao.resume(intent.id).reasons


def test_telemetry_does_not_invalidate_but_temporal_and_support_do(sql_storage, ao_factory, clock):
    ao = ao_factory()
    item = fact(sql_storage, clock)
    intent, lease, refs = based(ao, item)
    sql_storage.save(
        "owner", [replace(item, access_count=10, importance=0.1, tags=["a"], version=10)]
    )
    assert ao.capture_evidence((("owner", item.id),)) == refs
    sql_storage.save("owner", [replace(item, supported=False)])
    assert "changed" in ao.resume(intent.id).reasons


@pytest.mark.parametrize("waiting", ["time", "slot", "change"])
def test_wait_survives_restart_without_recall(sql_storage, ao_factory, clock, waiting):
    ao = ao_factory()
    item = fact(sql_storage, clock, slot_key="ci::status", value_text="pending")
    refs = ao.capture_evidence((("owner", item.id),))
    intent = ao.create_intent(client_key="wait", goal="ship")
    lease = ao.claim(intent.id, worker_id="w")
    condition = {
        "time": WaitUntil(clock() + timedelta(seconds=10)),
        "slot": WaitForSlot("owner", "ci::status", "passed"),
        "change": WaitForChange(refs[0]),
    }[waiting]
    checkpoint(
        ao, intent, lease, state=IntentState.WAITING, waiting_for=condition, next_step="deploy"
    )
    assert not ao_factory().resume(intent.id).ready
    if waiting == "time":
        clock.advance(10)
    else:
        sql_storage.save("owner", [replace(item, value_text="passed", content="ci passed")])
    assert ao_factory().resume(intent.id).ready
    assert sql_storage.load("owner")[0].access_count == 0


def test_freshness_failure_blocks_before_old_validity_and_backoff(sql_storage, ao_factory, clock):
    ao = ao_factory()
    item = fact(sql_storage, clock)
    ref = ao.capture_evidence((("owner", item.id),))[0]
    policy = ao.set_freshness(ref, source_ref="host:build", recheck_after_seconds=600)
    assert policy.checked_at is None
    assert ao.due_verifications()[0].ref == ref
    confirmed = ao.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"etag": "1"}, "v1")
    )
    assert not ao.due_verifications()
    intent, lease, _ = based(ao, item)
    action = ao.prepare_action(intent.id, lease_token=lease.token, action_key="deploy", request={})
    unavailable = ao.record_verification(
        ref, result=VerificationResult(VerificationOutcome.UNAVAILABLE, {"error": "outage"}, "")
    )
    assert unavailable.verified_until == confirmed.verified_until
    assert unavailable.checked_at == confirmed.checked_at
    assert unavailable.retry_at == clock() + timedelta(seconds=30)
    assert not ao.due_verifications()
    with pytest.raises(StaleEvidence):
        ao.begin_action(action.id, lease_token=lease.token)
    clock.advance(30)
    assert ao.due_verifications()[0].ref == ref
    again = ao.record_verification(
        ref, result=VerificationResult(VerificationOutcome.UNAVAILABLE, {}, "")
    )
    assert again.retry_at == clock() + timedelta(seconds=60)


def test_freshness_slot_replacement_due_immediately(sql_storage, ao_factory, clock):
    ao = ao_factory()
    item = fact(sql_storage, clock, slot_key="build::version", value_text="v1")
    ref = ao.capture_evidence((("owner", item.id),))[0]
    ao.set_freshness(ref, source_ref="host:build", recheck_after_seconds=600)
    ao.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"etag": "1"}, "1")
    )
    replacement = replace(item, id="replacement", content="v2", value_text="v2")
    sql_storage.save("owner", [replacement])
    due = ao.due_verifications()
    assert due[0].ref.fact_id == replacement.id
    with pytest.raises(StaleEvidence):
        ao.record_verification(
            ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"etag": "old"}, "1")
        )
    ao.record_verification(
        due[0].ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"etag": "2"}, "2")
    )
    assert not ao.due_verifications()


def test_action_idempotency_single_permit_and_finish(ao_factory):
    ao = ao_factory()
    intent = ao.create_intent(client_key="action", goal="ship")
    lease = ao.claim(intent.id, worker_id="w")
    action = ao.prepare_action(
        intent.id, lease_token=lease.token, action_key="deploy", request={"a": 1}
    )
    assert (
        ao_factory().prepare_action(
            intent.id, lease_token=lease.token, action_key="deploy", request={"a": 1}
        )
        == action
    )
    with pytest.raises(IdempotencyConflict):
        ao.prepare_action(intent.id, lease_token=lease.token, action_key="deploy", request={"a": 2})
    ao.begin_action(action.id, lease_token=lease.token)
    with pytest.raises(RevisionConflict):
        ao.begin_action(action.id, lease_token=lease.token)
    done = ao.finish_action(
        action.id, lease_token=lease.token, outcome="succeeded", result={"id": 1}
    )
    assert (
        ao_factory().finish_action(
            action.id, lease_token=lease.token, outcome="succeeded", result={"id": 1}
        )
        == done
    )
    with pytest.raises(RevisionConflict):
        ao.finish_action(action.id, lease_token=lease.token, outcome="succeeded", result={"id": 2})


@pytest.mark.parametrize(
    "operation",
    [
        lambda ao: ao.list_intents(limit=True),
        lambda ao: ao.list_intents(limit=501),
        lambda ao: ao.create_intent(client_key="", goal="x"),
        lambda ao: ao.create_intent(
            client_key="x", goal="x", deadline=__import__("datetime").datetime(2026, 1, 1)
        ),
    ],
)
def test_input_rejected_without_rows(ao_factory, operation):
    ao = ao_factory()
    with pytest.raises(ValueError):
        operation(ao)
    assert ao.list_intents() == []


def test_shared_acl_revocation_redacts_new_evidence(sql_storage, clock):
    from ai_knot.pool import SharedMemoryPool

    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True, clock=clock)
    pool.grant_read("owner", "team:ops")
    item = Fact(
        "private deployment target",
        origin_agent_id="publisher",
        visibility_scope="team:ops",
        memory_tier="pool",
        created_at=clock(),
        last_accessed=clock(),
        valid_from=clock(),
    )
    sql_storage.save("__shared__", [item])
    ao = AlwaysOnMemory(KnowledgeBase("owner", storage=sql_storage), pool=pool, clock=clock)
    refs = ao.capture_evidence((("__shared__", item.id),))
    intent = ao.create_intent(client_key="shared", goal="ship")
    lease = ao.claim(intent.id, worker_id="w")
    checkpoint(ao, intent, lease, basis=refs)
    pool.revoke_read("owner", "team:ops")
    packet = ao.resume(intent.id)
    assert packet.stale_basis[0].reason == "access_changed"
    assert packet.stale_basis[0].ref is None
    with pytest.raises(AccessDenied):
        ao.capture_evidence((("__shared__", item.id),))


def test_retryable_reconciliation_preserves_external_key(ao_factory, clock):
    ao = ao_factory()
    intent = ao.create_intent(client_key="retry", goal="ship")
    lease = ao.claim(intent.id, worker_id="w")
    action = ao.prepare_action(intent.id, lease_token=lease.token, action_key="deploy", request={})
    ao.begin_action(action.id, lease_token=lease.token)
    clock.advance(60)
    ao.resume(intent.id)
    lease = ao.claim(intent.id, worker_id="recover")
    retry = ao.reconcile_action(
        action.id,
        lease_token=lease.token,
        outcome="retryable",
        evidence={"remote_proves_absent": True},
    )
    assert retry.state == "prepared"
    assert retry.idempotency_key == action.idempotency_key
    assert ao.get_intent(intent.id).state == IntentState.RUNNING
    permit = ao.begin_action(action.id, lease_token=lease.token)
    assert permit.idempotency_key == action.idempotency_key


def test_proven_success_survives_stale_historical_basis(sql_storage, ao_factory, clock):
    ao = ao_factory()
    item = fact(sql_storage, clock)
    intent, lease, refs = based(ao, item)
    action = ao.prepare_action(intent.id, lease_token=lease.token, action_key="deploy", request={})
    ao.begin_action(action.id, lease_token=lease.token)
    replacement = replace(item, content="new target")
    sql_storage.save("owner", [replacement])
    clock.advance(60)
    ao.resume(intent.id)
    lease = ao.claim(intent.id, worker_id="recover")
    resolved = ao.reconcile_action(
        action.id,
        lease_token=lease.token,
        outcome="succeeded",
        evidence={"remote_effect": "recorded"},
    )
    assert resolved.state == "succeeded"
    assert ao.get_intent(intent.id).state == IntentState.BLOCKED
    current = ao.capture_evidence((("owner", item.id),))
    checkpoint(ao, intent, lease, basis=current)
    lease = ao.claim(intent.id, worker_id="finish")
    checkpoint(ao, intent, lease, state=IntentState.COMPLETED, basis=current)
    assert ao.resume(intent.id).intent.state == IntentState.COMPLETED


def test_bad_json_and_confirmation_do_not_write(sql_storage, ao_factory, clock):
    ao = ao_factory()
    intent = ao.create_intent(client_key="bad", goal="ship")
    lease = ao.claim(intent.id, worker_id="w")
    for bad in ({"x": float("nan")}, {"x": "я" * 40000}, {"": 1}, {"x": object()}):
        with pytest.raises(ValueError):
            checkpoint(ao, intent, lease, progress=bad)
    item = fact(sql_storage, clock)
    ref = ao.capture_evidence((("owner", item.id),))[0]
    ao.set_freshness(ref, source_ref="host", recheck_after_seconds=10)
    with pytest.raises(ValueError):
        ao.record_verification(
            ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {}, "")
        )
    assert ao.due_verifications()[0].ref == ref


def test_evidence_and_wait_models_reject_malformed_fields():
    from ai_knot.always_on import EvidenceRef

    with pytest.raises(ValueError):
        EvidenceRef("owner", "id", "")
    with pytest.raises(ValueError):
        WaitForSlot("owner", "", "passed")
    with pytest.raises(ValueError):
        WaitUntil(__import__("datetime").datetime(2026, 1, 1))


def test_state_schema_version_and_postgres_timestamp_types(sql_storage, ao_factory):
    from ai_knot.always_on.storage import SQLStatement, sql

    ao = ao_factory()
    with ao._transaction(write=False) as session:
        assert session.fetch_one(sql("SELECT version FROM ao_schema WHERE id=1"))["version"] >= 5
        if type(sql_storage).__name__ == "PostgresStorage":
            rows = session.fetch_all(
                SQLStatement(
                    "SELECT 1",
                    "SELECT column_name,data_type FROM information_schema.columns "
                    "WHERE table_schema=current_schema() "
                    "AND table_name='ai-knot_ao_freshness'",
                )
            )
            columns = {row["column_name"]: row["data_type"] for row in rows}
            assert columns["retry_at"] == "timestamp with time zone"
            assert columns["verified_until"] == "timestamp with time zone"


def test_yaml_explicitly_unsupported_without_sidecar(tmp_path):
    from ai_knot.always_on import UnsupportedAlwaysOnBackend
    from ai_knot.storage.yaml_storage import YAMLStorage

    kb = KnowledgeBase("owner", storage=YAMLStorage(str(tmp_path)))
    with pytest.raises(UnsupportedAlwaysOnBackend):
        AlwaysOnMemory(kb)
    assert not list(tmp_path.rglob("*.db"))


def test_opt_in_installs_complete_schema_and_lazy_maintenance_exports(ao_factory):
    import ai_knot.always_on as public
    from ai_knot.always_on.storage import sql

    names = (
        "MaintenanceBudget",
        "MaintenanceReport",
        "ConsolidationProposal",
        "VerificationProvider",
        "ConsolidationProvider",
        "run_maintenance",
    )
    for name in names:
        assert getattr(public, name) is not None
        assert name in public.__all__
    ao = ao_factory()
    with ao._transaction(write=False) as session:
        assert session.fetch_one(sql("SELECT version FROM ao_schema WHERE id=1"))["version"] == 6
        assert session.fetch_all(sql("SELECT * FROM ao_maintenance_runs")) == []


def test_freshness_internal_revision_fences_identical_policy_reset(sql_storage, ao_factory, clock):
    from ai_knot.always_on.storage import sql

    ao = ao_factory()
    item = fact(sql_storage, clock)
    ref = ao.capture_evidence((("owner", item.id),))[0]

    def revision():
        with ao._transaction(write=False) as session:
            return session.fetch_one(
                sql("SELECT revision FROM ao_freshness WHERE namespace=? AND subject_key=?"),
                ("owner", item.id),
            )["revision"]

    ao.set_freshness(ref, source_ref="host", recheck_after_seconds=10)
    first = revision()
    ao.set_freshness(ref, source_ref="host", recheck_after_seconds=10)
    assert revision() == first + 1
    ao.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "")
    )
    assert revision() == first + 2


def test_due_limit_uses_filtered_fact_reads_for_many_policies(
    sql_storage, ao_factory, clock, monkeypatch
):
    from ai_knot.always_on.changes import evidence_revision
    from ai_knot.always_on.models import FreshnessRecord
    from ai_knot.always_on.state_storage import save_freshness

    ao = ao_factory()
    items = [
        Fact(
            f"fact {i}",
            id=f"f{i:04}",
            created_at=clock(),
            last_accessed=clock(),
            valid_from=clock(),
        )
        for i in range(1000)
    ]
    sql_storage.save("owner", items)
    with ao._transaction() as session:
        for item in items:
            from ai_knot.always_on import EvidenceRef

            save_freshness(
                session,
                FreshnessRecord(
                    EvidenceRef("owner", item.id, evidence_revision(item)),
                    "fact",
                    item.id,
                    "host",
                    60,
                    None,
                    clock(),
                    clock(),
                    None,
                    0,
                ),
            )
    reads = []
    original = sql_storage._load_on_connection

    def tracked(conn, namespace, **filters):
        reads.append(filters)
        return original(conn, namespace, **filters)

    monkeypatch.setattr(sql_storage, "_load_on_connection", tracked)
    result = ao.due_verifications(limit=1)
    assert len(result) == 1
    assert len(reads) <= 3
    assert all(filters.get("fact_id") or filters.get("slot_key") for filters in reads)


def test_future_confirmed_policy_becomes_due_on_same_id_change(sql_storage, ao_factory, clock):
    ao = ao_factory()
    item = fact(sql_storage, clock)
    ref = ao.capture_evidence((("owner", item.id),))[0]
    ao.set_freshness(ref, source_ref="host", recheck_after_seconds=600)
    ao.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "")
    )
    assert ao.due_verifications() == []
    sql_storage.save("owner", [replace(item, content="changed")])
    assert ao.due_verifications()[0].ref != ref


def test_due_preserves_microsecond_activity_boundary(sql_storage, ao_factory, clock):
    ao = ao_factory()
    item = fact(sql_storage, clock, valid_until=clock() + timedelta(microseconds=1))
    ref = ao.capture_evidence((("owner", item.id),))[0]
    ao.set_freshness(ref, source_ref="host", recheck_after_seconds=60)
    assert ao.due_verifications()[0].ref == ref
    clock.advance(microseconds=1)
    assert ao.due_verifications() == []


def test_due_slot_frontier_orders_offset_timestamps_as_instants(sql_storage, ao_factory, clock):
    from datetime import timezone

    ao = ao_factory()
    current = Fact(
        "value",
        id="current",
        slot_key="s::value",
        value_text="same",
        version=1,
        created_at=clock() - timedelta(hours=1),
        valid_from=clock() - timedelta(hours=2),
    )
    older = replace(
        current,
        id="older",
        created_at=(clock() - timedelta(hours=1, minutes=30)).astimezone(
            timezone(timedelta(hours=3))
        ),
    )
    sql_storage.save("owner", [current, older])
    ref = ao.capture_evidence((("owner", current.id),))[0]
    ao.set_freshness(ref, source_ref="host", recheck_after_seconds=60)
    assert ao.due_verifications()[0].ref == ref


def test_existing_v6_policy_columns_upgrade_before_plain_fact_writer(
    sql_storage, ao_factory, clock
):
    from ai_knot.always_on.storage import sql

    ao = ao_factory()
    item = fact(sql_storage, clock)
    ref = ao.capture_evidence((("owner", item.id),))[0]
    ao.set_freshness(ref, source_ref="host", recheck_after_seconds=600)
    ao.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "")
    )
    with ao._transaction() as session:
        session.execute(sql("DROP INDEX ao_freshness_due"))
        session.execute(sql("ALTER TABLE ao_freshness DROP COLUMN dirty"))
        session.execute(sql("ALTER TABLE ao_freshness DROP COLUMN revision"))
    if type(sql_storage).__name__ == "PostgresStorage":
        from ai_knot.storage.postgres_storage import PostgresStorage

        reopened = PostgresStorage(sql_storage._dsn)
    else:
        from ai_knot.storage.sqlite_storage import SQLiteStorage

        reopened = SQLiteStorage(sql_storage._db_path)
    reopened.save("owner", [replace(item, content="new value")])
    assert ao.due_verifications()[0].ref != ref


def test_due_detects_scheduled_slot_frontier_without_new_write(sql_storage, ao_factory, clock):
    ao = ao_factory()
    old = Fact(
        "old",
        id="old",
        slot_key="release::version",
        value_text="v1",
        version=1,
        created_at=clock(),
        valid_from=clock(),
        valid_until=clock() + timedelta(seconds=30),
    )
    future = replace(
        old,
        id="future",
        content="future",
        value_text="v2",
        version=2,
        valid_from=clock() + timedelta(seconds=30),
        valid_until=None,
    )
    sql_storage.save("owner", [old, future])
    ref = ao.capture_evidence((("owner", old.id),))[0]
    ao.set_freshness(ref, source_ref="host", recheck_after_seconds=3600)
    ao.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "")
    )
    assert ao.due_verifications() == []
    clock.advance(31)
    current = ao.capture_evidence((("owner", future.id),))[0]
    assert ao.due_verifications()[0].ref == current
