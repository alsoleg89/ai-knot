import asyncio
from datetime import timedelta

import pytest

from ai_knot.always_on.maintenance import (
    ConsolidationProposal,
    MaintenanceBudget,
    run_maintenance,
)
from ai_knot.types import Fact, MemoryType


class Consolidator:
    config_key = "test-summary-v1"
    model_cost = 1

    def __init__(self, ao, mode="valid"):
        self.ao, self.mode, self.calls = ao, mode, 0

    async def propose(self, episodes):
        self.calls += 1
        refs = self.ao.capture_evidence(tuple(("owner", e.id) for e in episodes))
        ids = tuple(e.id for e in episodes)
        if self.mode == "empty":
            return ConsolidationProposal((), (), refs)
        import json

        sources = [*ids, ids[0]] if self.mode == "duplicate" else list(ids)
        fact = Fact("Summary", qualifiers={"consolidation_ids": json.dumps(sources)})
        return ConsolidationProposal((fact,), ids, refs)


def episode(kb, clock):
    item = Fact(
        "Old event",
        type=MemoryType.EPISODIC,
        created_at=clock() - timedelta(days=2),
        valid_from=clock() - timedelta(days=2),
    )
    kb._storage.save("owner", [item])
    return item


def test_zero_model_budget_never_calls_consolidator(ao_factory, kb, clock):
    ao = ao_factory()
    item = episode(kb, clock)
    provider = Consolidator(ao)
    report = asyncio.run(
        run_maintenance(ao, run_key="zero", budget=MaintenanceBudget(), consolidator=provider)
    )
    assert provider.calls == report.model_calls_reserved == 0
    assert kb.get(item.id).is_active(clock())


@pytest.mark.parametrize("mode", ["empty", "duplicate"])
def test_invalid_consolidation_never_closes_episodes(ao_factory, kb, clock, mode):
    ao = ao_factory()
    item = episode(kb, clock)
    provider = Consolidator(ao, mode)
    report = asyncio.run(
        run_maintenance(
            ao,
            run_key="invalid",
            budget=MaintenanceBudget(max_model_calls=1),
            consolidator=provider,
        )
    )
    assert report.consolidated == 0
    assert kb.get(item.id).is_active(clock())
    assert len(kb.list_facts()) == 1


def test_valid_consolidation_replay_and_lineage(ao_factory, kb, clock):
    ao = ao_factory()
    item = episode(kb, clock)
    provider = Consolidator(ao)
    budget = MaintenanceBudget(max_model_calls=1)
    first = asyncio.run(run_maintenance(ao, run_key="valid", budget=budget, consolidator=provider))
    second = asyncio.run(
        run_maintenance(ao_factory(), run_key="valid", budget=budget, consolidator=provider)
    )
    assert first == second
    assert provider.calls == first.model_calls_reserved == first.consolidated == 1
    assert not kb.get(item.id).is_active(clock())
    assert next(f for f in kb.list_facts() if f.id != item.id).provenance.consolidation_ids == (
        item.id,
    )


def test_foreground_slot_change_and_new_episode_survive(ao_factory, kb, clock):
    ao = ao_factory()
    old = episode(kb, clock)
    kb.add_resolved([Fact("Old deploy", slot_key="deploy::target", value_text="old")])

    class Racing(Consolidator):
        async def propose(self, episodes):
            proposal = await super().propose(episodes)
            kb.add_resolved([Fact("New deploy", slot_key="deploy::target", value_text="new")])
            fresh = kb.add_episodic("Foreground event")
            self.fresh_id = fresh.id
            proposal.facts[0].slot_key = "deploy::target"
            proposal.facts[0].value_text = "background"
            return proposal

    provider = Racing(ao)
    report = asyncio.run(
        run_maintenance(
            ao, run_key="race", budget=MaintenanceBudget(max_model_calls=1), consolidator=provider
        )
    )
    assert report.status == "conflict"
    assert kb.get(old.id).is_active(clock())
    assert kb.get(provider.fresh_id).is_active(clock())
    assert next(f for f in kb.list_facts() if f.value_text == "new").is_active(clock())


def test_input_budget_and_age_cutoff(ao_factory, kb, clock):
    ao = ao_factory()
    episode(kb, clock)
    provider = Consolidator(ao)
    report = asyncio.run(
        run_maintenance(
            ao,
            run_key="bytes",
            budget=MaintenanceBudget(max_model_calls=1, max_input_bytes=1),
            consolidator=provider,
        )
    )
    assert report.status == "budget_exhausted" and provider.calls == 0
    kb.clear_all()
    kb.add_episodic("Recent event")
    report = asyncio.run(
        run_maintenance(
            ao, run_key="recent", budget=MaintenanceBudget(max_model_calls=1), consolidator=provider
        )
    )
    assert provider.calls == report.consolidated == 0


def test_verifier_outage_does_not_confirm(ao_factory, kb, clock):
    from ai_knot.always_on.models import VerificationOutcome

    ao = ao_factory()
    fact = kb.add("Deployment basis")
    (ref,) = ao.capture_evidence((("owner", fact.id),))
    ao.set_freshness(ref, source_ref="connector:build", recheck_after_seconds=60)

    class Offline:
        config_key = "offline-v1"
        model_cost = 0

        async def verify(self, request):
            raise TimeoutError("source unavailable")

    report = asyncio.run(
        run_maintenance(ao, run_key="offline", budget=MaintenanceBudget(), verifier=Offline())
    )
    assert report.verified == 0
    with ao._transaction(write=False) as session:
        policy = ao._policy(session, "owner", kb.get(fact.id))
    assert policy.last_outcome == VerificationOutcome.UNAVAILABLE
    assert policy.checked_at is None


def test_concurrent_run_and_cancellation_preserve_reservations(ao_factory, kb, clock):
    from ai_knot.always_on.models import RevisionConflict
    from ai_knot.always_on.storage import sql

    ao = ao_factory()
    item = kb.add("Basis")
    (ref,) = ao.capture_evidence((("owner", item.id),))
    ao.set_freshness(ref, source_ref="connector", recheck_after_seconds=60)

    async def scenario():
        started = asyncio.Event()
        stop = asyncio.Event()

        class Waiting:
            config_key = "waiting-v1"
            model_cost = 1

            async def verify(self, request):
                started.set()
                await stop.wait()

        provider = Waiting()
        budget = MaintenanceBudget(max_model_calls=1, max_wall_seconds=100)
        task = asyncio.create_task(
            run_maintenance(ao, run_key="cancel", budget=budget, verifier=provider)
        )
        await started.wait()
        with pytest.raises(RevisionConflict):
            await run_maintenance(ao_factory(), run_key="cancel", budget=budget, verifier=provider)
        clock.advance(seconds=101)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with ao._transaction(write=False) as session:
            row = session.fetch_one(
                sql(
                    "SELECT reserved,report_json FROM ao_maintenance_runs "
                    "WHERE owner=? AND run_key=?"
                ),
                ("owner", "cancel"),
            )
        assert row["reserved"] == 1 and row["report_json"] is None
        retry = await run_maintenance(
            ao_factory(), run_key="cancel", budget=budget, verifier=provider
        )
        assert retry.status == "budget_exhausted" and retry.model_calls_reserved == 1

    asyncio.run(scenario())


def test_prune_cannot_skip_unacked_subscription(ao_factory, kb):
    from ai_knot.always_on.subscriptions import WatchSelector

    ao = ao_factory()
    ao.subscribe(subscription_id="w", selector=WatchSelector("owner"))
    fact = kb.add("Change")
    batch = ao.poll("w")
    asyncio.run(run_maintenance(ao, run_key="prune", budget=MaintenanceBudget()))
    assert ao_factory().poll("w") == batch
    assert fact.id in {event.fact_id for event in batch.events}


@pytest.mark.parametrize("fault", ["takeover", "diagnostics"])
def test_cancellation_survives_takeover_and_failed_diagnostics(
    ao_factory, kb, clock, monkeypatch, fault
):
    from ai_knot.always_on.storage import sql

    ao = ao_factory()
    fact = kb.add("Protected")
    (ref,) = ao.capture_evidence((("owner", fact.id),))
    ao.set_freshness(ref, source_ref="checker", recheck_after_seconds=60)

    async def scenario():
        started = asyncio.Event()

        class Waiting:
            config_key = "wait"
            model_cost = 1

            async def verify(self, request):
                started.set()
                await asyncio.Event().wait()

        provider = Waiting()
        budget = MaintenanceBudget(max_model_calls=1, max_wall_seconds=100)
        task = asyncio.create_task(
            run_maintenance(ao, run_key="fault", budget=budget, verifier=provider)
        )
        await started.wait()
        successor = None
        original = ao._transaction
        if fault == "takeover":
            clock.advance(seconds=101)
            successor = await run_maintenance(
                ao_factory(), run_key="fault", budget=budget, verifier=provider
            )
        else:

            def unavailable(*args, **kwargs):
                raise RuntimeError("database unavailable")

            monkeypatch.setattr(ao, "_transaction", unavailable)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        monkeypatch.setattr(ao, "_transaction", original)
        with ao._transaction(write=False) as session:
            row = session.fetch_one(
                sql(
                    "SELECT reserved,report_json FROM ao_maintenance_runs "
                    "WHERE owner=? AND run_key=?"
                ),
                ("owner", "fault"),
            )
        assert row["reserved"] == 1
        if successor:
            from ai_knot.always_on.codec import canonical_json

            assert row["report_json"] == canonical_json(successor)

    asyncio.run(scenario())


def test_legacy_consolidation_prepares_before_atomic_commit(kb, clock, monkeypatch):
    item = episode(kb, clock)
    kb._default_provider = "test"
    monkeypatch.setattr(kb, "_extract_phase", lambda turns: [Fact("Summary")])
    assert asyncio.run(kb.consolidate_episodic()) == 1
    assert not kb.get(item.id).is_active(clock())
    assert next(f for f in kb.list_facts() if f.id != item.id).provenance.consolidation_ids == (
        item.id,
    )


def test_legacy_consolidation_does_not_close_inputs_on_empty_output(kb, clock, monkeypatch):
    item = episode(kb, clock)
    kb._default_provider = "test"
    monkeypatch.setattr(kb, "_extract_phase", lambda turns: [])
    assert asyncio.run(kb.consolidate_episodic()) == 0
    assert kb.get(item.id).is_active(clock())


@pytest.mark.parametrize("pruner", ["public", "maintenance"])
def test_pruning_during_provider_preserves_foreground_slot(ao_factory, kb, clock, pruner):
    ao = ao_factory()
    source = episode(kb, clock)

    class Pruning(Consolidator):
        async def propose(self, episodes):
            proposal = await super().propose(episodes)
            kb.add_resolved(
                [Fact("Foreground", slot_key="deploy::target", value_text="foreground")]
            )
            if pruner == "public":
                ao.prune_changes()
            else:
                # A different run can safely prune its own journal while this one awaits.
                await run_maintenance(ao_factory(), run_key="pruner", budget=MaintenanceBudget())
            proposal.facts[0].slot_key = "deploy::target"
            proposal.facts[0].value_text = "stale-background"
            return proposal

    report = asyncio.run(
        run_maintenance(
            ao,
            run_key="pruned-race",
            budget=MaintenanceBudget(max_model_calls=1),
            consolidator=Pruning(ao),
        )
    )
    assert report.status == "conflict" and report.consolidated == 0
    assert kb.get(source.id).is_active(clock())
    assert next(f for f in kb.list_facts() if f.value_text == "foreground").is_active(clock())
    assert not any(f.value_text == "stale-background" for f in kb.list_facts())


@pytest.mark.parametrize("mutation", ["source", "interval", "changed", "same_reset"])
@pytest.mark.parametrize("outcome", ["confirmed", "error", "cancelled"])
def test_outstanding_verifier_cannot_change_reconfigured_policy(ao_factory, kb, mutation, outcome):
    from ai_knot.always_on.models import VerificationOutcome, VerificationResult

    ao = ao_factory()
    fact = kb.add("Protected basis")
    (ref,) = ao.capture_evidence((("owner", fact.id),))
    ao.set_freshness(ref, source_ref="old-source", recheck_after_seconds=60)

    class RacingVerifier:
        config_key = "policy-cas-v1"
        model_cost = 0
        expected = None

        async def verify(self, request):
            assert request.source_ref == "old-source"
            if mutation == "changed":
                self.expected = ao.record_verification(
                    ref,
                    result=VerificationResult(
                        VerificationOutcome.CHANGED, {"observation": "newer"}, "newer-v2"
                    ),
                )
            else:
                self.expected = ao.set_freshness(
                    ref,
                    source_ref="new-source" if mutation == "source" else "old-source",
                    recheck_after_seconds=999 if mutation == "interval" else 60,
                )
            if outcome == "error":
                raise TimeoutError("old source failed")
            if outcome == "cancelled":
                raise asyncio.CancelledError
            return VerificationResult(
                VerificationOutcome.CONFIRMED, {"observation": "old-source"}, "old-v1"
            )

    provider = RacingVerifier()
    operation = run_maintenance(
        ao, run_key="policy-cas", budget=MaintenanceBudget(), verifier=provider
    )
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(operation)
    else:
        report = asyncio.run(operation)
        assert report.status == "conflict" and report.verified == 0
    with ao._transaction(write=False) as session:
        assert ao._policy(session, "owner", kb.get(fact.id)) == provider.expected


@pytest.mark.parametrize("invalid_time", ["expired", "future", "naive", "malformed"])
def test_inactive_or_malformed_output_cannot_close_sources(ao_factory, kb, clock, invalid_time):
    ao = ao_factory()
    source = episode(kb, clock)

    class InvalidTemporal(Consolidator):
        async def propose(self, episodes):
            proposal = await super().propose(episodes)
            output = proposal.facts[0]
            if invalid_time == "expired":
                output.valid_until = clock() - timedelta(seconds=1)
            elif invalid_time == "future":
                output.valid_from = clock() + timedelta(days=1)
            elif invalid_time == "naive":
                output.valid_from = clock().replace(tzinfo=None)
            else:
                output.valid_from = "not-a-date"
            return proposal

    report = asyncio.run(
        run_maintenance(
            ao,
            run_key="invalid-time",
            budget=MaintenanceBudget(max_model_calls=1),
            consolidator=InvalidTemporal(ao),
        )
    )
    assert report.consolidated == 0 and report.status == "failed"
    assert kb.get(source.id).is_active(clock())
    assert len(kb.list_facts()) == 1


def _assert_legacy_waiting_expiry(kb, clock, monkeypatch):
    from datetime import UTC, datetime

    from ai_knot import learning

    clock.now = datetime.now(UTC)
    source = episode(kb, clock)
    kb._default_provider = "test"

    class ControlledDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock()

    monkeypatch.setattr(learning, "datetime", ControlledDatetime)
    monkeypatch.setattr(
        kb,
        "_extract_phase",
        lambda turns: [
            Fact(
                "Summary",
                created_at=clock(),
                valid_from=clock(),
                valid_until=clock() + timedelta(seconds=1),
            )
        ],
    )
    storage = kb._storage
    if hasattr(storage, "always_on_transaction"):
        original = storage.always_on_transaction

        def delayed(namespaces, *, write):
            if write:
                clock.advance(seconds=2)
            return original(namespaces, write=write)

        monkeypatch.setattr(storage, "always_on_transaction", delayed)
    else:
        original_load = storage.load
        calls = 0

        def delayed_load(agent_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                clock.advance(seconds=2)
            return original_load(agent_id)

        monkeypatch.setattr(storage, "load", delayed_load)
    with pytest.raises(ValueError):
        asyncio.run(kb.consolidate_episodic())
    assert kb.get(source.id).is_active(clock())
    assert len(kb.list_facts()) == 1


def test_legacy_output_expiry_while_waiting_for_sql_lock(kb, clock, monkeypatch):
    _assert_legacy_waiting_expiry(kb, clock, monkeypatch)


def test_legacy_output_expiry_while_waiting_for_yaml_commit(tmp_path, clock, monkeypatch):
    from ai_knot.knowledge import KnowledgeBase
    from ai_knot.storage.yaml_storage import YAMLStorage

    kb = KnowledgeBase("owner", storage=YAMLStorage(str(tmp_path)), embed_url="")
    _assert_legacy_waiting_expiry(kb, clock, monkeypatch)


def test_wall_budget_after_reservation_does_not_construct_verifier_coroutine(
    ao_factory, kb, monkeypatch
):
    import warnings
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ai_knot.always_on import maintenance
    from ai_knot.always_on.storage import sql

    ao = ao_factory()
    source = kb.add("Deployment verification source")
    (ref,) = ao.capture_evidence((("owner", source.id),))
    ao.set_freshness(ref, source_ref="deployment", recheck_after_seconds=60)
    elapsed = 0.0
    reserve = maintenance._reserve

    def delayed_reserve(*args, **kwargs):
        nonlocal elapsed
        token = reserve(*args, **kwargs)
        elapsed = 2.0
        return token

    async def verify(request):
        raise AssertionError("verifier must not run after the wall deadline")

    provider = SimpleNamespace(
        config_key="wall-after-reserve", model_cost=1, verify=Mock(wraps=verify)
    )
    monkeypatch.setattr(maintenance, "_reserve", delayed_reserve)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", RuntimeWarning)
        report = asyncio.run(
            maintenance.run_maintenance(
                ao,
                run_key="wall-after-reserve",
                budget=MaintenanceBudget(max_model_calls=1, max_wall_seconds=1),
                verifier=provider,
                monotonic=lambda: elapsed,
            )
        )
    assert report.status == "budget_exhausted"
    assert "wall_budget" in report.reasons
    assert report.model_calls_reserved == 1
    with ao._transaction(write=False) as session:
        row = session.fetch_one(
            sql("SELECT reserved FROM ao_maintenance_runs WHERE owner=? AND run_key=?"),
            ("owner", "wall-after-reserve"),
        )
    assert row["reserved"] == 1
    provider.verify.assert_not_called()
    assert not [warning for warning in observed if issubclass(warning.category, RuntimeWarning)]
