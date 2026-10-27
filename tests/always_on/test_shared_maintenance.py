import pytest

from ai_knot.always_on.models import AccessDenied
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact


def test_shared_reader_cannot_change_policy(ao_factory, sql_storage, clock):
    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True, clock=clock)
    fact = Fact("public", created_at=clock(), valid_from=clock())
    sql_storage.save("__shared__", [fact])
    memory = ao_factory(pool=pool)
    (ref,) = memory.capture_evidence((("__shared__", fact.id),))
    with pytest.raises(AccessDenied):
        memory.set_freshness(ref, source_ref="source", recheck_after_seconds=60)


def test_shared_reader_can_run_private_maintenance(ao_factory, sql_storage, clock):
    import asyncio

    from ai_knot.always_on import MaintenanceBudget, run_maintenance

    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock)
    memory = ao_factory(pool=pool)
    report = asyncio.run(
        run_maintenance(memory, run_key="private-only", budget=MaintenanceBudget())
    )
    assert report.status == "completed"
    assert report.model_calls_reserved == 0


def shared_pair(sql_storage, clock):
    from ai_knot.always_on import AlwaysOnMemory
    from ai_knot.always_on.shared_maintenance import configure_workspace_budget
    from ai_knot.always_on.workspace import WorkspaceMemory
    from ai_knot.knowledge import KnowledgeBase

    workspace = WorkspaceMemory(sql_storage, "team")
    pool = SharedMemoryPool(storage=sql_storage, persist_stats=True, clock=clock)
    fact = Fact("public", created_at=clock(), valid_from=clock())
    sql_storage.save("__shared__", [fact])
    memories = []
    from ai_knot.storage.sqlite_storage import SQLiteStorage

    for owner in ("one", "two"):
        storage = (
            SQLiteStorage(str(sql_storage._db_path))
            if isinstance(sql_storage, SQLiteStorage)
            else type(sql_storage)(sql_storage._dsn)
        )
        owner_workspace = WorkspaceMemory(storage, "team")
        workspace.grant(owner, "maintain")
        memories.append(
            AlwaysOnMemory(
                KnowledgeBase(owner, storage=storage),
                pool=pool,
                workspace=owner_workspace,
                clock=clock,
            )
        )
    configure_workspace_budget(workspace, max_model_calls_per_day=2)
    (ref,) = memories[0].capture_evidence((("__shared__", fact.id),))
    memories[0].set_freshness(ref, source_ref="source", recheck_after_seconds=60)
    return workspace, memories, ref


@pytest.mark.anyio
async def test_unowned_shared_policies_do_not_starve_private_work(sql_storage, clock):
    from ai_knot.always_on import (
        MaintenanceBudget,
        VerificationOutcome,
        VerificationResult,
        run_maintenance,
    )

    workspace, (reader, _), _ = shared_pair(sql_storage, clock)
    workspace.revoke("one", "maintain")
    private = reader._kb.add("Private basis")
    (ref,) = reader.capture_evidence((("one", private.id),))
    reader.set_freshness(ref, source_ref="private", recheck_after_seconds=60)

    class Provider:
        config_key = "private-only"
        model_cost = 1
        calls = []

        async def verify(self, request):
            self.calls.append(request.ref.namespace)
            return VerificationResult(VerificationOutcome.CONFIRMED, {"source": "private"}, "v1")

    provider = Provider()
    report = await run_maintenance(
        reader,
        run_key="private-priority",
        budget=MaintenanceBudget(max_items=1, max_model_calls=1),
        verifier=provider,
    )
    assert provider.calls == ["one"]
    assert report.verified == 1


@pytest.mark.anyio
async def test_overlapping_owners_call_once_and_revoked_commit_fails(sql_storage, clock):
    import asyncio

    from ai_knot.always_on.maintenance import MaintenanceBudget, run_maintenance
    from ai_knot.always_on.models import VerificationOutcome, VerificationResult

    workspace, (one, two), ref = shared_pair(sql_storage, clock)
    started, release = asyncio.Event(), asyncio.Event()

    class Provider:
        config_key = "v1"
        model_cost = 1
        calls = 0

        async def verify(self, request):
            self.calls += 1
            started.set()
            await release.wait()
            return VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "v1")

    provider = Provider()
    budget = MaintenanceBudget(max_model_calls=2, max_wall_seconds=10)
    pending = asyncio.create_task(
        run_maintenance(one, run_key="a", budget=budget, verifier=provider)
    )
    await started.wait()
    second = await run_maintenance(two, run_key="b", budget=budget, verifier=provider)
    assert second.status == "conflict"
    assert provider.calls == 1
    workspace.revoke("one", "maintain")
    release.set()
    first = await pending
    assert first.verified == 0
    assert first.status == "failed"
    with pytest.raises(AccessDenied):
        one.record_verification(
            ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "v1")
        )


@pytest.mark.anyio
async def test_unknown_outcome_holds_claim_and_daily_cost_survives_recovery(sql_storage, clock):
    from ai_knot.always_on.maintenance import MaintenanceBudget, run_maintenance
    from ai_knot.always_on.shared_maintenance import configure_workspace_budget

    workspace, (one, two), ref = shared_pair(sql_storage, clock)
    configure_workspace_budget(workspace, max_model_calls_per_day=1)

    class Provider:
        config_key = "v1"
        model_cost = 1
        calls = 0

        async def verify(self, request):
            self.calls += 1
            raise TimeoutError("remote outcome unknown")

    provider = Provider()
    # This checks daily accounting and recovery, not shared database latency.
    budget = MaintenanceBudget(max_model_calls=10, max_wall_seconds=10)
    first = await run_maintenance(one, run_key="a", budget=budget, verifier=provider)
    assert first.model_calls_reserved == 1
    # A policy/source change cannot make the old in-flight refresh safe to replay.
    two.set_freshness(ref, source_ref="new-source", recheck_after_seconds=60)
    second = await run_maintenance(two, run_key="b", budget=budget, verifier=provider)
    assert second.status == "conflict"
    assert provider.calls == 1
    clock.advance(seconds=61)
    third = await run_maintenance(two, run_key="c", budget=budget, verifier=provider)
    assert "workspace_model_budget" in third.reasons
    assert provider.calls == 1
    clock.advance(days=1)
    fourth = await run_maintenance(two, run_key="d", budget=budget, verifier=provider)
    assert fourth.model_calls_reserved == 1
    assert provider.calls == 2


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["verification", "consolidation"])
async def test_workspace_budget_covers_private_provider_work(sql_storage, clock, kind):
    import asyncio
    import json
    from datetime import timedelta

    from ai_knot import KnowledgeBase
    from ai_knot.always_on import (
        AlwaysOnMemory,
        ConsolidationProposal,
        MaintenanceBudget,
        VerificationOutcome,
        VerificationResult,
        WorkspaceMemory,
        configure_workspace_budget,
        run_maintenance,
    )
    from ai_knot.types import MemoryType

    workspace = WorkspaceMemory(sql_storage, "private-budget")
    configure_workspace_budget(workspace, max_model_calls_per_day=1)
    memories = []
    for owner in ("first", "second"):
        kb = KnowledgeBase(owner, storage=sql_storage, embed_url="")
        memory = AlwaysOnMemory(kb, workspace=workspace, clock=clock)
        if kind == "verification":
            fact = kb.add("Private deployment state " + owner)
            (ref,) = memory.capture_evidence(((owner, fact.id),))
            memory.set_freshness(ref, source_ref=owner, recheck_after_seconds=60)
        else:
            sql_storage.save(
                owner,
                [
                    Fact(
                        "Old episode",
                        type=MemoryType.EPISODIC,
                        created_at=clock() - timedelta(days=2),
                        valid_from=clock() - timedelta(days=2),
                    )
                ],
            )
        memories.append(memory)

    class Provider:
        config_key = "private-budget-v1"
        model_cost = 1
        calls = 0

        async def verify(self, request):
            self.calls += 1
            return VerificationResult(
                VerificationOutcome.CONFIRMED, {"source": request.source_ref}, "v1"
            )

        async def propose(self, episodes):
            self.calls += 1
            memory = next(m for m in memories if m._storage.load(m._owner)[0].id == episodes[0].id)
            refs = memory.capture_evidence(tuple((memory._owner, f.id) for f in episodes))
            ids = tuple(f.id for f in episodes)
            summary = Fact("Summary", qualifiers={"consolidation_ids": json.dumps(ids)})
            return ConsolidationProposal((summary,), ids, refs)

    provider = Provider()
    kwargs = {"verifier" if kind == "verification" else "consolidator": provider}
    reports = await asyncio.gather(
        *(
            run_maintenance(
                m, run_key="tick", budget=MaintenanceBudget(max_model_calls=1), **kwargs
            )
            for m in memories
        )
    )
    assert provider.calls == sum(r.model_calls_reserved for r in reports) == 1
    assert any("workspace_model_budget" in r.reasons for r in reports)


@pytest.mark.anyio
async def test_cancelled_refresh_retains_claim_and_cost(sql_storage, clock):
    import asyncio

    from ai_knot.always_on.maintenance import MaintenanceBudget, run_maintenance
    from ai_knot.always_on.models import VerificationOutcome, VerificationResult

    workspace, (one, two), ref = shared_pair(sql_storage, clock)
    started = asyncio.Event()

    class Provider:
        config_key = "cancel"
        model_cost = 1
        calls = 0

        async def verify(self, request):
            self.calls += 1
            started.set()
            await asyncio.Event().wait()
            return VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "v1")

    provider = Provider()
    budget = MaintenanceBudget(max_model_calls=2)
    pending = asyncio.create_task(
        run_maintenance(one, run_key="cancel", budget=budget, verifier=provider)
    )
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    # Reset policy retry time, which cannot reset the remote claim or reserved costs.
    two.set_freshness(ref, source_ref="new", recheck_after_seconds=1)
    blocked = await run_maintenance(two, run_key="blocked", budget=budget, verifier=provider)
    assert blocked.status == "conflict"
    assert provider.calls == 1
    clock.advance(seconds=61)

    class Finished(Provider):
        async def verify(self, request):
            self.calls += 1
            return VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "v2")

    finished = Finished()
    recovered = await run_maintenance(two, run_key="recover", budget=budget, verifier=finished)
    assert recovered.verified == 1
    assert recovered.model_calls_reserved == 1


@pytest.mark.anyio
async def test_budget_across_subjects_and_reconfiguration_does_not_reset(sql_storage, clock):
    from ai_knot.always_on.maintenance import MaintenanceBudget, run_maintenance
    from ai_knot.always_on.models import VerificationOutcome, VerificationResult
    from ai_knot.always_on.shared_maintenance import configure_workspace_budget

    workspace, (one, two), ref = shared_pair(sql_storage, clock)
    configure_workspace_budget(workspace, max_model_calls_per_day=1)

    class Provider:
        config_key = "ok"
        model_cost = 1
        calls = 0

        async def verify(self, request):
            self.calls += 1
            return VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "v1")

    provider = Provider()
    budget = MaintenanceBudget(max_model_calls=10)
    assert (
        await run_maintenance(one, run_key="first", budget=budget, verifier=provider)
    ).verified == 1
    extra = Fact("second", created_at=clock(), valid_from=clock())
    sql_storage.save("__shared__", [*sql_storage.load("__shared__"), extra])
    (extra_ref,) = two.capture_evidence((("__shared__", extra.id),))
    two.set_freshness(extra_ref, source_ref="second", recheck_after_seconds=1)
    configure_workspace_budget(workspace, max_model_calls_per_day=1)
    blocked = await run_maintenance(two, run_key="second", budget=budget, verifier=provider)
    assert "workspace_model_budget" in blocked.reasons
    assert provider.calls == 1
    configure_workspace_budget(workspace, max_model_calls_per_day=2)
    assert (
        await run_maintenance(two, run_key="raised", budget=budget, verifier=provider)
    ).verified == 1
    assert provider.calls == 2


@pytest.mark.anyio
async def test_shared_claim_is_global_across_workspace_instances(sql_storage, clock):
    import asyncio

    from ai_knot.always_on import AlwaysOnMemory
    from ai_knot.always_on.maintenance import MaintenanceBudget, run_maintenance
    from ai_knot.always_on.models import VerificationOutcome, VerificationResult
    from ai_knot.always_on.shared_maintenance import configure_workspace_budget
    from ai_knot.always_on.workspace import WorkspaceMemory
    from ai_knot.knowledge import KnowledgeBase

    workspace, (one, two), ref = shared_pair(sql_storage, clock)
    other = WorkspaceMemory(sql_storage, "other")
    other.grant("three", "maintain")
    configure_workspace_budget(other, max_model_calls_per_day=2)
    three = AlwaysOnMemory(
        KnowledgeBase("three", storage=sql_storage), workspace=other, pool=one._pool, clock=clock
    )
    started, release = asyncio.Event(), asyncio.Event()

    class Provider:
        config_key = "ok"
        model_cost = 1
        calls = 0

        async def verify(self, request):
            self.calls += 1
            started.set()
            await release.wait()
            return VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "v1")

    provider = Provider()
    budget = MaintenanceBudget(max_model_calls=2, max_wall_seconds=10)
    pending = asyncio.create_task(
        run_maintenance(one, run_key="first", budget=budget, verifier=provider)
    )
    await started.wait()
    blocked = await run_maintenance(three, run_key="second", budget=budget, verifier=provider)
    assert blocked.status == "conflict"
    assert provider.calls == 1
    release.set()
    assert (await pending).verified == 1


@pytest.mark.anyio
async def test_stale_response_cannot_release_replacement_claim(sql_storage, clock):
    import asyncio

    from ai_knot.always_on.maintenance import MaintenanceBudget, run_maintenance
    from ai_knot.always_on.models import VerificationOutcome, VerificationResult

    workspace, (one, two), ref = shared_pair(sql_storage, clock)
    started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    class Provider:
        config_key = "slow"
        model_cost = 1
        calls = 0

        async def verify(self, request):
            index = self.calls
            self.calls += 1
            started[index].set()
            await release[index].wait()
            return VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, "v1")

    provider = Provider()
    budget = MaintenanceBudget(max_model_calls=2, max_wall_seconds=10)
    old = asyncio.create_task(run_maintenance(one, run_key="old", budget=budget, verifier=provider))
    await started[0].wait()
    clock.advance(seconds=61)
    two.set_freshness(ref, source_ref="replacement", recheck_after_seconds=60)
    new = asyncio.create_task(run_maintenance(two, run_key="new", budget=budget, verifier=provider))
    await started[1].wait()
    release[0].set()
    assert (await old).status == "conflict"
    # A late old generation must leave the replacement exclusion in place.
    blocked = await run_maintenance(one, run_key="third", budget=budget, verifier=provider)
    assert blocked.status == "conflict"
    assert provider.calls == 2
    release[1].set()
    assert (await new).verified == 1
