"""Copied evidence retains its roots, permissions and revision requirements."""

import pytest

from ai_knot import KnowledgeBase
from ai_knot.always_on import AccessDenied, AlwaysOnMemory, StaleEvidence, WorkspaceMemory
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact


def setup_team(sql_storage, clock):
    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock)
    workspace = WorkspaceMemory(sql_storage, "project")
    workspace.grant("writer", "maintain")
    source = Fact(
        "Deployment uses verified build 42",
        id="root",
        origin_agent_id="writer",
        visibility_scope="team:ops",
        created_at=clock(),
        last_accessed=clock(),
        valid_from=clock(),
    )
    sql_storage.save("__shared__", [source])
    agents = {}
    for name in ("writer", "reader"):
        pool.grant_read(name, "team:ops")
        agents[name] = AlwaysOnMemory(
            KnowledgeBase(name, storage=sql_storage, embed_url=""),
            pool=pool,
            workspace=workspace,
            clock=clock,
        )
    return pool, agents, source


def test_copies_flatten_roots_and_survive_reopen(sql_storage, clock):
    pool, agents, source = setup_team(sql_storage, clock)
    writer = agents["writer"]
    refs = writer.capture_evidence((("__shared__", source.id),))
    first = writer.derive_fact("Build 42 is verified", basis=refs, namespace="__shared__")
    copied = writer.derive_fact(
        "Deployment evidence: build 42",
        namespace="__shared__",
        basis=writer.capture_evidence((("__shared__", first.id),)),
    )
    reopened = AlwaysOnMemory(
        KnowledgeBase("reader", storage=sql_storage, embed_url=""),
        pool=pool,
        clock=clock,
    )
    evidence = reopened.capture_evidence(
        tuple(("__shared__", f.id) for f in (source, first, copied))
    )
    assert reopened.independent_evidence(evidence) == refs


def test_revocation_hides_derived_shared_content(sql_storage, clock):
    pool, agents, source = setup_team(sql_storage, clock)
    writer, reader = agents["writer"], agents["reader"]
    derived = writer.derive_fact(
        "Deployment evidence confidential summary",
        namespace="__shared__",
        basis=writer.capture_evidence((("__shared__", source.id),)),
        visibility_scope="global",
    )
    refs = reader.capture_evidence((("__shared__", derived.id),))
    pool.revoke_read("reader", "team:ops")
    with pytest.raises(AccessDenied):
        reader.capture_evidence((("__shared__", derived.id),))
    with pytest.raises(AccessDenied):
        reader.independent_evidence(refs)
    assert not pool.recall("confidential summary", "reader")


def test_changed_root_blocks_prepared_action(sql_storage, clock):
    _, agents, source = setup_team(sql_storage, clock)
    writer, reader = agents["writer"], agents["reader"]
    derived = writer.derive_fact(
        "Deployment summary",
        namespace="__shared__",
        basis=writer.capture_evidence((("__shared__", source.id),)),
    )
    refs = reader.capture_evidence((("__shared__", derived.id),))
    intent = reader.create_intent(client_key="deploy", goal="Deploy")
    lease = reader.claim(intent.id, worker_id="worker")
    action = reader.prepare_action(
        intent.id,
        lease_token=lease.token,
        action_key="deploy",
        request={},
        basis=refs,
    )
    facts = sql_storage.load("__shared__")
    next(f for f in facts if f.id == source.id).content = "Build 42 verification failed"
    sql_storage.save("__shared__", facts)
    with pytest.raises(StaleEvidence):
        reader.begin_action(action.id, lease_token=lease.token)


def test_reader_cannot_publish_shared_derivation(sql_storage, clock):
    _, agents, source = setup_team(sql_storage, clock)
    reader = agents["reader"]
    refs = reader.capture_evidence((("__shared__", source.id),))
    with pytest.raises(AccessDenied):
        reader.derive_fact("Unapproved summary", basis=refs, namespace="__shared__")


def test_private_derivation_does_not_leak_on_pool_publish(sql_storage, clock):
    pool, agents, _ = setup_team(sql_storage, clock)
    writer, reader = agents["writer"], agents["reader"]
    source = writer._kb.add("Secret private deployment credential")
    derived = writer.derive_fact(
        "Secret private credential copied into summary",
        basis=writer.capture_evidence((("writer", source.id),)),
    )
    pool.register("writer")
    assert pool.publish("writer", [derived.id], kb=writer._kb)
    with pytest.raises(AccessDenied):
        reader.capture_evidence((("__shared__", derived.id),))
    assert derived.id not in {f.id for f, _ in pool.recall("Secret private credential", "reader")}


def test_private_summary_of_shared_source_can_be_guarded(sql_storage, clock):
    pool, agents, source = setup_team(sql_storage, clock)
    reader = agents["reader"]
    derived = reader.derive_fact(
        "Private plan from shared evidence",
        basis=reader.capture_evidence((("__shared__", source.id),)),
    )
    (ref,) = reader.capture_evidence((("reader", derived.id),))
    reader.set_freshness(ref, source_ref="private-plan", recheck_after_seconds=60)
    from ai_knot.always_on import VerificationOutcome, VerificationResult

    reader.record_verification(
        ref, result=VerificationResult(VerificationOutcome.CONFIRMED, {"source": "checked"}, "v1")
    )
    intent = reader.create_intent(client_key="private", goal="Use derived evidence")
    lease = reader.claim(intent.id, worker_id="worker")
    reader.prepare_action(
        intent.id,
        lease_token=lease.token,
        action_key="read",
        request={},
        basis=(ref,),
    )
    assert derived.id in {f.id for f in reader._kb.recall_facts("Private plan")}
    pool.revoke_read("reader", "team:ops")
    assert derived.id not in {f.id for f in reader._kb.recall_facts("Private plan")}


def test_derived_action_cannot_bypass_root_freshness(sql_storage, clock):
    from ai_knot.always_on import VerificationOutcome, VerificationResult

    _, agents, source = setup_team(sql_storage, clock)
    writer, reader = agents["writer"], agents["reader"]
    source_refs = writer.capture_evidence((("__shared__", source.id),))
    writer.set_freshness(source_refs[0], source_ref="deployment", recheck_after_seconds=60)
    writer.record_verification(
        source_refs[0],
        result=VerificationResult(VerificationOutcome.CONFIRMED, {"build": "42"}, "v1"),
    )
    summary = writer.derive_fact("Deployment summary", basis=source_refs, namespace="__shared__")
    basis = reader.capture_evidence((("__shared__", summary.id),))
    intent = reader.create_intent(client_key="fresh-root", goal="Deploy")
    lease = reader.claim(intent.id, worker_id="host", ttl_seconds=300)
    action = reader.prepare_action(
        intent.id,
        lease_token=lease.token,
        action_key="deploy",
        request={},
        basis=basis,
    )
    clock.advance(seconds=61)
    with pytest.raises(StaleEvidence):
        reader.begin_action(action.id, lease_token=lease.token)


def test_archival_preserves_freshness_deadline_without_renewing(sql_storage, clock):
    import json
    from datetime import timedelta

    from ai_knot.always_on import ConsolidationProposal, VerificationOutcome, VerificationResult
    from ai_knot.always_on.maintenance import commit_consolidation
    from ai_knot.types import MemoryType

    memory = AlwaysOnMemory(KnowledgeBase("owner", storage=sql_storage, embed_url=""), clock=clock)
    source = Fact(
        "Old event",
        type=MemoryType.EPISODIC,
        created_at=clock() - timedelta(days=2),
        valid_from=clock() - timedelta(days=2),
    )
    sql_storage.save("owner", [source])
    refs = memory.capture_evidence((("owner", source.id),))
    memory.set_freshness(refs[0], source_ref="remote", recheck_after_seconds=60)
    original_policy = memory.record_verification(
        refs[0],
        result=VerificationResult(VerificationOutcome.CONFIRMED, {"source": "checked"}, "v1"),
    )
    clock.advance(seconds=20)
    summary = Fact("Summary", qualifiers={"consolidation_ids": json.dumps([source.id])})
    commit_consolidation(memory, ConsolidationProposal((summary,), (source.id,), refs), refs)
    summary_refs = memory.capture_evidence((("owner", summary.id),))
    intent = memory.create_intent(client_key="archive", goal="Use summary")
    lease = memory.claim(intent.id, worker_id="host", ttl_seconds=300)
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="use", request={}, basis=summary_refs
    )
    with memory._transaction(write=False) as session:
        policy = memory._policy(session, "owner", source)
    assert policy.checked_at == original_policy.checked_at
    assert policy.verified_until == original_policy.verified_until
    clock.advance(seconds=41)
    with pytest.raises(StaleEvidence):
        memory.begin_action(action.id, lease_token=lease.token)


def test_intermediate_revision_and_acl_are_dependencies(sql_storage, clock):
    pool, agents, source = setup_team(sql_storage, clock)
    # A public root cannot erase a restriction on an intermediate shared view.
    source.visibility_scope = "global"
    sql_storage.save("__shared__", [source])
    writer, reader = agents["writer"], agents["reader"]
    intermediate = writer.derive_fact(
        "Restricted intermediate",
        namespace="__shared__",
        visibility_scope="team:ops",
        basis=writer.capture_evidence((("__shared__", source.id),)),
    )
    final = writer.derive_fact(
        "Public outer summary",
        namespace="__shared__",
        visibility_scope="global",
        basis=writer.capture_evidence((("__shared__", intermediate.id),)),
    )
    refs = reader.capture_evidence((("__shared__", final.id),))
    pool.revoke_read("reader", "team:ops")
    with pytest.raises(AccessDenied):
        reader.independent_evidence(refs)
    pool.grant_read("reader", "team:ops")
    facts = sql_storage.load("__shared__")
    next(f for f in facts if f.id == intermediate.id).content = "Corrected intermediate"
    sql_storage.save("__shared__", facts)
    with pytest.raises(StaleEvidence):
        reader.independent_evidence(refs)
