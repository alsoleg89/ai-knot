"""Cross-agent operation identity and resource exclusion on real SQL backends."""

import pytest

from ai_knot import KnowledgeBase
from ai_knot.always_on import AccessDenied, AlwaysOnMemory, IdempotencyConflict, RevisionConflict


def setup_agents(storage, clock):
    from ai_knot.always_on.workspace import WorkspaceMemory

    workspace = WorkspaceMemory(storage, "team")
    memories = []
    for name in ("alice", "bob"):
        workspace.grant(name, "execute")
        memory = AlwaysOnMemory(
            KnowledgeBase(name, storage=storage), clock=clock, workspace=workspace
        )
        intent = memory.create_intent(client_key="ship", goal="Ship")
        lease = memory.claim(intent.id, worker_id=name)
        memories.append((memory, intent, lease))
    return workspace, memories


def prepare(agent, operation="deploy", resource="production", request=None):
    memory, intent, lease = agent
    return memory.prepare_action(
        intent.id,
        lease_token=lease.token,
        action_key=operation,
        operation_key=operation,
        resource_key=resource,
        request={"v": 1} if request is None else request,
    )


def test_handoff_fences_previous_agent_and_preserves_identity(sql_storage, clock):
    workspace, (a, b) = setup_agents(sql_storage, clock)
    first = prepare(a)
    second = prepare(b)
    assert first.idempotency_key == second.idempotency_key
    with pytest.raises(RevisionConflict):
        a[0].begin_action(first.id, lease_token=a[2].token)
    permit = b[0].begin_action(second.id, lease_token=b[2].token)
    assert permit.fencing_token >= 2
    with pytest.raises(RevisionConflict):
        prepare(a)
    with pytest.raises(RevisionConflict):
        prepare(a, operation="another")
    workspace.revoke("alice", "execute")
    with pytest.raises(AccessDenied):
        prepare(a)


def test_payload_resource_mismatch_and_revoked_start(sql_storage, clock):
    workspace, (a, b) = setup_agents(sql_storage, clock)
    action = prepare(a)
    for kwargs in ({"request": {"v": 2}}, {"resource": "other"}):
        with pytest.raises(IdempotencyConflict):
            prepare(b, **kwargs)
    workspace.revoke("alice", "execute")
    with pytest.raises(AccessDenied):
        a[0].begin_action(action.id, lease_token=a[2].token)


def test_unknown_holds_resource_until_reconciled(sql_storage, clock):
    _, (a, b) = setup_agents(sql_storage, clock)
    action = prepare(a)
    permit = a[0].begin_action(action.id, lease_token=a[2].token)
    a[0].finish_action(action.id, lease_token=a[2].token, outcome="unknown", result={})
    with pytest.raises(RevisionConflict):
        prepare(b)
    a[0].reconcile_action(
        action.id, lease_token=a[2].token, outcome="retryable", evidence={"checked": True}
    )
    second = prepare(b)
    next_permit = b[0].begin_action(second.id, lease_token=b[2].token)
    assert next_permit.idempotency_key == permit.idempotency_key
    assert next_permit.fencing_token > permit.fencing_token


def test_restart_terminal_replay_and_resource_generation(sql_storage, clock):
    from ai_knot.always_on.workspace import WorkspaceMemory

    _, (a, b) = setup_agents(sql_storage, clock)
    first = prepare(a)
    permit = a[0].begin_action(first.id, lease_token=a[2].token)
    a[0].finish_action(first.id, lease_token=a[2].token, outcome="succeeded", result={"done": True})
    restarted = AlwaysOnMemory(
        KnowledgeBase("bob", storage=sql_storage),
        clock=clock,
        workspace=WorkspaceMemory(sql_storage, "team"),
    )
    b = (restarted, b[1], b[2])
    replay = prepare(b)
    assert replay.state == "succeeded"
    assert replay.result == {"done": True}
    assert replay.idempotency_key == permit.idempotency_key
    next_action = prepare(b, operation="next")
    next_permit = b[0].begin_action(next_action.id, lease_token=b[2].token)
    assert next_permit.fencing_token > permit.fencing_token


def test_workspace_isolation_and_omitted_binding(sql_storage, clock):
    from ai_knot.always_on.workspace import WorkspaceMemory

    _, (a, _) = setup_agents(sql_storage, clock)
    action = prepare(a)
    with pytest.raises(ValueError):
        a[0].prepare_action(a[1].id, lease_token=a[2].token, action_key="deploy", request={"v": 1})
    other = WorkspaceMemory(sql_storage, "other")
    other.grant("carol", "execute")
    memory = AlwaysOnMemory(
        KnowledgeBase("carol", storage=sql_storage), workspace=other, clock=clock
    )
    intent = memory.create_intent(client_key="ship", goal="Ship")
    lease = memory.claim(intent.id, worker_id="carol")
    unrelated = prepare((memory, intent, lease))
    assert unrelated.idempotency_key != action.idempotency_key


def test_revocation_blocks_result_replay_and_finish(sql_storage, clock):
    workspace, (a, b) = setup_agents(sql_storage, clock)
    action = prepare(a)
    a[0].begin_action(action.id, lease_token=a[2].token)
    workspace.revoke("alice", "execute")
    with pytest.raises(AccessDenied):
        a[0].finish_action(action.id, lease_token=a[2].token, outcome="succeeded", result={})
    workspace.grant("alice", "execute")
    a[0].finish_action(
        action.id, lease_token=a[2].token, outcome="succeeded", result={"secret": True}
    )
    workspace.revoke("bob", "execute")
    with pytest.raises(AccessDenied):
        prepare(b)


def test_reconciliation_retry_fences_old_permit(sql_storage, clock):
    _, (a, _) = setup_agents(sql_storage, clock)
    action = prepare(a)
    first = a[0].begin_action(action.id, lease_token=a[2].token)
    a[0].finish_action(action.id, lease_token=a[2].token, outcome="unknown", result={})
    a[0].reconcile_action(
        action.id, lease_token=a[2].token, outcome="retryable", evidence={"remote_absent": True}
    )
    second = a[0].begin_action(action.id, lease_token=a[2].token)
    assert second.fencing_token > first.fencing_token


@pytest.mark.parametrize("same_operation", [False, True])
def test_concurrent_separate_connections_serialize_resource(sql_storage, clock, same_operation):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from ai_knot.always_on.workspace import WorkspaceMemory
    from ai_knot.storage.postgres_storage import PostgresStorage
    from ai_knot.storage.sqlite_storage import SQLiteStorage

    _, agents = setup_agents(sql_storage, clock)
    barrier = Barrier(2)

    def attempt(index):
        storage = (
            SQLiteStorage(sql_storage._db_path)
            if isinstance(sql_storage, SQLiteStorage)
            else PostgresStorage(sql_storage._dsn)
        )
        workspace = WorkspaceMemory(storage, "team")
        name = ("alice", "bob")[index]
        memory = AlwaysOnMemory(
            KnowledgeBase(name, storage=storage), clock=clock, workspace=workspace
        )
        barrier.wait(timeout=10)
        try:
            action = prepare(
                (memory, agents[index][1], agents[index][2]),
                operation="deploy" if same_operation else name,
            )
            return memory.begin_action(action.id, lease_token=agents[index][2].token)
        except RevisionConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, (0, 1)))
    assert sum(result is not None for result in results) == 1


def test_abandon_prepared_releases_resource_but_never_unknown(sql_storage, clock):
    workspace, (a, b) = setup_agents(sql_storage, clock)
    action = prepare(a)
    with pytest.raises(AccessDenied):
        workspace.abandon("bob", "deploy")
    workspace.abandon("alice", "deploy")
    replacement = prepare(b, operation="replacement")
    with pytest.raises(RevisionConflict):
        a[0].begin_action(action.id, lease_token=a[2].token)
    b[0].begin_action(replacement.id, lease_token=b[2].token)
    with pytest.raises(RevisionConflict):
        workspace.abandon("bob", "replacement")


def test_workspace_rejects_unrelated_database(tmp_path, clock):
    from ai_knot.always_on.workspace import WorkspaceMemory
    from ai_knot.storage.sqlite_storage import SQLiteStorage

    workspace = WorkspaceMemory(SQLiteStorage(str(tmp_path / "a.db")), "team")
    with pytest.raises(ValueError, match="storage"):
        AlwaysOnMemory(
            KnowledgeBase("alice", storage=SQLiteStorage(str(tmp_path / "b.db"))),
            workspace=workspace,
            clock=clock,
        )


@pytest.mark.parametrize("reopen_workspace", [None, "other"])
def test_reopen_cannot_downgrade_persisted_workspace_action(sql_storage, clock, reopen_workspace):
    from ai_knot.always_on.workspace import WorkspaceMemory

    workspace, (a, b) = setup_agents(sql_storage, clock)
    old = prepare(a)
    current = prepare(b)
    b[0].begin_action(current.id, lease_token=b[2].token)
    workspace.revoke("alice", "execute")
    other = None
    if reopen_workspace is not None:
        other = WorkspaceMemory(sql_storage, reopen_workspace)
        other.grant("alice", "execute")
    reopened = AlwaysOnMemory(
        KnowledgeBase("alice", storage=sql_storage), clock=clock, workspace=other
    )
    with pytest.raises(AccessDenied):
        reopened.begin_action(old.id, lease_token=a[2].token)
    with pytest.raises(AccessDenied):
        reopened.resume(a[1].id)
    with pytest.raises(AccessDenied):
        reopened.prepare_action(
            a[1].id, lease_token=a[2].token, action_key="deploy", request={"v": 1}
        )
    with pytest.raises(AccessDenied):
        reopened.finish_action(old.id, lease_token=a[2].token, outcome="succeeded", result={})
    with pytest.raises(AccessDenied):
        reopened.reconcile_action(
            old.id, lease_token=a[2].token, outcome="retryable", evidence={"checked": True}
        )
    if other is not None:
        with pytest.raises(AccessDenied):
            prepare((reopened, a[1], a[2]))
