"""Persisted workspace refresh exclusion and UTC calendar-day call accounting.

Refresh exclusion is keyed by the actual shared namespace/fact across all workspaces.
Unknown provider outcomes keep their claim until its deadline. A successful response
releases exclusion only atomically with its revision-checked freshness update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from .models import AccessDenied, RevisionConflict, VerificationRequest
from .state_storage import timestamp
from .storage import sql

if TYPE_CHECKING:
    from .memory import AlwaysOnMemory
    from .storage import AlwaysOnSession
    from .workspace import WorkspaceMemory


def initialize(session: AlwaysOnSession) -> None:
    session.execute(
        sql("""CREATE TABLE IF NOT EXISTS ao_refresh_claims (
        workspace TEXT NOT NULL, namespace TEXT NOT NULL, fact_id TEXT NOT NULL,
        generation TEXT NOT NULL, claim_key TEXT NOT NULL, expires_at TEXT NOT NULL,
        PRIMARY KEY(namespace,fact_id))""")
    )
    session.execute(
        sql("""CREATE TABLE IF NOT EXISTS ao_workspace_call_limits (
        workspace TEXT PRIMARY KEY, max_calls BIGINT NOT NULL)""")
    )
    session.execute(
        sql("""CREATE TABLE IF NOT EXISTS ao_workspace_call_usage (
        workspace TEXT NOT NULL, utc_day TEXT NOT NULL, reserved BIGINT NOT NULL,
        PRIMARY KEY(workspace,utc_day))""")
    )


def configure_workspace_budget(workspace: WorkspaceMemory, *, max_model_calls_per_day: int) -> None:
    """Trusted host sets cost-unit ceiling per UTC day (00:00 inclusive to next 00:00).

    No configuration means zero paid workspace maintenance calls. Reservations are never refunded,
    including cancellation or crashes. Changing the ceiling does not reset usage.
    """
    if type(max_model_calls_per_day) is not int or max_model_calls_per_day < 0:
        raise ValueError("max_model_calls_per_day must be a nonnegative integer")
    with workspace._storage.always_on_transaction(
        ("__always_on_schema__", workspace.lock_namespace), write=True
    ) as session:
        initialize(session)
        session.execute(
            sql("""INSERT INTO ao_workspace_call_limits(workspace,max_calls)
            VALUES(?,?) ON CONFLICT(workspace) DO UPDATE SET max_calls=excluded.max_calls"""),
            (workspace.lock_namespace, max_model_calls_per_day),
        )


def require_maintain(memory: AlwaysOnMemory, session: AlwaysOnSession, namespace: str) -> None:
    if namespace == memory._owner:
        return
    workspace = getattr(memory, "_workspace", None)
    if workspace is None:
        raise AccessDenied("shared maintenance requires workspace authority")
    workspace.require(session, memory._owner, "maintain")


def locks(memory: AlwaysOnMemory, namespace: str) -> tuple[str, ...]:
    workspace = getattr(memory, "_workspace", None)
    return (
        (namespace, workspace.lock_namespace)
        if workspace and namespace != memory._owner
        else (namespace,)
    )


def claim(
    memory: AlwaysOnMemory,
    session: AlwaysOnSession,
    request: VerificationRequest,
    claim_key: str,
    expires_at: str,
    cost: int,
) -> str:
    """Called in the same transaction as per-run cost reservation."""
    require_maintain(memory, session, request.ref.namespace)
    assert memory._workspace is not None
    workspace = memory._workspace.lock_namespace
    initialize(session)
    row = session.fetch_one(
        sql("""SELECT * FROM ao_refresh_claims WHERE namespace=? AND fact_id=?"""),
        (request.ref.namespace, request.ref.fact_id),
    )
    if row and str(row["expires_at"]) > str(timestamp(memory._now())):
        raise RevisionConflict("shared refresh already claimed")
    reserve_budget(memory, session, cost)
    token = uuid4().hex
    session.execute(
        sql("""INSERT INTO ao_refresh_claims
        (workspace,namespace,fact_id,generation,claim_key,expires_at) VALUES(?,?,?,?,?,?)
        ON CONFLICT(namespace,fact_id) DO UPDATE SET
        workspace=excluded.workspace,generation=excluded.generation,claim_key=excluded.claim_key,expires_at=excluded.expires_at"""),
        (workspace, request.ref.namespace, request.ref.fact_id, token, claim_key, expires_at),
    )
    return token


def reserve_budget(memory: AlwaysOnMemory, session: AlwaysOnSession, cost: int) -> None:
    """Charge every workspace maintenance provider, including private consolidation."""
    if memory._workspace is None:
        return
    initialize(session)
    workspace = memory._workspace.lock_namespace
    day = memory._now().date().isoformat()
    limit = session.fetch_one(
        sql("SELECT max_calls FROM ao_workspace_call_limits WHERE workspace=?"), (workspace,)
    )
    usage = session.fetch_one(
        sql("SELECT reserved FROM ao_workspace_call_usage WHERE workspace=? AND utc_day=?"),
        (workspace, day),
    )
    if cost + (int(usage["reserved"] or 0) if usage else 0) > (
        int(limit["max_calls"] or 0) if limit else 0
    ):
        from .maintenance import _BudgetExhausted

        raise _BudgetExhausted("workspace_model_budget")
    session.execute(
        sql("""INSERT INTO ao_workspace_call_usage(workspace,utc_day,reserved)
        VALUES(?,?,?) ON CONFLICT(workspace,utc_day) DO UPDATE SET
        reserved=ao_workspace_call_usage.reserved+excluded.reserved"""),
        (workspace, day, cost),
    )


def guard_claim(
    memory: AlwaysOnMemory,
    session: AlwaysOnSession,
    request: VerificationRequest,
    token: str,
) -> None:
    assert memory._workspace is not None
    require_maintain(memory, session, request.ref.namespace)
    row = session.fetch_one(
        sql("""SELECT generation,expires_at FROM ao_refresh_claims
        WHERE workspace=? AND namespace=? AND fact_id=?"""),
        (memory._workspace.lock_namespace, request.ref.namespace, request.ref.fact_id),
    )
    if (
        not row
        or row["generation"] != token
        or str(row["expires_at"]) <= str(timestamp(memory._now()))
    ):
        raise RevisionConflict("shared refresh generation expired")


def release_claim(
    memory: AlwaysOnMemory,
    session: AlwaysOnSession,
    request: VerificationRequest,
    token: str,
) -> None:
    assert memory._workspace is not None
    session.execute(
        sql("""DELETE FROM ao_refresh_claims WHERE workspace=? AND namespace=?
        AND fact_id=? AND generation=?"""),
        (memory._workspace.lock_namespace, request.ref.namespace, request.ref.fact_id, token),
    )
