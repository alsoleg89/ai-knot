"""Trusted-host workspace authority and durable external-operation coordination."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .models import (
    AccessDenied,
    ActionPermit,
    ActionRecord,
    ActionState,
    IdempotencyConflict,
    RevisionConflict,
    digest,
    nonempty,
)
from .storage import AlwaysOnSession, AlwaysOnStorageCapable, SQLRow, sql


def initialize_workspace_bindings(session: AlwaysOnSession) -> None:
    """Persist action association even when a facade has no workspace configured."""
    session.execute(
        sql(
            "CREATE TABLE IF NOT EXISTS ao_workspace_bindings (workspace "
            "TEXT NOT NULL, agent TEXT NOT NULL, action TEXT NOT NULL, "
            "operation TEXT NOT NULL, PRIMARY KEY(workspace,agent,action))"
        )
    )
    session.execute(
        sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS ao_workspace_action_binding "
            "ON ao_workspace_bindings(agent,action)"
        )
    )


def require_action_workspace(
    session: AlwaysOnSession,
    agent: str,
    action_id: str,
    workspace: WorkspaceMemory | None,
) -> None:
    """A persisted shared action cannot become private when its facade reopens."""
    bound = session.fetch_one(
        sql("SELECT workspace FROM ao_workspace_bindings WHERE agent=? AND action=?"),
        (agent, action_id),
    )
    if bound is None:
        return
    if workspace is None or bound["workspace"] != workspace.workspace_id:
        raise AccessDenied("action requires its persisted workspace")
    workspace.require(session, agent, "execute")


@dataclass(frozen=True)
class WorkspaceActionPermit(ActionPermit):
    resource_key: str
    fencing_token: int


class WorkspaceMemory:
    """Host-managed grants; this object is not a tenant authentication boundary."""

    def __init__(self, storage: AlwaysOnStorageCapable, workspace_id: str) -> None:
        nonempty(workspace_id, "workspace_id")
        self._storage = storage
        self.workspace_id = workspace_id
        self.lock_namespace = "__workspace__:" + workspace_id
        with storage.always_on_transaction(
            ("__always_on_schema__", self.lock_namespace), write=True
        ) as session:
            for statement in (
                (
                    "CREATE TABLE IF NOT EXISTS ao_workspace_grants (workspace "
                    "TEXT NOT NULL, agent TEXT NOT NULL, permission TEXT NOT "
                    "NULL, PRIMARY KEY(workspace,agent,permission))"
                ),
                (
                    "CREATE TABLE IF NOT EXISTS ao_workspace_operations "
                    "(workspace TEXT NOT NULL, operation TEXT NOT NULL, resource "
                    "TEXT NOT NULL, request_digest TEXT NOT NULL, agent TEXT NOT "
                    "NULL, action TEXT NOT NULL, state TEXT NOT NULL, result "
                    "TEXT, generation BIGINT NOT NULL, PRIMARY "
                    "KEY(workspace,operation))"
                ),
                (
                    "CREATE TABLE IF NOT EXISTS ao_workspace_resources (workspace"
                    " TEXT NOT NULL, resource TEXT NOT NULL, operation TEXT, "
                    "generation BIGINT NOT NULL, PRIMARY KEY(workspace,resource))"
                ),
            ):
                session.execute(sql(statement))
            initialize_workspace_bindings(session)

    def check_storage(self, storage: AlwaysOnStorageCapable) -> None:
        """Allow reopened connections to the same configured SQL database only."""
        if storage is self._storage:
            return
        if type(storage) is type(self._storage):
            path = getattr(storage, "_db_path", None)
            own_path = getattr(self._storage, "_db_path", None)
            if (
                path is not None
                and own_path is not None
                and path != ":memory:"
                and Path(path).resolve() == Path(own_path).resolve()
            ):
                return
            dsn = getattr(storage, "_dsn", None)
            if dsn is not None and dsn == getattr(self._storage, "_dsn", None):
                return
        raise ValueError("workspace storage must match memory storage")

    def abandon(self, agent_id: str, operation_key: str) -> None:
        """Cancel this agent's prepared reservation; never release a possible effect.

        Future preparation replays the failed result. A trusted host may explicitly
        reconcile it as retryable before making another attempt.
        """
        nonempty(operation_key, "operation_key")
        with self._storage.always_on_transaction((self.lock_namespace,), write=True) as session:
            self.require(session, agent_id, "execute")
            row = session.fetch_one(
                sql("SELECT * FROM ao_workspace_operations WHERE workspace=? AND operation=?"),
                (self.workspace_id, operation_key),
            )
            if row is None or row["agent"] != agent_id:
                raise AccessDenied("operation unavailable")
            if row["state"] != "prepared":
                raise RevisionConflict("only a prepared operation can be abandoned")
            session.execute(
                sql(
                    "UPDATE ao_workspace_operations SET state='failed',result=? "
                    "WHERE workspace=? AND operation=?"
                ),
                (json.dumps({"abandoned": True}), self.workspace_id, operation_key),
            )
            session.execute(
                sql(
                    "UPDATE ao_workspace_resources SET operation=NULL "
                    "WHERE workspace=? AND resource=? AND operation=?"
                ),
                (self.workspace_id, row["resource"], operation_key),
            )

    @staticmethod
    def _permission(permission: str) -> None:
        if permission not in ("execute", "maintain"):
            raise ValueError("permission must be execute or maintain")

    def grant(self, agent_id: str, permission: str) -> None:
        self._permission(permission)
        nonempty(agent_id, "agent_id")
        with self._storage.always_on_transaction((self.lock_namespace,), write=True) as session:
            session.execute(
                sql("INSERT INTO ao_workspace_grants VALUES(?,?,?) ON CONFLICT DO NOTHING"),
                (self.workspace_id, agent_id, permission),
            )

    def revoke(self, agent_id: str, permission: str) -> None:
        self._permission(permission)
        with self._storage.always_on_transaction((self.lock_namespace,), write=True) as session:
            session.execute(
                sql(
                    "DELETE FROM ao_workspace_grants WHERE workspace=? AND agent=? AND permission=?"
                ),
                (self.workspace_id, agent_id, permission),
            )

    def require(self, session: AlwaysOnSession, agent_id: str, permission: str) -> None:
        self._permission(permission)
        if (
            session.fetch_one(
                sql(
                    "SELECT 1 AS granted FROM ao_workspace_grants WHERE "
                    "workspace=? AND agent=? AND permission=?"
                ),
                (self.workspace_id, agent_id, permission),
            )
            is None
        ):
            raise AccessDenied("workspace permission required")

    def reserve(
        self,
        session: AlwaysOnSession,
        agent: str,
        action: ActionRecord,
        operation: str,
        resource: str,
    ) -> ActionRecord:
        require_action_workspace(session, agent, action.id, self)
        self.require(session, agent, "execute")
        row = session.fetch_one(
            sql("SELECT * FROM ao_workspace_operations WHERE workspace=? AND operation=?"),
            (self.workspace_id, operation),
        )
        if row is None and action.state != ActionState.PREPARED:
            raise RevisionConflict("only prepared actions may join a workspace operation")
        if row is not None:
            if row["resource"] != resource or row["request_digest"] != action.request_digest:
                raise IdempotencyConflict("operation key reused for another request or resource")
            if row["state"] in ("started", "unknown") and (row["agent"], row["action"]) != (
                agent,
                action.id,
            ):
                raise RevisionConflict("operation requires reconciliation by its owning agent")
            if row["state"] in ("succeeded", "failed"):
                action = replace(
                    action,
                    state=ActionState(str(row["state"])),
                    result=json.loads(str(row["result"])) if row["result"] else None,
                )
        bound = session.fetch_one(
            sql(
                "SELECT operation FROM ao_workspace_bindings WHERE "
                "workspace=? AND agent=? AND action=?"
            ),
            (self.workspace_id, agent, action.id),
        )
        if bound is not None and bound["operation"] != operation:
            raise IdempotencyConflict("action already bound to another operation")
        if row is None or row["state"] == "prepared":
            held = session.fetch_one(
                sql("SELECT * FROM ao_workspace_resources WHERE workspace=? AND resource=?"),
                (self.workspace_id, resource),
            )
            if held is not None and held["operation"] not in (None, operation):
                raise RevisionConflict("resource reserved by another operation")
            same = row is not None and (row["agent"], row["action"]) == (agent, action.id)
            generation = int(held["generation"] or 0) if held else 0
            if not same:
                generation += 1
            session.execute(
                sql(
                    "INSERT INTO ao_workspace_resources VALUES(?,?,?,?) ON "
                    "CONFLICT(workspace,resource) DO UPDATE SET "
                    "operation=excluded.operation,generation=excluded.generation"
                ),
                (self.workspace_id, resource, operation, generation),
            )
            session.execute(
                sql(
                    "INSERT INTO ao_workspace_operations "
                    "VALUES(?,?,?,?,?,?,?,?,?) ON "
                    "CONFLICT(workspace,operation) DO UPDATE SET "
                    "agent=excluded.agent,action=excluded.action,generation=excluded.generation"
                ),
                (
                    self.workspace_id,
                    operation,
                    resource,
                    action.request_digest,
                    agent,
                    action.id,
                    "prepared",
                    None,
                    generation,
                ),
            )
        session.execute(
            sql("INSERT INTO ao_workspace_bindings VALUES(?,?,?,?) ON CONFLICT DO NOTHING"),
            (self.workspace_id, agent, action.id, operation),
        )
        return replace(action, idempotency_key=digest([self.workspace_id, operation]))

    def check(self, session: AlwaysOnSession, agent: str, action: ActionRecord) -> SQLRow | None:
        row = session.fetch_one(
            sql(
                "SELECT o.* FROM ao_workspace_operations o JOIN "
                "ao_workspace_bindings b ON o.workspace=b.workspace AND "
                "o.operation=b.operation WHERE b.workspace=? AND b.agent=? "
                "AND b.action=?"
            ),
            (self.workspace_id, agent, action.id),
        )
        if row is None:
            return None
        self.require(session, agent, "execute")
        if (row["agent"], row["action"]) != (agent, action.id):
            raise RevisionConflict("workspace operation handed off")
        return row

    def update(
        self, session: AlwaysOnSession, agent: str, action: ActionRecord
    ) -> WorkspaceActionPermit | None:
        row = self.check(session, agent, action)
        if row is None:
            return None
        if action.state == ActionState.STARTED and row["state"] != "prepared":
            raise RevisionConflict("workspace operation is no longer prepared")
        session.execute(
            sql(
                "UPDATE ao_workspace_operations SET state=?,result=? WHERE "
                "workspace=? AND operation=?"
            ),
            (action.state.value, json.dumps(action.result), self.workspace_id, row["operation"]),
        )
        if action.state in (ActionState.SUCCEEDED, ActionState.FAILED):
            session.execute(
                sql(
                    "UPDATE ao_workspace_resources SET operation=NULL WHERE "
                    "workspace=? AND resource=? AND operation=?"
                ),
                (self.workspace_id, row["resource"], row["operation"]),
            )
        elif action.state == ActionState.PREPARED:
            held = session.fetch_one(
                sql(
                    "SELECT operation,generation FROM ao_workspace_resources "
                    "WHERE workspace=? AND resource=?"
                ),
                (self.workspace_id, row["resource"]),
            )
            if held is not None and held["operation"] not in (None, row["operation"]):
                raise RevisionConflict("resource reserved by another operation")
            assert held is not None
            generation = int(row["generation"] or 0)
            if row["state"] != "prepared":
                generation = int(held["generation"] or 0) + 1
                session.execute(
                    sql(
                        "UPDATE ao_workspace_operations SET generation=? "
                        "WHERE workspace=? AND operation=?"
                    ),
                    (generation, self.workspace_id, row["operation"]),
                )
            session.execute(
                sql(
                    "UPDATE ao_workspace_resources SET operation=?,generation=? "
                    "WHERE workspace=? AND resource=?"
                ),
                (row["operation"], generation, self.workspace_id, row["resource"]),
            )
        return WorkspaceActionPermit(
            action.id, action.idempotency_key, str(row["resource"]), int(row["generation"] or 0)
        )
