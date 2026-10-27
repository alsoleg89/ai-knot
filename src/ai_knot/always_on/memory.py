"""Durable agent work and pre-action checks on the existing SQL backend."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from ai_knot.knowledge import KnowledgeBase
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact

from . import _state as transitions
from .changes import access_projection, evidence_revision, visible
from .models import (
    AccessDenied,
    ActionPermit,
    ActionRecord,
    BasisIssue,
    EvidenceRef,
    FreshnessRecord,
    IdempotencyConflict,
    Intent,
    IntentState,
    JSONObject,
    Lease,
    ResumePacket,
    RevisionConflict,
    StaleEvidence,
    UnsupportedAlwaysOnBackend,
    VerificationRequest,
    VerificationResult,
    WaitCondition,
    WaitForChange,
    WaitForSlot,
    WaitUntil,
    canonical,
    digest,
    nonempty,
    page_limit,
    positive_integer,
    utc,
)
from .state_storage import (
    action_from_row,
    due_policy_rows,
    freshness_from_row,
    initialize,
    insert_intent,
    intent_from_row,
    save_action,
    save_freshness,
    save_intent,
    timestamp,
)
from .storage import AlwaysOnSession, AlwaysOnStorageCapable, enable_stream, sql
from .subscriptions import SubscriptionsMixin


def utc_now() -> datetime:
    return datetime.now(UTC)


class _ExpandedLocks(Exception):
    pass


class AlwaysOnMemory(SubscriptionsMixin):
    """Explicitly enable durable memory state; external actions remain the host's job."""

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        pool: SharedMemoryPool | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if not isinstance(kb._storage, AlwaysOnStorageCapable):
            raise UnsupportedAlwaysOnBackend("always-on requires SQLite or PostgreSQL storage")
        self._kb = kb
        self._storage = kb._storage
        self._owner = kb._agent_id
        self._pool = pool
        self._clock = clock
        self._now()
        from .maintenance import initialize_maintenance

        with self._transaction(("__always_on_schema__",)) as session:
            initialize(session)
            initialize_maintenance(session)
            enable_stream(session, self._owner, self._now())

    def _now(self) -> datetime:
        return utc(self._clock())

    def _transaction(
        self, namespaces: tuple[str, ...] = (), *, write: bool = True
    ) -> AbstractContextManager[AlwaysOnSession]:
        return self._storage.always_on_transaction(
            tuple(sorted({self._owner, *namespaces})), write=write
        )

    def _intent(self, session: AlwaysOnSession, intent_id: str) -> Intent:
        row = session.fetch_one(
            sql("SELECT * FROM ao_intents WHERE owner=? AND id=?"), (self._owner, intent_id)
        )
        if row is None:
            raise AccessDenied("intent unavailable")
        return intent_from_row(row)

    def _action(self, session: AlwaysOnSession, action_id: str) -> ActionRecord:
        row = session.fetch_one(
            sql("SELECT * FROM ao_actions WHERE owner=? AND id=?"), (self._owner, action_id)
        )
        if row is None:
            raise AccessDenied("action unavailable")
        return action_from_row(row)

    def _actions(self, session: AlwaysOnSession, intent_id: str) -> tuple[ActionRecord, ...]:
        return tuple(
            action_from_row(row)
            for row in session.fetch_all(
                sql("SELECT * FROM ao_actions WHERE owner=? AND intent_id=? ORDER BY id"),
                (self._owner, intent_id),
            )
        )

    @staticmethod
    def _dependencies(intent: Intent, actions: tuple[ActionRecord, ...]) -> set[str]:
        scopes = {ref.namespace for ref in intent.basis}
        for action in actions:
            scopes.update(ref.namespace for ref in action.basis)
        if isinstance(intent.waiting_for, WaitForSlot):
            scopes.add(intent.waiting_for.namespace)
        elif isinstance(intent.waiting_for, WaitForChange):
            scopes.add(intent.waiting_for.ref.namespace)
        return scopes

    @contextmanager
    def _intent_transaction(
        self, intent_id: str, refs: tuple[EvidenceRef, ...] = ()
    ) -> Iterator[tuple[AlwaysOnSession, Intent, tuple[ActionRecord, ...]]]:
        namespaces = {self._owner, *(ref.namespace for ref in refs)}
        for _attempt in range(3):
            with self._transaction(write=False) as snapshot:
                item = self._intent(snapshot, intent_id)
                namespaces.update(self._dependencies(item, self._actions(snapshot, intent_id)))
            try:
                with self._transaction(tuple(sorted(namespaces))) as session:
                    item = self._intent(session, intent_id)
                    actions = self._actions(session, intent_id)
                    current = self._dependencies(item, actions)
                    if not current <= namespaces:
                        namespaces.update(current)
                        raise _ExpandedLocks
                    yield session, item, actions
                return
            except _ExpandedLocks:
                continue
        raise RevisionConflict("dependency scopes changed repeatedly")

    def _refs_for_intent(self, intent_id: str) -> tuple[EvidenceRef, ...]:
        with self._transaction(write=False) as session:
            return self._intent(session, intent_id).basis

    def _check_namespace(self, namespace: str) -> None:
        nonempty(namespace, "namespace")
        if namespace == self._owner:
            return
        if namespace != "__shared__" or self._pool is None or not self._pool._persist_stats:
            raise AccessDenied("memory scope unavailable")
        other = self._pool._storage
        if other is not self._kb._storage:
            paths = ("_db_path", "db_path", "_dsn", "dsn")
            if not any(
                hasattr(other, key) and getattr(other, key) == getattr(self._storage, key, None)
                for key in paths
            ):
                raise AccessDenied("shared pool storage does not match")

    def _evidence_grants(self, session: AlwaysOnSession) -> set[str]:
        rows = session.fetch_all(
            sql("SELECT scope FROM acl_grants WHERE agent_id=?"), (self._owner,)
        )
        return {str(row["scope"]) for row in rows}

    def _readable(self, session: AlwaysOnSession, namespace: str, fact: Fact) -> bool:
        if namespace == self._owner:
            return True
        try:
            self._check_namespace(namespace)
        except AccessDenied:
            return False
        return visible(access_projection(fact), self._owner, self._evidence_grants(session))

    def _policy(
        self, session: AlwaysOnSession, namespace: str, fact: Fact
    ) -> FreshnessRecord | None:
        kind, key = ("slot", fact.slot_key) if fact.slot_key else ("fact", fact.id)
        row = session.fetch_one(
            sql(
                "SELECT * FROM ao_freshness WHERE namespace=? AND subject_kind=? AND subject_key=?"
            ),
            (namespace, kind, key),
        )
        return None if row is None else freshness_from_row(row)

    def _view(
        self, session: AlwaysOnSession, ref: EvidenceRef, now: datetime, *, freshness: bool = True
    ) -> transitions.EvidenceView:
        try:
            self._check_namespace(ref.namespace)
        except AccessDenied:
            return transitions.EvidenceView(ref.namespace, ref.fact_id, None, readable=False)
        facts = session.load_facts(ref.namespace, fact_id=ref.fact_id)
        fact = next((fact for fact in facts if fact.id == ref.fact_id), None)
        if fact is None:
            return transitions.EvidenceView(ref.namespace, ref.fact_id, None)
        if not self._readable(session, ref.namespace, fact):
            return transitions.EvidenceView(ref.namespace, ref.fact_id, None, readable=False)
        supported = fact.supported
        if fact.slot_key:
            facts = session.load_facts(ref.namespace, slot_key=fact.slot_key)
            current = [
                item
                for item in facts
                if item.slot_key == fact.slot_key
                and item.is_active(now)
                and self._readable(session, ref.namespace, item)
            ]
            values = {item.value_text or item.content for item in current if item.supported}
            supported = supported and len(values) == 1
            if (
                current
                and fact.id
                != max(current, key=lambda item: (item.version, item.created_at, item.id)).id
            ):
                return transitions.EvidenceView(ref.namespace, ref.fact_id, None)
        return transitions.EvidenceView(
            ref.namespace,
            ref.fact_id,
            evidence_revision(fact),
            active=fact.is_active(now),
            supported=supported,
            freshness=self._policy(session, ref.namespace, fact) if freshness else None,
        )

    def _views(
        self,
        session: AlwaysOnSession,
        refs: tuple[EvidenceRef, ...],
        now: datetime,
        *,
        freshness: bool = True,
    ) -> tuple[transitions.EvidenceView, ...]:
        return tuple(self._view(session, ref, now, freshness=freshness) for ref in refs)

    def _validate_basis(
        self,
        session: AlwaysOnSession,
        refs: tuple[EvidenceRef, ...],
        now: datetime,
        *,
        freshness: bool = True,
    ) -> tuple[BasisIssue, ...]:
        return transitions.validate_basis(
            refs, self._views(session, refs, now, freshness=freshness), now
        )

    def _fact_for_ref(self, session: AlwaysOnSession, ref: EvidenceRef, now: datetime) -> Fact:
        issues = self._validate_basis(session, (ref,), now, freshness=False)
        if issues:
            if issues[0].reason == "access_changed":
                raise AccessDenied("memory scope unavailable")
            raise StaleEvidence("fact is no longer current")
        return session.load_facts(ref.namespace, fact_id=ref.fact_id)[0]

    def create_intent(
        self, *, client_key: str, goal: str, next_step: str = "", deadline: datetime | None = None
    ) -> Intent:
        nonempty(client_key, "client_key")
        nonempty(goal, "goal")
        if not isinstance(next_step, str):
            raise ValueError("next_step must be text")
        if deadline is not None:
            utc(deadline)
        create_digest = digest(
            {"goal": goal, "next_step": next_step, "deadline": timestamp(deadline)}
        )
        with self._transaction() as session:
            row = session.fetch_one(
                sql("SELECT * FROM ao_intents WHERE owner=? AND client_key=?"),
                (self._owner, client_key),
            )
            if row is not None:
                if row["create_digest"] != create_digest:
                    raise IdempotencyConflict("client key reused with different input")
                return intent_from_row(row)
            intent = Intent(
                uuid4().hex,
                self._owner,
                client_key,
                goal,
                IntentState.READY,
                {},
                next_step,
                None,
                1,
                (),
                None,
                deadline,
                self._now(),
            )
            insert_intent(session, intent, create_digest)
            return intent

    def get_intent(self, intent_id: str) -> Intent:
        nonempty(intent_id, "intent_id")
        with self._transaction(write=False) as session:
            return self._intent(session, intent_id)

    def list_intents(self, *, limit: int = 100) -> list[Intent]:
        page_limit(limit)
        with self._transaction(write=False) as session:
            return [
                intent_from_row(row)
                for row in session.fetch_all(
                    sql("SELECT * FROM ao_intents WHERE owner=? ORDER BY id LIMIT ?"),
                    (self._owner, limit),
                )
            ]

    def _settle(
        self,
        session: AlwaysOnSession,
        intent: Intent,
        actions: tuple[ActionRecord, ...],
        now: datetime,
    ) -> tuple[Intent, tuple[ActionRecord, ...]]:
        current, changed = transitions.expire(intent, actions, now)
        save_intent(session, intent, current)
        for old, new in zip(actions, changed, strict=True):
            save_action(session, self._owner, old, new)
        return current, changed

    def claim(self, intent_id: str, *, worker_id: str, ttl_seconds: int = 60) -> Lease:
        nonempty(worker_id, "worker_id")
        positive_integer(ttl_seconds, "ttl_seconds", 3600)
        error = None
        with self._intent_transaction(intent_id) as (session, intent, actions):
            now = self._now()
            intent, _ = self._settle(session, intent, actions, now)
            try:
                current, lease = transitions.claim(
                    intent,
                    worker_id=worker_id,
                    token=secrets.token_urlsafe(32),
                    ttl_seconds=ttl_seconds,
                    now=now,
                )
                save_intent(session, intent, current)
            except RevisionConflict as exc:
                error = exc
        if error is not None:
            raise error
        return lease

    def renew(self, intent_id: str, *, lease_token: str, ttl_seconds: int = 60) -> Lease:
        positive_integer(ttl_seconds, "ttl_seconds", 3600)
        with self._intent_transaction(intent_id) as (session, intent, _):
            current = transitions.renew(
                intent, token=lease_token, ttl_seconds=ttl_seconds, now=self._now()
            )
            save_intent(session, intent, current)
            assert current.lease is not None
            return current.lease

    def checkpoint(
        self,
        intent_id: str,
        *,
        lease_token: str,
        expected_revision: int,
        state: IntentState,
        progress: JSONObject,
        next_step: str = "",
        waiting_for: WaitCondition | None = None,
        basis: tuple[EvidenceRef, ...] = (),
    ) -> Intent:
        state = IntentState(state)
        positive_integer(expected_revision, "expected_revision")
        if not isinstance(progress, dict) or not isinstance(next_step, str):
            raise ValueError("progress object and next_step text required")
        canonical(progress)
        refs = transitions.union_basis(basis)
        extra = list(refs)
        if isinstance(waiting_for, WaitUntil):
            utc(waiting_for.at)
        elif isinstance(waiting_for, WaitForSlot):
            self._check_namespace(waiting_for.namespace)
            nonempty(waiting_for.slot_key, "slot_key")
            extra.append(EvidenceRef(waiting_for.namespace, "wait-slot", "wait-slot"))
        elif isinstance(waiting_for, WaitForChange):
            extra.append(waiting_for.ref)
        elif waiting_for is not None:
            raise ValueError("unknown wait condition")
        with self._intent_transaction(intent_id, tuple(extra)) as (session, intent, actions):
            now = self._now()
            current = transitions.checkpoint(
                intent,
                actions,
                token=lease_token,
                expected_revision=expected_revision,
                state=state,
                progress=progress,
                next_step=next_step,
                waiting_for=waiting_for,
                basis=refs,
                views=self._views(session, refs, now),
                now=now,
            )
            save_intent(session, intent, current)
            return current

    def _condition(
        self, session: AlwaysOnSession, wait: WaitCondition | None, now: datetime
    ) -> bool:
        if isinstance(wait, WaitUntil):
            return now >= wait.at
        if isinstance(wait, WaitForChange):
            view = self._view(session, wait.ref, now, freshness=False)
            if not view.readable:
                raise AccessDenied("wait scope unavailable")
            return view.revision != wait.ref.revision
        if isinstance(wait, WaitForSlot):
            self._check_namespace(wait.namespace)
            facts = [
                fact
                for fact in session.load_facts(wait.namespace, slot_key=wait.slot_key)
                if fact.slot_key == wait.slot_key
                and fact.is_active(now)
                and fact.supported
                and self._readable(session, wait.namespace, fact)
            ]
            return {fact.value_text or fact.content for fact in facts} == {wait.value_text}
        return False

    def resume(self, intent_id: str) -> ResumePacket:
        with self._intent_transaction(intent_id) as (session, intent, actions):
            now = self._now()
            current, changed = transitions.expire(intent, actions, now)
            try:
                condition = self._condition(session, current.waiting_for, now)
            except AccessDenied:
                condition = False
                if current.state not in transitions.TERMINAL:
                    current = transitions.transition(
                        current,
                        now,
                        state=IntentState.BLOCKED,
                        blocked_reasons=tuple(
                            sorted(set(current.blocked_reasons) | {"access_changed"})
                        ),
                    )
            packet = transitions.resume(
                current,
                changed,
                self._views(session, current.basis, now),
                condition_met=condition,
                now=now,
            )
            save_intent(session, intent, packet.intent)
            for old, new in zip(actions, changed, strict=True):
                save_action(session, self._owner, old, new)
            return packet

    def capture_evidence(self, refs: tuple[tuple[str, str], ...]) -> tuple[EvidenceRef, ...]:
        for namespace, fact_id in refs:
            self._check_namespace(namespace)
            nonempty(fact_id, "fact_id")
        scopes = tuple(sorted({namespace for namespace, _ in refs}))
        if "__shared__" in scopes:
            with self._transaction(scopes) as session:
                enable_stream(session, "__shared__", self._now())
        with self._transaction(scopes, write=False) as session:
            found = []
            now = self._now()
            for namespace, fact_id in refs:
                fact = next(iter(session.load_facts(namespace, fact_id=fact_id)), None)
                if fact is None or not self._readable(session, namespace, fact):
                    raise AccessDenied("fact unavailable")
                ref = EvidenceRef(namespace, fact_id, evidence_revision(fact))
                self._fact_for_ref(session, ref, now)
                found.append(ref)
            return transitions.union_basis(tuple(found))

    def prepare_action(
        self,
        intent_id: str,
        *,
        lease_token: str,
        action_key: str,
        request: JSONObject,
        basis: tuple[EvidenceRef, ...] = (),
    ) -> ActionRecord:
        nonempty(action_key, "action_key")
        if not isinstance(request, dict):
            raise ValueError("request must be object")
        canonical(request)
        refs = transitions.union_basis(basis)
        with self._intent_transaction(intent_id, refs) as (session, intent, _):
            row = session.fetch_one(
                sql("SELECT * FROM ao_actions WHERE owner=? AND intent_id=? AND action_key=?"),
                (self._owner, intent_id, action_key),
            )
            old = None if row is None else action_from_row(row)
            now = self._now()
            effective = transitions.union_basis(intent.basis, refs)
            current = transitions.prepare_action(
                intent,
                old,
                action_id=uuid4().hex,
                token=lease_token,
                action_key=action_key,
                request=request,
                basis=refs,
                views=self._views(session, effective, now),
                now=now,
            )
            save_action(session, self._owner, old, current)
            return current

    def _action_intent_id(self, action_id: str) -> str:
        with self._transaction(write=False) as session:
            return self._action(session, action_id).intent_id

    def begin_action(self, action_id: str, *, lease_token: str) -> ActionPermit:
        intent_id = self._action_intent_id(action_id)
        error = None
        with self._intent_transaction(intent_id) as (session, intent, actions):
            action = self._action(session, action_id)
            now = self._now()
            transitions.require_lease(intent, lease_token, now)
            refs = transitions.union_basis(intent.basis, action.basis)
            issues = self._validate_basis(session, refs, now)
            if issues:
                reasons = tuple(
                    sorted(set(intent.blocked_reasons) | {issue.reason for issue in issues})
                )
                current = transitions.transition(
                    intent, now, state=IntentState.BLOCKED, blocked_reasons=reasons
                )
                save_intent(session, intent, current)
                error = StaleEvidence("pre-action guard rejected basis")
            else:
                started_action, permit = transitions.begin_action(
                    intent,
                    action,
                    actions,
                    token=lease_token,
                    views=self._views(session, refs, now),
                    now=now,
                )
                save_action(session, self._owner, action, started_action)
        if error is not None:
            raise error
        return permit

    def finish_action(
        self,
        action_id: str,
        *,
        lease_token: str,
        outcome: Literal["succeeded", "failed", "unknown"],
        result: JSONObject,
    ) -> ActionRecord:
        if not isinstance(result, dict):
            raise ValueError("result must be object")
        canonical(result)
        with self._intent_transaction(self._action_intent_id(action_id)) as (session, intent, _):
            old = self._action(session, action_id)
            current = transitions.finish_action(
                intent, old, token=lease_token, outcome=outcome, result=result, now=self._now()
            )
            save_action(session, self._owner, old, current)
            return current

    def reconcile_action(
        self,
        action_id: str,
        *,
        lease_token: str,
        outcome: Literal["succeeded", "failed", "retryable"],
        evidence: JSONObject,
    ) -> ActionRecord:
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be object")
        canonical(evidence)
        with self._intent_transaction(self._action_intent_id(action_id)) as (
            session,
            intent,
            actions,
        ):
            old = self._action(session, action_id)
            now = self._now()
            refs = tuple({*intent.basis, *old.basis})
            current, action = transitions.reconcile_action(
                intent,
                old,
                actions,
                token=lease_token,
                outcome=outcome,
                evidence=evidence,
                views=self._views(session, refs, now),
                now=now,
            )
            save_intent(session, intent, current)
            save_action(session, self._owner, old, action)
            return action

    def set_freshness(
        self, ref: EvidenceRef, *, source_ref: str, recheck_after_seconds: int
    ) -> FreshnessRecord:
        nonempty(source_ref, "source_ref")
        positive_integer(recheck_after_seconds, "recheck_after_seconds")
        with self._transaction((ref.namespace,)) as session:
            now = self._now()
            fact = self._fact_for_ref(session, ref, now)
            policy = transitions.set_freshness(
                ref,
                slot_key=fact.slot_key,
                source_ref=source_ref,
                interval=recheck_after_seconds,
                now=now,
            )
            save_freshness(session, policy)
            return policy

    def due_verifications(self, *, limit: int = 100) -> list[VerificationRequest]:
        page_limit(limit)
        namespaces = (
            (self._owner, "__shared__")
            if self._pool is not None and self._pool._persist_stats
            else (self._owner,)
        )
        with self._transaction(namespaces, write=False) as session:
            now = self._now()
            due: list[VerificationRequest] = []
            offset = 0
            batch_size = max(32, limit)
            while True:
                rows = due_policy_rows(
                    session, namespaces, self._owner, now, limit=batch_size, offset=offset
                )
                for row in rows:
                    policy = freshness_from_row(row)
                    facts = session.load_facts(
                        policy.ref.namespace, fact_id=str(row["current_fact_id"])
                    )
                    if not facts:
                        continue
                    ref = EvidenceRef(
                        policy.ref.namespace, facts[0].id, evidence_revision(facts[0])
                    )
                    if self._validate_basis(session, (ref,), now, freshness=False):
                        continue
                    if transitions.verification_due(policy, ref, now):
                        due.append(VerificationRequest(ref, policy.source_ref))
                        if len(due) == limit:
                            return due
                if len(rows) < batch_size:
                    return due
                offset += len(rows)

    def record_verification(
        self, ref: EvidenceRef, *, result: VerificationResult
    ) -> FreshnessRecord:
        """Apply a trusted host verdict to the current policy for this exact fact.

        Hosts performing their own asynchronous checks must coordinate policy changes.
        ``run_maintenance`` additionally fences its provider response by policy generation.
        """
        if not isinstance(result, VerificationResult):
            raise ValueError("typed verification result required")
        from .models import VerificationOutcome

        result = VerificationResult(
            VerificationOutcome(result.outcome), result.evidence, result.source_version
        )
        if not isinstance(result.evidence, dict) or not isinstance(result.source_version, str):
            raise ValueError("invalid verification result")
        canonical(result.evidence)
        with self._transaction((ref.namespace,)) as session:
            now = self._now()
            fact = self._fact_for_ref(session, ref, now)
            policy = self._policy(session, ref.namespace, fact)
            if policy is None:
                raise RevisionConflict("freshness policy missing")
            current = transitions.record_verification(
                policy, ref, current=ref, result=result, now=now
            )
            save_freshness(session, current)
            return current
