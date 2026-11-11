"""Durable agent work and pre-action checks on the existing SQL backend."""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from ai_knot.knowledge import KnowledgeBase
from ai_knot.pool import SharedMemoryPool
from ai_knot.types import Fact, MemoryType

from . import _state as transitions
from .changes import ChangeBatch, FactChange, access_projection, evidence_revision, visible
from .changes import canonical as encode_changes
from .lineage import (
    DEPENDENCIES_KEY,
    ROOTS_KEY,
    canonical_observations,
    encode_roots,
    evidence_dependencies,
    evidence_roots,
    is_derived,
    lineage_status,
)
from .models import (
    AccessDenied,
    ActionPermit,
    ActionRecord,
    BasisIssue,
    ContextFact,
    ContextReview,
    EvidenceRef,
    FreshnessRecord,
    IdempotencyConflict,
    Intent,
    IntentState,
    JSONObject,
    Lease,
    RecordedExperience,
    ResumePacket,
    RevisionConflict,
    SnapshotTooLarge,
    StaleEvidence,
    TaskContext,
    TaskContextGuard,
    UnsupportedAlwaysOnBackend,
    VerificationRequest,
    VerificationResult,
    WaitCondition,
    WaitForChange,
    WaitForSlot,
    WaitUntil,
    canonical,
    context_label,
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
from .storage import AlwaysOnSession, AlwaysOnStorageCapable, SQLStatement, enable_stream, sql
from .subscriptions import SubscriptionsMixin
from .workspace import WorkspaceMemory, initialize_workspace_bindings, require_action_workspace


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
        workspace: WorkspaceMemory | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if not isinstance(kb._storage, AlwaysOnStorageCapable):
            raise UnsupportedAlwaysOnBackend("always-on requires SQLite or PostgreSQL storage")
        self._kb = kb
        self._storage = kb._storage
        self._owner = kb._agent_id
        self._pool = pool
        self._workspace = workspace
        if workspace is not None:
            workspace.check_storage(self._storage)
        self._clock = clock
        self._now()
        from .maintenance import initialize_maintenance

        with self._transaction(("__always_on_schema__",)) as session:
            initialize(session)
            initialize_workspace_bindings(session)
            initialize_maintenance(session)
            session.execute(sql("UPDATE ao_schema SET version=8 WHERE id=1 AND version<8"))
            enable_stream(session, self._owner, self._now())

    def _now(self) -> datetime:
        return utc(self._clock())

    def _transaction(
        self, namespaces: tuple[str, ...] = (), *, write: bool = True
    ) -> AbstractContextManager[AlwaysOnSession]:
        return self._storage.always_on_transaction(
            tuple(
                sorted(
                    {
                        self._owner,
                        *namespaces,
                        *((self._workspace.lock_namespace,) if self._workspace else ()),
                        *(("__shared__",) if not write and self._pool is not None else ()),
                    }
                )
            ),
            write=write,
        )

    def _intent(self, session: AlwaysOnSession, intent_id: str) -> Intent:
        row = session.fetch_one(
            sql("SELECT * FROM ao_intents WHERE owner=? AND id=?"), (self._owner, intent_id)
        )
        if row is None:
            raise AccessDenied("intent unavailable")
        return intent_from_row(row)

    def _action(self, session: AlwaysOnSession, action_id: str) -> ActionRecord:
        require_action_workspace(session, self._owner, action_id, self._workspace)
        row = session.fetch_one(
            sql("SELECT * FROM ao_actions WHERE owner=? AND id=?"), (self._owner, action_id)
        )
        if row is None:
            raise AccessDenied("action unavailable")
        return action_from_row(row)

    def _actions(self, session: AlwaysOnSession, intent_id: str) -> tuple[ActionRecord, ...]:
        rows = session.fetch_all(
            sql("SELECT * FROM ao_actions WHERE owner=? AND intent_id=? ORDER BY id"),
            (self._owner, intent_id),
        )
        for row in rows:
            require_action_workspace(session, self._owner, str(row["id"]), self._workspace)
        return tuple(action_from_row(row) for row in rows)

    def _root_scopes(self, session: AlwaysOnSession, refs: tuple[EvidenceRef, ...]) -> set[str]:
        scopes: set[str] = set()
        for ref in refs:
            try:
                self._check_namespace(ref.namespace)
            except AccessDenied:
                continue
            try:
                fact = next(iter(session.load_facts(ref.namespace, fact_id=ref.fact_id)), None)
            except ValueError:
                continue  # The outer dependency loop first expands direct scopes.
            if fact is not None and ROOTS_KEY in fact.qualifiers:
                with suppress(TypeError, ValueError, KeyError):
                    scopes.update(
                        root.namespace
                        for root in (
                            *evidence_roots(ref.namespace, fact),
                            *evidence_dependencies(fact),
                        )
                        if root.namespace in (self._owner, "__shared__")
                    )
        return scopes

    def _dependencies(
        self, session: AlwaysOnSession, intent: Intent, actions: tuple[ActionRecord, ...]
    ) -> set[str]:
        scopes = {ref.namespace for ref in intent.basis}
        refs = list(intent.basis)
        for action in actions:
            scopes.update(ref.namespace for ref in action.basis)
            refs.extend(action.basis)
        scopes.update(self._root_scopes(session, tuple(refs)))
        from .experience import experience_scopes

        scopes.update(experience_scopes(session, self._owner, intent.id))
        if isinstance(intent.waiting_for, WaitForSlot):
            scopes.add(intent.waiting_for.namespace)
        elif isinstance(intent.waiting_for, WaitForChange):
            scopes.add(intent.waiting_for.ref.namespace)
        return scopes

    def _evidence_scopes(self, refs: tuple[EvidenceRef, ...]) -> tuple[str, ...]:
        scopes = {ref.namespace for ref in refs}
        with self._transaction(tuple(sorted(scopes)), write=False) as snapshot:
            scopes.update(self._root_scopes(snapshot, refs))
        return tuple(sorted(scopes))

    @contextmanager
    def _intent_transaction(
        self,
        intent_id: str,
        refs: tuple[EvidenceRef, ...] = (),
        *,
        extra_namespaces: tuple[str, ...] = (),
    ) -> Iterator[tuple[AlwaysOnSession, Intent, tuple[ActionRecord, ...]]]:
        namespaces = {self._owner, *extra_namespaces, *(ref.namespace for ref in refs)}
        for _attempt in range(3):
            with self._transaction(write=False) as snapshot:
                item = self._intent(snapshot, intent_id)
                namespaces.update(
                    self._dependencies(snapshot, item, self._actions(snapshot, intent_id))
                )
                namespaces.update(self._root_scopes(snapshot, refs))
            try:
                with self._transaction(tuple(sorted(namespaces))) as session:
                    item = self._intent(session, intent_id)
                    actions = self._actions(session, intent_id)
                    current = self._dependencies(session, item, actions) | self._root_scopes(
                        session, refs
                    )
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
        lineage_readable = lineage_current = True
        if is_derived(fact):
            lineage_readable, lineage_current = lineage_status(
                ref.namespace,
                fact,
                self._owner,
                self._evidence_grants(session),
                lambda namespace, fact_id: session.load_facts(namespace, fact_id=fact_id),
            )
            if freshness and lineage_readable and lineage_current:
                for root in {*evidence_roots(ref.namespace, fact), *evidence_dependencies(fact)}:
                    original = session.load_facts(root.namespace, fact_id=root.fact_id)[0]
                    policy = self._policy(session, root.namespace, original)
                    if policy is not None and transitions.freshness_blocks(policy, root, now):
                        lineage_current = False
                        break
        return transitions.EvidenceView(
            ref.namespace,
            ref.fact_id,
            evidence_revision(fact),
            active=fact.is_active(now),
            supported=supported and lineage_current,
            readable=lineage_readable,
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
        context_guard: TaskContextGuard | None = None,
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
        guard_scopes = self._task_guard_scopes(context_guard)
        with self._intent_transaction(intent_id, tuple(extra), extra_namespaces=guard_scopes) as (
            session,
            intent,
            actions,
        ):
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
            if context_guard is not None:
                self._check_task_guard(session, context_guard, now)
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

    def _snapshot_current(self, session: AlwaysOnSession, event: FactChange, now: datetime) -> bool:
        if event.after_revision is None:
            return False
        ref = EvidenceRef(event.namespace, event.fact_id, event.after_revision)
        return not self._validate_basis(session, (ref,), now, freshness=False)

    def resync(self, subscription_id: str, *, limit: int = 500) -> ChangeBatch:
        """Replace a pending delivery with a complete bounded working-set manifest.

        Persist the replacement cache and receipt before ack. No cursor advances
        here, and writes after the captured head remain in the delta journal.
        """
        nonempty(subscription_id, "subscription_id")
        page_limit(limit)
        with self._transaction(write=False) as session:
            namespace = str(self._subrow(session, subscription_id)["namespace"])
        self._subscription_namespace(namespace)
        scopes = (self._owner, "__shared__") if self._pool is not None else (self._owner,)
        with self._transaction(scopes) as session:
            row = self._subrow(session, subscription_id)
            if row["namespace"] != namespace:
                raise RevisionConflict("subscription changed")
            selector = self._subscription(row).selector
            stream = session.fetch_one(
                sql("SELECT head_seq FROM ao_streams WHERE namespace=?"), (namespace,)
            )
            assert stream is not None
            head = int(stream["head_seq"] or 0)
            now = self._now()
            events: list[FactChange] = []
            after: str | None = None
            while True:
                clause = " AND id>?" if after is not None else ""
                candidates = session.fetch_all(
                    SQLStatement(
                        "SELECT id,slot_key FROM facts WHERE agent_id=?"
                        + clause
                        + " ORDER BY id LIMIT 500",
                        'SELECT id,slot_key FROM "ai-knot_facts" WHERE agent_id=%s'
                        + clause.replace("?", "%s")
                        + " ORDER BY id LIMIT 500",
                    ),
                    (namespace, after) if after is not None else (namespace,),
                )
                for candidate in candidates:
                    fact_id = str(candidate["id"])
                    if (selector.fact_ids or selector.slot_keys) and (
                        fact_id not in selector.fact_ids
                        and candidate["slot_key"] not in selector.slot_keys
                    ):
                        continue
                    fact = session.load_facts(namespace, fact_id=fact_id)[0]
                    if not fact.supported or not fact.is_active(now):
                        continue
                    event = FactChange(
                        namespace,
                        head,
                        "snapshot",
                        fact.id,
                        fact.slot_key,
                        None,
                        evidence_revision(fact),
                    )
                    if not self._snapshot_current(session, event, now):
                        continue
                    events.append(event)
                    if len(events) > limit:
                        raise SnapshotTooLarge("Narrow the selected working set before recovery")
                if len(candidates) < 500:
                    break
                after = str(candidates[-1]["id"])
            token = uuid4().hex
            grants = self._grants(session, namespace)
            acl_digest = hashlib.sha256(encode_changes(sorted(grants)).encode()).hexdigest()
            session.execute(
                sql(
                    "UPDATE ao_subscriptions SET pending_since=NULL,inflight_token=?,"
                    "inflight_through_seq=?,inflight_events_json=?,inflight_resync=0,"
                    "inflight_acl_digest=?,inflight_snapshot=1 WHERE owner=? AND id=?"
                ),
                (
                    token,
                    head,
                    encode_changes([asdict(event) for event in events]),
                    acl_digest,
                    self._owner,
                    subscription_id,
                ),
            )
            return ChangeBatch(
                token, tuple(events), through_seq=head, head_seq=head, is_snapshot=True
            )

    def review_context(
        self, refs: tuple[EvidenceRef, ...], *, max_chars: int = 12000
    ) -> ContextReview:
        """Admit exact saved revisions within a whole-fact character budget.

        Replace cached model context with the accepted set. This read-only view
        is point-in-time evidence checking, not permission for a later effect.
        """
        positive_integer(max_chars, "max_chars", 65536)
        if len(refs) > 128 or any(not isinstance(ref, EvidenceRef) for ref in refs):
            raise ValueError("context requires at most 128 evidence references")
        transitions.union_basis(refs)
        scopes = (self._owner, "__shared__") if self._pool is not None else (self._owner,)
        with self._transaction(scopes, write=False) as session:
            from .reads import ReadCache

            session = ReadCache(session)
            now = self._now()
            accepted: list[ContextFact] = []
            issues: list[BasisIssue] = []
            omitted: list[EvidenceRef] = []
            remaining = max_chars
            for ref in dict.fromkeys(refs):
                problems = self._validate_basis(session, (ref,), now)
                if problems:
                    issues.extend(problems)
                    continue
                fact = session.load_facts(ref.namespace, fact_id=ref.fact_id)[0]
                if len(fact.content) > remaining:
                    omitted.append(ref)
                    continue
                accepted.append(ContextFact(ref, fact.content, fact.slot_key))
                remaining -= len(fact.content)
            return ContextReview(
                tuple(accepted), tuple(issues), tuple(omitted), now, not issues and not omitted
            )

    def independent_evidence(self, refs: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
        """Flatten copied observations into distinct current source revisions."""
        scopes = {ref.namespace for ref in refs}
        if self._pool is not None:
            scopes.add("__shared__")
        with self._transaction(tuple(sorted(scopes)), write=False) as session:
            roots: set[EvidenceRef] = set()
            for ref in refs:
                fact = self._fact_for_ref(session, ref, self._now())
                roots.update(evidence_roots(ref.namespace, fact))
            return canonical_observations(
                roots, lambda namespace, fact_id: session.load_facts(namespace, fact_id=fact_id)
            )

    def select_context(
        self,
        requirements: tuple[str, ...],
        *,
        task_key: str = "",
        role: str = "",
        max_facts: int = 8,
        max_chars: int = 12000,
        required_refs: tuple[EvidenceRef, ...] = (),
    ) -> TaskContext:
        """Select current task evidence within a whole-fact context budget."""
        from .context import select_context

        return select_context(
            self,
            requirements,
            task_key=task_key,
            role=role,
            max_facts=max_facts,
            max_chars=max_chars,
            required_refs=required_refs,
        )

    def task_snapshot(
        self,
        *,
        task_key: str,
        max_facts: int = 8,
        max_chars: int = 12000,
        include_derived: bool = True,
    ) -> tuple[ContextFact, ...]:
        """Read all current task evidence, or raise SnapshotTooLarge without a prefix."""
        from .context import task_snapshot

        return task_snapshot(
            self,
            task_key=task_key,
            max_facts=max_facts,
            max_chars=max_chars,
            include_derived=include_derived,
        )

    def task_guard(self, *, task_key: str, max_facts: int = 128) -> TaskContextGuard:
        """Capture the admissible universe before selection; not semantic completeness."""
        from .context import task_guard, task_scopes

        context_label(task_key, "task_key")
        positive_integer(max_facts, "max_facts", 128)
        with self._transaction(task_scopes(self), write=False) as session:
            return task_guard(
                self, session, task_key=task_key, max_facts=max_facts, now=self._now()
            )

    def _task_guard_scopes(self, guard: TaskContextGuard | None) -> tuple[str, ...]:
        from .context import task_scopes

        if guard is not None and not isinstance(guard, TaskContextGuard):
            raise ValueError("context_guard must be TaskContextGuard")
        return task_scopes(self) if guard is not None else ()

    def _check_task_guard(
        self, session: AlwaysOnSession, guard: TaskContextGuard, now: datetime
    ) -> None:
        from .context import task_guard

        try:
            current = task_guard(
                self, session, task_key=guard.task_key, max_facts=guard.max_facts, now=now
            )
        except SnapshotTooLarge:
            raise StaleEvidence("task context no longer matches") from None
        if current != guard:
            raise StaleEvidence("task context no longer matches")

    def derive_fact(
        self,
        content: str,
        *,
        basis: tuple[EvidenceRef, ...],
        namespace: str | None = None,
        visibility_scope: str = "global",
        memory_type: MemoryType = MemoryType.SEMANTIC,
        task_key: str = "",
        for_role: str = "",
    ) -> Fact:
        """Store a host-declared derivation with flattened, revision-bound roots.

        Shared derivation requires maintain authority and readable inputs. This
        creates a dependency record, not a new independent confirmation.
        """
        namespace = namespace or self._owner
        self._check_namespace(namespace)
        nonempty(content, "content")
        nonempty(visibility_scope, "visibility_scope")
        memory_type = MemoryType(memory_type)
        context_label(task_key, "task_key")
        context_label(for_role, "for_role")
        if not basis:
            raise ValueError("derivation requires evidence")
        scopes = {namespace, *(ref.namespace for ref in basis)}
        if self._pool is not None:
            scopes.add("__shared__")
        from .shared_maintenance import require_maintain

        with self._transaction(tuple(sorted(scopes))) as session:
            require_maintain(self, session, namespace)
            roots: set[EvidenceRef] = set()
            dependencies: set[EvidenceRef] = set()
            for ref in transitions.union_basis(basis):
                fact = self._fact_for_ref(session, ref, self._now())
                roots.update(evidence_roots(ref.namespace, fact))
                dependencies.update((ref, *evidence_dependencies(fact)))
            now = self._now()
            output = Fact(
                content,
                type=memory_type,
                created_at=now,
                last_accessed=now,
                valid_from=now,
                origin_agent_id=self._owner,
                visibility_scope=visibility_scope,
                visibility="shared" if namespace == "__shared__" else "private",
                memory_tier="pool" if namespace == "__shared__" else "private",
                qualifiers={
                    ROOTS_KEY: encode_roots(tuple(roots)),
                    DEPENDENCIES_KEY: encode_roots(tuple(dependencies)),
                    **({"task_key": task_key} if task_key else {}),
                    **({"for_role": for_role} if for_role else {}),
                },
            )
            session.replace_facts(namespace, [*session.load_facts(namespace), output])
            return output

    def record_experience(
        self,
        action_id: str,
        *,
        expected_action_revision: int,
        experience_key: str,
        content: str,
        basis: tuple[EvidenceRef, ...],
        namespace: str | None = None,
        visibility_scope: str = "global",
        task_key: str = "",
        for_role: str = "",
    ) -> RecordedExperience:
        """Record a host-authored lesson bound to this action's current outcome."""
        from .experience import record_experience

        return record_experience(
            self,
            action_id,
            expected_action_revision=expected_action_revision,
            experience_key=experience_key,
            content=content,
            basis=basis,
            namespace=namespace,
            visibility_scope=visibility_scope,
            task_key=task_key,
            for_role=for_role,
        )

    def prepare_action(
        self,
        intent_id: str,
        *,
        lease_token: str,
        action_key: str,
        request: JSONObject,
        basis: tuple[EvidenceRef, ...] = (),
        operation_key: str | None = None,
        resource_key: str | None = None,
    ) -> ActionRecord:
        if (operation_key is None) != (resource_key is None):
            raise ValueError("operation_key and resource_key are required together")
        if operation_key is not None:
            nonempty(operation_key, "operation_key")
            assert resource_key is not None
            nonempty(resource_key, "resource_key")
            if self._workspace is None:
                raise ValueError("workspace required for shared operations")
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
            if old is not None:
                require_action_workspace(session, self._owner, old.id, self._workspace)
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
            if operation_key is not None:
                assert self._workspace is not None and resource_key is not None
                current = self._workspace.reserve(
                    session, self._owner, current, operation_key, resource_key
                )
            elif self._workspace is not None:
                bound = self._workspace.check(session, self._owner, current)
                if bound is not None:
                    raise ValueError("shared action requires operation_key and resource_key")
            save_action(session, self._owner, old, current)
            return current

    def _action_intent_id(self, action_id: str) -> str:
        with self._transaction(write=False) as session:
            return self._action(session, action_id).intent_id

    def begin_action(
        self,
        action_id: str,
        *,
        lease_token: str,
        context_guard: TaskContextGuard | None = None,
    ) -> ActionPermit:
        guard_scopes = self._task_guard_scopes(context_guard)
        intent_id = self._action_intent_id(action_id)
        error = None
        with self._intent_transaction(intent_id, extra_namespaces=guard_scopes) as (
            session,
            intent,
            actions,
        ):
            action = self._action(session, action_id)
            if self._workspace is not None:
                self._workspace.check(session, self._owner, action)
            now = self._now()
            transitions.require_lease(intent, lease_token, now)
            refs = transitions.union_basis(intent.basis, action.basis)
            issues = self._validate_basis(session, refs, now)
            if context_guard is not None and not issues:
                try:
                    self._check_task_guard(session, context_guard, now)
                except StaleEvidence:
                    issues = (BasisIssue("changed", None),)
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
                if self._workspace is not None:
                    permit = self._workspace.update(session, self._owner, started_action) or permit
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
            if self._workspace is not None:
                self._workspace.update(session, self._owner, current)
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
            if self._workspace is not None:
                self._workspace.update(session, self._owner, action)
            save_action(session, self._owner, old, action)
            return action

    def set_freshness(
        self, ref: EvidenceRef, *, source_ref: str, recheck_after_seconds: int
    ) -> FreshnessRecord:
        nonempty(source_ref, "source_ref")
        positive_integer(recheck_after_seconds, "recheck_after_seconds")
        from .shared_maintenance import locks, require_maintain

        with self._transaction(
            (*locks(self, ref.namespace), *self._evidence_scopes((ref,)))
        ) as session:
            require_maintain(self, session, ref.namespace)
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

    def due_verifications(
        self, *, limit: int = 100, include_shared: bool = True
    ) -> list[VerificationRequest]:
        page_limit(limit)
        if type(include_shared) is not bool:
            raise ValueError("include_shared must be a boolean")
        namespaces = (
            (self._owner, "__shared__")
            if include_shared and self._pool is not None and self._pool._persist_stats
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
        from .shared_maintenance import locks, require_maintain

        with self._transaction(
            (*locks(self, ref.namespace), *self._evidence_scopes((ref,)))
        ) as session:
            require_maintain(self, session, ref.namespace)
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
