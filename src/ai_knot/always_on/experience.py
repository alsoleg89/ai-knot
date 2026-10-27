"""Explicit host-authored lessons with durable current-action dependencies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Literal

from ai_knot.types import Fact, MemoryType

from .changes import evidence_revision
from .models import (
    ActionRecord,
    ActionState,
    ContextFact,
    EvidenceRef,
    IdempotencyConflict,
    RecordedExperience,
    RevisionConflict,
    StaleEvidence,
    context_label,
    digest,
    nonempty,
    positive_integer,
)
from .storage import AlwaysOnSession, sql

if TYPE_CHECKING:
    from .memory import AlwaysOnMemory


def experience_scopes(session: AlwaysOnSession, owner: str, intent_id: str) -> set[str]:
    return {
        str(row["namespace"])
        for row in session.fetch_all(
            sql("""SELECT DISTINCT e.namespace FROM ao_experiences e
            JOIN ao_actions a ON a.owner=e.owner AND a.id=e.action_id
            WHERE a.owner=? AND a.intent_id=?"""),
            (owner, intent_id),
        )
    }


def guard_experience_writes(
    session: AlwaysOnSession, namespace: str, facts: list[Fact]
) -> list[Fact]:
    """Fact restore/import cannot roll durable action outcomes back."""
    bound = {f.id for f in facts if "_ai_knot_action_binding" in f.qualifiers}
    if not bound:
        return facts
    schema = session.fetch_one(sql("SELECT version FROM ao_schema WHERE id=1"))
    valid: set[str] = set()
    if schema is not None and int(schema["version"] or 0) >= 8:
        ids = sorted(bound)
        for start in range(0, len(ids), 128):
            batch = ids[start : start + 128]
            placeholders = ",".join("?" for _ in batch)
            rows = session.fetch_all(
                sql(
                    "SELECT e.fact_id FROM ao_experiences e JOIN ao_actions a "
                    "ON a.owner=e.owner AND a.id=e.action_id WHERE e.namespace=? "
                    "AND a.revision=e.action_revision AND e.fact_id IN (" + placeholders + ")"
                ),
                (namespace, *batch),
            )
            valid.update(str(row["fact_id"]) for row in rows)
    return [
        replace(f, supported=False) if f.id in bound and f.id not in valid else f for f in facts
    ]


def invalidate_experiences(session: AlwaysOnSession, owner: str, action: ActionRecord) -> None:
    rows = session.fetch_all(
        sql("""SELECT namespace,fact_id FROM ao_experiences
        WHERE owner=? AND action_id=? AND action_revision<>?"""),
        (owner, action.id, action.revision),
    )
    by_scope: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_scope[str(row["namespace"])].add(str(row["fact_id"]))
    for namespace, ids in by_scope.items():
        facts = session.load_facts(namespace)
        if any(f.id in ids and f.supported for f in facts):
            session.replace_facts(
                namespace, [replace(f, supported=False) if f.id in ids else f for f in facts]
            )


def record_experience(
    memory: AlwaysOnMemory,
    action_id: str,
    *,
    expected_action_revision: int,
    experience_key: str,
    content: str,
    basis: tuple[EvidenceRef, ...],
    namespace: str | None,
    visibility_scope: str,
    task_key: str,
    for_role: str,
) -> RecordedExperience:
    positive_integer(expected_action_revision, "expected_action_revision")
    nonempty(experience_key, "experience_key")
    context_label(experience_key, "experience_key")
    nonempty(content, "content")
    nonempty(visibility_scope, "visibility_scope")
    context_label(task_key, "task_key")
    context_label(for_role, "for_role")
    namespace = memory._owner if namespace is None else namespace
    memory._check_namespace(namespace)
    if (
        not isinstance(basis, (tuple, list))
        or not 1 <= len(basis) <= 128
        or any(not isinstance(ref, EvidenceRef) for ref in basis)
    ):
        raise ValueError("experience requires 1..128 evidence references")
    refs = tuple(sorted(set(basis)))
    request_digest = digest(
        {
            "content": content,
            "basis": [asdict(ref) for ref in refs],
            "action_revision": expected_action_revision,
            "visibility_scope": visibility_scope,
            "task_key": task_key,
            "for_role": for_role,
        }
    )
    from .shared_maintenance import require_maintain

    intent_id = memory._action_intent_id(action_id)
    with memory._intent_transaction(intent_id, refs, extra_namespaces=(namespace,)) as (
        session,
        _,
        _,
    ):
        require_maintain(memory, session, namespace)
        action = memory._action(session, action_id)
        if memory._workspace is not None:
            memory._workspace.check(session, memory._owner, action)
        if action.revision != expected_action_revision:
            raise RevisionConflict("action revision changed")
        if action.state not in (ActionState.SUCCEEDED, ActionState.FAILED) or not action.result:
            raise RevisionConflict("recorded success/failure with nonempty result required")
        outcome: Literal["succeeded", "failed"] = (
            "succeeded" if action.state == ActionState.SUCCEEDED else "failed"
        )
        now = memory._now()
        if memory._validate_basis(session, refs, now):
            raise StaleEvidence("experience evidence is not current")
        row = session.fetch_one(
            sql("""SELECT * FROM ao_experiences WHERE owner=? AND action_id=?
            AND namespace=? AND experience_key=?"""),
            (memory._owner, action_id, namespace, experience_key),
        )
        if row is not None:
            if row["input_digest"] != request_digest:
                raise IdempotencyConflict("experience key reused with different input")
            ref = EvidenceRef(namespace, str(row["fact_id"]), str(row["fact_revision"]))
            lesson = memory._fact_for_ref(session, ref, now)
        else:
            sources = [memory._fact_for_ref(session, ref, now) for ref in refs]
            # Sources can span namespaces; flatten their original roots explicitly.
            from .lineage import (
                DEPENDENCIES_KEY,
                ROOTS_KEY,
                encode_roots,
                evidence_dependencies,
                evidence_roots,
            )

            roots = {
                r
                for ref, fact in zip(refs, sources, strict=True)
                for r in evidence_roots(ref.namespace, fact)
            }
            dependencies = {
                r
                for ref, fact in zip(refs, sources, strict=True)
                for r in (ref, *evidence_dependencies(fact))
            }
            lesson = Fact(
                f"Recorded outcome: {outcome}.\n{content}",
                id="experience-" + digest([memory._owner, action_id, namespace, experience_key]),
                type=MemoryType.PROCEDURAL,
                created_at=now,
                last_accessed=now,
                valid_from=now,
                origin_agent_id=memory._owner,
                visibility_scope=visibility_scope,
                visibility="shared" if namespace == "__shared__" else "private",
                memory_tier="pool" if namespace == "__shared__" else "private",
                qualifiers={
                    ROOTS_KEY: encode_roots(tuple(roots)),
                    DEPENDENCIES_KEY: encode_roots(tuple(dependencies)),
                    "_ai_knot_action_binding": digest([memory._owner, action_id, action.revision]),
                    **({"task_key": task_key} if task_key else {}),
                    **({"for_role": for_role} if for_role else {}),
                },
            )
            ref = EvidenceRef(namespace, lesson.id, evidence_revision(lesson))
            session.execute(
                sql("""INSERT INTO ao_experiences
                (owner,action_id,namespace,experience_key,fact_id,action_revision,input_digest,fact_revision)
                VALUES(?,?,?,?,?,?,?,?)"""),
                (
                    memory._owner,
                    action_id,
                    namespace,
                    experience_key,
                    lesson.id,
                    action.revision,
                    request_digest,
                    ref.revision,
                ),
            )
            session.replace_facts(namespace, [*session.load_facts(namespace), lesson])
        return RecordedExperience(
            ContextFact(ref, lesson.content, lesson.slot_key), outcome, action.revision
        )
