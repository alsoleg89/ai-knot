"""SQL statements and lossless codecs for durable work state."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from typing import Any, Literal, cast

from .models import (
    ActionRecord,
    ActionState,
    EvidenceRef,
    FreshnessRecord,
    Intent,
    IntentState,
    JSONObject,
    Lease,
    VerificationOutcome,
    WaitCondition,
    WaitForChange,
    WaitForSlot,
    WaitUntil,
)
from .storage import AlwaysOnSession, SQLRow, SQLScalar, SQLStatement, migrate_freshness, sql

_DDL = (
    """
CREATE TABLE IF NOT EXISTS ao_intents ( owner TEXT NOT NULL,  id TEXT NOT NULL,
client_key TEXT NOT NULL,  create_digest TEXT NOT NULL,  goal TEXT NOT NULL,  state
TEXT NOT NULL,  progress_json TEXT NOT NULL,  next_step TEXT NOT NULL,  waiting_json
TEXT,  revision INTEGER NOT NULL CHECK(revision>0),  basis_json TEXT NOT NULL,
lease_token TEXT,  lease_generation INTEGER NOT NULL DEFAULT 0,  lease_worker TEXT,
lease_expires_at TEXT,  deadline TEXT,  blocked_reasons_json TEXT NOT NULL,
last_transition_at TEXT NOT NULL,  PRIMARY KEY(owner,  id),  UNIQUE(owner,
client_key),  CHECK(state IN ('ready',  'running',  'waiting',  'blocked',
'completed',  'cancelled')),  CHECK((state='waiting' AND waiting_json IS NOT NULL)
OR (state<>'waiting' AND waiting_json IS NULL)),  CHECK((lease_token IS NULL AND
lease_worker IS NULL AND lease_expires_at IS NULL) OR (lease_token IS NOT NULL AND
lease_worker IS NOT NULL AND lease_expires_at IS NOT NULL)))
""",
    "CREATE INDEX IF NOT EXISTS ao_intents_owner_state ON ao_intents(owner,state,id)",
    """
CREATE TABLE IF NOT EXISTS ao_actions ( owner TEXT NOT NULL,  id TEXT NOT NULL,
intent_id TEXT NOT NULL,  action_key TEXT NOT NULL,  request_json TEXT NOT NULL,
request_digest TEXT NOT NULL,  basis_json TEXT NOT NULL,  state TEXT NOT NULL,
revision INTEGER NOT NULL CHECK(revision>0),  idempotency_key TEXT NOT NULL,
result_json TEXT,  error_json TEXT,  reconciliation_evidence_json TEXT,  created_at
TEXT NOT NULL,  updated_at TEXT NOT NULL,  PRIMARY KEY(owner,  id),  UNIQUE(owner,
intent_id,  action_key),  CHECK(state IN ('prepared',  'started',  'succeeded',
'failed',  'unknown')))
""",
    "CREATE INDEX IF NOT EXISTS ao_actions_intent_state ON ao_actions(owner,intent_id,state,id)",
    """
CREATE TABLE IF NOT EXISTS ao_freshness ( namespace TEXT NOT NULL,  subject_kind
TEXT NOT NULL,  subject_key TEXT NOT NULL,  bound_fact_id TEXT NOT NULL,
bound_revision TEXT NOT NULL,  source_ref TEXT NOT NULL,  interval_seconds INTEGER
NOT NULL CHECK(interval_seconds>0),  checked_at TEXT,  verified_until TEXT NOT NULL,
retry_at TEXT NOT NULL,  last_outcome TEXT,  failure_count INTEGER NOT NULL
CHECK(failure_count>=0),  evidence_json TEXT,  source_version TEXT NOT NULL,
revision BIGINT NOT NULL DEFAULT 1, dirty INTEGER NOT NULL DEFAULT 1,
PRIMARY KEY(namespace,  subject_kind,  subject_key),  CHECK(subject_kind IN ('slot',
'fact')),  CHECK(last_outcome IS NULL OR last_outcome IN ('confirmed',  'changed',
'unavailable')))
""",
    "CREATE INDEX IF NOT EXISTS ao_freshness_retry "
    "ON ao_freshness(retry_at,namespace,bound_fact_id)",
)


def initialize(session: AlwaysOnSession) -> None:
    for statement in _DDL:
        pair = sql(statement)
        pg = pair.postgres
        for field in (
            "lease_expires_at",
            "deadline",
            "last_transition_at",
            "created_at",
            "updated_at",
            "checked_at",
            "verified_until",
            "retry_at",
        ):
            pg = re.sub(r"\b" + field + r"\s+TEXT\b", field + " TIMESTAMPTZ", pg)
        pg = re.sub(r"\b(revision|lease_generation)\s+INTEGER\b", r"\1 BIGINT", pg)
        session.execute(SQLStatement(pair.sqlite, pg))
    migrate_freshness(session)
    session.execute(
        sql(
            "CREATE INDEX IF NOT EXISTS ao_freshness_due "
            "ON ao_freshness(namespace,dirty,retry_at,subject_kind,subject_key)"
        )
    )
    session.execute(
        SQLStatement(
            "CREATE INDEX IF NOT EXISTS ao_fact_subject ON facts(agent_id,slot_key)",
            'CREATE INDEX IF NOT EXISTS "ai-knot_ao_fact_subject" '
            'ON "ai-knot_facts"(agent_id,slot_key)',
        )
    )
    session.execute(sql("UPDATE ao_schema SET version=5 WHERE version<5"))


def timestamp(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat(timespec="microseconds")


def dt(value: SQLScalar) -> datetime:
    return datetime.fromisoformat(str(value))


def dumps(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def refs_json(refs: tuple[EvidenceRef, ...]) -> str:
    return dumps([asdict(ref) for ref in refs])


def parse_refs(value: SQLScalar) -> tuple[EvidenceRef, ...]:
    return tuple(EvidenceRef(**item) for item in json.loads(str(value)))


def wait_json(wait: WaitCondition | None) -> str | None:
    if wait is None:
        return None
    if isinstance(wait, WaitUntil):
        return dumps({"kind": "until", "at": timestamp(wait.at)})
    if isinstance(wait, WaitForSlot):
        return dumps(
            {
                "kind": "slot",
                "namespace": wait.namespace,
                "slot_key": wait.slot_key,
                "value_text": wait.value_text,
            }
        )
    return dumps({"kind": "change", "ref": asdict(wait.ref)})


def parse_wait(value: SQLScalar) -> WaitCondition | None:
    if value is None:
        return None
    item = json.loads(str(value))
    if item["kind"] == "until":
        return WaitUntil(datetime.fromisoformat(item["at"]))
    if item["kind"] == "slot":
        return WaitForSlot(item["namespace"], item["slot_key"], item["value_text"])
    return WaitForChange(EvidenceRef(**item["ref"]))


def intent_from_row(row: SQLRow) -> Intent:
    lease = (
        None
        if row["lease_token"] is None
        else Lease(
            str(row["lease_token"]),
            int(cast(int, row["lease_generation"])),
            str(row["lease_worker"]),
            dt(row["lease_expires_at"]),
        )
    )
    return Intent(
        str(row["id"]),
        str(row["owner"]),
        str(row["client_key"]),
        str(row["goal"]),
        IntentState(str(row["state"])),
        json.loads(str(row["progress_json"])),
        str(row["next_step"]),
        parse_wait(row["waiting_json"]),
        int(cast(int, row["revision"])),
        parse_refs(row["basis_json"]),
        lease,
        None if row["deadline"] is None else dt(row["deadline"]),
        dt(row["last_transition_at"]),
        int(cast(int, row["lease_generation"])),
        tuple(json.loads(str(row["blocked_reasons_json"]))),
    )


def insert_intent(session: AlwaysOnSession, intent: Intent, create_digest: str) -> None:
    session.execute(
        sql(
            """
INSERT INTO ao_intents(owner,  id,  client_key,  create_digest,  goal,  state,
progress_json,  next_step,  waiting_json,  revision,  basis_json,  lease_token,
lease_generation,  lease_worker,  lease_expires_at,  deadline,
blocked_reasons_json,  last_transition_at) VALUES (?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,
?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?)
"""
        ),
        (
            intent.owner_agent_id,
            intent.id,
            intent.client_key,
            create_digest,
            intent.goal,
            intent.state.value,
            dumps(intent.progress),
            intent.next_step,
            wait_json(intent.waiting_for),
            intent.revision,
            refs_json(intent.basis),
            None,
            0,
            None,
            None,
            timestamp(intent.deadline),
            dumps(intent.blocked_reasons),
            timestamp(intent.last_transition_at),
        ),
    )


def save_intent(session: AlwaysOnSession, previous: Intent, current: Intent) -> None:
    if previous == current:
        return
    lease = current.lease
    count = session.execute(
        sql(
            """
UPDATE ao_intents SET state=?,  progress_json=?,  next_step=?,  waiting_json=?,
revision=?,  basis_json=?,  lease_token=?,  lease_generation=?,  lease_worker=?,
lease_expires_at=?,  blocked_reasons_json=?,  last_transition_at=? WHERE owner=? AND
id=? AND revision=?
"""
        ),
        (
            current.state.value,
            dumps(current.progress),
            current.next_step,
            wait_json(current.waiting_for),
            current.revision,
            refs_json(current.basis),
            None if lease is None else lease.token,
            current.lease_generation,
            None if lease is None else lease.worker_id,
            None if lease is None else timestamp(lease.expires_at),
            dumps(current.blocked_reasons),
            timestamp(current.last_transition_at),
            current.owner_agent_id,
            current.id,
            previous.revision,
        ),
    )
    if count != 1:
        from .models import RevisionConflict

        raise RevisionConflict("intent revision changed")


def action_from_row(row: SQLRow) -> ActionRecord:
    def obj(key: str) -> JSONObject | None:
        return None if row[key] is None else cast(JSONObject, json.loads(str(row[key])))

    return ActionRecord(
        str(row["id"]),
        str(row["intent_id"]),
        str(row["action_key"]),
        cast(JSONObject, obj("request_json")),
        str(row["request_digest"]),
        parse_refs(row["basis_json"]),
        ActionState(str(row["state"])),
        str(row["idempotency_key"]),
        obj("result_json"),
        obj("error_json"),
        dt(row["created_at"]),
        dt(row["updated_at"]),
        obj("reconciliation_evidence_json"),
        int(cast(int, row["revision"])),
    )


def save_action(
    session: AlwaysOnSession, owner: str, previous: ActionRecord | None, current: ActionRecord
) -> None:
    values: tuple[SQLScalar, ...] = (
        refs_json(current.basis),
        current.state.value,
        current.revision,
        current.idempotency_key,
        None if current.result is None else dumps(current.result),
        None if current.error is None else dumps(current.error),
        None if current.reconciliation_evidence is None else dumps(current.reconciliation_evidence),
        timestamp(current.created_at),
        timestamp(current.updated_at),
    )
    if previous is None:
        session.execute(
            sql(
                """
INSERT INTO ao_actions(owner,  id,  intent_id,  action_key,  request_json,
request_digest,  basis_json,  state,  revision,  idempotency_key,  result_json,
error_json,  reconciliation_evidence_json,  created_at,  updated_at) VALUES (?,  ?,
?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?)
"""
            ),
            (
                owner,
                current.id,
                current.intent_id,
                current.action_key,
                dumps(current.request),
                current.request_digest,
                *values,
            ),
        )
    elif previous != current:
        count = session.execute(
            sql(
                """
UPDATE ao_actions SET basis_json=?,  state=?,  revision=?,  idempotency_key=?,
result_json=?,  error_json=?,  reconciliation_evidence_json=?,  created_at=?,
updated_at=? WHERE owner=? AND id=? AND revision=?
"""
            ),
            (*values, owner, current.id, previous.revision),
        )
        if count != 1:
            from .models import RevisionConflict

            raise RevisionConflict("action revision changed")


def freshness_from_row(row: SQLRow) -> FreshnessRecord:
    kind = str(row["subject_kind"])
    assert kind in ("slot", "fact")
    return FreshnessRecord(
        EvidenceRef(str(row["namespace"]), str(row["bound_fact_id"]), str(row["bound_revision"])),
        cast(Literal["slot", "fact"], kind),
        str(row["subject_key"]),
        str(row["source_ref"]),
        int(cast(int, row["interval_seconds"])),
        None if row["checked_at"] is None else dt(row["checked_at"]),
        dt(row["verified_until"]),
        dt(row["retry_at"]),
        None if row["last_outcome"] is None else VerificationOutcome(str(row["last_outcome"])),
        int(cast(int, row["failure_count"])),
        None if row["evidence_json"] is None else json.loads(str(row["evidence_json"])),
        str(row["source_version"]),
    )


def save_freshness(session: AlwaysOnSession, value: FreshnessRecord) -> None:
    session.execute(
        sql(
            """
INSERT INTO ao_freshness(namespace,  subject_kind,  subject_key,  bound_fact_id,
bound_revision,  source_ref,  interval_seconds,  checked_at,  verified_until,
retry_at,  last_outcome,  failure_count,  evidence_json,  source_version, dirty) VALUES (?,
?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?, 0) ON CONFLICT(namespace,
subject_kind,  subject_key) DO UPDATE SET bound_fact_id=excluded.bound_fact_id,
bound_revision=excluded.bound_revision,  source_ref=excluded.source_ref,
interval_seconds=excluded.interval_seconds,  checked_at=excluded.checked_at,
verified_until=excluded.verified_until,  retry_at=excluded.retry_at,
last_outcome=excluded.last_outcome,  failure_count=excluded.failure_count,
evidence_json=excluded.evidence_json,  source_version=excluded.source_version,
revision=ao_freshness.revision+1, dirty=0
"""
        ),
        (
            value.ref.namespace,
            value.subject_kind,
            value.subject_key,
            value.ref.fact_id,
            value.ref.revision,
            value.source_ref,
            value.recheck_after_seconds,
            timestamp(value.checked_at),
            timestamp(value.verified_until),
            timestamp(value.retry_at),
            None if value.last_outcome is None else value.last_outcome.value,
            value.failure_count,
            None if value.evidence is None else dumps(value.evidence),
            value.source_version,
        ),
    )


def due_policy_rows(
    session: AlwaysOnSession,
    namespaces: tuple[str, ...],
    owner: str,
    now: datetime,
    *,
    limit: int,
    offset: int,
) -> list[SQLRow]:
    """Select current eligible subjects using indexes, before decoding Fact objects."""

    def statement(postgres: bool) -> tuple[str, tuple[SQLScalar, ...]]:
        params: list[SQLScalar] = []
        mark = "%s" if postgres else "?"
        facts = '"ai-knot_facts"' if postgres else "facts"
        policy = '"ai-knot_ao_freshness"' if postgres else "ao_freshness"
        acl = '"ai-knot_acl_grants"' if postgres else "acl_grants"
        now_text = timestamp(now)

        def bind(value: SQLScalar) -> str:
            params.append(value)
            return mark

        def instant(value: str) -> str:
            return f"CAST({value} AS TIMESTAMPTZ)" if postgres else f"ai_knot_utc({value})"

        def active(alias: str) -> str:
            def moment(column: str) -> str:
                value = f"COALESCE(NULLIF({alias}.{column},''),{alias}.created_at)"
                return instant(value)

            lower, upper = bind(now_text), bind(now_text)
            if postgres:
                lower, upper = f"CAST({lower} AS TIMESTAMPTZ)", f"CAST({upper} AS TIMESTAMPTZ)"
            return (
                f"({moment('valid_from')}<={lower} AND "
                f"({alias}.valid_until IS NULL OR {moment('valid_until')}>{upper}))"
            )

        def readable(alias: str) -> str:
            return (
                f"({alias}.agent_id<>'__shared__' OR "
                f"COALESCE(NULLIF({alias}.visibility_scope,''),'global')='global' OR "
                f"{alias}.origin_agent_id={bind(owner)} OR EXISTS "
                f"(SELECT 1 FROM {acl} grants WHERE grants.agent_id={bind(owner)} "
                f"AND grants.scope={alias}.visibility_scope))"
            )

        # Bind parameters in textual SQL order.
        effective = (
            f"CASE WHEN p.dirty=1 OR f.id<>p.bound_fact_id "
            f"THEN {bind(now_text)} ELSE p.retry_at END"
        )
        scopes = ",".join(bind(namespace) for namespace in namespaces)
        retry_now = bind(now_text)
        is_active = active("f")
        can_read = readable("f")
        newer_active = active("newer")
        newer_readable = readable("newer")
        peer_active = active("peer")
        peer_readable = readable("peer")
        newer_created, current_created = instant("newer.created_at"), instant("f.created_at")
        query = f"""SELECT p.*, f.id AS current_fact_id, {effective} AS effective_due
FROM {policy} p JOIN {facts} f ON f.agent_id=p.namespace
AND ((p.subject_kind='fact' AND f.id=p.subject_key)
 OR (p.subject_kind='slot' AND f.slot_key=p.subject_key))
WHERE p.namespace IN ({scopes}) AND (p.dirty=1 OR p.retry_at<={retry_now} OR f.id<>p.bound_fact_id)
AND f.supported=1 AND {is_active} AND {can_read}
AND (p.subject_kind='fact' OR (
 NOT EXISTS (SELECT 1 FROM {facts} newer WHERE newer.agent_id=f.agent_id
  AND newer.slot_key=f.slot_key AND {newer_active} AND {newer_readable}
  AND (newer.version>f.version OR (newer.version=f.version
       AND ({newer_created}>{current_created}
         OR ({newer_created}={current_created} AND newer.id>f.id)))))
 AND NOT EXISTS (SELECT 1 FROM {facts} peer WHERE peer.agent_id=f.agent_id
  AND peer.slot_key=f.slot_key AND peer.supported=1 AND {peer_active} AND {peer_readable}
  AND COALESCE(NULLIF(peer.value_text,''),peer.content)
      <>COALESCE(NULLIF(f.value_text,''),f.content))))
ORDER BY effective_due,p.namespace,f.id,p.subject_kind,p.subject_key
LIMIT {bind(limit)} OFFSET {bind(offset)}"""
        return query, tuple(params)

    sqlite, params = statement(False)
    postgres, _ = statement(True)
    return session.fetch_all(SQLStatement(sqlite, postgres), params)
