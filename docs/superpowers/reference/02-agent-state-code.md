# P2: точный кодовый контракт состояния агента

Статус: архивный проект алгоритмов; production реализация и проверки описаны в [implementation-final.md](../implementation-final.md). [Спецификация](../specs/2026-10-26-always-on-memory-design.md) задаёт публичный API; [план P2](../plans/2026-10-26-always-on-02-agent-state.md) — порядок внедрения. [always_on_state_reference.py](always_on_state_reference.py) — stdlib-only эталон чистых переходов. Его unittest проверяет переданные значения и не доказывает SQL durability, конкурентность, ACL или отсутствие внешних дублей.

## Точные типы и размещение

В src/ai_knot/always_on/models.py перенести enum/dataclass declarations из эталона: IntentState, ActionState, VerificationOutcome, EvidenceRef, три wait-типа, Lease, Intent, ActionRecord, ActionPermit, FreshnessRecord, VerificationRequest, VerificationResult, BasisIssue, ResumePacket. JSONValue recursive, JSONObject=dict[str,JSONValue]. EvidenceView и TypedDict changes — внутренние детали memory.py, не публичные facade APIs. Эталон не импортируется production-кодом.

| Тип | Все поля, в порядке declaration эталона |
|---|---|
| Intent | id, owner_agent_id, client_key, goal, state, progress, next_step, waiting_for, revision, basis, lease, deadline, last_transition_at, lease_generation, blocked_reasons |
| Lease | token:str, generation:int, worker_id:str, expires_at:datetime |
| ActionRecord | id, intent_id, action_key, request, request_digest, basis, state, idempotency_key, result, error, created_at, updated_at, reconciliation_evidence, revision |
| FreshnessRecord | ref, subject_kind:Literal['slot','fact'], subject_key, source_ref, recheck_after_seconds, checked_at, verified_until, retry_at, last_outcome, failure_count, evidence, source_version |
| BasisIssue | reason:str, ref:EvidenceRef\|None; access_changed имеет ref=None |
| ActionPermit | action_id, idempotency_key; выдаётся после COMMIT started |

Конкретные annotations/defaults зафиксированы в исполняемом файле. Неизвестные enum, пустые/non-string JSON keys, NaN/Infinity, неподдерживаемые Python objects, naive/non-UTC datetime и payload >65536 UTF-8 bytes на отдельное caller value отвергаются до SQL; response batch целиком этим размером не ограничен. Canonical JSON: json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False). JSON fields копируются, чтобы caller не менял saved value через исходный dict. Токены secrets.token_urlsafe(32), IDs uuid4().hex. Revision начинается с 1. Идемпотентный no-op не меняет revision/timestamps. Renew меняет только expires_at; generation отдельно переживает очистку lease.

## Схема и SQL CAS

P1 предоставляет AlwaysOnStorageCapable.always_on_transaction(namespaces,write=...) и session: load_facts(namespace), replace_facts(namespace,facts) с journal hook, fetch_one/fetch_all/execute(SQLStatement(sqlite,postgres), params). Rows — dict. P2 не добавляет session.save_intent/query. Row codec и SQLStatement constants размещаются в always_on/storage.py, orchestration — в memory.py.

Ниже SQLite DDL. PostgreSQL использует "ai-knot_ao_*", BIGINT для revision/generation и TIMESTAMPTZ вместо TEXT times; placeholder %s вместо ?. JSON остаётся TEXT. SQLite times — fixed UTC ISO microseconds, например 2026-10-26T00:00:00.000000+00:00: SQL сравнение лексикографическое. Внутри пакета owner берётся из существующего `kb._agent_id`, store — из `kb._storage`; публичных свойств `kb.agent_id`/`kb.storage` в текущей базе нет. Host example использует только facade и публичные KB методы.

~~~sql
CREATE TABLE ao_intents (
 owner TEXT NOT NULL, id TEXT NOT NULL, client_key TEXT NOT NULL,
 create_digest TEXT NOT NULL, goal TEXT NOT NULL, state TEXT NOT NULL,
 progress_json TEXT NOT NULL, next_step TEXT NOT NULL, waiting_json TEXT,
 revision INTEGER NOT NULL CHECK(revision>0), basis_json TEXT NOT NULL,
 lease_token TEXT, lease_generation INTEGER NOT NULL DEFAULT 0,
 lease_worker TEXT, lease_expires_at TEXT, deadline TEXT,
 blocked_reasons_json TEXT NOT NULL, last_transition_at TEXT NOT NULL,
 PRIMARY KEY(owner,id), UNIQUE(owner,client_key),
 CHECK(state IN ('ready','running','waiting','blocked','completed','cancelled')),
 CHECK((state='waiting' AND waiting_json IS NOT NULL)
    OR (state<>'waiting' AND waiting_json IS NULL)),
 CHECK((lease_token IS NULL AND lease_worker IS NULL AND lease_expires_at IS NULL)
    OR (lease_token IS NOT NULL AND lease_worker IS NOT NULL AND lease_expires_at IS NOT NULL))
);
CREATE INDEX ao_intents_owner_state ON ao_intents(owner,state,id);
CREATE TABLE ao_actions (
 owner TEXT NOT NULL, id TEXT NOT NULL, intent_id TEXT NOT NULL,
 action_key TEXT NOT NULL, request_json TEXT NOT NULL, request_digest TEXT NOT NULL,
 basis_json TEXT NOT NULL, state TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>0),
 idempotency_key TEXT NOT NULL, result_json TEXT, error_json TEXT,
 reconciliation_evidence_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY(owner,id), UNIQUE(owner,intent_id,action_key),
 CHECK(state IN ('prepared','started','succeeded','failed','unknown'))
);
CREATE INDEX ao_actions_intent_state ON ao_actions(owner,intent_id,state,id);
CREATE TABLE ao_freshness (
 namespace TEXT NOT NULL, subject_kind TEXT NOT NULL, subject_key TEXT NOT NULL,
 bound_fact_id TEXT NOT NULL, bound_revision TEXT NOT NULL, source_ref TEXT NOT NULL,
 interval_seconds INTEGER NOT NULL CHECK(interval_seconds>0), checked_at TEXT,
 verified_until TEXT NOT NULL, retry_at TEXT NOT NULL, last_outcome TEXT,
 failure_count INTEGER NOT NULL CHECK(failure_count>=0),
 evidence_json TEXT, source_version TEXT NOT NULL,
 PRIMARY KEY(namespace,subject_kind,subject_key),
 CHECK(subject_kind IN ('slot','fact')),
 CHECK(last_outcome IS NULL OR last_outcome IN ('confirmed','changed','unavailable'))
);
CREATE INDEX ao_freshness_retry ON ao_freshness(retry_at,namespace,bound_fact_id);
~~~

Migrations: v1 streams/changes, v2 subscriptions/receipts из P1; P2 добавляет v3 intents, v4 actions, v5 freshness; P3 — v6 maintenance. Python validates wait JSON kind until/slot/change and state consistency; любой переход из waiting в другое состояние очищает waiting_json. create_digest hashes original goal/next_step/deadline; later checkpoint does not change it. Duplicate client_key returns the same intent only for matching digest, otherwise IdempotencyConflict. No FK to deletable facts. Every query includes owner, including action→intent lookup. Missing/foreign owner-scoped record → AccessDenied. Get/list never expose other owner; list ordered by id, limit default100/max500.

Claim after settling expiry/deadline under namespace lock:

~~~sql
UPDATE ao_intents SET state=?, lease_token=?, lease_generation=lease_generation+1,
 lease_worker=?, lease_expires_at=?, revision=revision+1, last_transition_at=?
WHERE owner=? AND id=? AND revision=? AND lease_token IS NULL
 AND state IN ('ready','blocked');
~~~

First state parameter is running for ready, blocked for blocked. Require one affected row; zero → RevisionConflict. Checkpoint and heartbeat:

~~~sql
UPDATE ao_intents SET state=?, progress_json=?, next_step=?, waiting_json=?,
 basis_json=?, blocked_reasons_json=?, revision=revision+1, last_transition_at=?,
 lease_token=NULL, lease_worker=NULL, lease_expires_at=NULL
WHERE owner=? AND id=? AND revision=? AND lease_generation=?
 AND lease_token=? AND lease_expires_at>?;

UPDATE ao_intents SET lease_expires_at=?
WHERE owner=? AND id=? AND lease_generation=? AND lease_token=? AND lease_expires_at>?;
~~~

Differentiate errors by reread under lock: invalid/expired token → LeaseLost before revision mismatch; valid token plus wrong revision → RevisionConflict. Renew does not increment business revision. Clock is read after locks. Invalid token never updates action/result. Expiry settlement can commit its own valid transition; it does not authorize stale worker mutation.

Action CAS runs in the same transaction as guards:

~~~sql
UPDATE ao_actions SET state='started', revision=revision+1, updated_at=?
WHERE owner=? AND id=? AND intent_id=? AND revision=? AND state='prepared';
~~~

Require one row. Finish/reconcile/expiry use the same CAS pattern with expected states and action revision. updated_at never serves as CAS token. Lease predicate concerns owning intent in that transaction. No permit before COMMIT. A lost begin response cannot be replayed into a second permit: started requires reconciliation.

## Полная таблица переходов

| Операция / исходное состояние | Результат и ограничения |
|---|---|
| create | ready, revision=1, generation=0, empty basis/progress |
| claim ready | running, new generation/token, revision+1 |
| claim blocked | blocked, recovery lease; begin forbidden |
| claim waiting/running/terminal | settle first; must become ready/blocked with no lease, else RevisionConflict |
| renew | current unexpired lease only; same token/generation |
| checkpoint running | ready/waiting/blocked/completed/cancelled при чистом action guard; clear lease, revision+1 |
| checkpoint blocked | ready/waiting/cancelled; valid supplied basis; clear lease |
| checkpoint completed | reject with started/unknown; terminals never reopen |
| checkpoint при STARTED | reject для всех targets; сначала finish либо expiry→unknown, lease не теряется |
| checkpoint при UNKNOWN | только blocked с reason unknown_action или явная cancelled; READY/WAITING/COMPLETED reject |
| deadline | unfinished→blocked(deadline_exceeded); only checkpoint cancelled |
| expiry running without unresolved | ready, clear lease; prepared remain prepared |
| expiry with started/unknown | started→unknown, intent blocked(unknown_action), clear lease |
| expiry recovery lease | remains blocked; next claim gets new generation |
| resume waiting condition met | ready, clear waiting_for |
| resume stale basis | unfinished→blocked with reasons; live lease retained |
| prepare new key | prepared, union intent+explicit basis, revision=1 |
| prepare same key/payload | same ID/key; only prepared can validly rebase, revision+1 only if changed |
| begin prepared | started only with running + live lease + clean guards; never second permit |
| finish started | succeeded/failed/unknown; facts unchanged |
| finish repeated outcome/result | live lease required; canonical equal no-op, otherwise RevisionConflict |
| reconcile unknown/failed | succeeded/failed or retryable→prepared; nonempty evidence, stable key |
| reconcile repeated outcome/evidence | live lease required; equal no-op, conflicting stored result RevisionConflict |
| reconcile removes sole unknown blocker | blocked→running, same recovery lease, if no other unresolved and clean guards |
| reconcile stale/deadline/host blocker | save action outcome; intent remains blocked with current reasons |

External key = SHA-256 canonical JSON array [owner,intent_id,action_key]. New request under existing action_key → IdempotencyConflict. Prepared rebasis preserves request, ID, key, created_at. Started/unknown/terminal preserve historical basis. Conflicting revisions of one fact → StaleEvidence. Retryable requires host evidence supporting safe retry; missing response alone proves no absence of effect.

## Pre-action transaction и recovery

1. Owner-scoped read discovers namespaces from intent.basis + stored action.basis + wait ref/slot. This read discovers locks, never grants permission.
2. Use always_on_transaction(tuple(sorted({owner,*dependencies})),write=True): SQLite BEGIN IMMEDIATE; PostgreSQL READ COMMITTED and stable advisory locks in that order. No nested connection.
3. Reread rows. If dependency namespace set expanded, rollback and retry full set at most three times, then RevisionConflict. ACL and freshness writers acquire the same namespace locks.
4. Read Clock inside locks. Validate owner, current token/generation/expires_at. Settle deadline/expired lease; stale caller gets no mutation permit. Read all actions for owner+intent; any started/unknown blocks another begin.
5. Require running intent and prepared action. Union current intent.basis and stored action.basis; conflicting revisions → StaleEvidence. Never replace refs with latest automatically.
6. Load facts transaction-locally; durable ACL projection precedes closed content/hash access. Apply P1 evidence_revision, Fact.is_active(at=now), supported, frontier and freshness. Missing/inactive/unsupported/changed/revoked/due basis rejects with action still prepared. Persist unfinished intent blocked with safe reasons in the transaction result branch; COMMIT first, then facade raises StaleEvidence. Raising inside the context manager would roll back that blocker.
7. Action CAS, COMMIT, return ActionPermit. Host calls external tool outside SQL with stable idempotency_key and conditional/fencing API if supported.
8. Finish stores result under live lease. After crash resume changes expired started→unknown; claim blocked grants recovery lease. Host performs lookup outside SQL and supplies evidence to reconcile. No automatic external dispatch.

Reconcile conservatively checks union current intent basis + immutable action basis. Proven external success is stored even if old action basis is stale; intent remains blocked. Host checkpoint new valid intent basis→ready under recovery lease, claim ready→running, checkpoint completed. Terminal action is neither rebased nor retried. Cancelling with UNKNOWN history leaves that history unknown; terminal intent cannot be claimed again. STARTED cannot release its lease via checkpoint, including cancellation: first record a finish outcome or let lease expire. Regression tests cover every lease-releasing target and subsequent recovery claim.

Resume does not call recall or increment access_count. Since it can write expiry/wait/stale reconciliation, use a write transaction with the same lock protocol. Get/list/capture use one read snapshot: SQLite read transaction / PG REPEATABLE READ. Ready=False for live lease, waiting/blocked/terminal, stale/due basis or unresolved action. Open actions = prepared/started/unknown; complete action history stays in durable table.

WaitUntil: now>=at. WaitForChange: with current read access, missing fact or changed revision satisfies; revoke produces access_changed blocker. WaitForSlot: current ACL projection, supported, is_active(now), frontier; exactly one canonical active value must equal value_text. Competing values, future/expired-only and absent slot remain waiting. next_step text is never code.

## Freshness: validity отдельно от schedule

Subject = (namespace,'slot',slot_key) for slotted fact, else (namespace,'fact',fact_id). Bound columns correspond to FreshnessRecord.ref. set_freshness validates current readable/supported/active ref under namespace lock and creates/replaces policy unverified: checked_at=None, verified_until=retry_at=now, failures=0. Repeated set never manufactures confirmation. Absent policy preserves legacy behavior.

record_verification rereads facts/ACL/frontier under lock. Old revision/old slot ID/ambiguous slot → StaleEvidence, with no update to current policy. Replacement slot ID retains policy but invalidates bound confirmation. An unslotted new ID is a new subject; old intent ref remains stale. If P1 writer did not eagerly rebind, every guard/read/due selector detects mismatch and treats it unverified immediately.

| Outcome | checked_at / verified_until | retry_at / failure_count | Fact/basis |
|---|---|---|---|
| confirmed, nonempty evidence | now / now+interval | now+interval / 0 | fresh only for exact current ref |
| changed | now / min(previous,now) | now+interval / 0 | no content/support/valid_until mutation; basis blocked |
| unavailable | preserve both | now+min(3600,30*2**min(count-1,7)) / count+1 | blocks immediately even if old verified_until is future |

Freshness guard blocks mismatch, checked_at=None, last_outcome!=confirmed or now>=verified_until. It never uses retry_at. due_verifications selects retry_at<=now OR bound mismatch regardless of retry_at; returns current-ref VerificationRequest only for accessible unambiguous active subject. Ordering: effective_due, namespace, current_fact_id; mismatch effective_due=now, otherwise retry_at; limit100/max500. Query due index plus active slot frontier join to detect mismatch, owner/ACL filter before pagination; SQL LIMIT is on eligible records so hidden rows cannot starve later visible work. No arbitrary first-1000 policy scan that silently omits all eligible work.

Model/provider/URL fetch runs in host/P3 outside transaction. evidence uses JSON codec; source_version is an opaque string. Write-time ACL is checked again. Facts are never made unsupported/expired merely because source is unavailable.

## Integration points и границы доказательства

memory.py reads package-internal kb._agent_id/kb._storage and P1 session; current KB has no public agent_id/storage properties. changes.py owns evidence hash: include topic_channel/memory_tier, exclude telemetry/retention/importance/tags/version/MESI, per canonical spec. Fact.version and cached pool scopes cannot authorize guards. Shared capture requires same store pool with persist_stats=True; arbitrary namespace does not grant __shared__ access. Durable ACL is reread transaction-locally. Revocation issue has ref=None without newly returned closed ID/text.

Existing seams: knowledge.py get/list/replace_facts, types.py Fact.is_active, SQL _load_on_connection; P1 adds transaction-local frontier/ACL projection. P3 HTTP/MCP/TS call the same facade, no second state machine. Snapshot/restore replaces facts only; AO rows survive.

The pure reference accepts already projected EvidenceView and condition_met; those are explicit computation inputs, not new public/test-only facade methods. Production computes them under actual locks. Before release, SQLite child-process and live PostgreSQL tests must prove dual claim, CAS race, expiry fencing, fact/ACL/freshness mutation between prepare/begin, crash after started/external success, restart reconciliation and snapshot action history. The in-memory checks do not replace those tests.
