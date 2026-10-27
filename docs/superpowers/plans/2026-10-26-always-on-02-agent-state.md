# Always-on 02 — Agent State and Action Guards Implementation Plan

**Execution status (2026-10-27): production work complete.** See [implementation and final verification](../implementation-final.md). The original checklist below is retained as the implementation brief; per-task commits were not created.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сохранять незавершённую работу, блокировать действие на устаревших основаниях и восстанавливать неоднозначный внешний исход без слепого повтора.

**Architecture:** Отдельные SQL records для intentions/actions/freshness; transitions выполняются под owner/dependency locks с проверкой lease и числовой revision. Host выполняет внешние действия и предоставляет reconciliation evidence.

**Tech Stack:** Python >=3.11, dataclasses/enums, sqlite3, существующий psycopg extra и pytest; без новых обязательных dependencies.

**Spec:** [2026-10-26-always-on-memory-design.md](../specs/2026-10-26-always-on-memory-design.md).

**Status (2026-10-26):** финальный кодовый план; production feature не реализован. [Спецификация](../specs/2026-10-26-always-on-memory-design.md), [точные типы/DDL/CAS/алгоритм](../reference/02-agent-state-code.md), [исполняемый эталон](../reference/always_on_state_reference.py). Эталон — чистые функции, не SQL реализация и не доказательство restart/concurrency/ACL guarantees.

**Dependencies:** завершённый P1: атомарные fact writers, evidence_revision, durable ACL projection, namespace locks, AlwaysOnSession. Python >=3.11, sqlite3/существующий psycopg extra; без ORM, broker и новых обязательных dependencies. P3 добавит host verifiers и transport surfaces.

## Global Constraints

- Python >=3.11; Node >=18; существующие runtime dependencies без новых обязательных пакетов.
- Новые durable гарантии: SQLite и PostgreSQL; YAML явно возвращает UnsupportedAlwaysOnBackend, прежний YAML API сохраняется.
- Always-on включается явно для namespace; по умолчанию нет модельных вызовов, подписок или фоновых потоков.
- Все новые времена timezone-aware UTC; Clock внедряется; тесты не ждут реальных минут и часов.
- Модели и внешние tools никогда не вызываются внутри SQL-транзакции.
- Доставка изменений at-least-once; внешние эффекты не объявляются exactly-once.
- HTTP/MCP/CLI работают от настроенного agent_id; идентификатор владельца не принимается из тела пользовательского запроса.
- Начальные пределы: page limit 100 (max 500), scan limit 1000, lease 60s (1..3600), debounce 0s (0..3600), JSON payload 65536 UTF-8 bytes.
- Ruff, mypy strict, Python coverage >=80%, текущие retrieval/eval и S8–S26 gates сохраняются.
- Snapshot/restore относится к фактам; intentions/actions/cursors/verification/jobs не откатываются вместе с фактами.

## Review Focus

- Lease истекла, старый worker ещё работает: P2.1 проверяет отказ checkpoint/renew со старым token.
- Action без explicit basis: P2.3 проверяет наследование basis намерения и отказ на stale evidence.
- External success без finish: P2.3 проверяет unknown и отсутствие слепого retry после restart.
- Ответ verifier относится к прежней revision: P2.4 проверяет StaleEvidence без refresh нового факта.
- Источник unavailable до verified_until: P2.4 проверяет немедленную блокировку независимо retry_at.

## Обязательные контракты

- Конструктор AlwaysOnMemory — explicit private stream opt-in; обычные KB/storage без AO ничего не запускают. YAML отвергает AO с UnsupportedAlwaysOnBackend без sidecar DB.
- Public signatures строго из spec: create_intent/get_intent/list_intents/claim/renew/checkpoint/resume/capture_evidence/prepare_action/begin_action/finish_action/reconcile_action/set_freshness/due_verifications/record_verification. now берётся только constructor Clock; никакого публичного now/owner input.
- UTC aware Clock, payload limit 65536 UTF-8 bytes на каждое caller JSON value, strict enum/finite JSON codec; limit default100/max500, TTL default60 и 1..3600. SQL всегда bound parameters.
- Write guards охватывают owner + dependency namespaces в sorted order, перечитывают facts/ACL после locks; lock-set expansion → rollback/retry максимум3, затем RevisionConflict. SQLite BEGIN IMMEDIATE; PostgreSQL READ COMMITTED + stable advisory locks. Read snapshots PG REPEATABLE READ.
- Lease expires при now>=expires_at. Old token не получает checkpoint/begin/finish/reconcile/renew, включая повтор. Recovery claim сохраняет blocked. Generation durable, action/intent revision INTEGER CAS; timestamp не CAS.
- Basis intent всегда наследуется action; started/unknown/terminal basis неизменен. Любой started/unknown action блокирует другой begin. Внешние tools/model calls выполняются вне SQL.
- Checkpoint не освобождает lease с STARTED action: такой переход отвергается. С UNKNOWN допускается только blocked(unknown_action) либо явная cancelled с сохранением неопределённого audit outcome; READY/WAITING/COMPLETED запрещены.
- Freshness validity не schedule: unavailable блокирует немедленно, не продлевает verified_until; retry_at только планирует запрос. Slot replacement сохраняет policy и сбрасывает confirmation.
- Missing/foreign owner-scoped record → AccessDenied; invalid body → ValueError/InvalidInput в P3; wrong revision → RevisionConflict; stale token → LeaseLost. AccessChanged issue не раскрывает закрытые ID/text.
- Snapshot/restore меняет facts, не откатывает AO state. Внешний эффект не exactly-once: после permit гонку закрывает только external conditional/fencing API.
- Ruff, mypy strict, coverage>=80%, current retrieval/eval/S8–S26 gates остаются обязательными.

## Карта файлов и fixtures

| Файл будущей реализации | Изменение |
|---|---|
| src/ai_knot/always_on/models.py | dataclasses/enums/errors/JSON+wait codecs по reference |
| src/ai_knot/always_on/memory.py | фасад, guards, owner/lease/revision checks, transaction orchestration |
| src/ai_knot/always_on/storage.py | row codecs, SQLStatement constants; AO migrations v3 intents/v4 actions/v5 freshness |
| src/ai_knot/always_on/changes.py | reuse P1 evidence_revision, не создавать второй hash |
| src/ai_knot/always_on/__init__.py | public type/facade exports |
| tests/always_on/conftest.py | дополнить P1 fixtures, без test-only facade methods |
| tests/always_on/test_intents.py | state/lease/restart/owner/CAS |
| tests/always_on/test_evidence.py | current fact/ACL/time guards |
| tests/always_on/test_actions.py | durable fake external service, crashes/reconcile/idempotency |
| tests/always_on/test_freshness.py | exact revision/slot replacement/backoff |

До behavior tests определить fixtures:

- clock: P1 FakeClock с UTC now, callable и advance(seconds=...). Facts устанавливают valid_from/created_at/last_accessed из clock: реальные Fact defaults не смешивать с historical clock.
- store_factory(): новый backend instance на том же temporary SQLite path / PG schema. PG: AI_KNOT_TEST_PG_DSN, уникальная schema через search_path, cleanup; локальный skip без DSN, отдельный CI job с live PG обязателен.
- ao_factory(agent_id='owner'): KnowledgeBase(agent_id=agent_id,storage=store_factory()) → AlwaysOnMemory(kb,clock=clock); каждая call — новый объект, прежний SQL state сохраняется. Shared tests используют pool persist_stats=True того же store.
- kb: KnowledgeBase тех же owner/store; mutate через get/replace_facts/add_resolved. Historical Fact fixture задаёт timestamps явно.
- external: отдельная temporary SQLite DB с effects(idempotency_key TEXT PRIMARY KEY,result_json TEXT). perform(key): INSERT ON CONFLICT DO NOTHING; lookup(key): durable read. Сервис не исчезает с рестартом agent. Это только test helper.
- Process fault tests: multiprocessing spawn, Barrier/Event и Pipe. Child сообщает parent после public begin_action (COMMIT завершён) либо durable external.perform, ждёт Event; parent terminate/join(timeout), проверяет exitcode. Не вводить production pause/crash_after hooks.

KnowledgeBase сейчас имеет _agent_id/_storage: package-internal facade использует их, новые public properties не предполагаются. Session имеет P1 load_facts/replace_facts/fetch_one/fetch_all/execute с SQLStatement(sqlite,postgres); save_intent/validate_for_test не вводить.

## Task P2.1: intents, checkpoints, leases, resume

**Files:** Modify `always_on/{models,memory,storage}.py`; Create `tests/always_on/test_intents.py`; Extend `tests/always_on/conftest.py`. Пути production — относительно `src/ai_knot/`.

**Interfaces:** Consumes P1 AlwaysOnSession/Clock. Produces create_intent/get_intent/list_intents/claim/renew/checkpoint/resume и Intent/Lease/ResumePacket из spec; SQL migration v3.

- [ ] **1. Создать failing tests** test_intents.py и fixtures. Public сценарий: create→claim→checkpoint waiting(WaitUntil)→новый ao_factory.resume не ready→clock.advance до границы→resume ready с прежними progress/next_step. Два processes claim одного intent: одна lease; expiry/reclaim, старый token LeaseLost. Проверить exact expiry boundary, renewal, CAS expected_revision, generation после checkpoint+claim, terminal non-reopen, deadline only cancelled, owner isolation, payload/enum/limit validation.
- [ ] **2. Запустить red:** .venv/bin/python -m pytest tests/always_on/test_intents.py -q --no-cov. Fixtures импортируются; отказ относится к новому контракту.
- [ ] **3. Добавить модели/миграцию v3** по DDL reference. Revision=1, UNIQUE(owner,client_key), create_digest исходных goal/next_step/deadline. Same create digest возвращает запись, другой input конфликтует. Wait JSON kind until/slot/change, nullable wait iff waiting. Generation и blocked_reasons durable.
- [ ] **4. Реализовать create/get/list/claim/renew/checkpoint/resume** в memory.py. Перед claim settle expiry/deadline. Running checkpoint → ready/waiting/blocked/completed/cancelled; blocked → ready/waiting/cancelled; clears lease, CAS revision+1. Deadline-exceeded только cancelled; completed нельзя при unresolved action. Resume не вызывает recall; wait slot — supported active unambiguous ACL frontier. Basis guard вызывается уже здесь, полный P2 не публиковать до P2.4.
- [ ] **5. Повторить targeted + P1 suite** SQLite/live PG; separate-instance restart и отсутствие access_count writes. Commit: feat: persist agent intentions with fenced leases and checkpoints.

## Task P2.2: evidence и transactional guards

**Files:** Modify `always_on/{models,memory,changes}.py`; Create `tests/always_on/test_evidence.py`.

**Interfaces:** Consumes P1 evidence_revision/ACL и P2.1 Intent.basis. Produces capture_evidence и shared validate_basis(session,refs,owner_id,now); public signatures из spec, точный internal algorithm из code appendix.

- [ ] **1. Создать failing tests** test_evidence.py. Unslotted Fact.version=0 меняет content через kb.get/replace_facts: прежний basis блокирует resume. Metadata-only recall не меняет evidence revision. valid_until/future valid_from/deletion/support/source changes блокируют. Snapshot restore инвалидирует refs. Grant/revoke из другого instance проверяется durable ACL; closed issue.ref=None без text. Terminal intent не открывается новым grant.
- [ ] **2. Запустить red:** .venv/bin/python -m pytest tests/always_on/test_evidence.py -q --no-cov.
- [ ] **3. Реализовать capture_evidence**: one consistent read snapshot current accessible/supported/active facts → sorted tuple EvidenceRef. Private owner не получает __shared__ без допустимого pool/durable ACL; unknown/closed requested ref отвергать без раскрытия content. Revision из changes.evidence_revision P1, не Fact.version; topic_channel/memory_tier включены, telemetry/ranking поля исключены по spec.
- [ ] **4. Добавить внутренний validate_basis(session,refs,owner_id,now)** для checkpoint/resume/begin/reconcile: durable ACL до content/hash, exact revision, Fact.is_active(now), supported, frontier, optional freshness. Reasons changed/missing/inactive/unsupported/access_changed/verification_due. Reference EvidenceView — internal projected input, не facade API. Different revisions одного fact в union → StaleEvidence.
- [ ] **5. Внедрить lock-set discovery/retry** по reference. Supplied checkpoint basis валидируется, stale reasons сбрасываются; latest refs автоматически не подставляются. Для persisted blocked + exception COMMIT transaction result branch до raise, иначе context rollback потеряет blocker.
- [ ] **6. Запустить P2.1/P2.2 и** tests/test_temporal_validity.py tests/test_acl_ledger_storage.py tests/test_shared_pool.py с --no-cov; live PG cross-instance ACL race. Commit: feat: track decision evidence and block stale plans.

## Task P2.3: action journal, permit, unknown reconciliation

Дополнительные regressions после независимого review: `test_checkpoint_cannot_orphan_started_action` отвергает все lease-releasing checkpoints с STARTED; `test_unknown_checkpoint_only_blocks_or_explicitly_cancels` отвергает ready/waiting/completed, допускает blocked→recovery claim→reconcile и явную cancellation. Эти проверки воспроизводят прежнее зависание waiting/started/lease=None и входят в test_actions.py.

**Files:** Modify `always_on/{models,memory,storage}.py`; Create `tests/always_on/test_actions.py`.

**Interfaces:** Consumes P2.1 lease и P2.2 guard. Produces prepare_action/begin_action/finish_action/reconcile_action, ActionRecord/ActionPermit и SQL migration v4.

- [ ] **1. Создать failing crash test** test_actions.py с durable external. Child: create/claim/prepare/begin→external.perform→parent terminate→clock+60s→новая factory.resume: unknown action, blocked intent, effect_count=1. Recovery claim остаётся blocked; external.lookup→reconcile succeeded возвращает running при clear guards; checkpoint completed, effect_count=1. Без remote idempotency unknown остаётся до evidence.
- [ ] **2. Покрыть boundaries:** inherited basis без explicit refs; prepare duplicate same/different payload; only prepared rebases; fact/ACL/freshness mutation между prepare/begin запрещает permit; repeated begin after response loss не выдаёт второй permit; started-before-dispatch crash всё равно unknown; stale token не finish/reconcile; unknown блокирует другой action_key; explicit retryable с прежним key; repeated finish/reconcile canonical equal no-op при live lease, mismatch RevisionConflict.
- [ ] **3. Запустить red:** .venv/bin/python -m pytest tests/always_on/test_actions.py -q --no-cov.
- [ ] **4. Добавить v4 action row codec/DDL** по reference. INTEGER revision; UNIQUE(owner,intent_id,action_key); request_digest canonical; idempotency_key=SHA256 canonical [owner,intent_id,action_key]. Prepared rebase сохраняет ID/request/key/created_at; started/unknown/terminal не rebase. Reconciliation evidence отдельно; timestamp не CAS.
- [ ] **5. Реализовать prepare/begin/finish/reconcile**. Begin transaction: live lease, running, union current intent+stored action basis, current freshness/ACL, no unresolved; CAS prepared→started; permit после COMMIT. Finish не меняет facts. Expiry started→unknown, никогда не failed.
- [ ] **6. Реализовать recovery**: reconcile unknown/failed с непустым trusted host evidence → terminal outcome или retryable→prepared. Blocked→running только sole unknown blocker + clear guards, с той же recovery lease. При stale историческом action basis outcome сохраняется, blocked остаётся: checkpoint нового basis→ready, claim→running, checkpoint completed; terminal action не запускается заново.
- [ ] **7. Запустить P2 suite SQLite/live PG** и child process tests, проверить actions после snapshot restore. Commit: feat: journal external action attempts and reconcile uncertain outcomes.

## Task P2.4: freshness policy и source verification

**Files:** Modify `always_on/{models,memory,storage}.py`; Create `tests/always_on/test_freshness.py`.

**Interfaces:** Consumes EvidenceRef, Clock и guarded current slot frontier. Produces set_freshness/due_verifications/record_verification, FreshnessRecord/VerificationResult и SQL migration v5; P3 consumes эти методы.

- [ ] **1. Создать failing tests** test_freshness.py. Set policy unverified немедленно; confirmed exact ref даёт interval; unavailable до/после verified_until сохраняет его и блокирует basis; retry_at=now+30s/60s/...cap3600. До retry_at due пусто, но resume/begin blocked. Fake clock, никаких sleeps.
- [ ] **2. Добавить tests** old revision StaleEvidence без refresh новой policy; changed basis blocked без fact mutation; confirmed empty evidence reject; revoke до record deny; slot replacement unverified сразу несмотря на old retry_at; ambiguous slot reject; deterministic due order/limit; unslotted new ID требует intent replan.
- [ ] **3. Запустить red:** .venv/bin/python -m pytest tests/always_on/test_freshness.py -q --no-cov.
- [ ] **4. Добавить v5 freshness DDL/models** по reference. PK(namespace,subject_kind,subject_key); bound fact/revision; source/interval/checked_at/verified_until/retry_at/outcome/failures/evidence/source_version. Slot subject переживает ID replacement; no FK cascade.
- [ ] **5. Реализовать set/due/record**. Set initial/repeated unverified. Record locks scope, rereads current ACL/frontier/ref. Confirmed→checked_at=now, verified_until=retry_at=now+interval, failures=0. Changed→checked_at=now, verified_until=min(old,now), retry_at=now+interval, failures=0. Unavailable сохраняет checked_at/verified_until, failures+1, retry_at по exact backoff spec. Providers вне SQL; verification не меняет content/support/valid_until.
- [ ] **6. Интегрировать guard/due selection**: guard blocks unverified/mismatch/nonconfirmed/expired независимо retry_at. Due selects retry_at<=now OR slot binding mismatch; eligible/ACL filter до SQL LIMIT, join active frontier. Ordering effective_due,namespace,current_fact_id, default100/max500. Новый EvidenceRef/ID не стирает slot policy.
- [ ] **7. Запустить весь P1+P2 regression**, .venv/bin/ruff check src tests; .venv/bin/mypy src/ai_knot; .venv/bin/python -m pytest tests/always_on. Live PG CI без skip; coverage>=80%. Commit: feat: revalidate decision critical facts with explicit freshness policies.

## P2 exit gate и проверка документа

- [ ] SQLite/live PG одинаково проходят restart/lease/CAS/unknown/freshness; mock psycopg не засчитывается.
- [ ] Scenario: wait CI→stop→CI/basis changed→resume stale blocked→host replan→external effect→crash before finish→reconcile→completed.
- [ ] Non-idempotent external branch честно unknown; TOCTOU после permit и PG clock skew описаны.
- [ ] Нет transport state machine, scheduler daemon или модельного эксперимента; production P2 не объявлен завершённым по эталону.

Проверка только предоставленного дизайна, не выполнение будущих implementation tests:

~~~sh
.venv/bin/python docs/superpowers/reference/always_on_state_reference.py
.venv/bin/ruff check docs/superpowers/reference/always_on_state_reference.py
.venv/bin/mypy --strict --follow-imports=skip docs/superpowers/reference/always_on_state_reference.py
~~~
