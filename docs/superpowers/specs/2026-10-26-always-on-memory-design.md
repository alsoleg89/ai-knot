# Always-on memory: спецификация и порядок реализации

Дата начала: 2026-10-26; итоговая сверка: 2026-10-27. База: AI Knot `69e2adf`, Python/package version `0.12.0`.
Статус: реализовано по P1–P3. Production API описан в [always-on.md](../../always-on.md); итоговая проверка — [implementation-final.md](../implementation-final.md). Исполняемые образцы в `docs/superpowers/reference/` сохраняются как отдельные модели алгоритмов.

## Цель и решения автора плана

Агент сохраняет незавершённую работу, переживает остановки, получает пропущенные изменения, проверяет основания решения перед действием и обслуживает память с ограниченной стоимостью.

Покрываются все шесть согласованных направлений обсуждения: намерения; подписки; зависимости решений; журнал действий; повторная проверка знаний; фоновое обслуживание. Предложенные ниже ограничения и значения — решения автора плана, а не ранее высказанные требования пользователя.

Выбран слой памяти со структурированным состоянием работы. Запуск процесса, расписание, внешние tools и исполнение действий остаются у host/runtime. AI Knot возвращает готовность, причины блокировки, пакет восстановления и устойчивые delivery/action IDs.

Рассмотренные подходы:

| Подход | Выигрыш | Цена / решение |
|---|---|---|
| Хранить всё как `Fact` + snapshot | Мало изменений | Нет строгих transitions, leases, ack; snapshots откатывали бы историю действий. Отклонён |
| Отдельные SQL-записи состояния + журнал памяти, существующий runtime исполняет | Переиспользует SQLite/PostgreSQL и интеграции; проверяемое восстановление | Выбран; нужны транзакционные изменения SQL-хранилищ |
| Встроить scheduler, task broker и workflow runtime | Единый исполнитель | Отдельный продукт и избыточная ответственность. За пределами этой версии |

## Планы, зависимости и выпуск

1. [P1: атомарность, журнал и подписки](../plans/2026-10-26-always-on-01-storage.md) — задачи P1.1–P1.4. Самостоятельный результат: устойчивое чтение изменений с ack.
2. [P2: намерения, основания, действия и актуальность](../plans/2026-10-26-always-on-02-agent-state.md) — P2.1–P2.4. Самостоятельный результат: агент восстанавливает работу и блокирует устаревшее действие.
3. [P3: обслуживание, интерфейсы и приёмка](../plans/2026-10-26-always-on-03-maintenance-surfaces.md) — P3.1–P3.5.

Порядок: `P1.1 → P1.2 → P1.3 → P1.4 → P2.1 → P2.2 → P2.3 → P2.4 → P3.1 → P3.2 → P3.3 → P3.4 → P3.5`. После фиксации contracts P3.2/P3.3 могут выполняться независимо. Один task — один проверяемый commit; связанные commits одного плана объединяются в reviewable PR. Точное число строк и сроки не выдаются за измеренные до реализации.

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

## Точки изменения текущего кода

`StorageBackend.save()` заменяет весь namespace. SQLite имеет `atomic_update`, PostgreSQL — только `save_atomic` без защиты read-modify-write. У `KnowledgeBase.add/add_many/add_episodic`, `learning._commit_phase`, recall metadata, decay, pool promotion/GC есть разнесённые load/save. Их необходимо привести к безопасным изменениям, прежде чем фоновые writers начнут работать одновременно с foreground.

`pool.sync_dirty/sync_slot_deltas` сохраняют `_known_version` в процессе и продвигают его при чтении. Это остаётся legacy API, но новый durable feed строится на собственном sequence, а не на `Fact.version`: private facts, удаления и GC не обеспечивают нужную семантику version.

`consolidate_episodic` сейчас вызывает `learn_async` до закрытия эпизодов и не отделяет proposal от commit. Для фонового исполнения нужен двухфазный путь с проверкой входных revisions.

## Структура модулей

Новый пакет `src/ai_knot/always_on/`:

- `models.py`: dataclasses/enums, ошибки, чистая валидация контрактов.
- `storage.py`: узкий внутренний SQL seam `AlwaysOnStorageCapable` / `AlwaysOnSession`, migration constants. Две реальные реализации — SQLite/PostgreSQL.
- `memory.py`: `AlwaysOnMemory`, public Python interface; транзакционные операции над состоянием и событиями. SQL-значения всегда параметризованы.
- `changes.py`: evidence revision, diff и правила scope projection/coalescing. Без I/O и моделей.
- `maintenance.py`: планирование ограниченного batch, verify/consolidate preparation, commit и ledger.
- `__init__.py`: публичные типы и facade.

Меняются существующие `storage/sqlite_storage.py`, `storage/postgres_storage.py`, `storage/base.py`, `knowledge.py`, `learning.py`, `pool.py`, `_pool_recall.py`. В SQL-хранилищах остаются конкретные transaction/locking и fact serialization; никаких новых ORM, очередей или общего plugin framework.

Для нескольких scopes брать locks в лексикографическом порядке namespace. Набор locks для checkpoint/prepare/begin/verification включает owner namespace и **все namespaces читаемых dependencies**, даже когда там нет записи; grant/revoke/freshness mutation берут соответствующий namespace lock. SQLite: одна `BEGIN IMMEDIATE` транзакция. PostgreSQL: `pg_advisory_xact_lock` по стабильному signed 64-bit digest namespace; не Python `hash()`. В write transaction читать актуальные facts/ACL после получения всех locks (READ COMMITTED). Если набор dependency namespaces был предварительно прочитан ради lock set, после locks перечитать запись; при изменении состава отпустить transaction и повторить с полным набором, максимум 3 попытки, затем RevisionConflict. Read snapshot: одна SQLite read transaction / PostgreSQL REPEATABLE READ transaction. Внутренний `AlwaysOnSession` позволяет читать факты и AO-таблицы в одной транзакции, без вложенных connections.

```python
class AlwaysOnStorageCapable(Protocol):
    def always_on_transaction(
        self, namespaces: tuple[str, ...], *, write: bool
    ) -> AbstractContextManager[AlwaysOnSession]: ...
```

Это внутренний seam для двух SQL-backends; пользователь вызывает facade, а не пишет SQL callbacks. `StorageBackend` не расширять обязательными методами. Включение YAML завершается понятной ошибкой без создания побочного SQLite-файла.

Внутренний session имеет один согласованный интерфейс:

```python
SQLScalar = str | int | float | bytes | None
SQLRow = dict[str, SQLScalar]

@dataclass(frozen=True)
class SQLStatement:
    sqlite: str
    postgres: str

class AlwaysOnSession(Protocol):
    def load_facts(self, namespace: str, *, fact_id: str | None = None,
                   slot_key: str | None = None) -> list[Fact]: ...
    def replace_facts(self, namespace: str, facts: list[Fact]) -> None: ...
    def fetch_one(self, statement: SQLStatement,
                  params: Sequence[SQLScalar] = ()) -> SQLRow | None: ...
    def fetch_all(self, statement: SQLStatement,
                  params: Sequence[SQLScalar] = ()) -> list[SQLRow]: ...
    def execute(self, statement: SQLStatement,
                params: Sequence[SQLScalar] = ()) -> int: ...
```

SQLStatement содержит фиксированные SQL constants двух dialects, params — только bind values. execute возвращает rowcount; fetch возвращает dict rows. replace_facts использует общий journal hook в той же connection. Session не может открывать вложенную backend transaction; методы работают только по scopes, запрошенным у always_on_transaction. Обращение к незаявленному namespace отвергается.

Новые AO timestamps в SQLite сохраняются как UTC ISO строки с фиксированной microsecond precision. PostgreSQL TIMESTAMPTZ возвращает datetime: session нормализует такие значения в те же UTC ISO строки при построении SQLRow, сохраняя заявленный SQLScalar contract. Domain codec преобразует строки обратно в aware datetime; naive значения отвергаются.

AO schema marker: v1 streams/changes; v2 subscriptions/delivery_receipts; v3 intents; v4 actions; v5 freshness; v6 maintenance_runs. Upgrade additive и идемпотентный; marker и DDL коммитятся вместе. До enable в существующие namespace не добавляется выдуманная история. Rollback релиза не удаляет AO history автоматически.

Уточнения production review: freshness хранит внутренние `revision` и `dirty`, без изменения public JSON. Revision увеличивается при регистрации/результате и логическом изменении основания; ответ встроенного verifier коммитится только для сохранённой policy revision. Dirty обновляется вместе с facts; due selection использует индексированные subject queries и фильтрованные fact/slot reads. Новая временная frontier становится due без дополнительной записи. Старые schema 5/6 получают эти поля аддитивно ещё при открытии backend обычным writer.

## Доменные типы и состояния

Типы JSON: recursive `JSONValue`, `JSONObject = dict[str, JSONValue]`. Строки ключей непустые; неизвестные enum значения, NaN/Infinity, naive datetime, размер >65536 bytes отвергаются до записи. JSON digest считается после canonical serialization с сортировкой ключей; один ключ с другим payload даёт `IdempotencyConflict`.

Предел 65536 UTF-8 bytes относится к каждому входному logical JSON value (progress, request, result, evidence, selector), а не к суммарному ChangeBatch/списку ответов или внутреннему inflight snapshot. Длина выдачи ограничена page limit; общий response byte cap в этой версии не заявляется.

```python
@dataclass(frozen=True)
class EvidenceRef:
    namespace: str
    fact_id: str
    revision: str  # SHA-256 логического состояния, не Fact.version

@dataclass(frozen=True)
class WaitUntil:
    at: datetime

@dataclass(frozen=True)
class WaitForSlot:
    namespace: str
    slot_key: str
    value_text: str

@dataclass(frozen=True)
class WaitForChange:
    ref: EvidenceRef

WaitCondition = WaitUntil | WaitForSlot | WaitForChange
```

`Intent`: id, owner_agent_id, client_key, goal, state, progress, next_step, waiting_for, revision, basis, lease, lease_generation, blocked_reasons, deadline, last_transition_at. States: `ready`, `running`, `waiting`, `blocked`, `completed`, `cancelled`. Новое намерение `ready`; claim ready переводит в running. claim blocked выдаёт recovery lease и сохраняет blocked: можно reconcile и checkpoint нового плана, нельзя begin внешнего действия. checkpoint из running разрешает ready/waiting/blocked/completed/cancelled; из blocked — ready с новым валидным basis, waiting или cancelled. completed/cancelled терминальны. waiting_for обязателен только для waiting. Deadline переводит незавершённое намерение в blocked с reason `deadline_exceeded`; такой intent допускает только cancelled, для новой попытки нужен новый client_key. lease_generation хранится независимо от текущей lease и увеличивается при каждой новой выдаче, включая выдачу после checkpoint/restart.

`Lease`: token, generation, worker_id, expires_at. claim атомарен. Любой checkpoint/begin/finish внешнего действия проверяет текущий token и время в той же транзакции; устаревший worker получает `LeaseLost`. heartbeat продлевает только ещё действующую lease. Для SQLite Clock берётся host; для нескольких PostgreSQL hosts требуется синхронизированное время и документированный допустимый skew — fencing остаётся обязательным. Не обещать внешнее fencing для API, которое его не понимает.

После lease expiry: running без started/unknown action становится ready; с незавершённым started action — blocked, действие unknown до reconciliation. Reconciliation разрешён новой действующей lease; прежний worker не может дописать результат.

Checkpoint всегда освобождает lease, поэтому при любом STARTED action он запрещён для всех целевых состояний: сначала finish_action или expiry→unknown. При UNKNOWN разрешены только checkpoint(blocked), который сохраняет reason unknown_action для recovery lease, либо явный cancelled с сохранением unknown audit history. READY/WAITING/COMPLETED при UNKNOWN запрещены. Отмена не объявляет внешний эффект отменённым; cancelled intent не возобновляется автоматически.

`ActionRecord`: id, intent_id, action_key, request, request_digest, basis, state, revision, idempotency_key, result, error, created_at, updated_at. States: `prepared`, `started`, `succeeded`, `failed`, `unknown`. revision — монотонный INTEGER для compare-and-set; updated_at не заменяет revision, поскольку timestamps могут совпадать. idempotency_key = стабильный digest `(owner, intent_id, action_key)`, не случайный новый key на каждый retry. Эффективный basis — union текущего intent.basis и explicit action basis; разные revisions одного факта конфликтуют. prepare с тем же ключом и payload возвращает прежнюю запись, с другим payload — conflict. Только ещё prepared action может при повторном prepare принять новый эффективный basis под действующей lease; started/terminal/unknown сохраняют исходные ссылки. begin допускает только prepared и running intent; повтор begin не выдаёт второй permit. started после сбоя не объявляется failed автоматически. reconcile retryable переводит unknown/failed обратно в prepared с прежним idempotency_key; succeeded/failed outcomes записывают конечный результат. Повтор finish/reconcile с тем же конечным outcome и canonical result/evidence при действующей lease — no-op; несовпадающий результат — RevisionConflict. Наличие любого unresolved started/unknown action блокирует begin других действий. raw external call AI Knot не выполняет.

Ошибки: `UnsupportedAlwaysOnBackend`, `RevisionConflict`, `LeaseLost`, `IdempotencyConflict`, `StaleEvidence`, `CursorExpired`, `InvalidDelivery`, `AccessDenied`. Проблемы проверки источника — результаты verification, а не ошибка достоверности самого факта.

## Python interface — целевой контракт

Все названия ниже фиксированы для планов. `Lease.token` и delivery token непрозрачны. `now` для новых операций берётся только из constructor Clock.

Создание `AlwaysOnMemory(kb, ...)` является явным opt-in: private stream namespace этой KB включается атомарно и идемпотентно. Shared stream включается при первом допустимом shared subscribe/capture после проверки pool/durable ACL. Создание обычной KnowledgeBase или SQLStorage само по себе не включает journal.

```python
class AlwaysOnMemory:
    def __init__(self, kb: KnowledgeBase, *, pool: SharedMemoryPool | None = None,
                 clock: Callable[[], datetime] = utc_now) -> None: ...

    def create_intent(self, *, client_key: str, goal: str,
                      next_step: str = "", deadline: datetime | None = None) -> Intent: ...
    def get_intent(self, intent_id: str) -> Intent: ...
    def list_intents(self, *, limit: int = 100) -> list[Intent]: ...
    def claim(self, intent_id: str, *, worker_id: str,
              ttl_seconds: int = 60) -> Lease: ...
    def renew(self, intent_id: str, *, lease_token: str,
              ttl_seconds: int = 60) -> Lease: ...
    def checkpoint(self, intent_id: str, *, lease_token: str, expected_revision: int,
                   state: IntentState, progress: JSONObject, next_step: str = "",
                   waiting_for: WaitCondition | None = None,
                   basis: tuple[EvidenceRef, ...] = ()) -> Intent: ...
    def resume(self, intent_id: str) -> ResumePacket: ...

    def capture_evidence(self, refs: tuple[tuple[str, str], ...]) -> tuple[EvidenceRef, ...]: ...
    def subscribe(self, *, subscription_id: str, selector: WatchSelector,
                  start: Literal["now", "beginning"] = "now",
                  debounce_seconds: float = 0) -> Subscription: ...
    def poll(self, subscription_id: str, *, limit: int = 100) -> ChangeBatch: ...
    def ack(self, subscription_id: str, *, delivery_id: str) -> None: ...
    def unsubscribe(self, subscription_id: str) -> None: ...

    def prepare_action(self, intent_id: str, *, lease_token: str,
                       action_key: str, request: JSONObject,
                       basis: tuple[EvidenceRef, ...] = ()) -> ActionRecord: ...
    def begin_action(self, action_id: str, *, lease_token: str) -> ActionPermit: ...
    def finish_action(self, action_id: str, *, lease_token: str,
                      outcome: Literal["succeeded", "failed", "unknown"],
                      result: JSONObject) -> ActionRecord: ...
    def reconcile_action(self, action_id: str, *, lease_token: str,
                         outcome: Literal["succeeded", "failed", "retryable"],
                         evidence: JSONObject) -> ActionRecord: ...

    def set_freshness(self, ref: EvidenceRef, *, source_ref: str,
                      recheck_after_seconds: int) -> FreshnessRecord: ...
    def due_verifications(self, *, limit: int = 100) -> list[VerificationRequest]: ...
    def record_verification(self, ref: EvidenceRef, *, result: VerificationResult) -> FreshnessRecord: ...
```

`IntentState` — enum из models.py. `ResumePacket`: intent, open_actions, stale_basis, freshness_due, ready, reasons. `resume` детерминированно сверяет актуальные факты/ACL/время, ничего не запускает; разрешены только описанные state reconciliation записи. `ready=False` при любой живой lease (resume не принимает worker identity), waiting без выполненного условия, blocked, терминальном состоянии, stale/недоступном basis, due verification и unknown action. Revoked basis issue показывается как `access_changed` без нового чтения закрытого текста. Owner-owned progress/request и ранее сохранённые evidence references не являются новой shared-read выдачей; ACL revoke не обещает вычистить произвольные цитаты из старого progress JSON или памяти внешнего процесса.

`checkpoint(state=ready)` после перепланирования должен принять новый подтверждённый basis и сбросить stale reason. waiting условие переводит в ready при выполнении во время resume; внешнее CI-событие host заранее записывает через `add_resolved` в согласованный slot. Человеческий текст условия не исполняется как код/LLM prompt.

При policy due verification для basis `resume` блокирует только явно включённые freshness records. Срок действия по умолчанию не вводится всем старым фактам.

## Журнал и доставка

Logical tables (SQLite `ao_*`, PostgreSQL `"ai-knot_ao_*"`):

| Таблица | Ключ / содержимое |
|---|---|
| streams | namespace PK; enabled, head_seq, min_retained_seq |
| changes | (namespace,seq) PK; kind, fact_id, slot_key, before_slot_key, before/after revision, old/new scope и origin, optional ACL subject; content не копируется |
| subscriptions | (owner,id) PK; incarnation, namespace, selector JSON, cursor, debounce, pending window, inflight delivery token/through_seq и зафиксированный список seq |
| delivery_receipts | (owner,subscription_id,incarnation,delivery_id) PK; through_seq; запись создаётся при успешном ack, повтор ack старой подтверждённой batch остаётся no-op |
| intents | (owner,id) PK; UNIQUE(owner,client_key); поля Intent и lease fencing generation |
| actions | (owner,id) PK; UNIQUE(owner,intent_id,action_key); поля ActionRecord |
| freshness | (namespace,subject_kind,subject_key) PK; subject=slot_key для slotted или fact_id для unslotted; bound fact_id/revision, source_ref, interval, checked_at, verified_until, retry_at, last_outcome, failure_count |
| maintenance_runs | (namespace,run_id) PK; input refs, budget reservation, lease generation, status, counters и error |

Операции состояния Intent/Action и delivery не кодируются фиктивными facts. JSON поля удобны для progress и результатов; поля фильтрации, revision, lease, due time — отдельные SQL columns с индексами. Без внешних ключей на удаляемые facts: deletion должен инвалидировать ссылку, не удалять историю намерения.

Включение stream + baseline фиксируется атомарно: событие `resync_required` при существующих фактах; historical events не выдумываются. subscribe beginning увидит baseline; subscribe now начинается с текущего head. До enable нет journal writes. После enable все обновлённые SQL writer instances проверяют persisted flag; mixed deployment со старым бинарником, не умеющим журнал, не поддерживается — все writers обновляются до включения.

Fact save/delete/atomic_update + logical diff + новые events + head update коммитятся вместе. Важна **очерёдность commit**: PostgreSQL sequence/identity без namespace lock не подходит — ранний uncommitted seq может быть пропущен читателем. Head увеличивается под тем же namespace transaction lock. Времени/Fact.version для курсора недостаточно.

Evidence revision включает content/type, slot/entity/value, canonical/witness/prompt surfaces, qualifiers, sources, support/state confidence, temporal validity/event_time, origin/visibility/scope/claim_key, topic_channel и memory_tier. Не включает access_count/last_accessed/access_intervals, retention, importance, tags, MESI и ranking version. Это консервативная проверка основания, не hash всех полей dataclass. Изменение только статистики не порождает fact-change wakeup; promotion с изменением memory_tier порождает. Любая смена evidence revision, insert/delete создаёт событие; expiry без записи определяется на resume/begin_action по времени. В service events (`resync_required`, ACL) public fact_id/slot_key — пустые строки. Изменение slot учитывает before_slot_key при подборе старых подписчиков.

WatchSelector: namespace, fact_ids tuple, slot_keys tuple; пустые selectors = весь доступный namespace; объединение fact_ids/slot_keys по OR. Сначала поддерживаются точные ID/slot, без semantic query subscriptions. Для shared selector требуется переданный pool на том же SQL store, созданный с `persist_stats=True` (в текущем API это включает durable ACL). Если flag выключен, private AO работает, а shared subscribe/capture возвращает понятную configuration error до создания подписки. ACL проверяется заново из durable grants при каждом poll/resume/begin_action. Отсутствие shared pool не даёт доступ к `__shared__`. Поддерживаемые изменения ACL: существующий `pool.grant_read`, новый `pool.revoke_read(agent_id,scope)`, существующие SQL `save_grant/revoke_grant`; SQL entry points атомарно добавляют адресованный ACL event при включённом shared stream, поэтому facade не пишет второй event. При shared AO existing recall тоже обновляет cached scopes из durable grants.

poll не двигает подтверждённый cursor. Одна подписка имеет максимум один inflight batch; повторный poll после process restart возвращает тот же delivery_id и тот же разрешённый набор событий. При ACL change набор может быть редактирован до безопасного resync_required, но delivery_id остаётся; newly granted events не добавляются задним числом в уже зафиксированный batch. ack идемпотентен только для выданного batch, чужой/будущий token отвергается. Receipts сохраняются до unsubscribe, поэтому повтор ack старого подтверждённого batch после более новых доставок тоже no-op. Монотонность cursor и owner binding проверяются в транзакции. unsubscribe идемпотентно удаляет owner subscription/inflight/receipt state; прежние delivery tokens становятся InvalidDelivery, подписка перестаёт удерживать retention frontier. Повторное создание того же subscription_id получает новую incarnation. Receipt history растёт с количеством batches живой подписки; это явно учитывается в storage benchmark, constant-space гарантия не заявляется.

На revoke/сужении scope разрешён только общий `resync_required` без закрытых fact IDs/contents. Shared ACL grant/revoke дописывает адресованное субъекту служебное событие в том же commit. Другие агенты его не видят. Кэшированные `_read_scopes` не являются актуальным разрешением для новых AO API.

Полностью отфильтрованные страницы не застревают: batch содержит пустые events, opaque token и next scan boundary; его можно ack. Пустой tail без новых events не создаёт delivery. Scan bounded 1000. Coalescing по fact_id сохраняет последнюю операцию и earliest before revision; supersede двух fact IDs не теряет invalidation прежнего ID. Debounce окно начинается при первом замеченном подходящем изменении, persisted в subscription; poll до due ничего не будит. Без открытого inflight batch новые события могут объединяться в это окно.

Retention: автоматическое удаление только до минимального подтверждённого cursor живых подписок; ручное принудительное pruning требует явной операции и переводит отставших в CursorExpired. Никаких молчаливых пропусков. Без подписчиков разрешено удалять до head. min_retained_seq — первая доступная seq или head+1 при пустом журнале; cursor expired, если cursor < min_retained_seq-1. subscribe(beginning) после pruning возвращает CursorExpired без создания подписки, а не выдаёт retained suffix за полную историю. Host восстанавливается явно: unsubscribe старой подписки, subscribe(now) с новым incarnation, затем читает текущий разрешённый memory state; всё изменившееся после нового anchor остаётся в журнале. Порядок «сначала snapshot, потом subscribe(now)» запрещён: он мог бы потерять события между операциями.

## Проверка оснований и журнал действия

capture_evidence — консистентный snapshot доступных фактов. checkpoint сохраняет ссылки. begin_action в одной транзакции проверяет lease, текущие revisions, is_active(now), freshness и ACL и лишь затем ставит started. Чтение факта до транзакции и begin после него не считается проверкой.

Эта проверка фиксирует корректность на момент начала действия. Если факт поменялся после выдачи ActionPermit, строгую защиту внешнего эффекта даёт только внешняя conditional API/fencing, если она существует. Не обещать атомарность локальной БД с внешним сервисом.

Host использует `ActionPermit.idempotency_key` при внешнем вызове, затем finish_action. Сбой в любом окне сохраняет наблюдаемый статус. Для неизвестного исхода host запрашивает внешнюю систему и передаёт evidence в reconcile; ядро не считает отсутствие ответа доказательством отсутствия действия. После reconcile с recovery lease intent возвращается из blocked в running только если причиной блокировки был unknown action и все остальные guards чисты; текущая recovery lease сохраняется. Иначе intent остаётся blocked для перепланирования. Это позволяет после доказанного external success завершить checkpoint, не требуя невозможного claim blocked→running.

## Freshness и фоновое обслуживание

Freshness — отдельная policy, не retention и не valid_until. Policy привязана к slot_key, если он есть, иначе к fact_id: замена версии/ID в том же slot сохраняет policy, но делает её unverified. При set_freshness checked_at=None, verified_until=now, retry_at=now: регистрация или повторная настройка не является подтверждением. Outcome: confirmed, changed, unavailable. confirmed для точного актуального EvidenceRef ставит checked_at=now, verified_until=retry_at=now+interval. changed ставит checked_at=now, verified_until=min(previous,now), retry_at=now+interval, failure_count=0 и требует перепланирования; новое валидное значение host записывает через add_resolved с provenance. unavailable сохраняет факт, checked_at и прежний verified_until, повышает failure_count и меняет retry_at по backoff `min(3600, 30*2**min(failure_count-1,7))` seconds. last_outcome=changed/unavailable блокирует basis сразу, даже если старый verified_until ещё в будущем; retry_at отвечает только за расписание проверки. Проверка старой revision не подтверждает новую. due_verifications учитывает mismatch policy.bound_revision с текущим active slot frontier независимо от старого retry_at; новый ref становится due немедленно. Слоты с неоднозначными активными значениями не подтверждаются автоматически.

`MaintenanceBudget(max_items=100,max_model_calls=0,max_input_bytes=65536,max_wall_seconds=1.0)`; positive finite values, max_model_calls >=0. `max_wall_seconds` измеряется monotonic timer (подменяемым в тесте) и означает cooperative scheduling/deadline bound, не hard realtime guarantee для неотменяемого внешнего вызова. Без configured verifier/model возвращаются due work/proposals и явные skip counters, а не фиктивный success. Предельное число вызовов резервируется до вызова; при timeout reservation считается потраченной, чтобы retry не обходил бюджет. Дневной/денежный лимит оставляется host, не выдумывается цена токена.

```python
async def run_maintenance(
    memory: AlwaysOnMemory, *, run_key: str, budget: MaintenanceBudget,
    verifier: VerificationProvider | None = None,
    consolidator: ConsolidationProvider | None = None,
) -> MaintenanceReport: ...
```

Оба provider protocols имеют `config_key: str` (стабильная identity/config revision без credentials) и `model_cost: int >=0` — резервируемую верхнюю границу числа модельных вызовов одной invocation; bool не является допустимым int. `VerificationProvider.verify(request: VerificationRequest) -> Awaitable[VerificationResult]`; `ConsolidationProvider.propose(episodes: tuple[Fact, ...]) -> Awaitable[ConsolidationProposal]`. Стоимость резервируется до вызова; provider обязан соблюдать заявленный model_cost и не скрывать дополнительные вызовы. Счётчик отражает reservations, не выдуманное точное число реально оплаченных запросов; неиспользованный/неоднозначный резерв не возвращается. `max_input_bytes` — суммарный объём переданных provider inputs за run, включая повторные attempts. run_key связан с canonical budget и identities/config revisions обоих providers.

Порядок: due verifications активного basis → обработка выбранных старых эпизодов → GC/prune. Эпизоды: active semantic-buffer записи типа episodic, created_at не позже now−24h (существующий default consolidate_episodic), bounded batch. Подготовка proposal вне write lock; commit сверяет refs+lease generation и атомарно добавляет проверенные semantic facts и закрывает только покрытые ими эпизоды. Пустой/невалидный proposal ничего не закрывает. provenance `consolidation_ids` должна round-trip через текущую сериализацию qualifiers/Provenance. Коммит не затирает новые эпизоды foreground.

Maintenance read-only verification по injected provider не превращается в произвольный URL fetcher. Provider возвращает типизированный результат и evidence pointer. Сеть/credentials/SSRF policy принадлежат конкретному host connector. Новых connectors в этом плане нет.

asyncio.CancelledError всегда передаётся вызывающему task. Попытки сохранить unavailable/diagnostics/report при cancellation выполняются best-effort: потеря lease, смена generation или ошибка записи не подменяют cancellation обычным Conflict/result. Для обычных ошибок хранения, когда cancellation отсутствует, сохраняется явный отказ. Source IDs внутри каждого consolidation output уникальны; duplicates отвергают весь proposal до изменения facts/validity.

## Внешние интерфейсы

Python — полный контракт. HTTP/MCP/CLI используют один facade, единый JSON codec и одинаковые domain errors. Для HTTP owner привязан к `kb` приложения; shared AO в HTTP/MCP первой версии отключён, пока host явно не inject-нет pool и субъект доступа. Существующий bearer token — доступ к экземпляру, не новая multi-tenant authentication.

HTTP: `/v1/always-on/intents`, `/intents/{id}`, `/intents/{id}/{claim,renew,checkpoint,resume}`, `/subscriptions`, `/subscriptions/{id}/{poll,ack}`, `/actions`, `/actions/{id}/{begin,finish,reconcile}`, `/evidence/capture`, `/freshness`, `/freshness/due`, `/freshness/record`, `/maintenance/run`. GET list/get/due, остальные POST; unsubscribe — `DELETE /subscriptions/{id}`. `/resume` — POST, поскольку может reconcile expired lease/wait condition. Errors: 409 conflict/LeaseLost/StaleEvidence, 403 AccessDenied, 410 CursorExpired, 422 invalid input, 501 unsupported backend. Собственный auth остаётся 401. Ни один route не принимает owner/namespace вне scoped selector contract.

Wire JSON conditions: `{"kind":"until","at":"2026-10-26T10:00:00Z"}`, `{"kind":"slot","namespace":"owner","slot_key":"ci::status","value_text":"passed"}`, `{"kind":"change","ref":{"namespace":"owner","fact_id":"f1","revision":"..."}}`. capture_evidence input refs — массив пар `[namespace,fact_id]`. HTTP success — raw encoded public object с 200, ack/unsubscribe — 204 без body; MCP success для этих двух методов — JSON null. InvalidDelivery — 409; malformed/oversize JSON — 422 с code InvalidInput. Missing и foreign owner-scoped objects возвращают AccessDenied (403), не раскрывая существование чужого объекта. Lease/delivery tokens никогда не копируются в error message.

MCP tools: `intent_create/get/list/claim/renew/checkpoint/resume`, `memory_subscribe/poll/ack/unsubscribe`, `action_prepare/begin/finish/reconcile`, `memory_evidence_capture`, `memory_freshness_set/due/record`, `memory_maintenance`. Ввод через typed JSON dictionaries проверяется тем же codec; ошибки — structured error code, не stacktrace. capabilities сохраняет array contract и добавляет enabled tools с описанием supported backends.

CLI group `ai-knot work`: create/list/resume/claim/checkpoint; `ai-knot changes`: subscribe/poll/ack/unsubscribe; `ai-knot maintenance run` — один batch. Все команды поддерживают `--json`, не содержат вечного daemon loop. Остальные операции доступны Python/HTTP/MCP, а не дублируются десятками CLI flags.

TypeScript: отдельный `HttpAlwaysOnMemory` поверх существующего transport; `client.alwaysOn` у HTTP и MCP клиентов, без фонового polling при constructor. Публичные types экспортируются из npm/src/index.ts. Поддержка новых методов не зависит от Vercel/другого framework.

## Проверка результата и критерии приёмки

Проверять через public interfaces и реальный SQL. Барьеры/Events управляют гонками; fake clock управляет expiry/backoff/debounce; sleeps не имитируют дни. SQLite: отдельные connections и реальные child processes. PostgreSQL: live service в CI, не mock psycopg как доказательство atomicity. Каждый тест получает отдельный PG schema через DSN search_path; introspection migrations ограничена current_schema().

Обязательная fault matrix:

| Сбой / ввод | Ожидаемый результат | Задача |
|---|---|---|
| Падение между fact write и journal write | Откат обоих либо commit обоих | P1.3 |
| PostgreSQL seq выделен, commit задержан | Consumer не пропускает поздно committed изменение | P1.3 |
| Recall завершился после конкурентной коррекции | Коррекция сохраняется; обновляется только telemetry | P1.2 |
| Сбой после poll до ack | Тот же delivery после restart | P1.4 |
| Отозван доступ между poll и replay | Закрытые payload/IDs не возвращаются | P1.4/P2.2 |
| Два worker претендуют на intent | Одна действующая lease; старый token fenced | P2.1 |
| Fact изменён после планирования | begin_action блокируется | P2.2 |
| Сбой после external success до finish | unknown → reconciliation; эффект не повторяется вслепую | P2.3 |
| Внешний источник недоступен | Факт не удалён; проверки не считаются успешными | P2.4 |
| Consolidation сталкивается с foreground update | Proposal не затирает новое состояние | P3.1 |
| Старый snapshot восстановлен | Facts changed events есть; action/cursor history не откатывается | P3.5 |
| Бюджет закончился / провайдер timeout | Нет незарезервированного следующего вызова; отчёт объясняет остановку | P3.1 |

Приёмочный deterministic сценарий: intent ждёт CI → процесс остановлен → CI и deployment basis меняются → restart доставляет изменения → старый план blocked → host фиксирует новый basis → действие с external idempotency key → crash before finish → reconcile → completed. Вторая ветка использует внешний API без idempotency: правильный результат — blocked/unknown, а не обещание отсутствия дублей.

Binding metrics на фиксированных сценариях: missing committed changes=0; stale action starts=0; accepted stale lease mutations=0; unintended duplicate effects=0 у idempotent fake external API; false auto-retries unknown actions=0; cross-owner/scope leaks=0; deterministic task completion=100% после предоставленной host reconciliation. Это software invariants на fixtures, не статистическая гарантия всем внешним системам.

Performance report: 1k/10k фактов, 1/10/100 подписок, restart replay, p50/p95 poll/resume/write, scanned events, delivered batches и receipt growth. На одном runner сравнивать enabled/disabled с baseline; цель <=20% p95 overhead для legacy recall с AO disabled — первоначальный regression budget, требующий проверки, не обещанный результат. Проверить SQLite query plan по `(namespace,seq)`, owner/state, retry_at; event scan bounded. Набор 100k events — отдельный slow test, без абсолютных wall-time требований в flaky CI.
