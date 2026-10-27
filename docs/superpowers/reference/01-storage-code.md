# P1 — конкретный storage/journal contract

Дата: 2026-10-26. Статус: архивный implementation blueprint и отдельная SQLite модель. Production реализован; актуальные проверки — [implementation-final.md](../implementation-final.md). [Spec](../specs/2026-10-26-always-on-memory-design.md) задаёт public API; [P1 plan](../plans/2026-10-26-always-on-01-storage.md) задаёт порядок commits. Ни одна будущая проверка PostgreSQL ниже не объявляется выполненной.

Запуск standalone reference:

```bash
.venv/bin/python docs/superpowers/reference/always_on_storage_reference.py
.venv/bin/ruff check docs/superpowers/reference/always_on_storage_reference.py
.venv/bin/ruff format --check docs/superpowers/reference/always_on_storage_reference.py
```

`Store`/`Fact`/`Batch` в reference — уменьшенные типы примера. Не импортировать их в `src/`, не публиковать как Python facade и не подменять ими integration fixtures. Reference использует временную SQLite DB и стандартную библиотеку. Имена production SQL ниже окончательные; reference таблица `facts(namespace,id,payload)` намеренно не копирует 37-column serializer AI Knot.

## Точные internal signatures

Создать `always_on/storage.py`; `StorageBackend` и current `AtomicUpdateCapable` не расширять обязательными методами. `StorageBackend` импортировать для capability check на SQL backend. `SQLScalar` допускает только значения driver binding, domain JSON сериализуется заранее. Fixed statements объявляются кодом, не передаются внешним запросом.

```python
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

SQLScalar: TypeAlias = str | int | float | bytes | None
SQLRow: TypeAlias = dict[str, SQLScalar]

@dataclass(frozen=True)
class SQLStatement:
    sqlite: str
    postgres: str

class AlwaysOnSession(Protocol):
    def load_facts(self, namespace: str) -> list[Fact]: ...
    def replace_facts(self, namespace: str, facts: list[Fact]) -> None: ...
    def fetch_one(self, statement: SQLStatement,
                  params: Sequence[SQLScalar] = ()) -> SQLRow | None: ...
    def fetch_all(self, statement: SQLStatement,
                  params: Sequence[SQLScalar] = ()) -> list[SQLRow]: ...
    def execute(self, statement: SQLStatement,
                params: Sequence[SQLScalar] = ()) -> int: ...

@runtime_checkable
class AlwaysOnStorageCapable(Protocol):
    def always_on_transaction(
        self, namespaces: tuple[str, ...], *, write: bool
    ) -> AbstractContextManager[AlwaysOnSession]: ...
```

`Fact` импортируется из `ai_knot.types`; методы session не открывают connections и не делают commit. `execute` возвращает driver rowcount (для DML — изменённые rows; `-1` для команд без count допустим и не используется как success marker). `fetch_*` возвращают копии mappings по SQL column names; PG использует existing psycopg optional package, cursor description/dict_row, не новый dependency. `replace_facts` проверяет membership namespace в lock set и `write=True`; нарушение — internal programming error. Наружу native connection не выдаётся. Read-only session не исполняет write statements; caller выбирает только заранее объявленные read statements, backend transaction дополнительно выставляет PG READ ONLY / SQLite `PRAGMA query_only=ON`.

Существующий shared helper в `storage/base.py`:

```python
T = TypeVar("T")
def mutate_facts(storage: StorageBackend, agent_id: str,
                 fn: Callable[[list[Fact]], tuple[list[Fact], T]]) -> T: ...
```

`mutate_facts` передаёт existing `atomic_update` wrapper, который сохраняет callback result локально и возвращает только новый список. Helper отдаёт `T` после успешного commit; нет второго load и нет лишнего save после callback. Non-SQL fallback — прежний load/transform/save с single-writer semantics. Никакой callback retry внутри helper: random IDs, candidate creation и side effects не должны повторяться скрыто. P2 dependency-lock retry — отдельный bounded protocol с полным re-read, максимум 3 transactions.

Для новых AO timestamps SQL session выдаёт нормализованные UTC ISO строки с фиксированными microseconds: SQLite хранит их как TEXT, PostgreSQL datetime из TIMESTAMPTZ преобразуется при построении SQLRow. Это сохраняет общий SQLScalar contract; domain row codec затем восстанавливает aware datetime. Нативный PG datetime не должен незаметно попасть в dict, объявленный как SQLRow со скалярными строками/числами.

## Production schema, migration и sequence

Использовать текущие `facts` / `"ai-knot_facts"` и existing serializers. Additive AO migration запускается при инициализации SQL backend, но пустые AO tables **не** означают включённый namespace. Нет записи журнала до persisted `enabled=1`.

| SQLite table | Колонки / constraints |
|---|---|
| `ao_schema` | `id INTEGER PRIMARY KEY CHECK(id=1)`, `version INTEGER NOT NULL`; singleton migration marker |
| `ao_streams` | `namespace TEXT PRIMARY KEY`, `enabled INTEGER NOT NULL CHECK(enabled IN(0,1))`, `head_seq INTEGER NOT NULL DEFAULT 0 CHECK(head_seq>=0)`, `min_retained_seq INTEGER NOT NULL DEFAULT 1 CHECK(min_retained_seq>=1 AND min_retained_seq<=head_seq+1)` |
| `ao_changes` | `(namespace TEXT, seq INTEGER)` composite PK; `kind TEXT NOT NULL`, `fact_id TEXT NOT NULL`, `slot_key TEXT NOT NULL`, `before_slot_key TEXT NOT NULL DEFAULT ''`, `before_revision TEXT NULL`, `after_revision TEXT NULL`, `old_access_json TEXT NULL`, `new_access_json TEXT NULL`, `acl_subject TEXT NULL` |
| `ao_subscriptions` | `(owner TEXT,id TEXT)` PK; `incarnation TEXT NOT NULL`; `namespace TEXT NOT NULL`, `selector_json TEXT NOT NULL`, `cursor INTEGER NOT NULL CHECK(cursor>=0)`, `debounce_seconds REAL NOT NULL CHECK(debounce_seconds BETWEEN 0 AND 3600)`, `pending_since TEXT NULL`, `inflight_token TEXT NULL`, `inflight_through_seq INTEGER NULL`, `inflight_events_json TEXT NULL`, `inflight_resync INTEGER NULL`, `inflight_acl_digest TEXT NULL` |
| `ao_delivery_receipts` | `(owner TEXT, subscription_id TEXT, incarnation TEXT, delivery_id TEXT)` PK; `through_seq INTEGER NOT NULL`; confirmations retained until unsubscribe |

Все non-PK columns таблицы receipts NOT NULL. В subscription либо все inflight columns NULL, либо все заполнены (ACL digest для private — SHA-256 пустого набора grants); enforce CHECK. `pending_since` timezone-aware UTC text. `kind` CHECK IN `('insert','update','delete','resync_required','acl_changed')`. Служебные события используют empty fact_id/slot_key и NULL revisions. Изменение факта сохраняет old/new scopes **и** origin, visibility, topic_channel, memory_tier в access JSON, но ни content, ни snippets/surfaces. Access JSON — fixed shape internal data. Нет FK на удаляемые facts. На subscriptions индекс `(namespace,cursor)` для retention; changes composite PK уже обеспечивает `(namespace,seq)` scan. Добавить changes index `(namespace,fact_id,seq)` для проверки последней projection удалённых фактов при replay; другие индексы только под P2/P3 queries.

P1.3 мигрирует AO schema до version=1 (marker, streams, changes). P1.4 до version=2 (subscriptions, receipts). Marker обновляется в той же DDL transaction после успешных statements. Более новая неизвестная schema version — fail closed с понятной unsupported-schema error, без изменения marker. SQLite migration сериализуется `BEGIN IMMEDIATE`; PG — отдельным stable advisory migration lock до проверки marker. Все AO миграции additive и повторяемые; facts column upgrades сохраняют все существующие поля и применяются до enable. Не оборачивать SQLite `executescript` внутри уже открытой transaction: он может закоммитить её; выполнять отдельные фиксированные statements, как reference.

PostgreSQL counterpart: таблицы `"ai-knot_ao_schema"`, `"ai-knot_ao_streams"`, `"ai-knot_ao_changes"`, `"ai-knot_ao_subscriptions"`, `"ai-knot_ao_delivery_receipts"`; schema `current_schema()` из configured search_path. `seq/head_seq/cursor/through_seq` — BIGINT, flags — INTEGER с теми же CHECK, `debounce_seconds` — DOUBLE PRECISION, JSON/timestamps — TEXT для единообразного codec. Не использовать SERIAL/IDENTITY/nextval. `min_retained_seq` — first retained boundary, не последний удалённый seq. Receipt growth не bounded constant space: очистка только при unsubscribe в v1; отдельная будущая retention policy потребует изменения ack contract.

Для stream append под namespace lock:

```sql
-- SQLite, ? bindings; PG использует quoted table и %s bindings
UPDATE ao_streams SET head_seq = head_seq + ? WHERE namespace = ? AND enabled = 1;
SELECT head_seq FROM ao_streams WHERE namespace = ?;
-- при N events назначить [head-N+1, ..., head] и INSERT каждого event
```

Никакого промежуточного commit. Записывать только если `N>0`; стабильный diff order — sorted fact IDs. Head update можно делать один раз на весь diff, а reference делает по одному event: invariant одинаковый. Explicit `save([])` и `delete` сохраняют tombstones; missing delete даёт 0 events. Restore сравнивает current facts с snapshot и добавляет новые events; snapshots не содержат/не заменяют AO tables. `enable_stream` выполняет INSERT stream и один `resync_required` при уже существующих фактах в одной transaction; пустой stream head=0. Повторное enable ничего не меняет. No historical event fabrication.

### PostgreSQL locking — обязательная production реализация

Ключ должен совпадать у всех SQL entry points:

```python
def namespace_lock_key(namespace: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(namespace.encode("utf-8"), digest_size=8,
                        person=b"ai-knot-ao-v1").digest(),
        "big", signed=True,
    )
```

В READ COMMITTED transaction: `SELECT pg_advisory_xact_lock(%s)` для sorted unique namespace names; затем load facts/ACL/state; callback; `_save_on_connection`; events/head; commit. Включать lock даже для namespace без rows. `save_grant/revoke_grant` берут `__shared__` lock, читают существующий grant и append addressed ACL event только при реальной смене. Не делать предварительный namespace read на старом snapshot до ожидания lock. Для P2 lock set включает owner и все dependency namespaces: discover set снаружи, lock в sorted order, re-read intent/action; если состав изменился — rollback и новая attempt, максимум 3; не брать дополнительный lock в произвольном порядке внутри старой transaction.

Read-only snapshot — одна REPEATABLE READ READ ONLY transaction, без nested sessions. Writers держат head lock до commit: пока T1 с seq=1 не committed, T2 не сможет выделить seq=2 в том же namespace. Consumer с уже открытым snapshot видит старый head=0 и не может перепрыгнуть T1. PG sequence counter, даже под event INSERT ordering, этого не доказывает.

Live PG acceptance: T1 append, pause до commit; T2 writer ожидает namespace lock; independent poll не видит T1; release T1, T2 commit; poll+ack получает оба последовательных события. Повторить T1 rollback и проверить отсутствие consumed head/gap. Отдельно два subprocess writer к пустому namespace. Test schemas уникальны; исправить текущую information_schema query в `_migrate_db`: `table_schema=current_schema() AND table_name='ai-knot_facts'`. Mock psycopg не считается этим тестом.

## Semantic fingerprint и точные mutation patches

`evidence_revision(Fact)` включает ровно следующие Python поля:

```python
(
    "content", "type", "slot_key", "entity", "attribute", "value_text",
    "canonical_surface", "witness_surface", "prompt_surface", "qualifiers",
    "source_snippets", "source_spans", "source_verbatim", "supported",
    "support_confidence", "verification_source", "state_confidence",
    "valid_from", "valid_until", "event_time", "origin_agent_id", "visibility",
    "visibility_scope", "claim_key", "topic_channel", "memory_tier",
)
```

Не входят `id` (отдельный адрес), `created_at` (история вставки), `access_count/last_accessed/access_intervals`, `retention_score`, `importance`, `tags`, `version`, `mesi_state`, transient `op`. `qualifiers` включается целиком; поэтому lineage promotion meaningful. Нет Unicode/whitespace normalization content; JSON keys sorted, списки ordered; equivalent aware datetime нормализуется в UTC microseconds. Unknown enum и NaN/Infinity отвергаются до fact write. Старый serializer может содержать empty `valid_from`; его backfill должен стать стабильным один раз в migration, а не каждый load `datetime.now()`, иначе revision меняется от чтения. В SQLite `_fact_from_row` и PG `_fact_from_row` сейчас есть wall-clock fallback: перед enable миграция заменяет `valid_from=''` на persisted `created_at`, нормализованный через existing datetime parser. Не изменять старые nonempty даты без round-trip теста.

| Current location (method — устойчивый anchor) | Exact patch |
|---|---|
| `storage/sqlite_storage.py:SQLiteStorage._conn`, `atomic_update`, `save_atomic`, `_execute_save`, `load`, `delete` | Один outer transaction context, read/write mode. Existing `BEGIN EXCLUSIVE` atomic_update заменить `BEGIN IMMEDIATE`. Из `load` вынести `_load_on_connection`; из `_execute_save` — `_save_on_connection`; `delete` делает current-list removal внутри lock. `_build_rows`, `_fact_from_row`, `_SELECT_COLS` сохраняют порядок колонок. |
| `storage/postgres_storage.py:PostgresStorage._execute_save`, `save`, `save_atomic`, `load`, `delete` | Вынести connection-local helpers, добавить отсутствующий `atomic_update`; advisory lock до load; убрать `conn.commit()` из helpers. Не открывать `_get_conn()` повторно из callback. |
| `storage/base.py:AtomicUpdateCapable` | Добавить generic `mutate_facts`; existing protocol сигнатура без несовместимого изменения. |
| `knowledge.py:KnowledgeBase.add`, `add_many`, `add_episodic` | Pure candidate preparation снаружи; dedup/reference-window + append внутри `mutate_facts`. Сохранять parent/child dedup и original return Fact; не менять retrieval strategy. |
| `knowledge.py:KnowledgeBase._execute_recall` (current final full save) | Заменить на `_record_access(returned_ids, access_time)`; callback обновляет current telemetry только по существующим IDs. Никогда не возвращать stale content в storage. Max 20 intervals; не перемещать last_accessed назад. |
| `knowledge.py:replace_facts`, `forget`, `decay`, `clear_all`, `restore` | Replacement operations остаются replacements под writer lock; decay над current facts в callback; `restore` читает snapshot отдельно, затем explicit replacement+journal. Snapshot read не вызывает модель. |
| `learning.py:_LearningMixin.learn`, `add_resolved`, `_commit_phase` | Создать `_resolve_and_commit(verified: list[Fact], conflict_threshold: float) -> list[Fact]`; _resolve_phase + _consolidate_phase получают current facts внутри lock. `_commit_phase(existing,to_insert)` больше не сохраняет stale list; удалить/переименовать private helper и обновить обоих callers. |
| `learning.py:learn_async` | Embedding вне lock. Closure proposal хранит revisions old/new IDs. Compare-and-close в callback; changed/missing refs skip. Не перезаписывать `existing` snapshot из await. |
| `learning.py:consolidate_episodic` | P1 только guarded closing current episodic refs, чтобы не было stale whole save. P3 заменяет legacy two-phase method полным atomic proposal commit; P1 не объявляет consolidation transaction complete. |
| `pool.py:SharedMemoryPool._publish_locked`, `promote`, `gc_pool` | Все changes через `mutate_facts`; текущие supersession, version, claim policy сохраняются. Mutable trust/usage counters после commit; journal отдельный от trust/usage ledger. |
| `_pool_recall.py:_PoolRecallMixin.recall` auto-promote block | По current ID/tier выполнить promotion в callback. Не resurrect superseded/GC'd rows. |
| `pool.py:grant_read` и новый `revoke_read(agent_id,scope)`; SQL save_grant/revoke_grant | Durable grant write и addressed ACL event в одной shared transaction. `grant_read` in-memory cache обновляет после commit. При enabled shared stream current recall refreshes durable grants; `_scope_visible` semantics сохраняются. |

Не считать `save_atomic` full-snapshot API автоматическим merge: тот, кто намеренно передаёт целый список, всё ещё может заменить namespace. Все managed RMW callers обязаны использовать mutate/session. Raw external SQL, старый AI Knot binary и YAML не имеют journal guarantees. Rollout: upgrade всех writers, проверить migrations/backup, затем явно enable. Запретить запуск старых writers после enable; DB не может обнаружить произвольный внешний SQL.

## Poll/ack: пошаговый production algorithm

Owner всегда `kb._agent_id`, не request JSON. Private selector namespace должен равняться owner. Единственное другое разрешённое значение `__shared__` требует injected pool на том же backend/path или DSN+schema и `persist_stats=True`; запретить произвольный private namespace другого агента. Canonicalize fact_ids/slot_keys sorted unique; matching OR, empty tuple pair означает весь доступный namespace. Selector/debounce existing ID immutable; повтор identical subscribe возвращает текущую запись без reset. Constructor включает private stream; shared stream включается после первой успешно validated subscribe/capture.

1. Validate before write: limit integer 1..500 (bool reject), default100; debounce finite 0..3600, no bool; IDs непустые; `now=clock()` aware UTC. Subscribe stores current head (`now`) или 0 (`beginning`) под lock. Если cursor ниже `min_retained_seq-1`, `CursorExpired` без subscription insert.
2. Poll берёт owner+selected namespace locks. Re-read subscription и live grants в transaction. Проверить expired cursor **до** inflight replay, включая forced prune. Production missing/foreign owner-scoped subscription — AccessDenied (403); ack с неизвестным/устаревшим delivery token при существующей подписке — InvalidDelivery (409). Уменьшенный SQLite reference использует KeyError/InvalidDelivery локально и не является wire/error adapter. Tail без rows и без inflight возвращает `ChangeBatch(None,(),False)` без subscription update.
3. При inflight: replay сохранённых events только после current access checks, без добавления новых IDs, даже если новый limit больше. Изменился digest durable grants либо хотя бы один ранее выданный ID теперь закрыт — заменить **весь** persisted batch на empty events/resync_required=True с прежним token/through boundary; редактирование sticky до ack. Живой факт проверяется по текущей projection. Если факта больше нет, прочитать **последний** journal event для этого ID до current head: его new projection, либо old projection tombstone, задаёт актуальную границу доступа. Нельзя доверять только old projection первоначального inflight event: global insert мог затем стать private и быть удалён. Нужная поздняя запись удерживается, пока cursor не ack; force-pruned cursor expires до replay. Если достаточной projection нет, fail closed в generic resync. Это консервативное редактирование не возвращает закрытые IDs. Raw facts для HTTP payload не загружаются/не выводятся.
4. Без inflight читать `WHERE namespace=? AND seq>cursor ORDER BY seq LIMIT 1000`. Matching учитывает `before_slot_key` и `slot_key`. Subject ACL events видит только их subject; им соответствуют empty events+resync flag. Baseline тоже resync. Для факта: current grants + old/new projections определяют доступ; обе скрыты — skip; любое сужение, при котором новая projection закрыта после доступной старой, — только общий resync без ID. Обычный доступный insert/update/delete возвращает FactChange без текста. Deletion использует old projection. Re-evaluate current fact scope (либо latest tombstone projection как в шаге 3) перед возвратом исторического event, чтобы не выдать ID факта, который уже закрыт scope change позже текущего scan window.
5. Коалесцировать по fact_id: последняя kind/seq/after_revision и самая ранняя before_revision; service resync не расходует limit. При новом distinct visible ID сверх limit остановиться **до** его row, не продвигая through_seq через него. Raw scanned максимум1000. Hidden rows до/после видимого продвигают tentative through; distinct superseded old/new IDs не схлопываются в один slot. Не применять окончательный net-no-op drop к delete: invalidation должна остаться.
6. Если есть matching event/resync и debounce>0: `pending_since` устанавливается при первом наблюдении и persists; до due возвращается empty tokenless batch, cursor остаётся. После due rescanning bounded window coalesces latest rows. Hidden-only raw page получает token немедленно, чтобы можно было ack scan boundary; empty tail не начинает debounce. Pending window не extends при каждом poll.
7. Сохранить случайный opaque token, through_seq, projected events JSON, resync flag и current grants digest в той же transaction. Подтверждённый cursor не меняется. Ответ выдаётся после commit. Пустая отфильтрованная страница также сохраняется как inflight с token; полностью пустой tail — без token.
8. Ack под теми же locks: проверить owner/id/incarnation. Token уже в receipts — no-op, включая receipt A после ack B. Token равен current inflight — INSERT receipt, cursor=inflight_through, clear all inflight columns атомарно. Иначе InvalidDelivery без cursor mutation. Полезный failure injection — после INSERT receipt до cursor UPDATE: rollback обоих, restart всё ещё replay прежний batch.
9. Unsubscribe атомарно удаляет subscription и её receipts. Создание с прежним ID выдаёт новый incarnation. Cleanup journal ограничивается `MIN(cursor)` живых subscriptions namespace или head при отсутствии. Force prune повышает min_retained_seq; следующий poll отставших даёт CursorExpired, не перескакивает. Ack уже выданного token подтверждает обработку, но не обещает доступ к удалённой истории; последующий poll всё равно проверяет boundary.

Current `_scope_visible` rule — `visibility_scope or 'global'` публичен при `global`; иначе origin owner или durable grant на named scope. `visibility`, topic, tier сохраняются как evidence/projection, но эта версия не придумывает новый ACL policy поверх действующего pool rule. Отсутствие matching shared facts не означает запрет создать selector; пустой безопасный feed допустим. Все операции с scope snapshot и facts выполняются на одной connection.

64KiB — bound caller-supplied JSON field, а не сумма events в batch: persisted batch до500 events может быть больше64KiB. Reference deliberately `bounded=False` только для internal batch encoding. Не ослаблять валидацию progress/request/result/selector. Reference дополнительно ограничивает целый reduced Fact64KiB для небольших fixtures; это не новый лимит на existing production Fact API.

## Что проверяет executable reference

Фактически выполнено 2026-10-26: `.venv/bin/python docs/superpowers/reference/always_on_storage_reference.py` — `Ran 9 tests`, `OK`; Ruff check — `All checks passed!`; Ruff format --check — `1 file already formatted`; `compile(source, path, 'exec')` — syntax OK; `git diff --check` — exit 0. Production test suite и live PostgreSQL в этой documentation-only задаче не запускались.

9 unittest cases внутри файла проверяют: rollback callback и fault после facts/после event+head; telemetry suppression и semantic changes; tombstone; persisted enable/baseline; restart replay и old ack receipts; owner binding и ID incarnation; scan1000 с empty ackable page и limit immutable replay; coalescing old revision и old-slot matching; persistent debounce с fake UTC clock; safe/forced prune; SQLite writer lock timeout без partial facts/seq; invalid JSON/nonfinite/time/limit inputs. Connections пересоздаются при каждой transaction, restart — новый Store к тому же DB. SQLite WAL test вручную держит uncommitted writer, independent read видит старый head, второй writer получает bounded database-locked error; после commit retry получает следующий seq. SQLite busy timeout не является универсальной retry policy и не проглатывается.

Reference **не** проверяет: actual AI Knot serializers/callers, subprocess crash/fsync/power loss, real shared ACL/revocation, AO session PG implementation, PostgreSQL commit order/deadlocks/migrations, every evidence field separately, HTTP/MCP/CLI, retention performance/100k dataset, mypy strict production и coverage80%. `Clock` здесь injected function; temporal payload содержит ISO strings, production адаптер берёт datetime из Fact. `_record_access`, add/learn pipeline и snapshot adapter ещё предстоит реализовать и прогнать тестами P1. No models, schedulers, connectors or new dependencies.

Release gate остаётся P1: real SQLite multi-process + live PostgreSQL, all managed mutators и rollback injection, mixed-writer rollout constraint, existing regression suites, Ruff/mypy/coverage. Прохождение этого файла доказывает только сформулированные invariants уменьшенной SQLite модели.
