# Always-on 01 — Storage and Change Delivery Implementation Plan

**Execution status (2026-10-27): production work complete.** See [implementation and final verification](../implementation-final.md). The original checklist below is retained as the implementation brief; per-task commits were not created.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать атомарные изменения памяти и устойчивые подписки с подтверждением обработки.

**Architecture:** Сначала устранить read-modify-write races в managed Python API. Затем добавлять knowledge-change events в той же SQL-транзакции, что и facts, и доставлять через owner-scoped подписки. Новый журнал независим от legacy Fact.version и trust/usage ledger.

**Tech Stack:** Python 3.11+, sqlite3, существующий psycopg extra, dataclasses, pytest; без broker/ORM.

**Spec:** [2026-10-26-always-on-memory-design.md](../specs/2026-10-26-always-on-memory-design.md).

**Historical brief (2026-10-26):** исходный план до production-реализации; текущая приёмка приведена в implementation-final.md. Проверяемый stdlib SQLite prototype и точные SQL/integration решения: [01-storage-code.md](../reference/01-storage-code.md), [always_on_storage_reference.py](../reference/always_on_storage_reference.py). Прототип не импортирует AI Knot, не заменяет production и не доказывает PostgreSQL parity. Чекбоксы ниже сохраняют исходный brief; статус выполнения отражён в итоговом отчёте.

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

- Старый recall сохраняет telemetry после чужой коррекции — correction не теряется: P1.2 `test_recall_cannot_restore_stale_content`.
- PostgreSQL выдаёт seq раньше commit — consumer не перескакивает незавершённую запись: P1.3 `test_postgres_commit_order_no_gap`.
- Изменение записано через delete/restore/GC, а не add — появляется соответствующее событие: P1.3 `test_all_fact_mutators_emit_changes`.
- Все первые 1000 событий скрыты ACL — consumer может двигаться дальше без утечки: P1.4 `test_filtered_page_can_be_acked`.
- ACL отозван после poll до повторной доставки — replay безопасен: P1.4 `test_replay_after_acl_revoke_is_redacted`.

## File map

| Файл | Ответственность |
|---|---|
| Modify `src/ai_knot/storage/base.py` | shared `mutate_facts`, optional AtomicUpdateCapable contract |
| Modify `src/ai_knot/storage/{sqlite_storage,postgres_storage}.py` | единый transaction-local fact writer, locks, AO SQL seam, migration |
| Modify `src/ai_knot/{knowledge,learning,pool,_pool_recall}.py` | все managed mutations и telemetry используют безопасный путь |
| Create `src/ai_knot/always_on/{__init__,models,storage,changes,memory}.py` | contracts, SQL seam, diff/ACL projection, subscriptions facade |
| Create `tests/always_on/conftest.py` | fake Clock, isolated SQL fixtures, fresh instance factory |
| Create `tests/always_on/test_{atomicity,mutations,changes,subscriptions}.py` | public behavioral tests и transactional failure cases |

## Зафиксированные решения перед P1.1

1. `StorageBackend` остаётся четырёхметодным. `AtomicUpdateCapable` остаётся optional с текущей callback signature `Callable[[list[Fact]], list[Fact]]`. `mutate_facts` имеет generic `T = TypeVar("T")`; result возвращается только после успешного commit. YAML fallback не получает durable гарантий.
2. Одна backend transaction управляет load, transform, фактическим replace, diff, append и commit. `_save_on_connection` — единственная физическая запись facts; публичные `save/save_atomic/delete/atomic_update` и session `replace_facts` обязательно доходят до неё. Она получает `before` под lock и делает diff **один раз**. Snapshot tables, trust/usage ledger и pool_stats не проходят fact diff.
3. SQLite `BEGIN IMMEDIATE`, PostgreSQL READ COMMITTED + `pg_advisory_xact_lock(namespace_key)` до первого чтения mutable rows. Для read snapshot: SQLite `BEGIN`, PostgreSQL REPEATABLE READ; операция возвращает результат после завершения transaction. Session не вызывает публичный backend API, открывающий другую connection.
4. Ключ PG: `int.from_bytes(hashlib.blake2b(namespace.encode("utf-8"), digest_size=8, person=b"ai-knot-ao-v1").digest(), "big", signed=True)`. Везде одинаковый алгоритм, включая legacy writers и ACL writers. Lock set — sorted unique namespaces; пустая строка запрещена. Dependency lock discovery/re-read/retry ограничен 3 полными transactions; затем `RevisionConflict` (P2 использует этот же seam).
5. AO constructor явно включает private namespace; валидный shared subscribe/capture сначала проверяет идентичность SQL store и `persist_stats=True`, затем включает `__shared__` в transaction. Не включать shared stream до permission/config validation. Constructor YAML поднимает `UnsupportedAlwaysOnBackend`, без нового SQLite файла.
6. DDL и SQL constants из [reference notes](../reference/01-storage-code.md) — внутренняя реализация. Public facade signatures остаются ровно как в spec; reference `Store` — отдельная уменьшенная модель.
7. `min_retained_seq` — первая доступная последовательность (при пустом хвосте `head_seq+1`). Cursor expired iff `cursor < min_retained_seq-1`. `beginning` при уже pruned истории возвращает `CursorExpired`; восстановление — unsubscribe старой подписки, новая subscribe `now`, затем чтение текущего разрешённого состояния. Новый anchor фиксируется до snapshot, чтобы изменения между операциями оставались в feed; без silent jump.
8. Ack receipts хранятся до unsubscribe, включая старые подтверждения. `(owner,id)` имеет случайный persisted incarnation: повторно созданный ID не принимает старые tokens. Рост receipts линейный по числу ack; constant-space гарантий нет. Нельзя хранить только `last_ack`, иначе старый valid duplicate перестанет быть идемпотентным.
9. Evidence digest — SHA-256 UTF-8 canonical JSON: sorted keys, separators `(',', ':')`, `ensure_ascii=False`, `allow_nan=False`. Datetimes приводятся к aware UTC `isoformat(timespec='microseconds')`, enum — `.value`, lists сохраняют порядок, qualifiers сортируются как object. `id` адресует факт и не входит в digest; duplicate IDs в одном namespace reject до DELETE. Поля и exclusions перечислены в reference notes; `memory_tier/topic_channel` входят, promotion является meaningful change.
10. Journal сохраняет `before_slot_key` и old/new ACL projection, чтобы смена slot/scope не скрыла invalidation от прежних watchers. Public event использует `slot_key` нового факта (при delete — старого); service event `fact_id=slot_key=""`, `resync_required` в batch без фактического ID. Raw content не хранится.

## Task P1.1: единая атомарная операция SQL

**Files:** Modify `src/ai_knot/storage/base.py`, `sqlite_storage.py`, `postgres_storage.py`; Create `tests/always_on/conftest.py`, `test_atomicity.py`.

**Interfaces:**
- Consumes: `AtomicUpdateCapable.atomic_update(agent_id, fn) -> None`, существующие fact serializers.
- Produces: PostgreSQL реализует этот же метод; `mutate_facts(storage: StorageBackend, agent_id: str, fn: Callable[[list[Fact]], tuple[list[Fact], T]]) -> T` в `storage/base.py`.
- Backend helpers: `_load_on_connection(conn, agent_id: str) -> list[Fact]`, `_save_on_connection(conn, agent_id: str, facts: list[Fact]) -> None`; существующие save/save_atomic/delete/atomic_update вызывают их внутри одной transaction и namespace lock. В PostgreSQL фиксировать advisory lock до load, включая пустой namespace. Конкретные current methods и последовательность выноса — reference notes, таблица integration patches.

- [ ] **Step 1: добавить failing behavior tests.** SQL `store_factory` создаёт независимые storage instances к одному временному DB/schema; YAML не параметризует durable suite. Для PostgreSQL fixture использовать `AI_KNOT_TEST_PG_DSN`, уникальный schema и search_path; локально skip при отсутствии DSN, отдельный CI job обязан иметь DSN. В conftest определить fake `Clock` с UTC `now` property, `__call__() -> datetime`, `advance(*, seconds: float) -> None`; фиксированный Fact fixture задаёт одинаковые created_at/last_accessed/valid_from. Не оставлять реальные now() в тестовых фактах с historical Clock.

```python
def test_atomic_callback_rolls_back(store_factory):
    store = store_factory()
    store.save("owner", [Fact(id="f1", content="old")])
    def broken(facts):
        facts[0].content = "new"
        raise RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        store.atomic_update("owner", broken)
    assert store_factory().load("owner")[0].content == "old"
```

Также: `test_two_process_appends_keep_all_20_facts` — 2 subprocess writers ×10 appends через atomic_update, 20 уникальных IDs; `test_empty_namespace_lock_serializes_first_writes`; `test_postgres_schema_introspection_is_isolated`. Гонки управлять Barrier/Event, каждый child завершать с timeout и проверять exitcode.

- [ ] **Step 2: выполнить** `.venv/bin/python -m pytest tests/always_on/test_atomicity.py -q --no-cov`. На исходной базе PostgreSQL atomic_update отсутствует; ошибка должна быть по новому контракту, не из-за fixtures/import setup.
- [ ] **Step 3: реализовать helpers/locks/mutate_facts.** `mutate_facts` оборачивает existing atomic callback, удерживает `(new_facts, result)` локально и отдаёт result лишь после возврата `atomic_update`; исключение не проглатывает. Для YAML fallback существующий single-writer load/transform/save. Выделить SQLite `_build_rows/_fact_from_row/_SELECT_COLS` и PG `_build_rows/_fact_from_row/_SELECT_COLS` без изменения column order. Убрать внутренние commit из connection-local helpers; outer context отвечает за rollback даже если callback/serializer/event insert бросает exception. `_migrate_db` PostgreSQL фильтрует `table_schema=current_schema()`; PostgreSQL `executemany` вызывается через cursor. Полный replacement `save` сериализует writers, но намеренно заменяет весь namespace и не обещает merge заранее загруженного stale snapshot.
- [ ] **Step 4: повторить targeted tests и существующие** `tests/test_sqlite_storage.py tests/test_storage_compat.py tests/test_concurrent.py tests/test_pool_field_parity.py` с `--no-cov`. В PG CI повторить с live service; mock не засчитывать как parity.
- [ ] **Step 5: commit** `fix: serialize namespace read modify write across sql backends`.

## Task P1.2: безопасные callers и отсутствие stale telemetry overwrite

**Files:** Modify `knowledge.py`, `learning.py`, `pool.py`, `_pool_recall.py`; Create `tests/always_on/test_mutations.py`.

**Interfaces:**
- Consumes: `mutate_facts(...) -> T` из P1.1.
- Produces: прежние публичные add/add_many/add_episodic/learn/add_resolved/decay/forget/clear/restore/publish/promote/gc_pool сохраняют сигнатуры; все managed writes больше не коммитят stale full snapshots после вычислений вне lock.
- В learning добавить `_resolve_and_commit(verified: list[Fact], conflict_threshold: float) -> list[Fact]`: candidate verification/extraction остаётся снаружи, resolve/consolidate-pure/commit выполняются над текущими facts в atomic callback. Async semantic dedup коммитит только guarded IDs/revisions, не прежний `existing` целиком.

- [ ] **Step 1: написать failing concurrency regression.** Test helper `pause_recall_before_metadata_save` — monkeypatch internal persistence seam только для управляемого interleaving, не mock retrieval результата.

```python
def test_recall_cannot_restore_stale_content(kb_factory, pause_recall_before_metadata_save):
    kb = kb_factory()
    original = kb.add("Deployment uses blue service")
    with pause_recall_before_metadata_save(kb, query="Deployment"):
        peer = kb_factory()
        current = peer.get(original.id)
        current.content = "Deployment uses green service"
        peer.replace_facts([current])
    assert kb_factory().get(original.id).content == "Deployment uses green service"
```

Другие проверки: одновременные kb.add сохраняют оба разных факта; pool recall не воскрешает superseded claim; decay не удаляет concurrent add; async embed response не закрывает уже заменённую revision; restore является явной заменой facts; learn provider вызывается вне lock.

- [ ] **Step 2: выполнить** `.venv/bin/python -m pytest tests/always_on/test_mutations.py -q --no-cov`; воспроизвести исходный lost update.
- [ ] **Step 3: перевести все найденные callers.** Повторить `rg -n '\._storage\.(save|delete|atomic_update)' src/ai_knot/{knowledge,learning,pool,_pool_recall}.py`. Для каждого оставшегося save документировать: intentional full replacement в lock либо snapshot-only. Recall обновляет только существующие returned IDs: access_count/last_accessed/intervals, без перезаписи content/validity/ACL. В чистых callbacks не выполнять embeddings, LLM, telemetry HTTP или I/O.

  Точные преобразования (все сохраняют current return values):
  - `knowledge.add`: построение candidate/date enrichment снаружи; dedup parent/children против текущего reference window и append внутри callback. Duplicate возвращает текущий persisted Fact. `add_many/add_episodic` append к текущему списку.
  - `knowledge._execute_recall`: новый `_record_access(returned_ids: set[str], access_time: datetime) -> None` вызывает `mutate_facts`; increment только ещё существующих IDs, `last_accessed=max(current.last_accessed, access_time)`, interval `max(0, delta_hours)`, последние 20 intervals. Никакой записи `all_facts` из retrieval snapshot. Interleaving fixture patch-ит именно `_record_access`, вызывает peer correction перед original helper.
  - `learning.learn/add_resolved`: verification/extraction вне lock; новый `_resolve_and_commit(verified, conflict_threshold)` вызывает `_resolve_phase` и `_consolidate_phase` на текущем списке **внутри** callback, затем сохраняет current+to_insert+aggregates. Удалить stale `_commit_phase(existing,...)` path; возврат/логирование после commit.
  - `learning.learn_async`: embedding snapshots снаружи; proposal closures хранят `(old_id, old_revision, new_id, new_revision, closed_at)`. В callback закрываются только те old/new, которые всё ещё существуют, revisions совпадают и active; остальное skip, никогда не сохранять старый `existing`. Future-dated/уже superseded факты не закрывать повторно.
  - `knowledge.decay`: decay/purge policy над current facts в callback; `replace_facts/clear_all/restore` — намеренные full replacements через journaling writer. `forget/delete` — удаление по ID через тот же writer, missing ID no-op.
  - `pool._publish_locked`: существующий `_merge` через `mutate_facts`; counters/ledger/index invalidation после commit. `promote/gc_pool` и auto-promotion в `_pool_recall.recall` изменяют current rows по IDs, без stale save. `promote` меняет qualifiers/memory_tier и поэтому создаёт event; только rank/importance write — 0 events.
  - `learning.consolidate_episodic`: до замены P3.1 убрать stale full-save закрытия; closing callback сверяет исходные `(id,revision)` и обновляет только совпавшие episodic rows. Это предотвращает lost update, но **не** делает старый двухкоммитный consolidation атомарным; P3.1 заменяет whole prepare/commit path и его тест обязателен перед always-on maintenance release.
- [ ] **Step 4: targeted suite**, затем `tests/test_temporal_anchor_integration.py tests/test_shared_pool.py tests/test_recall_trace.py tests/test_event_time_persistence.py tests/test_pool_governance_durable.py` с `--no-cov`.
- [ ] **Step 5: commit** `fix: preserve concurrent memory updates during recall and learning`.

## Task P1.3: транзакционный journal, revisions и upgrade

**Files:** Create `always_on/models.py`, `storage.py`, `changes.py`, `__init__.py`; Modify SQL backend helpers из P1.1; Create `tests/always_on/test_changes.py`.

**Interfaces:**
- Consumes: transaction-local load/save, namespace locks.
- Produces: `evidence_revision(fact: Fact) -> str`, `diff_facts(before: list[Fact], after: list[Fact]) -> list[FactChange]`; `AlwaysOnStorageCapable.always_on_transaction(namespaces, write=...)`; `AlwaysOnSession` scoped fact read/write и AO SQL access на той же connection.
- Models: `FactChange(seq, kind, fact_id, slot_key, before_revision, after_revision)`; storage-only old/new access projection. `enable_stream(namespace)` внутри facade transaction создаёт baseline/head atomically. SQL identifiers — fixed constants; значения всегда bind parameters.

  `FactChange` — frozen dataclass, `kind: Literal['insert','update','delete','resync_required','acl_changed']`, seq positive; revisions `str | None`; service events используют пустые ID/slot. Внутренний `StoredChange` дополнительно несёт namespace, before_slot_key, old/new origin/visibility/scope/topic/tier, acl_subject. Session contract: `load_facts(namespace)`, `replace_facts(namespace,facts)`, `fetch_one/fetch_all/execute(SQLStatement, params)`. `SQLStatement` содержит две фиксированные строки (SQLite `?`, PG `%s`); rows — dict mappings. Не делать автоматическую замену `?` в SQL, f-string interpolation значений или dependency на SQL ORM. Полные типы в reference notes.

- [ ] **Step 1: добавить tests атомарности и suppression telemetry.** Факты в этом примере имеют фиксированные created_at/valid_from из fixture.

```python
def test_telemetry_does_not_emit_fact_changes(change_store, fixed_fact):
    change_store.enable("owner")
    change_store.save("owner", [fixed_fact])
    head = change_store.head("owner")
    fixed_fact.access_count += 1
    fixed_fact.retention_score = 0.9
    change_store.save("owner", [fixed_fact])
    assert change_store.head("owner") == head
```

`change_store` — thin test fixture over real session/store, не альтернативный in-memory implementation. Добавить `test_event_insert_failure_rolls_back_fact_write` с SQL failure после fact replacement до append и после первого event/head increment; `test_postgres_commit_order_no_gap` с двумя sessions/барьерами; `test_delete_keeps_tombstone`; parametrized `test_all_fact_mutators_emit_changes` (add, add_resolved, learn-stub, delete, clear, restore, pool publish/promote/GC); неизменный promote/rank-only write может давать 0 knowledge events, реальное изменение tier/qualifiers — минимум 1. `test_revision_normalizes_utc_and_qualifier_order` сравнивает timezone-equivalent timestamps и reordered qualifier keys; `test_each_evidence_field_changes_revision` параметризует весь whitelist; `test_telemetry_fields_never_change_revision` параметризует exclusions, а не только access_count.

Migration: `test_enable_existing_db_requires_resync`, `test_enable_idempotent_across_restart`, `test_two_instances_see_enabled_stream`, `test_old_fact_schema_upgrade_preserves_fields`, `test_failed_enable_is_atomic`, `test_independent_namespaces_have_independent_heads`.

- [ ] **Step 2: выполнить** `.venv/bin/python -m pytest tests/always_on/test_changes.py -q --no-cov`; до реализации нет новых event tables/capability.
- [ ] **Step 3: реализовать additive DDL** schema marker version=1, streams/changes по reference notes; не создавать заранее таблицы будущих intentions/jobs. `diff_facts` rejects duplicate IDs и идёт по `sorted(before_ids | after_ids)`: insert при отсутствии old, delete при отсутствии new, update при смене digest, иначе ничего. `_save_on_connection` читает persisted enabled flag каждый раз, пишет diff в этой же transaction, увеличивает stream head на число событий и вставляет contiguous seq; rollback возвращает и head. Baseline включения existing namespace — ровно один resync_required event; пустого — head=0. Повтор enable — no-op, даже после restart. Explicit delete пишет tombstone; restore делает новый diff. Нет SQL sequence/autoincrement для AO seq. Перед enable все writers обновлены, старые writer binaries запрещены операционным rollout gate.
- [ ] **Step 4: выполнить P1.1–P1.3 tests** на SQLite и live PG; `.venv/bin/ruff check src/ai_knot tests/always_on` и `.venv/bin/mypy src/ai_knot/`. Обновить backend round-trip tests, если serializer получил новое поле; не ослаблять текущие assertions.
- [ ] **Step 5: commit** `feat: add atomic memory change streams and evidence revisions`.

## Task P1.4: durable subscriptions, ack и ACL-safe replay

**Files:** Create `always_on/memory.py`; extend `models.py`, `changes.py`, AO DDL; Modify `pool.py` grant/revoke transaction path; Create `tests/always_on/test_subscriptions.py`.

**Interfaces:**
- Consumes: spec WatchSelector, SQL seam и FactChange; existing pool scope rule.
- Produces: `AlwaysOnMemory.__init__`, `subscribe`, `poll`, `ack`, `unsubscribe` с точными сигнатурами spec; models `WatchSelector`, `Subscription`, `ChangeBatch(delivery_id: str | None, events: tuple[FactChange,...], resync_required: bool)`. Create/get intents появятся в P2.
- Fresh instance из `ao_factory()` восстанавливает subscriptions из SQL, без прежних Python objects. Unsupported YAML constructor кидает UnsupportedAlwaysOnBackend.

- [ ] **Step 1: записать главный recovery test.** Фикстура `ao_factory` возвращает новый facade/KB/backend к тем же данным; `clock` — mutable fake UTC Clock.

```python
def test_poll_replays_until_ack_after_restart(ao_factory, kb):
    ao = ao_factory()
    ao.subscribe(subscription_id="watch", selector=WatchSelector(namespace="owner"))
    changed = kb.add("Pipeline moved to Prefect")
    first = ao.poll("watch")
    assert changed.id in {e.fact_id for e in first.events}
    second = ao_factory().poll("watch")
    assert second.delivery_id == first.delivery_id
    ao_factory().ack("watch", delivery_id=first.delivery_id)
    assert ao_factory().poll("watch").events == ()
```

Дополнительные tests с assertions:
- `test_foreign_or_future_ack_rejected`: InvalidDelivery, cursor прежний; повторный valid ack — no-op.
- `test_old_valid_ack_after_new_batch_is_noop`: ack A, ack B, restart, ack A — no-op; A после unsubscribe/recreate ID — InvalidDelivery.
- `test_shared_feed_requires_pool_and_live_grant`: deny без pool/при revoke на другом процессе.
- `test_replay_after_acl_revoke_is_redacted`: delivery_id прежний, закрытых IDs/text нет, resync_required=true.
- `test_replay_after_scope_narrow_then_delete_is_redacted`: первоначальный global insert уже inflight, затем scope private и delete; replay сверяет latest tombstone projection и не выдаёт старый global ID.
- `test_filtered_page_can_be_acked`: 1000 hidden events → empty batch с token → ack → видимое событие следующей страницы.
- `test_debounce_survives_restart`: debounce=30, Clock+29 → no delivery, +1 → один batch; обычный empty poll не пишет run/wakeup.
- `test_coalescing_keeps_old_id_invalidation`: supersede содержит invalidation старого fact_id и новый fact_id.
- `test_forced_prune_returns_cursor_expired`: нет silent cursor jump.
- `test_unsubscribe_releases_retention_and_rejects_old_ack`: повтор unsubscribe — no-op; прежний token недействителен, подписка не удерживает min cursor.
- `test_limit_and_selector_validation_before_write`: limit=0/501, naive datetime, NaN debounce, чужой namespace — reject.
- `test_slot_move_invalidates_old_slot_watch`: ID сохраняется, slot A→B; watcher A видит invalidation через before_slot_key.
- `test_page_limit_stops_before_unrepresented_fact`: limit=1 не продвигает through_seq за первое событие второго отличного ID; повтор poll с другим limit не меняет inflight.
- `test_debounce_tail_coalesces_only_until_scan_boundary`: 1000 raw rows maximum; hidden-only page получает token без ожидания debounce; empty tail не создаёт token/receipt и не пишет subscription.

- [ ] **Step 2: выполнить** `.venv/bin/python -m pytest tests/always_on/test_subscriptions.py -q --no-cov`.
- [ ] **Step 3: реализовать owner binding, persisted inflight batch и conservative redaction.** Shared AO требует pool `persist_stats=True`; private AO от этого flag не зависит. Добавить `pool.revoke_read(agent_id: str, scope: str) -> None`. SQL save_grant/revoke_grant атомарно записывают адресованный ACL event при enabled shared stream; no-op grant/revoke не создаёт event. AO и existing recall при enabled shared stream читают актуальные durable grants. Shared event payload не содержит raw fact content. Алгоритм poll/ack, проекция доступа и schema inflight/receipts — reference notes; scan <=1000, limit относится к distinct coalesced fact IDs, служебный resync не расходует fact limit. Cursor двигается только ack. Selector/debounce существующего subscription_id меняется только через новый ID: mismatch даёт RevisionConflict; повтор с тем же contract возвращает сохранённую подписку, без cursor reset. subscribe `now` записывает текущий head под lock и не получает чужую старую inflight delivery. `unsubscribe` удаляет subscription/receipts атомарно; повтор no-op.
- [ ] **Step 4: targeted tests +** `tests/test_acl_ledger_storage.py tests/test_pool_governance_durable.py tests/test_shared_pool.py tests/test_mcp_tools.py`. Проверить old sync_dirty/sync_slot_deltas API без несовместимого поведения.
- [ ] **Step 5: commit** `feat: deliver durable scoped memory changes with explicit acknowledgement`.

## P1 exit gate

- [ ] SQLite и live PostgreSQL: все P1 tests; требуемые tests реально выполнялись, не только skipped.
- [ ] Повторить transaction/restart сценарии в subprocess на одном DB path/schema.
- [ ] `.venv/bin/ruff check src/ tests/` и `.venv/bin/ruff format --check src/ tests/`; `.venv/bin/mypy src/ai_knot/`.
- [ ] Зафиксировать baseline timing AO disabled/enabled для P3.5, без публикации performance claims до измерения.
- [ ] В PR указать supported backends и запрет старых writer binaries после enable; перечислить фактически запущенные проверки.
