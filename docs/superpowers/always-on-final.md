# Always-on AI Knot: итоговый комплект реализации

Дата итоговой сверки: 2026-10-27. Исходная база: `69e2adf` (`0.12.0`); имена планов сохраняют дату начала 2026-10-26.

**Production-реализация P1–P3 завершена:** [документация API](../always-on.md), [код](../../src/ai_knot/always_on/), [проверки реализации и review](implementation-final.md). Описанные ниже references остаются изолированными моделями; результаты production-тестов приведены в новом отчёте.

Этот комплект фиксирует реализацию шести функций: незавершённые намерения, подписки на изменения, зависимости решений, журнал внешних действий, повторная проверка знаний и обслуживание памяти. Он включает спецификацию, 13 задач интеграции, конкретные схемы/алгоритмы и исполняемые образцы с проверками.

Итог независимых проверок: Astra и Ponytail подтвердили исправления четырёх найденных проблем; 36 executable checks проходят. Подробные случаи, исправления и границы проверки: [отчёт review](reviews/2026-10-27-astra-ponytail.md).

## Как читать документы

| Документ | Назначение |
|---|---|
| [Спецификация](specs/2026-10-26-always-on-memory-design.md) | Канонические public contracts, состояния, пределы, гарантии и ограничения |
| [P1: хранение и изменения](plans/2026-10-26-always-on-01-storage.md) | Четыре задачи: atomic update, безопасные callers, журнал, durable subscriptions |
| [P2: состояние работы](plans/2026-10-26-always-on-02-agent-state.md) | Четыре задачи: intent/lease, evidence, action journal, freshness |
| [P3: обслуживание и продукт](plans/2026-10-26-always-on-03-maintenance-surfaces.md) | Пять задач: maintenance, HTTP, MCP/CLI/TS, пример, acceptance/CI |
| [Код хранения и SQL](reference/01-storage-code.md) | DDL, транзакционные hooks, курсоры/ack, PostgreSQL counterpart и точные места изменений |
| [Код состояния и действий](reference/02-agent-state-code.md) | Dataclasses, state transitions, CAS, pre-action guards, неизвестный исход и freshness |
| [Код обслуживания и интерфейсов](reference/03-maintenance-surfaces-code.md) | Reservation/prepare/commit, wire contracts, transport bindings, host loop и матрица тестирования |

Спецификация определяет контракт; планы определяют порядок интеграции; приложения с кодом показывают конкретные тела алгоритмов. Исполняемые references изолированы от `src/ai_knot`: они проверяют решения до интеграции, а готовность продукта доказывается integration gates P1–P3.

## Проверяемый код

| Файл | Что проверяет локально |
|---|---|
| [always_on_storage_reference.py](reference/always_on_storage_reference.py) | Реальный SQLite: совместный rollback facts/events/head, подавление telemetry events, persisted delivery/ack/receipts, restart, pruning и ограничения чтения |
| [always_on_state_reference.py](reference/always_on_state_reference.py) | Чистые transitions: fencing, recovery lease, inherited basis, single permit, unknown/reconciliation, отдельные freshness validity/retry deadlines |
| [always_on_maintenance_reference.py](reference/always_on_maintenance_reference.py) | SQLite ledger + async providers: budget reservation, ограниченный batch, guarded proposal commit, timeout, concurrent generation и source outage |

Одна команда запускает проверки всех трёх файлов из корня репозитория:

```sh
.venv/bin/python -m unittest discover -s docs/superpowers/reference -p 'always_on_*_reference.py' -v
```

Требуется только Python >=3.11 и его стандартная библиотека. Внешние API и модели не вызываются. Каждый reference можно запустить отдельно:

```sh
.venv/bin/python docs/superpowers/reference/always_on_storage_reference.py
.venv/bin/python docs/superpowers/reference/always_on_state_reference.py
.venv/bin/python docs/superpowers/reference/always_on_maintenance_reference.py
```

Эти проверки не заменяют live PostgreSQL, реальные HTTP/MCP/TypeScript integration tests и product subprocess recovery suite. Их точные команды и условия приёмки указаны в P1–P3.

## Что получает реализация каждой функции

| Функция | Основные contracts | Интеграционные задачи | Главный тест отказа |
|---|---|---|---|
| Намерения и восстановление | create_intent, claim, checkpoint, resume | P2.1 | Restart после checkpoint и fencing старого worker |
| Подписки | subscribe, poll, ack, unsubscribe | P1.3–P1.4 | Crash до ack возвращает тот же batch; отзыв ACL редактирует replay |
| Зависимости решения | EvidenceRef, capture_evidence, validate_basis | P2.2 | Коррекция между prepare и begin блокирует permit |
| Внешние действия | prepare_action, begin_action, finish_action, reconcile_action | P2.3 | External success до потерянного finish восстанавливается по evidence |
| Актуальность знаний | set_freshness, due_verifications, record_verification | P2.4 | Недоступный источник не делает факт подтверждённым и не продлевает validity |
| Обслуживание | run_maintenance, MaintenanceBudget, proposal/commit | P3.1 | Late proposal не затирает concurrent update и не превышает reservation |

## Решения, которые нельзя менять случайно

1. `Fact.version` не служит новым event cursor или универсальной evidence revision.
2. Facts, journal и head коммитятся вместе; PostgreSQL head сериализуется по namespace до чтения состояния.
3. Все managed read-modify-write paths, включая recall telemetry и async learning, используют безопасный commit.
4. Delivery cursor двигается после ack; receipt привязан к owner/subscription incarnation. Receipts живут до unsubscribe, их рост измеряется.
5. Смена доступа проверяется заново при replay и перед действием. Owner-owned исторический progress не объявляется автоматически очищенным от ранее прочитанного текста.
6. Blocked intent получает recovery lease; это не разрешение внешнего действия. Lease generation и action revision монотонны.
7. Action наследует intent basis. Проверка и started commit находятся в одной транзакции со всеми dependency locks; локальная транзакция не охватывает внешний сервис.
8. Unknown outcome требует reconciliation. Remote idempotency key помогает избежать повторного эффекта только если внешняя система действительно его поддерживает.
9. Verification retry_at определяет расписание, verified_until — срок подтверждения. changed/unavailable блокируют основание сразу.
10. Подготовка consolidation выполняется вне SQL lock; commit повторно проверяет revisions/generation и закрывает только подтверждённо покрытые эпизоды.

## Порядок исполнения

Сначала P1.1–P1.4, затем P2.1–P2.4, затем P3.1–P3.5. Настоящие SQL concurrency tests нужны уже на P1, а не в конце. После каждого task выполнять его targeted tests, затем затронутые regression suites; весь набор gates — на P3.5.

Первый завершённый вертикальный сценарий после P2: агент сохраняет намерение, ждёт CI-slot, восстанавливается после остановки, обнаруживает смену основания, перепланирует и завершает действие после reconciliation. После P3 этот сценарий доступен через заявленные интерфейсы и воспроизводимый пример.

Все 13 задач сохраняют собственные Files/Interfaces/Steps и проверяемый результат. Новый scheduler, broker и framework запуска агентов не входят в этот слой памяти. SQL durability покрывает SQLite/PostgreSQL; YAML при включении новых гарантий выдаёт явную ошибку совместимости.
