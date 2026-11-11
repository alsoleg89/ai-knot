# Процедурный опыт, привязанный к исходу действия

Дата/граница источников: **2026-10-28**. Исходный код сверялся с `f877440`; внешние ссылки — первичные статьи, документация и официальный код. URL `main` и живая документация могут меняться. Это исследование контракта, без production/test изменений, запуска моделей или экспериментов.

## Вывод

`retained receipt → EvidenceRef → derive_fact(PROCEDURAL)` уже достаточно для **host-authored условного урока с отзывом по источнику**. Это показано существующим [примером](../../examples/task_context.py) и описано в [task-context](../task-context.md). Новый граф и ещё один слой «памяти» здесь не нужны.

Реальный дополнительный контракт уже: библиотека атомарно проверяет, что урок относится к **существующему owner-scoped action, точной revision и записанному terminal outcome**, и сохраняет эту зависимость вместе с текущими outcome evidence. Сейчас `EvidenceRef` адресует только факты, а `derive_fact` не читает action. Поэтому пример достаточен для доверенного host; он не даёт библиотечную гарантию action binding, защиты от гонки с reconciliation и идемпотентной записи урока после crash/retry. Новый API оправдан только ради этих проверяемых свойств.

Привязка к записанному outcome не доказывает внешнюю истинность результата или правильность обобщения. Даже exact action revision плюс receipt остаются host-declared evidence, пока отдельный connector не проверил внешний источник.

## Что подтверждают первоисточники

| Источник | Полезный механизм | Граница переноса |
|---|---|---|
| [ExpeL v3](https://arxiv.org/html/2308.10144v3), 2024-12-20; впервые 2023-08-20 | Сравнение failed/successful trajectories одной задачи и поиск закономерностей в успешных задачах; reusable insights отдельно от эпизодов | Нужны наблюдаемые outcomes и условия применения. Контроль action revision, ACL, crash consistency и отзыва статьи не предоставляют |
| [Reflexion v4](https://arxiv.org/abs/2303.11366v4), 2023-10-10 | Feedback → короткая reflection → память для следующей попытки; feedback может быть внешним либо внутренне симулированным | «Reflection записана» не означает «операция внешне подтверждена». Нельзя уравнивать self-report с независимой квитанцией |
| [G-Memory v2](https://arxiv.org/html/2506.07398v2), 2025-06-16; [официальный код](https://raw.githubusercontent.com/bingreeky/GMemory/main/mas/memory/mas_memory/GMemory.py) | `add_memory` требует boolean label; retrieval раздельно выбирает successful/failed примеры. Полезно сохранять отрицательный опыт рядом с положительным | Label — полученный сигнал, не доказательство. Графы и LLM reranking не нужны для локальной проверки журнал/receipt; ограниченные embodied/QA задачи не подтверждают always-on безопасность |
| [LangMem: episodic memories](https://langchain-ai.github.io/langmem/guides/extract_episodic_memories/), живая документация | Схема observation/action/result и извлечение удачных примеров | Демонстрация выводит успех из диалога; это не receipt binding. Для AI Knot достаточно наблюдения, действия, результата, evidence и краткого объяснения; полный private reasoning не требуется |
| [VRL-Bench v1](https://arxiv.org/html/2609.12404v1), 2026-09-11 | Разделяет запись reflection и её последующий replay при фиксированных первых попытках и бюджетах; memory помогает в одних условиях и мешает в других | Наличие урока само по себе не доказывает полезность его применения. Это finite-trial исследование, не проверка нашей реализации; нельзя переносить его числа на AI Knot |
| [Temporal: Activity Definition](https://docs.temporal.io/activity-definition), живая документация | Worker может выполнить внешний эффект и умереть до подтверждения; observed completion и число физических исполнений различаются; idempotency обеспечивает внешний сервис | Timeout/unknown не доказывают failure и тем более success. Перед повтором нужны внешняя идемпотентность или reconciliation |
| [W3C PROV-O](https://www.w3.org/TR/prov-o/), Recommendation 2013-04-30 | Разделяет activity, fixed-aspect entity, derivation, revision и invalidation | Полезная терминология: запись результата и текст урока — разные сущности. Это не требование использовать RDF/graph database и не гарантия truth |

## Текущий контракт AI Knot

- [`ActionRecord`](../../src/ai_knot/always_on/models.py) содержит `id`, `intent_id`, `request_digest`, `state`, `result`, `reconciliation_evidence`, integer `revision`. Реальная область идентичности включает owner; `_action` проверяет owner и привязанный workspace. Action ID сам по себе не tenant/ACL authority.
- [`finish_action`](../../src/ai_knot/always_on/_state.py) принимает только started action и текущий lease; сохраняет `succeeded`/`failed`/`unknown`. Совпадающий повтор результата возвращает прежнюю запись. `result={}` допустим даже при succeeded: внешний успех не проверяется библиотекой.
- `reconcile_action` принимает unknown/failed и непустой host JSON, переводит в succeeded/failed либо retryable→prepared. **Failed не неизменяемая конечная история:** после retry тот же action ID может получить новый исход. Каждое изменение повышает revision; одинаковый replay её не повышает.
- [`save_action`](../../src/ai_knot/always_on/state_storage.py) делает CAS по предыдущей revision, но хранит текущий snapshot строки; отдельной immutable истории всех action revisions здесь нет. После перезапуска публичного read/list terminal actions API в facade также нет; возвращённый `finish/reconcile` объект не заменяет durable binding.
- [`derive_fact`](../../src/ai_knot/always_on/memory.py) атомарно проверяет evidence и пишет typed/task/role-labelled fact с точными roots/dependencies. Shared запись требует maintain и доступа к источникам. Повтор вызова создаёт новый UUID, а не идемпотентный урок.
- [`lineage`](../../src/ai_knot/always_on/lineage.py) сохраняет независимые observation roots и зависимости от промежуточных revisions; correction/deletion/ACL ограничивают дальнейшую выдачу. Исторически архивированный источник может оставаться основанием, если сохранена точная допустимая revision. Эта семантика не означает, что любой исторический outcome является текущим состоянием операции.
- `independent_evidence` схлопывает копии до исходных revisions. Число агентов, lessons, retrievals и ActionRecord aliases не должно увеличивать число независимых наблюдений.

## Минимальный дополнительный контракт

Предлагаемая форма — небольшой guarded writer поверх существующих SQL transactions/facts, условно `remember_action_outcome`. Название несущественно; существенны следующие свойства.

1. **Вход.** Текущий owner, `action_id`, `expected_action_revision`, ожидаемый outcome (`succeeded`/`failed`), непустые `outcome_basis: tuple[EvidenceRef,...]`, короткий conditional lesson, стабильный `lesson_key`, task/role/scope. Caller не передаёт заменяющий ActionRecord или произвольную owner identity. Существующие bounds content/labels/roots сохраняются.
2. **Одна транзакция.** Прочитать доступный action; проверить workspace binding, точную revision, ожидаемый state и непустой сохранённый result; затем проверить currentness/readability/freshness outcome basis и записать PROCEDURAL fact вместе с binding. Prepared/started/unknown отвергаются, `failed` сохраняется как warning/failure experience. Не переводить action в success ради записи урока. Истёкший execution lease сам по себе не причина запретить чтение уже записанного результата; публикация всё равно требует текущего maintain/read права.
3. **Идентичность записи.** Ключ вроде `(owner, action_id, revision, lesson_key)` определяет один publication intent. Совпадающий повтор возвращает существующий fact только после проверок допустимости; другое содержание/basis/scope под тем же ключом даёт conflict. Старый replay после изменения action/evidence не должен возвращать отозванный lesson как current или восстанавливать `supported=True`. Ключ не должен быть только hash текста: одинаковый текст может относиться к разным событиям. Реализовать можно deterministic fact ID + existing transaction, без новой graph/runtime зависимости.
4. **Evidence и control dependency раздельны.** Roots урока — roots outcome evidence. Action revision/outcome binding — дополнительное условие допустимости, **не новое независимое свидетельство**. Два агента могут получить разные terminal ActionRecord при replay одной workspace business operation; два action ID не превращают одну receipt в два подтверждения. Связь evidence с внешней операцией всё ещё явно утверждает host; проверка JSON shape/digest не доказывает её семантику.
5. **Дальнейшее чтение.** Если заявлена семантика current-outcome lesson, все agent-facing recall/select/review/action guards должны проверять action binding, включая копии и descendants. Проверки только в writer недостаточно. Либо binding распространяется через lineage как dependency, либо action имеет guarded fact-проекцию, чья revision меняется/отзывается при action transition; обычный `derive_fact` работает уже от этой проекции. Не хранить непроверяемый `action_id` лишь в свободном qualifier и называть это гарантией.
6. **Отзыв.** Изменение/deletion/ACL evidence блокирует использование урока через существующие зависимости. Корректировка lesson пишет новую контролируемую revision; исправление исходного outcome отзывается в evidence. Совпадающий finish/reconcile replay не должен создавать новый lesson или лишнее invalidation. Новая action revision должна блокировать устаревший current-outcome lesson без idle model job.

Не требуется включать весь action request/result в общий текст: там могут быть данные с более узким доступом. Для team knowledge достаточно разрешённых outcome facts и opaque binding metadata. Read/maintain не дают права повторить действие.

### Сверка с выбранным проектным решением

Предложенный `record_experience(action_id, expected_action_revision, experience_key, content, basis, namespace/scope/task/role)` с одной небольшой SQL binding-таблицей соответствует этому контракту. Таблица полезна не как новый граф: она даёт устойчивый ключ, discovery затронутых fact namespaces и атомарное инвалидирование. Central `save_action` может выставлять mapped lessons `supported=False` при изменении action revision; `_intent_transaction` должен захватывать эти namespaces и после reopen без pool. Это сохраняет current-outcome семантику, а existing exact intermediate dependencies отзывают потомков.

Важные условия: invalidation использует обычную mutation факта, меняет evidence revision и публикует control change; одинаковый replay без смены action revision не отзывается; descendant не теряет dependency при копировании. Текстовый префикс лучше формулировать как «Recorded outcome: succeeded/failed». Если выбран «observed-success/failure», документация должна прямо сказать, что observation заявил host, а библиотека проверила binding, а не внешний сервис. Исторические failed-attempt warnings остаются отдельным receipt→derive путём ниже. Нового production-кода это исследование не добавляет.

## Как не потерять failure warnings

Нужно различать два утверждения:

- «Текущий записанный результат операции — failed». Оно зависит от **текущего** action revision и становится непригодным после retry/reconciliation.
- «Попытка при условиях X завершилась неудачей по квитанции R». Это исторический эпизод и основание условного предупреждения. Последующий успех другой попытки не опровергает прежнюю неудачу; исправление/отзыв самой R — опровергает.

Поскольку сейчас action хранится одним изменяемым snapshot, второй вид должен опираться на **отдельно сохранённую attempt/outcome receipt revision**, а не на обещание восстановить исчезнувший action revision. Её можно хранить обычным EPISODIC fact, а warning — процедурной проекцией. Raw episode не попадает в обычный task context. Не добавлять режим «игнорировать mismatch action revision» в первый API: использовать существующий исторический receipt-путь и явно описать его смысл. Failure warning — проверяемое ограниченное наблюдение, а не универсальное «этот способ никогда не работает».

## Когда достаточно примера

| Требование | Достаточно существующего host derive + receipt? |
|---|---|
| Сохранить условный урок, применимость/role и точные roots; отозвать после исправления receipt | Да; новый API ничего принципиально не добавляет |
| Сохранить историческую неудачу независимо от успешного retry | Да, если host сохраняет отдельную квитанцию попытки и не выдаёт её за current outcome |
| Атомарно отклонить урок от unknown/stale/чужого action | Нет; текущий derive не читает action |
| Повтор записи после crash не создаёт duplicate lesson | Нет; нужна guarded idempotent publication identity |
| При смене action revision автоматически исключить текущий outcome lesson и его копии | Нет; нужна проверяемая action dependency либо транзакционная fact-проекция |
| Доказать внешний успех и причинную правильность процедуры | Нет; новый binding API тоже этого не доказывает |

Рекомендация: реализовывать новый seam только с перечисленными action-binding/idempotency/currentness инвариантами. Если его scope ограничится несколькими labels поверх `derive_fact`, оставить пример и не вводить API. Автоматическую extraction/reflection оценивать отдельно от механической корректности привязки.

## Проверки без моделей

На SQLite/PostgreSQL отдельными connections/processes: exact terminal revision + readable receipt принимается; unknown/started/prepared, foreign owner/workspace, stale revision, пустое/недоступное/изменённое evidence отвергаются; concurrent reconcile против publication не допускает stale binding; matching replay даёт один lesson, divergent replay — conflict; failed→retryable→succeeded блокирует старый current-outcome lesson, но сохраняет отдельное историческое warning; correction/ACL/deletion receipt исключает lesson и копии; 20 копий/terminal aliases одной receipt дают один independent root. Проверить отсутствие отдельного wake/model вызова и утечки закрытых IDs/result payload.

Это тесты контракта, не результаты настоящего исследования. Для claims о learning нужны отдельные фиксированные traces, одинаковый доступный raw evidence и суммарный бюджет (formation + selection + answering), положительные и отрицательные outcomes, и измерение реальных решений/повторных ошибок. Детерминированный action binding сам по себе не доказывает прирост accuracy.
