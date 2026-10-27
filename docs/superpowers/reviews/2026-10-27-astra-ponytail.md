# Итоговая проверка Astra и Ponytail

Дата: 2026-10-27. Объект: спецификация, три плана, три кодовых приложения и три исполняемых reference-файла в `docs/superpowers/`.

## Результат

Astra и Ponytail независимо проверили комплект, затем адресно перепроверили исправления. Итог обоих повторных review: существенных незакрытых замечаний по исправленным случаям нет. Все четыре подтверждённые проблемы исправлены в коде примеров и соответствующих контрактах документов.

Это вердикт о комплекте реализации и executable design checks. Production-модули и live PostgreSQL ещё должны пройти предусмотренные P1–P3 integration gates.

## Найденные проблемы и исправления

| № | Кто подтвердил | Конкретный сбой | Исправление | Regression |
|---|---|---|---|---|
| 1 | Astra + Ponytail | После begin action checkpoint(waiting) освобождал lease; action оставался started, waiting intent не получал recovery lease даже через сутки | STARTED запрещает любой освобождающий lease checkpoint до finish/expiry. UNKNOWN допускает blocked с reason unknown_action или явную cancellation; ready/waiting/completed запрещены | `test_checkpoint_cannot_orphan_started_action`, `test_unknown_checkpoint_only_blocks_or_explicitly_cancels` |
| 2 | Astra; seam также проверил Ponytail | Новый ack использует POST/body, а текущий HTTP requestVoid принимает только DELETE и не передаёт body. Простое расширение типа оставляло бы delivery_id потерянным | В приложении дано полное тело будущего helper: POST/DELETE, JSON body, Content-Type/auth/custom headers, отсутствие JSON parsing для 204 | Strict tsc и Node assertions на временной копии фактического HttpKnowledgeBase с helper из документа |
| 3 | Astra | CancelledError при отмене verifier подменялся Conflict, если guarded_verify обнаруживал expiry/takeover; task возвращал обычный report вместо cancellation | Отдельный cancellation handler; unavailable/report diagnostics best-effort; оригинальная cancellation всегда пробрасывается. Обычные ошибки хранения не скрываются | `test_cancellation_survives_expiry_takeover_and_failed_diagnostics`: expiry, новая generation, закрытая diagnostic DB |
| 4 | Ponytail | source_ids=[e,e] принимался, эпизод закрывался, lineage содержала повтор, хотя контракт требует reject | Проверка уникальности source IDs после проверки типов, до изменения facts/validity; весь proposal отклоняется | `test_duplicate_sources_reject_entire_proposal`: episode active, summary отсутствует |

Исправления состояния: [always_on_state_reference.py](../reference/always_on_state_reference.py).

Исправления maintenance: [always_on_maintenance_reference.py](../reference/always_on_maintenance_reference.py).

HTTP helper и интеграционные contracts: [03-maintenance-surfaces-code.md](../reference/03-maintenance-surfaces-code.md).

## Выполненные проверки

1. Исходные 32 unittest проверки проходили до review. Четыре новых regression-теста сначала воспроизвели ошибки; после исправлений общий прогон: **36 tests, OK**. Оба reviewer повторили общий прогон с тем же результатом.
2. Ruff check всех трёх Python reference-файлов: passed. Ruff format --check: все три файла соответствуют формату.
3. Mypy `--strict --follow-imports=skip` для типизированного state reference: success. Это не заявление о strict typing остальных уменьшенных SQLite examples.
4. TypeScript: точное тело helper извлечено из документа и подставлено во временную копию текущего `npm/src/http.ts` рядом с текущим `types.ts`. Strict tsc прошёл. Node проверил POST delivery_id, JSON/auth/custom headers, прежний DELETE и отсутствие response.json() на 204. Исходные npm-файлы не изменялись.
5. Проверены локальные ссылки, Python syntax (в том числе grammar 3.11 для исполняемых файлов), одинаковые Global Constraints и наличие всех 13 задач.
6. SQLite выполнил DDL intentions/actions/freshness из кодового приложения. Это проверка DDL SQLite; PostgreSQL SQL не запускался.

Команды для повторения основных executable checks из корня репозитория:

```sh
.venv/bin/python -m unittest discover -s docs/superpowers/reference -p 'always_on_*_reference.py' -v
.venv/bin/ruff check docs/superpowers/reference/*.py
.venv/bin/ruff format --check docs/superpowers/reference/*.py
.venv/bin/mypy --strict --follow-imports=skip docs/superpowers/reference/always_on_state_reference.py
```

## Что остаётся проверять при интеграции

Live PostgreSQL locking/commit order/migrations; реальные namespace ACL и serializers; public facade persistence; subprocess crash tests actual package; HTTP/MCP/TypeScript product routes; 100k-event load и рост receipts; общую coverage и существующие regression suites. Эти проверки перечислены как будущие задачи, их прохождение не подменяется успехом reference-кода.

Ponytail не потребовал нового framework, слоя или обязательной зависимости. Все исправления локальны к найденным причинам; public API не расширялся ради обхода ошибок.
