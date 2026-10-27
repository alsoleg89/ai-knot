# Always-on 03 — Maintenance, Interfaces and Acceptance Implementation Plan

**Execution status (2026-10-27): production work complete.** See [implementation and final verification](../implementation-final.md). The original checklist below is retained as the implementation brief; per-task commits were not created.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Довести шесть always-on функций до используемого продукта и воспроизводимой проверки восстановления, стоимости и совместимости.

**Architecture:** Один bounded maintenance batch, подготовка внешних результатов вне lock и guarded commit. Python facade общий для HTTP/MCP/CLI и TypeScript transports; runtime запускает следующий batch самостоятельно. Приёмка воспроизводит полный цикл intent → изменение знания → crash → replan → reconciliation.

**Tech Stack:** Существующие FastAPI/Pydantic extra, MCP extra, Click, TypeScript/Vitest; pytest/subprocess/sqlite3 для offline acceptance.

**Spec:** [2026-10-26-always-on-memory-design.md](../specs/2026-10-26-always-on-memory-design.md). Требует P1/P2.

**Concrete reference:** [03-maintenance-surfaces-code.md](../reference/03-maintenance-surfaces-code.md) фиксирует текущие source symbols, все HTTP/MCP/TS mappings, JSON fixtures, error mapping, host recovery и обязательные assertions. [always_on_maintenance_reference.py](../reference/always_on_maintenance_reference.py) — исполняемая stdlib SQLite модель reservation/prepare/guarded-commit; её unittest не заменяют будущие integration/PG acceptance tests. Production/src и tests на этом этапе не изменяются; все task checkboxes остаются невыполненными до реализации.

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

- Provider вернул пустой/невалидный proposal — исходные эпизоды остаются: P3.1 `test_empty_consolidation_never_closes_episodes`.
- Provider завершился после expiry/replan — результат не затирает новое состояние: P3.1 `test_late_provider_result_is_fenced`.
- HTTP caller подставил owner/foreign delivery token — нет доступа к другому агенту: P3.2 `test_actor_cannot_be_spoofed`.
- MCP/HTTP replay потерял ответ после commit — повтор не создаёт второе намерение/действие: P3.3 `test_transport_retry_keeps_identity`.
- Repeated restores/deletions/долгая очередь дают неверное восстановление — end-to-end fault matrix: P3.5 `test_all_crash_windows`.

## File map

- Create `src/ai_knot/always_on/maintenance.py`, `codec.py`; extend `always_on/{models,memory,storage}.py`, `src/ai_knot/learning.py` and `types.py` for guarded consolidation/provenance.
- Create `src/ai_knot/server/always_on.py`, `src/ai_knot/_always_on_tools.py`; modify server/app.py, mcp_server.py, _mcp_tools.py, config.py, cli.py.
- Create `npm/src/always_on.ts`; modify `npm/src/{client,http,types,index}.ts` and respective tests. Public MCP `KnowledgeBase` находится в index.ts; client.ts содержит низкоуровневый McpClient, не публичную facade.
- Create `tests/always_on/test_{maintenance,codec,surfaces,cli,end_to_end}.py`, `tests/eval/always_on/{__init__,runner,scenarios}.py`.
- Create `examples/always_on_recovery.py`, `docs/always-on.md`; modify docs/production-readiness.md, docs/integrations.md, README.md, CHANGELOG.md, CI/benchmark workflows only where needed.

## Task P3.1: bounded maintenance и безопасная consolidation

Дополнительные regressions после независимого review: `test_cancellation_survives_expiry_takeover_and_failed_diagnostics` отменяет ожидающий verifier при expiry, takeover и ошибке записи diagnostics: вызывающий получает CancelledError, successor report не перезаписан. `test_duplicate_sources_reject_entire_proposal` оставляет episode active и не создаёт summary при source_ids=[id,id]. Включить оба в tests/always_on/test_maintenance.py.

**Files:** Create `always_on/maintenance.py`, `tests/always_on/test_maintenance.py`; Modify `learning.py`, `types.py`, AO models/storage.

**Interfaces:**
- Consumes: due_verifications/record_verification, EvidenceRef/lease guard, existing extractor/resolve stages.
- Produces: `MaintenanceBudget`, `MaintenanceReport`, `VerificationProvider`, `ConsolidationProvider`, `ConsolidationProposal`; `async run_maintenance(memory, *, run_key, budget, verifier=None, consolidator=None) -> MaintenanceReport` по spec.
- Providers: `async verify(request: VerificationRequest) -> VerificationResult`; `async propose(episodes: tuple[Fact,...]) -> ConsolidationProposal`. Proposal содержит structured `facts: tuple[Fact,...]`, `covered_episode_ids: tuple[str,...]`, `input_refs: tuple[EvidenceRef,...]`. Guarded commit принимает только refs текущей revision, valid source mappings и действующую generation maintenance lease.
- Каждый provider декларирует стабильный credential-free `config_key` и `model_cost: int >=0`; provider/config identity и budget входят в run digest. Внутренние extractor/embedder/resolver вызовы учитываются в model_cost либо резервируются отдельно, не скрываются за одним вызовом wrapper. Резерв model calls и aggregate UTF-8 input bytes сохраняется при crash/expired-run takeover; timeout не возвращает резерв. Одновременный live run_key не запускает второй batch; завершённый replay возвращает сохранённый report, recovery сохраняет cumulative counters.
- Report: run_id, status (`completed`, `budget_exhausted`, `conflict`, `failed`), inspected, verified, consolidated, expired, model_calls_reserved, skipped, reasons. Повтор run_key с тем же budget/config не повторяет завершённый batch; иной payload — IdempotencyConflict. Не создавать durable daily quota subsystem.
- `verified` считает accepted confirmed checks, `consolidated` — закрытые covered episodes, `expired` — committed GC removals. unavailable/changed не считаются verified. Budget-exhausted batch терминален для своего key; следующий batch получает новый host key.

- [ ] **Step 1: tests на бюджет и пустой proposal.** Использовать `asyncio.run`, а не добавлять pytest-asyncio.

```python
def test_zero_model_budget_never_calls_consolidator(ao, counting_consolidator):
    report = asyncio.run(run_maintenance(
        ao, run_key="batch-1", budget=MaintenanceBudget(max_model_calls=0),
        consolidator=counting_consolidator))
    assert counting_consolidator.calls == 0
    assert report.model_calls_reserved == 0
```

Добавить: `test_empty_consolidation_never_closes_episodes`; `test_invalid_source_mapping_rejects_entire_proposal`; `test_foreground_episode_survives_background_commit`; `test_late_provider_result_is_fenced`; `test_concurrent_run_key_reserves_once`; `test_timeout_consumes_reserved_call`; `test_crash_after_reservation_does_not_refund`; `test_budget_max_items_stops_at_100`; `test_input_utf8_budget_is_enforced_before_call`; `test_wall_budget_stops_new_calls`; `test_disabled_provider_reports_skipped`; `test_episode_age_cutoff_is_24_hours`; `test_unavailable_verification_never_authorizes_action`; `test_prune_cannot_skip_live_unacked_subscription`; `test_consolidation_lineage_round_trips_sql_and_yaml`. Assertions для source/race/budget перечислены в reference matrix; SQLite connection и process recovery — реальные, providers — только внешние fakes.

- [ ] **Step 2: выполнить** `.venv/bin/python -m pytest tests/always_on/test_maintenance.py -q --no-cov`. Provider fakes — только замена внешнего model/checker, SQL остаётся реальным.
- [ ] **Step 3: реализовать reservation/prepare/commit.** Создать maintenance_runs/index. Select active supported episodes с `created_at <= clock()-24h` и SQL LIMIT: сохраняется текущий `consolidate_episodic(older_than_hours=24.0)` default; explicit legacy override остаётся. Clock/deadline проверять между bounded units; network timeout передавать provider, ограничения cooperative явно документировать. Использовать общий prepare/guarded commit helper для run_maintenance и существующего consolidate_episodic; не запускать старый mutating learn_async из background до guarded commit. `_resolve_phase`/`_consolidate_phase` получают detached копии Fact, потому что меняют existing Fact in-place. В guarded commit reread текущие slot targets и input refs/ACL под P1 locks; `AlwaysOnSession.load_facts/replace_facts` + `fetch_one/fetch_all/execute(SQLStatement(sqlite,postgres),params)` используют ту же transaction, без nested connection. Fact diff/journal/head и ledger counters коммитятся вместе. Покрытие output provenance должно точно объяснять covered IDs; duplicate/unknown refs, output-ID collision, пустые facts и unrepresented covered IDs отвергают весь proposal. При конфликте сохранять bounded proposal/run error через generation CAS, не закрывать вход и не перезаписывать successor report. Serialize consolidation_ids как JSON array string в qualifiers и читать обратно в Fact.provenance, сохраняя existing keys; не добавлять новую обязательную Fact DB column только для списка.
- [ ] **Step 4: targeted tests +** `tests/test_shared_pool.py tests/test_temporal_anchor_integration.py tests/test_event_time_persistence.py tests/test_storage_compat.py` (существующий provenance accessor проверяется shared-pool suite; новых consolidation tests в текущей базе нет, они входят в этот task).
- [ ] **Step 5: commit** `feat: maintain agent memory with bounded guarded background batches`.

## Task P3.2: единый codec и HTTP interface

**Files:** Create `always_on/codec.py`, `server/always_on.py`, `tests/always_on/test_codec.py`, `test_surfaces.py`; Modify `server/app.py`, `config.py`.

**Interfaces:**
- Consumes: полные P1/P2/P3.1 methods.
- Produces: `encode_always_on(value: object) -> JSONValue`, typed request decoders с unknown-field rejection; `create_app(kb, *, token=None, always_on: AlwaysOnMemory | None = None) -> FastAPI`, backward compatible keyword-only addition; `build_always_on_router(memory, require_auth) -> APIRouter`.
- Config: `AI_KNOT_ALWAYS_ON=false` по умолчанию; parse в `AIKnotConfig.always_on_enabled: bool=False`. YAML+true даёт config error. HTTP включённый instance обслуживает только своего owner; shared scope требует явного server-side pool injection. New requests не используют permissive extra-ignore, включая вложенные condition/selector.
- Codec wire: WaitCondition `kind=until|slot|change`; refs — `{namespace,fact_id,revision}`; capture body `{"refs":[["owner","fact-id"]]}`. Success — encoded public object/list; create HTTP200; ack/unsubscribe HTTP204 и MCP JSON null. Error envelope `{"error":{"code":"LeaseLost","message":"The lease is no longer current."}}`; InvalidDelivery=409, malformed input422, AccessDenied403, CursorExpired410, unsupported501. Exact route/tool/body map — reference; ни один canonical method не исключён.

- [ ] **Step 1: tests wire contract/auth.**

```python
def test_actor_cannot_be_spoofed(authorized_client):
    response = authorized_client.post("/v1/always-on/intents", json={
        "client_key": "run-1", "goal": "Check deployment", "owner_agent_id": "other"})
    assert response.status_code == 422
```

Проверить unauthenticated401; same client_key retry возвращает тот же ID; typed unknown condition422; RevisionConflict409; CursorExpired410; YAML unsupported501 при explicit injection, env misconfig fails startup; JSON datetimes только UTC ISO; token/stacktrace не попадают в error. Невалидный JSON, duplicate object keys, NaN/Infinity, bool-as-integer и logical JSON fields >65536 UTF-8 bytes отвергаются до mutation; лимит применяется к progress/request/result/evidence/selector, не ко всему batch response; arbitrary keys в freeform progress/request/evidence разрешены. Missing и foreign owner-bound IDs одинаково дают AccessDenied, без existence oracle. Полный end-to-end HTTP create→claim→checkpoint→resume→subscribe→poll→ack проходит против SQLite; GET list/get/due не меняют revisions/cursors/leases.

- [ ] **Step 2: выполнить** `.venv/bin/python -m pytest tests/always_on/test_codec.py tests/always_on/test_surfaces.py -q --no-cov`.
- [ ] **Step 3: реализовать router и codec** для всех routes spec, включая `POST /freshness/record`. Scope owner берётся из facade, не query/header/body. Legacy app без always_on не регистрирует новые routes. Handler только decode→facade→encode и domain error mapping; процессы/бесконечные циклы не стартуют при import/app creation. GET endpoints не меняют доменное состояние.
- [ ] **Step 4:** `tests/test_server.py tests/test_config.py` плюс новые tests; весь existing JSON response contract прежних routes сохраняется.
- [ ] **Step 5: commit** `feat: expose scoped always on memory over http`.

## Task P3.3: MCP, CLI и TypeScript parity

Обязательный transport regression после независимого review: расширить существующий `HttpKnowledgeBase.requestVoid` до POST|DELETE + body; при ack сериализовать delivery_id и JSON headers, при delete сохранять прежний bodyless behavior. Проверить status204 без response.json(), bearer/custom headers и structured errors. Одного расширения TypeScript type недостаточно; конкретное тело helper дано в code appendix.

**Files:** Create `_always_on_tools.py`, `npm/src/always_on.ts`, `tests/always_on/test_cli.py`, `npm/src/__tests__/always_on.test.ts`; Modify `mcp_server.py`, `_mcp_tools.py`, cli.py, npm/src/{client,http,types,index}.ts, tests/test_mcp_e2e.py.

**Interfaces:**
- Consumes: codec/router/facade и existing MCP `call(toolName,args)` / HTTP request helper.
- Produces: все MCP tool names spec; `client.alwaysOn` с общим typed surface (`createIntent`, `getIntent`, `listIntents`, `claim`, `renew`, `checkpoint`, `resume`, `captureEvidence`, `subscribe`, `poll`, `ack`, `unsubscribe`, `prepareAction`, `beginAction`, `finishAction`, `reconcileAction`, `setFreshness`, `dueVerifications`, `recordVerification`, `runMaintenance`). captureEvidence transport доступен только scoped IDs.
- TS HTTP implementation `HttpAlwaysOnMemory`, MCP implementation `McpAlwaysOnMemory` реализуют `AlwaysOnClient`; внутренний transport callback использует текущий клиент, без нового fetch/subprocess stack. UTC dates на wire — strings; enum и discriminated WaitCondition совпадают с Python codec.
- В `npm/src/http.ts` constructor передаёт closures к private `requestJson` и `requestVoid`; unsubscribe использует существующий DELETE path. `buildHttpError` сохраняет structured AO code через `AlwaysOnError`; legacy routes сохраняют прежний формат. В `npm/src/index.ts::KnowledgeBase` constructor подключает `McpAlwaysOnMemory((name,args)=>this.client.call(name,args))`, index exports включают все AO types/error. `npm/src/client.ts::_handleLine` не превращает AO codes в regex-parsed text. Полные TS records включают ActionRecord integer revision, lease_generation/blocked_reasons и все resume/freshness fields.
- MCP capabilities сохраняет существующий JSON array contract; новые entries добавляются при explicit enable. Errors — `{error:{code,message}}`; TS `AlwaysOnError.code` сохраняет domain code, не regex по тексту.

- [ ] **Step 1: transport retry/parity tests.**

```typescript
it("does not start polling when constructed", () => {
  const fetch = vi.fn();
  const client = new HttpKnowledgeBase({ baseUrl: "http://localhost", fetch });
  expect(client.alwaysOn).toBeDefined();
  expect(fetch).not.toHaveBeenCalled();
});
```

Дополнительно: golden wire fixtures HTTP/MCP одинаковые inputs→IDs/status enums, 409→AlwaysOnError; `test_transport_retry_keeps_identity`; MCP subprocess restart после poll до ack сохраняет delivery; actor spoof в MCP reject; stdin CLI invalid JSON не оставляет intent; YAML work command понятная ошибка; importing npm не запускает процесс; ESM/CJS exports и build сохраняются.

- [ ] **Step 2: red commands:** `.venv/bin/python -m pytest tests/always_on/test_cli.py tests/test_mcp_e2e.py -q --no-cov`; `npm --prefix npm test -- src/__tests__/always_on.test.ts`.
- [ ] **Step 3: подключить tools/commands/SDK.** `ai-knot work create/list/resume/claim/checkpoint`, `changes subscribe/poll/ack/unsubscribe`, `maintenance run`, `--json`. Checkpoint сложного payload через JSON file/stdin, не shell-interpolated commands. Все routes/tools используют shared typed codec. Добавить capture evidence HTTP/MCP (`POST /evidence/capture`, `memory_evidence_capture`) для полноценной parity; не возвращать чужие records. В CLI `serve`/`serve-mcp` explicit enable передаёт facade; MCP `_make_server` получает backward-compatible keyword `always_on=None`; `_build_kb`/main учитывают config, но не создают daemon. Для CLI достаточно заявленного поднабора; action reconciliation не дублировать ещё одним CLI DSL. Tool capabilities остаётся array и добавляет новые записи только после enable.
- [ ] **Step 4:** `.venv/bin/python -m pytest tests/test_mcp_server.py tests/test_mcp_tools.py tests/test_mcp_e2e.py tests/always_on/test_cli.py -q --no-cov`; `npm --prefix npm run typecheck`; `npm --prefix npm run build`; `npm --prefix npm test`.
- [ ] **Step 5: commit** `feat: support always on memory in mcp cli and typescript clients`.

## Task P3.4: executable host example и документация гарантий

**Files:** Create `examples/always_on_recovery.py`, `docs/always-on.md`; Modify README.md, docs/production-readiness.md, docs/integrations.md, examples/README.md, CHANGELOG.md; Test `tests/always_on/test_example.py`.

**Interfaces:**
- Consumes: public Python API, no private `_storage` use in example.
- Produces: один offline runnable example с fake CI source и отдельным idempotent external store; JSON итог с stages и counters, без подключения настоящего deployment service.

- [ ] **Step 1: integration test запускает** `.venv/bin/python examples/always_on_recovery.py --data-dir <tmp>` в subprocess и проверяет: exitcode0; `completed=true`; `external_effects=1`; `stale_plan_blocked=true`; `model_calls=0`. Второй запуск не создаёт ещё один внешний эффект; отсутствующий idempotency support даёт понятную blocked/unknown ветку, не ложный completed.
- [ ] **Step 2: запустить red** `.venv/bin/python -m pytest tests/always_on/test_example.py -q --no-cov`.
- [ ] **Step 3: написать пример и docs.** Использовать public call sequence reference: claim→get revision→checkpoint wait, poll→durable local inbox receipt→ack, stale basis→replan→ready checkpoint→new claim, begin→fake external durable write→process crash→resume→recovery claim→lookup→reconcile→complete. Внешний fake store — отдельный SQLite-файл с UNIQUE(idempotency_key), не часть AO transaction. Для no-idempotency/no-proof ветки остановиться blocked/unknown без auto-repeat. Объяснить отличие фактов и намерения, enable/upgrade порядок, SQLite/PG/YAML возможности, ack after durable processing, receipts до unsubscribe и новую incarnation при recreate, live/recovery lease, verified_until vs retry_at, expiration без событий, boundary внешних effects и clock assumptions. Добавить versioned migration/backup/export AO tables и procedure restart; обычный facts export не объявлять полным AO backup. Не менять package version до release task, записать Unreleased changelog.
- [ ] **Step 4: test example +** `tests/test_docs_contracts.py tests/test_metadata_contracts.py`; проверить все local doc links.
- [ ] **Step 5: commit** `docs: demonstrate crash recovery and always on memory contracts`.

## Task P3.5: fault matrix, benchmark и CI gates

**Files:** Create `tests/always_on/test_end_to_end.py`, `tests/eval/always_on/{__init__,runner,scenarios}.py`; Modify `.github/workflows/ci.yml`, `.github/workflows/benchmark.yml`, pyproject.toml only to register markers if needed.

**Interfaces:**
- Consumes: public interfaces и пример.
- Produces: `.venv/bin/python -m tests.eval.always_on.runner --storage sqlite --output-dir <dir>`; PostgreSQL uses `--storage postgres` + `AI_KNOT_TEST_PG_DSN`. JSON output `always_on_report.json` содержит setup, per-scenario outcomes/counters/latency, aggregate invariants. Nonzero exit при нарушении binding invariant; не наследовать mock-judge QA score как correctness.
- 10 deterministic scenarios: waiting across restart; duplicate event; read-before-ack crash; stale plan; deleted/expired basis; lost lease; unknown external outcome; verifier outage; consolidation race; snapshot restore. Для replay и action crash использовать настоящие process exits на объявленных checkpoints.

- [ ] **Step 1: unit-sized acceptance test с фиксированным seed.**

```python
def test_all_crash_windows(recovery_scenario):
    for point in ("after_poll", "after_checkpoint", "after_begin", "after_external", "after_finish"):
        result = recovery_scenario(crash_at=point)
        assert result.missing_committed_changes == 0
        assert result.stale_action_starts == 0
        assert result.unintended_duplicate_effects == 0
        assert result.completed
```

`recovery_scenario` обязательно подаёт внешнее reconciliation evidence, когда исход unknown; без него expected blocked, не completed. Проверить дополнительные 5 inputs из Review Focus и таблицу fault matrix spec; new helper fixture не маскирует crash возвратом prepared success.

- [ ] **Step 2: выполнить red** `.venv/bin/python -m pytest tests/always_on/test_end_to_end.py -q --no-cov`.
- [ ] **Step 3: написать deterministic runner и CI wiring.** Стандартные unit/integration tests входят в существующий test job; live PG job обязан запускать новый folder с DSN (для required mode отсутствие DSN = fail, не skip). Performance slow cases идут в существующий benchmark workflow, smoke invariants — в PR CI. Текущий benchmark.yml запускается push/PR: 100k replay выполнять отдельным workflow_dispatch/явно добавленным schedule job, не утверждать наличие nightly cron. Для slow tests `--timeout` увеличить целевым маркером/командой, а не всей suite. Не менять S8–S26 thresholds для прохождения новой функции. В JSON report включить receipt rows/bytes и рост с batches до unsubscribe, наряду с journal bytes, scan counts, query plan, latency; constant-space гарантия не заявляется.
- [ ] **Step 4: полный gate:**

```sh
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
.venv/bin/mypy src/ai_knot/
.venv/bin/python -m pytest -m "not slow" --cov=ai_knot --cov-report=term-missing
.venv/bin/python -m pytest tests/eval/test_gates.py -q --no-cov
.venv/bin/python -m tests.eval.benchmark.runner --multi-agent --mock-judge --ma-storage sqlite --ma-gate
npm --prefix npm run typecheck
npm --prefix npm run build
npm --prefix npm test
```

Повторить required live PG subset с установленным postgres extra и CI DSN. Отдельно запустить runner 1k/10k facts, 1/10/100 subscriptions; 100k-event replay — slow workflow. Сравнить AO disabled p95 recall с baseline на том же runner, использовать >=5 repetitions/median; budget <=20% первоначальный, при нарушении investigate, не менять число в отчёте без review. Измерять scan counts/index plans и размер журнала, а не только wall time.

- [ ] **Step 5: commit** `test: gate always on recovery invariants and replay performance`.

## Финальная приёмка

- [ ] У каждого из шести направлений есть работающий public путь и свой recovery/regression test.
- [ ] Пример работает из нового процесса offline; реальный external side effect не вызывается.
- [ ] SQLite/live PostgreSQL дают одинаковые invariants; YAML остаётся совместимым и честно unsupported для AO.
- [ ] HTTP/MCP/TS JSON contracts совпадают; owner binding и data redaction проверены.
- [ ] CI и benchmark outputs отличают выполненные и пропущенные проверки.
- [ ] Reviewer проверяет весь branch и исходную spec до merge; release/version bump выполняется отдельным существующим release workflow.
