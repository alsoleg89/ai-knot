# 3-Agent Dev Plan: Phase A → A.5 → B → C → D → E (cycle-based)

## Context

Программа Lexical Sufficiency требует параллельной разработки тремя агентами через все фазы (A → E), с фазой E (Product Memory Surface) как финальным слоем. Предыдущая история проекта показала pattern «8 откатов подряд»: aggregate-only метрики не давали per-stage диагностики, single-2-conv runs не отличали ±3pp шум от сигнала, а попытки чинить cat1 поверх ranking knobs провалились (см. memory `project_locomo_phase1_retrieval_exhausted`, `project_locomo_phase1e_revert`, `project_locomo_claims_first_promotion_20260423`).

Этот план разрешает три проблемы сразу:
1. **Diagnostics-first execution** — Agent A0 шипит PROC harness ДО любой backend-работы, чтобы каждое изменение оценивалось per-bucket migration, а не aggregate-noise.
2. **Cycle-based gating вместо calendar-based** — каждый cycle закрывается acceptance gate (per-stage delta + full-10 valid) с двухступенчатой валидацией (2-conv → full-10), не часами или днями.
3. **Vertical slice ownership** — три агента (A0 Diagnostics / A Backend / U UX) owns свой слой через ВСЕ фазы, синхронизируясь только на phase combined gates.

Outcome: LOCOMO full-10 aggregate ≥ 77 % / cat1 ≥ 53 %, LongMemEval +12-15 pp vs Mem0 baseline, plus product surface (Promise Ledger / Decision Log / Memory Lane / Memory Time Travel) functional с реальным backend.

Companion: `research/lexical_sufficiency_program_20260426.md` (теория + per-metric targets).

---

## 0. Frame

Три агента работают параллельно через Claude Code, каждый owns vertical slice через все фазы:

- **Agent A0 (Diagnostics Engineer)** — PROC harness + расширения для каждой phase metric.
- **Agent A (Backend Engineer)** — production retrieval / pack / storage код.
- **Agent U (UX Engineer)** — clickable prototype + integrations с реальным backend.

**Cycle-based execution model:**
- Cycle = (одна задача → один PR → один gate review → ship-or-revert decision).
- Cycle закрывается acceptance gate, не временным окном.
- Параллельность: 3 агента работают одновременно; sync points между cycles (см. §5).
- Phase прогрессия: каждая phase открывается только после combined acceptance gate предыдущей.

**Phases в scope:**

| Phase | Scope | Trigger |
| --- | --- | --- |
| Phase A | Frame Lexical Bridge | always-on (стартовая) |
| Phase A.5 | CPL (Conversational Personal Lexicon) | conditional: lift +2-3pp но не +5pp |
| Phase B | Evidence Pack V2 + Lost-in-the-Middle reorder | conditional: LLM-fail bucket dominant в Phase A residual |
| Phase C | HeidelTime utterance-time grounding | conditional: cat2 / temporal bucket dominant |
| Phase D | Mention Graph projection | conditional: hard-miss bucket dominant |
| Phase E | Product Memory Surface API + Family Affordances | unconditional после A+B+C+D ship; финальный layer |

---

## 1. Cycle map: phase × agent → cycle

| Phase | Agent A0 (Diagnostics) | Agent A (Backend) | Agent U (UX) |
| --- | --- | --- | --- |
| Phase A.0 (prereq) | A0.proc PROC baseline harness | — (waits) | U.scaffold standalone proto scaffold + mock data |
| Phase A | A0.lexical LexicalExpansionUplift metric | A.lexicon query_lexicon.py + 4-channel | U.trace Inquiry Trace + Knot view (mock then real) |
| Phase A.5 (conditional) | A0.cpl CPL freshness metric | A.cpl ConversationalPersonalLexicon | — (UX unchanged) |
| Phase B (conditional) | A0.pack PackPositionEntropy + ReaderRecoveryRate | A.pack Evidence Pack V2 builder | U.pack Pack inspector view |
| Phase C (conditional) | A0.time TemporalResolutionAccuracy | A.heideltime HeidelTime ingest enrichment | U.timeline event-date strand viz |
| Phase D (conditional) | A0.graph EntityHopReach metric | A.graph Mention Graph projection | U.graph Knot crossings viz |
| Phase E | A0.product Promise Ledger / Decision Log harness + Memory Lane pilot tooling | A.surface memory_profile API + product commands + Memory Time Travel backend (fact_history) | U.surface Promise Ledger + Decision Log + Memory Lane + functional Memory Time Travel |

**Total cycles per agent:**
- A0: 7 cycles (proc, lexical, cpl, pack, time, graph, product)
- A: 6 cycles (lexicon, cpl, pack, heideltime, graph, surface) — A0.proc has no A counterpart
- U: 7 cycles (scaffold, trace, pack, timeline, graph, surface) + cpl skipped

**Hard dependencies:**
- A0.proc must ship before A.lexicon can validate P1.
- A0.* metric must ship before A.* can validate corresponding gate.
- U.* integration cycle requires A0.* + A.* merged (so prototype consumes real diagnostics output).
- Phase E backend (`fact_history` table) is hard prerequisite for U.surface Memory Time Travel functional implementation.

---

## 2. Agent A0 — Diagnostics Engineer (per-cycle breakdown)

**Mission:** дать каждому изменению per-stage delta вместо aggregate-noise. Останавливает «8 откатов подряд» pattern. Diagnostics scale с phase complexity — каждая phase добавляет свои metrics.

**Branch convention:** `feat/diag/<cycle-tag>` (e.g. `feat/diag/proc`, `feat/diag/lexical`)
**PR target:** `main`
**Reviewer:** один senior engineer + plan author sign-off

### Cycle A0.proc — PROC harness baseline (Phase A.0 prereq)

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `aiknotbench/src/aiknot.ts` | extend (~30 LOC) | `recallWithTrace(question): Promise<{context, trace}>` proxy для `kb.recall_facts_with_trace`. JS API only — не MCP. |
| `aiknotbench/src/runner.ts` | extend (~50 LOC) | `CheckpointResult` extended: `{stage1CandidateIds, stage4PackIds, stage0LexicalTrace, factContentMap}`. Запись `data/runs/<runId>/diagnostics_raw.jsonl`. |
| `aiknotbench/scripts/build_gold_mapper.ts` | CREATE (~80 LOC) | `D{dialog}:{turn} → fact.id` mapper. Critical: dated mode → 1 turn = до 3 facts (sliding window). Mapper помечает все 3 как gold-bearing. |
| `aiknotbench/scripts/diagnose_recall.ts` | CREATE (~150 LOC) | Compute `RawGoldExists`, `PoolGoldRecall@K`, `PackGoldRecall@Budget`, `GoldPackPosition`, `DistractorDensity`, `ReaderFailDespiteGold`. Emit `diagnostics.jsonl`. |
| `aiknotbench/scripts/eval_recall_buckets.ts` | CREATE (~80 LOC) | Bucket classifier (LLM-fail / partial-recall / low-recall / hard-miss); `--baseline` / `--candidate` diff mode. |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `aiknotbench/tests/test_gold_mapper.ts` | CREATE | `D1:3` → `(session=0, turn=2)`; sliding-window 3× inflation; empty evidence → null; fingerprint stability across re-ingest. |
| `aiknotbench/tests/test_runner_trace_capture.ts` | CREATE | Integration: runner на 1 conv с trace flag; assert `diagnostics_raw.jsonl` валиден. |
| `aiknotbench/tests/test_diagnose_recall.ts` | CREATE | На synthetic input с known gold positions → metrics match manual computation. |
| `aiknotbench/tests/test_eval_recall_buckets.ts` | CREATE | Каждый bucket-rule fires для edge cases; diff-mode классифицирует regressions. |

**Acceptance gate:**
1. Fresh re-run на `repro/dated-1167e70` (~10 min wall-clock; old checkpoints lack trace fields) воспроизводит **9 LLM-fail / 5 partial / 10 low-recall / 6 hard-miss** per memory `project_locomo_cat1_true_ceiling_20260423.md`. **Hard gate.**
2. Coverage cat1+cat2+cat3+cat4.
3. < 30 sec на conv для diagnostics phase (after bench run).
4. Zero LLM calls в harness.
5. `research/baseline_buckets_20260427.md` committed с pre-Phase-A bucket-таблицей.

**PR checklist:**
- [ ] no new LLM calls (runtime)
- [ ] gold-mapper handles dated-mode 3× sliding-window inflation
- [ ] trace capture is opt-in (env flag/CLI arg)
- [ ] `diagnostics_raw.jsonl` size < 10 MB per conv для full-10
- [ ] baseline 9/5/10/6 buckets reproduced на `repro/dated-1167e70`
- [ ] all 4 test files green в CI
- [ ] anti-overfit gates §6 verified

**Block-merge:** baseline buckets не воспроизводятся → re-investigate, не merge.

### Cycle A0.lexical — LexicalExpansionUplift metric (Phase A)

**Trigger:** Cycle A.lexicon ships flagged.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `aiknotbench/scripts/diagnose_recall.ts` | extend (~30 LOC) | Read `trace["stage0_lexical_bridge"]` при наличии; compute `LexicalExpansionUplift = PoolGoldRecall(with) − PoolGoldRecall(without)`; per-frame breakdown и per-intent. |
| `aiknotbench/scripts/eval_recall_buckets.ts` | extend (~20 LOC) | Diff-mode рендерит lexical bucket migration table. |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `aiknotbench/tests/test_diagnose_recall.ts` | extend | На synthetic trace с `stage0_lexical_bridge` block → `LexicalExpansionUplift` корректен. |

**Acceptance gate:**
1. Diff-mode comparison `baseline-pre-lexical` vs `phase-a-frames` рендерит valid bucket migration table.
2. `LexicalExpansionUplift` per-frame совпадает с manual recount на 5 random samples.

**PR checklist:**
- [ ] `LexicalExpansionUplift` computed только когда `stage0_lexical_bridge` присутствует
- [ ] no regression в baseline metric set
- [ ] diff-mode integration test passes

### Cycle A0.cpl — CPL freshness metric (Phase A.5, conditional)

**Trigger:** Phase A delivered +2-3pp но не +5pp → CPL activated.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `aiknotbench/scripts/diagnose_recall.ts` | extend (~25 LOC) | `CPLContribution` = доля expansion terms originating from CPL vs hardcoded frames; `CPLFreshness` = recency CPL update vs query time. |

**Tests:** один extend на `test_diagnose_recall.ts` для CPL trace fields.

**Acceptance gate:** metric computed на synthetic CPL fixture; consistent with manual recount.

### Cycle A0.pack — Pack V2 metrics (Phase B, conditional)

**Trigger:** A.pack ships flagged.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `aiknotbench/scripts/diagnose_recall.ts` | extend (~40 LOC) | Add: `PackPositionEntropy` (Shannon H по позициям gold в pack), `ReaderRecoveryRate` (fraction of LLM-fail fixed by V2), `LostInMiddleSignal` (gold в middle vs head/tail). |

**Tests:** synthetic packs с known gold positions.

**Acceptance gate:** metrics rendered в bucket-migration table; «LLM-fail → CORRECT» migration count visible.

### Cycle A0.time — Temporal grounding metric (Phase C, conditional)

**Trigger:** A.heideltime ships flagged.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `aiknotbench/scripts/diagnose_recall.ts` | extend (~35 LOC) | `TemporalResolutionAccuracy` (HeidelTime resolved date vs gold answer date), `EventDateMatchRate` (event_date tag matches BM25F query). |
| `aiknotbench/scripts/build_gold_mapper.ts` | extend (~20 LOC) | Если gold answer is a date string, parse it и attach to gold record для temporal scoring. |

**Tests:** synthetic conversations с known temporal phrases.

**Acceptance gate:** `TemporalResolutionAccuracy ≥ 85 %` на sample of 50 manually-annotated turns (one-time validation).

### Cycle A0.graph — Graph reachability metric (Phase D, conditional)

**Trigger:** A.graph ships flagged.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `aiknotbench/scripts/diagnose_recall.ts` | extend (~40 LOC) | `EntityHopReach` (fraction hard-miss где gold reachable via 1-hop entity link), `CoreferenceResolveRate` (pronoun → entity match rate). |

**Tests:** synthetic Mention Graph fixture; assert hop-reach metric on known examples.

**Acceptance gate:** hard-miss bucket migration tracked в diff-mode; explicit «hard-miss → low-recall» count.

### Cycle A0.product — Product scenario harness (Phase E)

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `aiknotbench/scripts/eval_promise_ledger.ts` | CREATE (~120 LOC) | Synthetic family-conversation suite (50 promises × 200 turns); recall ≥ 90 % unfulfilled w/ correct slots. |
| `aiknotbench/scripts/eval_decision_log.ts` | CREATE (~100 LOC) | Project-conversation suite (30 decisions × 150 turns); recall ≥ 85 %. |
| `aiknotbench/scripts/eval_memory_lane.ts` | CREATE (~150 LOC) | Weekly digest scoring; coverage ≥ 80 % flagged turns; supports 10-user pilot data ingestion. |
| `aiknotbench/data/synthetic/family_50_promises.json` | CREATE (~300 lines synth data) | Synthetic fixture for Promise Ledger eval. |
| `aiknotbench/data/synthetic/project_30_decisions.json` | CREATE (~250 lines synth data) | Synthetic fixture for Decision Log eval. |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `aiknotbench/tests/test_eval_promise_ledger.ts` | CREATE | Score computation correct on hand-checked subset. |
| `aiknotbench/tests/test_eval_decision_log.ts` | CREATE | Same. |
| `aiknotbench/tests/test_eval_memory_lane.ts` | CREATE | Coverage scoring deterministic. |

**Acceptance gate:**
1. Promise Ledger ≥ 70 % с production retrieval (Phase A baseline), ≥ 90 % с full Phase E.
2. Decision Log ≥ 65 % baseline, ≥ 85 % full.
3. Memory Lane: 10-user pilot avg ≥ 4/5 informativeness (gathered after E ships).

---

## 3. Agent A — Backend Engineer (per-cycle breakdown)

**Mission:** Production retrieval / pack / storage / API код. Strict no-LLM-on-recall path. Каждый cycle gated by Agent A0 metric.

**Branch convention:** `feat/<cycle-tag>` (e.g. `feat/lexicon`, `feat/pack-v2`)
**PR target:** `main`
**Reviewer:** один senior backend engineer + plan author sign-off

### Cycle A.lexicon — Frame Lexical Bridge (Phase A)

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `src/ai_knot/query_lexicon.py` | CREATE (~120 LOC) | `LexicalExpansion` dataclass; `LEXICON: dict[str, FrameDef]` с 6 frame families; `expand_query_lexically(query, intent, max_terms_per_intent)`. Pure-python, никаких deps. Все веса < 1.0. |
| `src/ai_knot/_query_intent.py` | extend (~10 LOC) | `lexical_expansion_max` поле в `PipelineConfig`; заполнить per IALE table из programme doc. |
| `src/ai_knot/knowledge.py::_execute_recall` | extend (~10-15 LOC) | После `intent = classify_recall_intent(query)`: gate на `AI_KNOT_LEXICAL_BRIDGE` env, expand, передать в Channel A (`index.score`), Channel B (rare-token с `RARE_EXP_CAP=5`); Channel D deferred. Trace block. |
| `src/ai_knot/_inverted_index.py::score` | verify only | Already accepts `expansion_weights`. Не трогаем. |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `tests/test_query_lexicon.py` | CREATE | Each frame matches только её intents; NAVIGATIONAL никогда не expanded; `max(weight) < 1.0`; deterministic. |
| `tests/test_recall_with_lexical_bridge.py` | CREATE | 2 mini-fixture KBs; assert trace presence, Channel A pool growth, Channel B rare-cap, NAVIGATIONAL identical. |
| `tests/test_lexicon_anti_overfit.py` | CREATE | Scan LEXICON; assert no add-term appears in LOCOMO answer set. **Blocking test.** |

**Acceptance gate (двухступенчатый):**

*Stage 1 — 2-conv* (closes when Agent A0.lexical green):
- `AI_KNOT_LEXICAL_BRIDGE=1` runs cleanly на conv0+conv1.
- Diagnostics: `LexicalExpansionUplift ≥ +2 pp` на cat1 PoolGoldRecall.
- DistractorDensity not up > 10 %.
- All 3 test files pass.

*Stage 2 — full-10:*
- Sequential 3-batch (per `dated_full10_analysis_20260426.md` RPD lessons).
- cat1 ≥ 41 % (baseline 38.8 + 2 pp).
- Aggregate ≥ 64 % (baseline 62.2 + 2 pp).
- Ни одна категория не падает > 1 pp.

**Decision tree:**
- P1 fail → halt; pivot to A.graph (Phase D).
- P1 pass + P2 fail → branch early to A.pack (Phase B).
- Both pass + lift < +5 pp → trigger A.cpl (Phase A.5).
- Both pass + lift ≥ +5 pp → ship default-on (separate follow-up PR).
- Any category drops > 2 pp → revert immediately.

**PR checklist:**
- [ ] no new LLM calls (runtime)
- [ ] no LOCOMO names/gold answers in lexicon
- [ ] stage diagnostics pass (requires Agent A0.lexical merged)
- [ ] mean tokens/query ≤ baseline +5%
- [ ] cat4/cat5 no-answer behavior stable (≤1pp regression)
- [ ] full-10 OR averaged 2-conv validation (не single 2-conv)
- [ ] `trace["stage0_lexical_bridge"]` always present when flag on

### Cycle A.cpl — CPL conditional (Phase A.5)

**Trigger:** Phase A lift +2-3pp но не +5pp.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `src/ai_knot/query_lexicon.py` | extend (~60 LOC) | `ConversationalPersonalLexicon` class; `top_k_pmi(term, k)` lookup; integration с `expand_query_lexically` (CPL terms appended после frames). |
| `src/ai_knot/storage/sqlite_storage.py` | extend (~50 LOC) | `cpl_pmi` table schema; `update_pmi(agent_id, terms)` + `top_pmi(agent_id, term, k)` accessors. |
| `src/ai_knot/knowledge.py::add_episodic` | extend (~10 LOC) | Hook на CPL update после fact saved (gated on `AI_KNOT_CPL=1`). |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `tests/test_cpl_pmi.py` | CREATE | PMI math correctness; incremental update consistency; θ threshold honored. |
| `tests/test_cpl_integration.py` | CREATE | Ingest 100 synthetic turns; query with vs без CPL; assert per-tenant PMI table grows linearly. |

**Acceptance gate:**
- 2-conv: additional `LexicalExpansionUplift` from CPL ≥ +1 pp on top of frames.
- CPL table size < 5 MB per tenant after 1000 turns.
- Phase A.5 combined acceptance: full-10 cat1 ≥ 43 % (baseline + 4 pp from frames + 1 from CPL).

**PR checklist:** same as A.lexicon plus:
- [ ] CPL table per-tenant (no cross-agent leakage)
- [ ] PMI recomputation cost < 100 ms for 1000-turn agent
- [ ] CPL falls back gracefully if table missing (older DBs)

### Cycle A.pack — Evidence Pack V2 (Phase B, conditional)

**Trigger:** Phase A residuals show LLM-fail bucket dominant (>30%).

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `src/ai_knot/pack.py` | CREATE (~200 LOC) | `EvidencePackBuilder` class; structured pack: `{header, raw_ribbons, what_we_dont_know}`. Lost-in-the-Middle reorder: rank-1→pos-1, rank-2→end, rank-3→pos-2, rank-4→end-1, … для FACTUAL/AGGREGATIONAL. Bounded budget (default 1500 tokens). |
| `src/ai_knot/knowledge.py` | extend (~20 LOC) | Pack builder gated on `AI_KNOT_PACK_V2=1`; replaces current sliding-window dump для context construction. |
| `aiknotbench/src/aiknot.ts::recall` | extend (~10 LOC) | If `AI_KNOT_PACK_V2=1`, return structured payload (TS-side) instead of pre-rendered string. |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `tests/test_pack_v2.py` | CREATE | Reorder positions correct (rank1→head, rank2→tail, …); budget bound respected; «what we don't know» rail emitted only when uncertainty signal present; structured fields parseable as JSON. |
| `tests/test_pack_no_answer_injection.py` | CREATE | Anti-overfit: pack никогда не contains final answer sentence; only evidence. |

**Acceptance gate (двухступенчатый):**
- 2-conv: `ReaderRecoveryRate` (от A0.pack) ≥ +25 % LLM-fail bucket migration to CORRECT.
- full-10: aggregate ≥ 70 % (Phase A baseline + 5 pp), cat1 ≥ 47 %.
- mean tokens/query ≤ baseline (Pack V2 плотнее текущего).

**PR checklist:** same +
- [ ] structured pack parseable as JSON (no free-text scrambling)
- [ ] no answer leak in pack (`test_pack_no_answer_injection.py`)
- [ ] rendered pack ≤ 1500 tokens default budget
- [ ] Lost-in-the-Middle reorder verified on FACTUAL queries

### Cycle A.heideltime — Utterance-time grounding (Phase C, conditional)

**Trigger:** Phase A+B residual cat2/temporal bucket dominant.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `pyproject.toml` | extend | Add `python-heideltime` (or wrapper) + `lxml` dep. |
| `src/ai_knot/_date_enrichment.py` | extend (~80 LOC) | HeidelTime pass: для каждого turn (с `[date]` reference) → resolve TIMEX3 → emit `event_date:` BM25F tags. New tag namespace separate from `session_date:`. |
| `src/ai_knot/knowledge.py::_execute_recall` | extend (~15 LOC) | EVENT/TEMPORAL queries: HeidelTime-pass на query → required-match `event_date:` tags. RRF: `event_date:` 4× weight `session_date:` для temporal intents. |
| `src/ai_knot/_query_intent.py` | extend (~5 LOC) | Add EVENT/TEMPORAL routing flag. |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `tests/test_heideltime_enrichment.py` | CREATE | «yesterday» w/ 2024-01-15 reference → 2024-01-14 tag; «last weekend» → range; ambiguous phrase → no fake precision. |
| `tests/test_temporal_query_routing.py` | CREATE | EVENT query with date phrase → resolved tags drive scoring; non-temporal query → no penalty. |
| `tests/test_phase_1e_regression.py` | CREATE | Replay 6 questions Phase 1E broke (from `project_locomo_phase1e_revert.md`); assert HeidelTime resolves correctly (no off-by-N-day). |

**Acceptance gate:**
- 2-conv: `TemporalResolutionAccuracy ≥ 85 %` on sample.
- full-10: cat2 ≥ 53 % (baseline 48.7 + 4 pp); cat1/3/4 stable.

**PR checklist:** same +
- [ ] HeidelTime adds searchable tags only — никогда не creates new claim
- [ ] Phase 1E replay test passes (no off-by-N-day regression)
- [ ] non-temporal queries unaffected (cat3 stable)

### Cycle A.graph — Mention Graph projection (Phase D, conditional)

**Trigger:** Phase A+B+C residual hard-miss bucket dominant.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `src/ai_knot/storage/sqlite_storage.py` | extend (~80 LOC) | `mention_graph` table: `(agent_id, source_fact_id, target_entity_id, edge_type, weight)`. Edge types: `pronoun-resolves-to`, `group-alias`, `coreference`. |
| `src/ai_knot/mention_graph.py` | CREATE (~150 LOC) | Builder: ingest-time entity/pronoun extraction; recall-time 1-hop expansion для hard queries. Gated on `AI_KNOT_MENTION_GRAPH=1`. |
| `src/ai_knot/knowledge.py::_execute_recall` | extend (~20 LOC) | Channel C (entity-hop) extended: traverse mention_graph для 1-hop entity expansion when Stage-1 pool empty (hard-miss recovery). |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `tests/test_mention_graph.py` | CREATE | Edge extraction: «she went there» → resolves to last-named entity + last-named place; group alias «the kids» → all child-entities. |
| `tests/test_mention_graph_recall.py` | CREATE | Synthetic conv с pronoun chain; assert hard-miss recoverable via 1-hop. |
| `tests/test_mention_graph_no_circular.py` | CREATE | Edge cycles detected; recall doesn't infinite-loop. |

**Acceptance gate:**
- 2-conv: `EntityHopReach ≥ 50 %` (half of hard-miss recoverable via 1-hop).
- full-10: hard-miss bucket size ≤ 50% baseline; aggregate ≥ 75 %.

**PR checklist:** same +
- [ ] mention_graph rebuildable from raw episodes (no source-of-truth violation)
- [ ] graph hop bounded (1-hop only, no recursive expansion in Phase D)
- [ ] graceful degrade if mention_graph table empty (older DBs)

### Cycle A.surface — Product Memory Surface API + fact_history backend (Phase E)

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `src/ai_knot/storage/sqlite_storage.py` | extend (~120 LOC) | `fact_history` table: `(fact_id, version, valid_from, valid_to, agent_id, snapshot_json)`. Append-only event log (replaces current save() replace-all). Migration script для existing DBs. |
| `src/ai_knot/memory_profile.py` | CREATE (~250 LOC) | `MemoryProfile` class: aggregates `stable_facts`, `current_preferences`, `open_commitments`, `recent_changes`, `uncertainty_conflicts`, `evidence_links`. Computed lazily от raw + projections. |
| `src/ai_knot/product_commands.py` | CREATE (~200 LOC) | Public API: `inspect_recall(query)`, `show_raw_evidence(fact_id)`, `show_last_changed(entity)`, `show_conflicts(entity)`, `export_memory_report()`, `recall_as_of(query, timestamp)` (Memory Time Travel backend — replays projections с frozen `as_of` cutoff против `fact_history`). |
| `src/ai_knot/mcp_server.py` (or equivalent) | extend (~50 LOC) | Expose product_commands через MCP surface (опционально — отдельное product решение). |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `tests/test_fact_history.py` | CREATE | Append-only behavior; migration не теряет existing facts; `valid_to` correctly set on overwrite. |
| `tests/test_memory_profile.py` | CREATE | Each section computed correctly от synthetic conv; conflict detection finds genuine contradictions. |
| `tests/test_product_commands.py` | CREATE | Each command returns expected shape; provenance chain preserved. |
| `tests/test_recall_as_of.py` | CREATE | Time-travel: `as_of=T1` returns answer based only on facts before T1; `as_of=T2 > T1` returns updated answer; никаких leaks из future. |
| `tests/test_recall_as_of_acceptance.py` | CREATE | Codex Phase 5 product scenarios pass: profile lookup, changed preference, forgotten/stale fact, commitment timeline, abstention. |

**Acceptance gate:**
- All product scenarios (Codex Phase 5) pass.
- Memory inspectable достаточно для dev debugging без чтения SQLite вручную.
- `recall_as_of` deterministic: same `(query, timestamp)` → same result across re-runs.
- Migration script runs cleanly на `repro/dated-1167e70` DB.

**PR checklist:**
- [ ] `fact_history` append-only (no destructive update); migration tested
- [ ] `recall_as_of` doesn't leak future facts (`test_recall_as_of.py` blocking)
- [ ] memory_profile sections rebuild from raw + projections (no source-of-truth violation)
- [ ] product commands have stable API surface (versioned)
- [ ] all 5 test files green

---

## 4. Agent U — UX Engineer (per-cycle breakdown)

**Mission:** clickable prototype + integrations с реальным backend. Никаких production файлов до Phase E (где backend ready) — только `prototypes/knot-ux/`.

**Branch convention:** `feat/ux/<cycle-tag>` (e.g. `feat/ux/scaffold`, `feat/ux/trace`)
**PR target:** `main` (но шипится в `prototypes/`)
**Reviewer:** один UI engineer + plan author sign-off; reviewer's special charge — find overclaims

### Cycle U.scaffold — Standalone scaffold + mock data (Phase A.0 prereq parallel)

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `prototypes/knot-ux/README.md` | CREATE (~50 LOC) | How to run; explicit «prototype only — not production» notice; backend dependency mapping. |
| `prototypes/knot-ux/package.json` | CREATE (~30 LOC) | Standalone (no aiknotbench workspace dep); Vite + React 18 + Tailwind. |
| `prototypes/knot-ux/src/types.ts` | CREATE (~80 LOC) | TypeScript types для trace JSON + Knot view types (strand, bead, crossing). |
| `prototypes/knot-ux/src/data/mock-trace.json` | CREATE (~150 LOC) | Synthetic trace fixtures, 3-5 example questions (factual / aggregational / exploratory). |
| `prototypes/knot-ux/src/data/mock-knot.json` | CREATE (~200 LOC) | Synthetic conv: 3-4 entities (Sarah/Mom/Apollo/Camping), 15-20 beads, 5-8 crossings. **No LOCOMO names.** |

**Acceptance gate:** `cd prototypes/knot-ux && npm install && npm run dev` — dev server starts cleanly.

**PR checklist:**
- [ ] all under `prototypes/knot-ux/` — no production paths touched
- [ ] standalone runnable (no aiknotbench dep)
- [ ] no LOCOMO names in mock fixtures
- [ ] README explains prototype scope

### Cycle U.trace — Inquiry Trace + Knot view (Phase A)

**Trigger:** A0.proc and A.lexicon merged (для real trace integration по завершении).

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `prototypes/knot-ux/src/components/InquiryTrace.tsx` | CREATE (~200 LOC) | Glass-box trace; 6-step format (intent → expansion → candidate count → top 3 evidence → answer rationale → confidence). Mock data first; real `diagnostics.jsonl` integration as second pass. |
| `prototypes/knot-ux/src/components/KnotView.tsx` | CREATE (~250 LOC) | Default home; vertical strands per entity, time axis, beads, ribbons. Visual demo с mock data. Tap → fact card stub. |
| `prototypes/knot-ux/src/components/MemoryTimeTravel.tsx` | CREATE stub (~80 LOC) | UI shell + visible disclaimer «Backend: not yet implemented — requires Phase E (`fact_history` + `recall_as_of`)». **No real time-travel logic.** |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `prototypes/knot-ux/src/__tests__/InquiryTrace.test.tsx` | CREATE | Trace JSON → 6 sections rendered; collapse/expand interaction; missing field → graceful degradation. |
| `prototypes/knot-ux/src/__tests__/KnotView.test.tsx` | CREATE | mock-knot.json → correct strand/bead count; bead click → fact card opens; zoom/pan don't crash. |

**Acceptance gate:**
1. Standalone runnable (carry-forward from U.scaffold).
2. After A0.proc + A.lexicon merged: prototype loads real `diagnostics.jsonl` от Agent A0 для one selected question и renders InquiryTrace с реальными expansion terms / candidate counts / evidence. **Hard gate** — validates Agent A trace format usable downstream.
3. Memory Time Travel disclaimer visible in UI (not just code comment).
4. Demo readiness: 2-min user walkthrough (Knot view → ask question → see Inquiry Trace → click Memory Time Travel → see disclaimer).

**PR checklist:**
- [ ] all under `prototypes/knot-ux/` — no production paths touched
- [ ] standalone runnable (no aiknotbench dep)
- [ ] Memory Time Travel honestly marked stub в UI (visible disclaimer, not just code comment)
- [ ] Inquiry Trace consumes real Agent A0 `diagnostics.jsonl` format (verified after A0+A merged)
- [ ] no LOCOMO data in mock fixtures
- [ ] README backend-dependency mapping accurate

### Cycle U.pack — Pack inspector view (Phase B)

**Trigger:** A0.pack + A.pack merged.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `prototypes/knot-ux/src/components/PackInspector.tsx` | CREATE (~180 LOC) | Visualize structured pack: header / raw_ribbons (with positions) / «what we don't know». Highlight Lost-in-the-Middle reorder visually (head/tail color-coded). |

**Tests:** `PackInspector.test.tsx` — render structured pack JSON; assert reorder visualization correct.

**Acceptance gate:** loads real Pack V2 JSON output от A.pack для one question; renders correctly.

### Cycle U.timeline — Event-date strand viz (Phase C)

**Trigger:** A0.time + A.heideltime merged.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `prototypes/knot-ux/src/components/TimelineStrand.tsx` | CREATE (~150 LOC) | Strand уже existed in KnotView; this adds `event_date` vs `session_date` distinction (visual: solid bead = event_date, hollow = session_date). |
| `prototypes/knot-ux/src/components/KnotView.tsx` | extend (~30 LOC) | Use TimelineStrand component для each strand; toggle «show resolved event dates» overlay. |

**Acceptance gate:** Real `event_date:` tag traces from A0.time consumed; visual distinction visible.

### Cycle U.graph — Knot crossings viz (Phase D)

**Trigger:** A0.graph + A.graph merged.

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `prototypes/knot-ux/src/components/CrossingsView.tsx` | CREATE (~200 LOC) | Render mention_graph edges as ribbons crossing between strands. Color-coded by edge type (pronoun/group-alias/coreference). Click ribbon → «why this crossing» explanation. |

**Acceptance gate:** Real mention_graph data от A.graph consumed; visualizations match test fixture.

### Cycle U.surface — Product surfaces + functional Memory Time Travel (Phase E)

**Trigger:** A.surface merged (`fact_history` + `recall_as_of` ready).

**Files:**

| Файл | Action | Назначение |
| --- | --- | --- |
| `prototypes/knot-ux/src/components/PromiseLedger.tsx` | CREATE (~180 LOC) | Filter view: commitments older than threshold; sortable by overdue. Backed by A.surface's `memory_profile.open_commitments`. |
| `prototypes/knot-ux/src/components/DecisionLog.tsx` | CREATE (~150 LOC) | Filter view: decisions с rationale. Backed by memory_profile. |
| `prototypes/knot-ux/src/components/MemoryLane.tsx` | CREATE (~200 LOC) | Weekly digest card; configurable threshold; «important» tag interaction. |
| `prototypes/knot-ux/src/components/MemoryTimeTravel.tsx` | REPLACE stub (~250 LOC) | Functional time-travel cursor; calls `recall_as_of(query, timestamp)` от A.surface; renders comparison «before vs after fact landed». **Removes stub disclaimer.** |
| `prototypes/knot-ux/src/components/DisagreeWithKnot.tsx` | CREATE (~120 LOC) | Bead → «this is wrong» → POST к backend; downgrades CPL-PMI weights (если CPL active). |

**Tests:**

| Тест-файл | Action | Coverage |
| --- | --- | --- |
| `prototypes/knot-ux/src/__tests__/PromiseLedger.test.tsx` | CREATE | Filter logic; sort order. |
| `prototypes/knot-ux/src/__tests__/DecisionLog.test.tsx` | CREATE | Same. |
| `prototypes/knot-ux/src/__tests__/MemoryLane.test.tsx` | CREATE | Threshold respected; weekly aggregation correct. |
| `prototypes/knot-ux/src/__tests__/MemoryTimeTravel.test.tsx` | REPLACE stub-only test | Real `recall_as_of` integration tested with mock backend; cursor interaction; before/after rendering. |
| `prototypes/knot-ux/src/__tests__/DisagreeWithKnot.test.tsx` | CREATE | Wrong-marking flow; backend call payload correct. |

**Acceptance gate:**
1. All 4 product surfaces (Promise Ledger / Decision Log / Memory Lane / Memory Time Travel) functional с real A.surface backend.
2. 10-user Memory Lane informativeness pilot ≥ 4/5 (ran AFTER PR merged; tracked separately).
3. Memory Time Travel disclaimer removed (functional now).
4. Disagree-with-Knot writes back to CPL-PMI (if A.cpl shipped) or no-op gracefully.

**PR checklist:**
- [ ] all 4 product surfaces consume real A.surface API (not mock)
- [ ] Memory Time Travel disclaimer removed (functional)
- [ ] no LOCOMO data in any fixture
- [ ] DisagreeWithKnot graceful no-op if CPL not active
- [ ] all 5 test files green

---

## 5. Sync points между агентами (cycle-gated)

| Sync point | Trigger | Что должно быть готово | Кто разблокируется |
| --- | --- | --- | --- |
| S1: trace API ready | A0.proc PR opens | `recallWithTrace` JS API callable | A may start A.lexicon scaffold; U may start integration plan для U.trace |
| S2: PROC baseline committed | A0.proc gate green | `research/baseline_buckets_20260427.md` committed; 9/5/10/6 reproduced | A.lexicon validation now possible; Phase A officially open |
| S3: Phase A combined gate | A0.lexical + A.lexicon + U.trace all green | Lexical bridge default-OFF flag merged; full-10 validated; UX integration on real diagnostics | Phase A.5 / Phase B / Phase C / Phase D decision (per §7 decision tree) |
| S4: Each subsequent phase combined gate | A0.* + A.* + U.* all green | Phase metric, backend, UX all merged | Next phase opens |
| S5: Phase E backend ready | A.surface PR opens | `fact_history` + `recall_as_of` callable | U.surface может replace MemoryTimeTravel stub functional |
| S6: Phase E combined gate | A0.product + A.surface + U.surface all green | Full product surface live; 10-user pilot kicked off | Program complete; ongoing maintenance mode |

**Communication discipline:**
- Sync points в shared notes (gate updates), не chat.
- Каждый агент пишет 3-line update at gate transition: что готово, что blocked, что next.
- Block escalation: если upstream agent blocked, downstream agent appends one comment к gate notes «blocked on …» и pauses. Никаких параллельных workarounds.

---

## 6. Per-cycle review process (universal)

Each cycle PR MUST satisfy 7-item Codex anti-overfit checklist PLUS cycle-specific PR checklist (each cycle defines its own above):

- [ ] no new LLM calls (runtime) — exception: Phase B opt-in offline-LLM extras explicit
- [ ] no LOCOMO names/gold answers anywhere в production code or fixtures
- [ ] stage diagnostics pass (PROC harness reports per-stage delta)
- [ ] token budget controlled (mean tokens/query ≤ baseline +5% Phase A; ≤ baseline для Phase B+)
- [ ] cat4/cat5 no-answer behavior stable (no regression > 1pp)
- [ ] full-10 OR averaged 2-conv validation (не single 2-conv noise)
- [ ] raw provenance visible в outputs (every derived item links to raw episode)

**Reviewer roles per agent:**
- A0 cycles: senior engineer + plan author sign-off (anchored to PROC spec).
- A cycles: senior backend engineer + plan author sign-off.
- U cycles: UI engineer + plan author sign-off; reviewer special charge — find overclaims, default assumption «prototype overclaims».

**Block-merge conditions:**
- Acceptance gate red.
- Any anti-overfit checklist item unchecked.
- LOCOMO name leak detected (`test_lexicon_anti_overfit.py` and equivalents are blocking).
- Trace serialization size unbounded (>10 MB per conv).

**Revert protocol:**
- Any phase gate failure → revert merged PR within 1 cycle; root-cause; re-attempt or pivot to alternative phase.
- «Stop-and-revert on regression» rule (memory `feedback_regression_stop_rule.md`): если 2-conv упал > 2 pp от baseline — revert, не fix поверх.

---

## 7. Risk register (cross-phase)

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Trace serialization > 10 MB per conv для full-10 | medium | medium | `factContentMap` хранит fact_id → content_hash; lazy-load full content на diagnose-time |
| Sliding-window mapper false positives (3× inflation) | high | high | `test_gold_mapper.ts` dedicated inflation test; report distinct-gold-turn-recall alongside fact-id recall |
| cat3 regression от lexical expansion (Move 5 history: −15.4 pp) | medium | high | NAVIGATIONAL = 0 expansion; test asserts NAVIGATIONAL identical with/without flag; cat3 ≤ −1 pp gate |
| cat4/cat5 abstention degrades | medium | medium | Anti-overfit gate blocks PR; checklist item explicit |
| UX prototype overpromises | medium | high | Visible UI disclaimer (not code comment); reviewer charged с finding overclaims; README explicit |
| Old checkpoints lack trace fields | certain | low | A0.proc explicit fresh re-run на conv0+conv1; old `repro/dated-1167e70` not consumed |
| Lexical hypothesis falsified (P1 fails) | low (~20%) | high | Stage 1 decision tree explicit pivot to A.graph (Phase D); diagnostics + UX deliver value regardless |
| HeidelTime regression repeats Phase 1E off-by-N-day | low | high | `test_phase_1e_regression.py` replays 6 questions Phase 1E broke; HeidelTime adds tags only — никогда не creates claim |
| Mention Graph circular references | low | medium | `test_mention_graph_no_circular.py`; 1-hop bound в Phase D |
| `fact_history` migration loses existing facts | low | critical | Migration test runs на `repro/dated-1167e70` DB; assert fact count preserved |
| `recall_as_of` leaks future facts (time-travel bug) | medium | critical | `test_recall_as_of.py` blocking; test asserts no future facts in result |
| Memory Lane pilot < 4/5 informativeness | medium | medium | Pilot iterative; threshold tuning is acceptable refinement, не ship-blocker |

---

## 8. Финальные acceptance gates per phase

### Phase A combined gate (S3)
- A0.proc PR merged + baseline committed.
- A0.lexical PR merged.
- A.lexicon PR merged behind flag; full-10 cat1 ≥ 41 %, agg ≥ 64 %, ни одна категория > 1 pp падает.
- U.scaffold + U.trace PRs merged; demo functional с real diagnostics.

### Phase A.5 gate (conditional)
- A0.cpl + A.cpl PRs merged.
- 2-conv: additional CPL contribution ≥ +1 pp on top of frames.
- full-10: cat1 ≥ 43 %.

### Phase B gate (conditional)
- A0.pack + A.pack + U.pack PRs merged.
- `ReaderRecoveryRate` ≥ +25 % LLM-fail migration to CORRECT.
- full-10: agg ≥ 70 %, cat1 ≥ 47 %.

### Phase C gate (conditional)
- A0.time + A.heideltime + U.timeline PRs merged.
- `TemporalResolutionAccuracy` ≥ 85 % on sample.
- full-10: cat2 ≥ 53 %.

### Phase D gate (conditional)
- A0.graph + A.graph + U.graph PRs merged.
- `EntityHopReach` ≥ 50 %.
- full-10: hard-miss ≤ 50% baseline; agg ≥ 75 %.

### Phase E gate (S6)
- A0.product + A.surface + U.surface PRs merged.
- All 5 product scenarios pass (Codex Phase 5).
- 10-user Memory Lane pilot ≥ 4/5 informativeness (post-merge).
- Promise Ledger ≥ 90 %; Decision Log ≥ 85 %.

### Program-final acceptance
- LOCOMO full-10 aggregate ≥ 77 %, cat1 ≥ 53 %.
- LongMemEval +12-15 pp vs Mem0 baseline.
- All anti-overfit gates verified для каждой phase shipped.

---

## 9. Что эти 3 агента НЕ делают

- Не commit'ятся к conditional phases до того, как diagnostics покажет соответствующий bottleneck. Phase A.5/B/C/D conditional на per-phase data signal.
- Не trigger flag default-on flips в same PR as feature merge — каждый default-on flip отдельный follow-up PR с observability gate.
- Не expose `recall_facts_with_trace` через MCP до Phase E — debug-only до того, как product-decision sign-off.
- Не touch `feedback_no_test_specific_hacks.md` rule — ни один improvement не может быть LOCOMO-tailored; всё должно держаться на synthetic product scenarios (Promise Ledger / Decision Log / Memory Lane).
- Не run full-10 multiple times per cycle — стоимость bench restricted (per `feedback_bench_cost.md`); 2-conv для validation, full-10 только на explicit gate.
- Не шипят product-сценарии (Promise Ledger, Decision Log, Memory Lane) — это Phase E (Product Memory Surface), отдельный assignment.

---

## 10. Critical files (paths to be touched)

**Backend (Agent A):**
- `src/ai_knot/query_lexicon.py` (CREATE Phase A; extend Phase A.5)
- `src/ai_knot/_query_intent.py` (extend Phase A, C)
- `src/ai_knot/knowledge.py` (extend Phase A, A.5, B, C, D)
- `src/ai_knot/_inverted_index.py` (verify only)
- `src/ai_knot/_date_enrichment.py` (extend Phase C)
- `src/ai_knot/pack.py` (CREATE Phase B)
- `src/ai_knot/mention_graph.py` (CREATE Phase D)
- `src/ai_knot/memory_profile.py` (CREATE Phase E)
- `src/ai_knot/product_commands.py` (CREATE Phase E)
- `src/ai_knot/storage/sqlite_storage.py` (extend Phase A.5, D, E)
- `src/ai_knot/mcp_server.py` (extend Phase E)
- `pyproject.toml` (extend Phase C)

**Diagnostics (Agent A0):**
- `aiknotbench/src/aiknot.ts` (extend Phase A.0)
- `aiknotbench/src/runner.ts` (extend Phase A.0)
- `aiknotbench/scripts/build_gold_mapper.ts` (CREATE A.0; extend C)
- `aiknotbench/scripts/diagnose_recall.ts` (CREATE A.0; extend A, A.5, B, C, D)
- `aiknotbench/scripts/eval_recall_buckets.ts` (CREATE A.0; extend A)
- `aiknotbench/scripts/eval_promise_ledger.ts` (CREATE Phase E)
- `aiknotbench/scripts/eval_decision_log.ts` (CREATE Phase E)
- `aiknotbench/scripts/eval_memory_lane.ts` (CREATE Phase E)
- `aiknotbench/data/synthetic/family_50_promises.json` (CREATE Phase E)
- `aiknotbench/data/synthetic/project_30_decisions.json` (CREATE Phase E)

**UX (Agent U):**
- `prototypes/knot-ux/` (entire tree CREATE; isolated from production)

**Companion docs (created at ExitPlanMode approval):**
- `research/three_agent_dev_plan_20260426.md` — standalone copy of this plan
- `research/baseline_buckets_20260427.md` — committed by A0.proc gate

---

## 11. Verification

Каждая phase верифицируется по своему combined gate (см. §8). Программа-final verification:

```bash
# 1. PROC harness baseline reproducible
cd aiknotbench && npm run diagnose -- --conv-range 0-9 --baseline
# expect: cat1 buckets 9 LLM-fail / 5 partial / 10 low-recall / 6 hard-miss

# 2. Production tests green (Phase A → E)
.venv/bin/ruff format src/ tests/ --check
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/ai_knot --strict
.venv/bin/pytest tests/ --ignore=tests/test_performance.py --ignore=tests/test_mcp_e2e.py -q

# 3. Anti-overfit lexicon scan
.venv/bin/pytest tests/test_lexicon_anti_overfit.py -v

# 4. LOCOMO full-10 final
cd aiknotbench && npm run bench -- --mode dated --conv-range 0-9 --backends ai_knot,memvid
# expect: aggregate ≥ 77 %, cat1 ≥ 53 %

# 5. Product scenarios pass
.venv/bin/pytest tests/test_recall_as_of_acceptance.py -v
cd aiknotbench && npm run eval:promise-ledger
cd aiknotbench && npm run eval:decision-log

# 6. UX prototype demo
cd prototypes/knot-ux && npm install && npm run dev
# manual: 2-min walkthrough — Knot view → Inquiry Trace → Memory Time Travel functional
```
