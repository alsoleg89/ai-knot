# Phase E — Enterprise Query-Shape Routing

**Дата:** 2026-04-13  
**Статус:** Реализовано, тесты зелёные (898 pass), бенчмарк pending

---

## Baseline (до Phase E)

| Метрика | Значение |
|---------|---------|
| Cat1 (single-hop) | 33% |
| Cat2 (multi-hop / temporal) | 46% |
| Cat3 (inference) | 49% |
| Cat4 (open-ended) | 78% |
| **TOT** | ~52% |

---

## Цели Phase E

Отказ от «парадокса универсального пайплайна» — один набор весов плохо работает для всех типов запросов. Вводим:

1. **Query Shape Router** — rule-based классификатор 6 интентов для single-agent пути
2. **Stage-3 RRF fusion** — замена greedy `_select_topk` на 6-сигнальный RRF с per-intent весами
3. **Enterprise hooks** — field_weights_override, memory_type_filter API

---

## E0 — Foundational Math Fixes

### E0.1 — Channel C safe token match

**Файл:** `src/ai_knot/knowledge.py::_execute_recall`

**Проблема:** `entity_index.get(hop_key, [])` — точный матч. "pottery class" не матчится с сущностью "pottery".

**Решение:** Token intersection с guard `len(t) > 2`:

```python
hop_tokens = frozenset(_tokenize(hop_key))
for known_entity, related_facts in entity_index.items():
    entity_tokens = frozenset(_tokenize(known_entity))
    shared = hop_tokens & entity_tokens
    if any(len(t) > 2 for t in shared):
        for related in related_facts:
            candidate_ids.add(related.id)
```

Guard `> 2` предотвращает FP: "car" ↔ "oscar" — после стемминга "car" длиной 3, но у "oscar" общих stem-токенов нет. "ai" (length 2) не триггерит матч.

**Тест:** `tests/test_channel_c_entity_hop.py` (6 тестов)

### E0.2 — MMR slot-aware Jaccard

**Файл:** `src/ai_knot/knowledge.py::_mmr_select`

**Проблема:** Элементы одного списка (одинаковый `slot_key`, разный `value_text`) подавляли друг друга в MMR — Jaccard был высоким из-за общего контента.

**Решение:** При `f.slot_key == sel_f.slot_key and f.value_text != sel_f.value_text` → `sim = 0.0`. Список-элементы не конкурируют между собой в MMR.

```python
sel_facts: list[Fact] = [pairs[0][0]]
# ...
for sel_f, st in zip(sel_facts, sel_tokens, strict=False):
    if f.slot_key and f.slot_key == sel_f.slot_key and f.value_text != sel_f.value_text:
        sim = 0.0  # list items don't compete
    # ... else normal Jaccard
sel_facts.append(fact)
```

**Тест:** `tests/test_mmr_slot_protection.py` (4 теста)

### E0.3 — BM25_B_CONTENT guard comment

`_BM25_B_CONTENT = 0.75` — inline-комментарий «Do not lower».

---

## E1 — Intent Classifier, PipelineConfig, RRF Fusion

### E1.1 — RecallIntent + PipelineConfig

**Файл:** `src/ai_knot/_query_intent.py`

Добавлены:
- `RecallIntent(StrEnum)` — 6 значений: FACTUAL, AGGREGATIONAL, EXPLORATORY, NAVIGATIONAL, PROCEDURAL, BROAD_CONTEXT
- `PipelineConfig` — frozen dataclass с полями: `skip_prf`, `rrf_weights` (6-tuple), `mmr_lambda`, `use_ddsa`, `sort_strategy`, `memory_type_filter`, `field_weights_override`
- `classify_recall_intent(query) -> RecallIntent` — rule-based, без LLM
- `get_pipeline_config(intent) -> PipelineConfig`

**Classifier — приоритетный порядок:**

1. `len(content_tokens) <= 1` → **BROAD_CONTEXT**  
   (порог ≤1, не ≤2: entity+attribute = 2 токена, должно быть FACTUAL, не BROAD_CONTEXT)
2. "how to" / "steps to" / "how do i" или токены {rule, policy, procedure, guideline, deploy, instruction} → **PROCEDURAL**
3. "meeting notes" или токены {find, show, open, file, document, log, transcript, report} → **NAVIGATIONAL**
4. Aggregation vocab (list, all, every, summarize, ...) или phrases ("what are", "how many", ...) → **AGGREGATIONAL**
5. "why", "how does/did", "before/after/between/during/history/related" или len(tokens) ≥ 10 → **EXPLORATORY**
6. Default → **FACTUAL**

**Важное решение: BROAD_CONTEXT threshold = 1, не 2.**  
При пороге ≤2: "what is Alice salary" → 2 content tokens → неверно BROAD_CONTEXT. "how to deploy" → 2 tokens → неверно BROAD_CONTEXT (не доходило до phrase check). Порог ≤1 фиксирует обе проблемы.

**PipelineConfig матрица:**

| Intent | skip_prf | rrf_weights (BM25,slot,trig,imp,ret,rec) | mmr_λ | ddsa | sort | mem_filter | field_override |
|--------|----------|------------------------------------------|-------|------|------|------------|----------------|
| FACTUAL | True | (10,5,2,0.5,0.5,0) | 0.85 | No | relevance | None | None |
| AGGREGATIONAL | False | (3,2,2,3,2,2) | 0.3 | No | sandwich | None | None |
| EXPLORATORY | False | (5,3,2,2,2,4) | 0.65 | Yes | chronological | None | None |
| NAVIGATIONAL | True | (2,1,8,0,0,5) | 0.9 | No | relevance | None | {tags:5,canonical:3} |
| PROCEDURAL | False | (8,4,2,5,0,0) | 0.7 | No | relevance | **None** | None |
| BROAD_CONTEXT | True | (3,1,1,6,5,2) | 0.5 | Yes | relevance | None | None |

**Почему PROCEDURAL.memory_type_filter = None:**  
Изначально планировалось `memory_type_filter=MemoryType.PROCEDURAL`. Это сломало 6 существующих тестов: запросы с "deploy" (и любым PROCEDURAL-токеном) автоматически фильтровали все SEMANTIC факты. Пример: `kb.add("Docker deployment")` + query "how should I deploy?" → пустой результат. Enterprise-изоляция по типу памяти должна быть opt-in на уровне конфигурации KnowledgeBase, не auto-apply из classifier.

### E1.2 — Stage-3 RRF fusion (замена greedy)

**Файл:** `src/ai_knot/knowledge.py::_execute_recall` (Stage 3)

**Было:** `_select_topk(candidates, query, index, top_k)` — greedy по BM25+importance composite score.

**Стало:** 6-сигнальный RRF:

```python
# 6 ranked lists
bm25_ranked      = sorted(candidates, key=bm25_raw.get, reverse=True)
slot_ranked      = sorted(candidates, key=lambda fid: _slot_exact_score(query_tokens_frozen, fact_map[fid]))
trigram_ranked   = sorted(candidates, key=_trig_score, reverse=True)
importance_ranked = sorted(candidates, key=lambda fid: fact_map[fid].importance, reverse=True)
retention_ranked  = sorted(candidates, key=lambda fid: fact_map[fid].retention_score, reverse=True)
recency_ranked    = sorted(candidates, key=lambda fid: fact_map[fid].created_at, reverse=True)

fused_scores = _rrf_fuse(
    [bm25_ranked, slot_ranked, trigram_ranked, importance_ranked, retention_ranked, recency_ranked],
    weights=list(config.rrf_weights),
)
selected_ids = sorted(candidates, key=lambda fid: fused_scores.get(fid, 0.0), reverse=True)[:top_k]
```

Dense guarantee и DDSA (Stage 3b, 4a) сохранены после RRF.

Trace-ключ переименован: `stage3_select` → `stage3_rrf` (обновлены `test_recall_trace.py` и `scripts/trace_cat1_misses.py`).

### E1.3 — Sort-strategy branch

**Файл:** `src/ai_knot/knowledge.py::recall`

**Было:** безусловный `self._sandwich_reorder(pairs)`.

**Стало:**
```python
config = get_pipeline_config(classify_recall_intent(query))
if config.sort_strategy == "sandwich":
    pairs = self._sandwich_reorder(pairs)   # AGGREGATIONAL only
elif config.sort_strategy == "chronological":
    head = list(pairs[:15])
    head.sort(key=lambda x: x[0].created_at)
    pairs = head + list(pairs[15:])         # EXPLORATORY: top-15 только
# else 'relevance' — уже отсортировано по RRF-score
```

### E1.4 — MMR lambda from config

`_execute_recall` → `self._mmr_select(pairs, top_k=top_k, lambda_=config.mmr_lambda)`. Lambda больше не хардкоженная 0.5.

### E1.5 — DDSA gate

`if _ddsa.DDSA_ENABLED and config.use_ddsa and pairs:` (вместо `if _ddsa.DDSA_ENABLED and pairs:`). DDSA включён только для EXPLORATORY и BROAD_CONTEXT.

**Исправление теста DDSA:** `test_ddsa_disabled_flag_skips_spreading_activation` использовал FACTUAL-query "what does Melanie play" (DDSA отключён для FACTUAL). Заменён на EXPLORATORY-query "why does Melanie play violin on Tuesdays" для части теста с DDSA_ENABLED=True.

---

## E2 — Enterprise Hooks

### E2.1 — memory_type_filter API

`PipelineConfig.memory_type_filter` поле существует и используется в фильтрации facts в `_execute_recall`. По умолчанию `None` для всех интентов. Может быть задан explicit через `dataclasses.replace(config, memory_type_filter=MemoryType.PROCEDURAL)`.

### E2.2 — field_weights_override в InvertedIndex.score

**Файл:** `src/ai_knot/_inverted_index.py::score`

```python
def score(self, query: str, *, ..., field_weights_override: dict[str, float] | None = None) -> dict[str, float]:
    _fw = field_weights_override or {}
    w_content = _fw.get("content", _W_CONTENT)
    w_tags    = _fw.get("tags", _W_TAGS)
    w_canonical = _fw.get("canonical", _W_CANONICAL)
    w_evidence  = _fw.get("evidence", _W_EVIDENCE)
```

NAVIGATIONAL queries автоматически получают `{"tags": 5.0, "canonical": 3.0}` — boost по тегам для поиска артефактов.

---

## Новые тестовые файлы (50 тестов)

| Файл | Тестирует | Кол-во |
|------|-----------|--------|
| `tests/test_channel_c_entity_hop.py` | E0.1 token intersection | 6 |
| `tests/test_mmr_slot_protection.py` | E0.2 slot-aware Jaccard | 4 |
| `tests/test_recall_intent.py` | E1.1 classifier + config matrix | 28 |
| `tests/test_recall_procedural_isolation.py` | E2.1 memory_type_filter API | 4 |
| `tests/test_bm25_field_weights_override.py` | E2.2 field_weights_override | 5 |
| `tests/test_recall_mmr_dispatch.py` | E1.4 lambda dispatch | 3 |

---

## Verification

```bash
# Pre-commit
ruff format src/ tests/
ruff check src/ tests/
mypy src/ai_knot --strict
pytest tests/ --ignore=tests/test_performance.py --ignore=tests/test_mcp_e2e.py -q
# Результат: 898 passed, 1 pre-existing failure (legal compliance scanner / research notes)
```

---

## Benchmark commands

```bash
# Пересборка npm (после изменений в src/ai_knot)
cd /path/to/ai-knot
cd npm && npm run build && cd ../aiknotbench && npm install

# LOCOMO full run — dated-learn (основной режим)
cd aiknotbench
npx tsx src/index.ts run -r phase-e-dated-learn --ingest-mode dated-learn --top-k 60 --force

# LOCOMO full run — dated (без LLM extraction)
npx tsx src/index.ts run -r phase-e-dated --ingest-mode dated --top-k 60 --force

# По категориям (быстрее для диагностики)
npx tsx src/index.ts run -r phase-e-cat1 --ingest-mode dated-learn --top-k 60 --types 1 --force
npx tsx src/index.ts run -r phase-e-cat2 --ingest-mode dated-learn --top-k 60 --types 2 --force

# Сравнение с предыдущим (list прогонов)
npx tsx src/index.ts list
```

---

## Ambition Targets

| Category | Baseline | Target | Механизм |
|----------|----------|--------|----------|
| Cat1 | 33% | ~70% | FACTUAL RRF(10,5,2,…) + MMR λ=0.85 + sandwich off |
| Cat2 | 46% | ~65% | Channel C token match + EXPLORATORY top-15 chrono |
| Cat3 | 49% | ~70% | MMR slot-protect + AGGREGATIONAL λ=0.3 + sandwich |
| Cat4 | 78% | ~80% | BROAD_CONTEXT + FACTUAL без sandwich |
| **TOT** | ~52% | **~70%** | Composite |

---

## Решения, принятые в ходе реализации

1. **BROAD_CONTEXT threshold = 1 (не 2)** — entity+attribute = 2 meaningful tokens. При пороге 2 реальные factual queries (alice salary, bob live) некорректно роутились как BROAD_CONTEXT.

2. **memory_type_filter убран из дефолтного PROCEDURAL config** — auto-filtering по типу из query intent ломает случай когда SEMANTIC факты описывают deployment/procedures. Enterprise isolation — opt-in, не auto.

3. **"meeting notes" убрано из _PROCEDURAL_PHRASES** — было в обоих (PROCEDURAL и NAVIGATIONAL), PROCEDURAL выигрывал. "Find meeting notes" должно быть NAVIGATIONAL.

4. **"summarize" / "summariz" добавлен в _AGGREGATION_TOKENS** — стем "summariz" ≠ "summary". Без этого "summarize everything" не попадало в AGGREGATIONAL.

5. **_query_specificity удалён из knowledge.py** — после замены Stage-3 greedy на RRF specificity перестала использоваться.

6. **DDSA test fix** — E1.5 интентно отключает DDSA для FACTUAL (use_ddsa=False). Тест `test_ddsa_disabled_flag_skips_spreading_activation` проверял FACTUAL query для DDSA_ENABLED=True — заменён на EXPLORATORY query.

---

## Изменённые файлы и функции (коммит 484a9bc)

### `src/ai_knot/_query_intent.py` — новый файл

| Сущность | Тип | Описание |
|----------|-----|----------|
| `RecallIntent` | class (StrEnum) | 6 значений: FACTUAL, AGGREGATIONAL, EXPLORATORY, NAVIGATIONAL, PROCEDURAL, BROAD_CONTEXT |
| `PipelineConfig` | frozen dataclass | skip_prf, rrf_weights (6-tuple), mmr_lambda, use_ddsa, sort_strategy, memory_type_filter, field_weights_override |
| `_PIPELINE_CONFIGS` | dict | Матрица RecallIntent → PipelineConfig |
| `classify_recall_intent()` | функция | Rule-based классификатор, priority order, без LLM |
| `get_pipeline_config()` | функция | Lookup в _PIPELINE_CONFIGS |
| `_STOP_SHORT_RECALL` | frozenset | Stopwords для content-token counting (local copy, без circular import) |
| `_PROCEDURAL_PHRASES/TOKENS` | tuple/frozenset | Vocabulary для PROCEDURAL intent |
| `_NAVIGATIONAL_PHRASES/TOKENS` | tuple/frozenset | Vocabulary для NAVIGATIONAL intent |
| `_EXPLORATORY_PHRASES/TOKENS` | tuple/frozenset | Vocabulary для EXPLORATORY intent |
| `_classify_pool_query()` | функция | Перенесена из knowledge.py (pool path, не трогаем) |
| `_query_specificity()` | функция | Перенесена из knowledge.py |
| `_PoolQueryIntent` | class | Перенесён из knowledge.py |
| `_RecallMeta` | dataclass | Перенесён из knowledge.py |

---

### `src/ai_knot/_inverted_index.py` — изменён

| Функция/property | Что изменилось |
|-----------------|----------------|
| `score()` | +kwarg `field_weights_override: dict[str, float] \| None = None`; локальные `w_content/w_tags/w_canonical/w_evidence` читаются из override или module-level default |
| `content_postings` | новый property (expose для RRF trigram lookup) |
| `tags_postings` | новый property |
| `idf()` | новый метод (expose IDF для внешних вызовов) |
| `median_idf()` | новый метод |
| `_BM25_B_CONTENT` | добавлен inline-guard comment "Do not lower" |

---

### `src/ai_knot/knowledge.py` — изменён

#### Новые/переработанные методы

| Метод | Тип изменения | Описание |
|-------|---------------|----------|
| `_expand_query_for_embed()` | новый | Упрощённая версия `_expand_query` только для embed-пути |
| `_build_entity_dictionary()` | новый | Вынесена логика построения словаря сущностей из `_execute_recall` |
| `_build_entity_mention_index()` | новый | Вынесен index сущностей (для Channel C) |
| `_select_topk()` | новый | Greedy select вынесен в отдельный метод (остался как helper, из main path убран) |
| `_sandwich_reorder()` | новый | Sandwich ordering вынесен из inline-кода `recall()` |
| `recall_facts_with_trace()` | новый | Diagnostic variant возвращает `(results, trace dict)` |
| `_expand_query()` | **удалён** | Заменён на `_expand_query_for_embed` |

#### Изменённые методы

| Метод | Что изменилось |
|-------|----------------|
| `_execute_recall()` | **Stage 0 (новый):** `classify_recall_intent` + `get_pipeline_config` в начале функции. **Фильтр фактов:** +`memory_type_filter`. **Channel A:** `index.score(field_weights_override=config.field_weights_override)`. **Channel C:** token intersection вместо `entity_index.get(hop_key, [])` — guard `len(t) > 2`. **Stage 3 (заменён):** greedy `_select_topk` → 6-сигнальный RRF fusion (`bm25 + slot_exact + trigram + importance + retention + recency`), веса из `config.rrf_weights`. **Stage 4a:** DDSA gate `config.use_ddsa`. **Stage 4b:** MMR lambda `config.mmr_lambda`. **Trace:** `stage3_select` → `stage3_rrf` с полями `intent`, `rrf_weights`, `selected_ids`. |
| `_mmr_select()` | +`sel_facts: list[Fact]` параллельно `sel_tokens`; в inner loop: `sim = 0.0` когда `f.slot_key == sel_f.slot_key and f.value_text != sel_f.value_text` (list items не конкурируют); `sel_facts.append(fact)` после выбора |
| `recall()` | Заменён безусловный `_sandwich_reorder` на sort_strategy branch: `"sandwich"` → только для AGGREGATIONAL; `"chronological"` → top-15 sort по `created_at`; `"relevance"` → без изменений (RRF-order) |
| `__init__` | Новые imports: `classify_recall_intent`, `get_pipeline_config`, `_char_trigrams`, `_slot_exact_score`, `_rrf_fuse` |
| `add` / `add_episodic` | Минорные правки из Phase C (не Phase E) |
| `recall_facts_with_scores()` | Минорные правки |

---

### Тесты — новые файлы

| Файл | Тестирует | Тестов |
|------|-----------|--------|
| `tests/test_recall_intent.py` | `classify_recall_intent` (18 parametrized cases) + `PipelineConfig` matrix (10 checks) | 28 |
| `tests/test_channel_c_entity_hop.py` | Token intersection: partial match, no FP oscar↔car, short tokens filtered, empty index no-op | 6 |
| `tests/test_mmr_slot_protection.py` | Slot-aware Jaccard: 4 hobbies → 3+ survive, same slot different value, raw mode compat | 4 |
| `tests/test_bm25_field_weights_override.py` | tags override boost, None = defaults, partial override, zero content weight, no mutation | 5 |
| `tests/test_recall_mmr_dispatch.py` | AGGREGATIONAL → λ≤0.4, FACTUAL → λ≥0.75, EXPLORATORY → λ∈[0.5,0.75] | 3 |
| `tests/test_recall_procedural_isolation.py` | memory_type_filter=None по умолчанию, SEMANTIC факты не дропаются, explicit filter API | 4 |
| `tests/test_recall_trace.py` | stage3_rrf в trace dict, selected_ids subset of pool, mmr dropped_ids disjoint | 6 |
| `tests/test_ddsa_integration.py` | DDSA_ENABLED flag, use_ddsa gate — EXPLORATORY query для DDSA_ENABLED=True check | (обновлён) |

---

### Прочие файлы

| Файл | Изменение |
|------|-----------|
| `scripts/trace_cat1_misses.py` | `stage3_select` → `stage3_rrf` в 3 местах (classify_miss, trace_summary.selected_count, gold_in_selected) |
| `research/phase_e_query_shape_routing.md` | этот файл |
| `research/INDEX.md` | Phase E добавлен в "Текущая работа" |
