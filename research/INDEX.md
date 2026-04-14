# Research Index — ai-knot

**Обновлён:** 2026-04-13

Навигация по всем материалам в `research/`. Организовано по темам.

---

## Текущая работа — Cat1/Cat2 улучшение

| Файл | Что внутри |
|------|-----------|
| [phase_e_query_shape_routing.md](phase_e_query_shape_routing.md) | **АКТИВНЫЙ.** Phase E: Query Shape Router (6 интентов), Stage-3 RRF fusion, MMR slot protection, Channel C token match. Baseline TOT=52%, target ~70%. Бенчмарк pending. |
| [phase_c_c6b_c6c_implementation.md](phase_c_c6b_c6c_implementation.md) | Phase C реализация: C6b enumeration split (все режимы) + C6c date enrichment. Baseline Cat1=39% Cat2=53%. |
| [aggregation_design.md](aggregation_design.md) | **АКТИВНЫЙ.** Дизайн lookup index для aggregation recall. Метрики: +14.4pp теоретический максимум Cat1. Два режима (learn-ON/OFF). API A/B/C сравнение. TurboQuant контекст. |
| [cat1_per_conv_analysis.md](cat1_per_conv_analysis.md) | **АКТИВНЫЙ.** Per-conv breakdown: aggregation vs point вопросы, MMR churn анализ, failure type по conv. Вывод: entity-grouped format — единственный lever для M-type. |
| [locomo_learn_off_partial_results.md](locomo_learn_off_partial_results.md) | Результаты v095-learn-off-all (conv 0–5): Cat1=35.2%, Cat2=50%. Breakdown: 83% M-type, 17% R-type. Off-by-1 и vague-gold патерны. |
| [locomo_fails_report.md](locomo_fails_report.md) | Детальный per-question анализ (conv 0, gpt-4o). Таблица 24 WRONG Cat1 с классификацией R/M/J. Промпты v1/v2/v3. learn() изоляционный эксперимент. |
| [phase2_research.md](phase2_research.md) | Phase 2 результаты — overfetch, raw-aware RRF, skip PRF. Все "мимо". Диагноз: bottleneck в post-retrieval selection. |
| [pipeline_diagnosis_plan.md](pipeline_diagnosis_plan.md) | Диагностика pipeline: где теряются очки. Анализ BM25 IDF, sliding window overlap, near-duplicate saturation. |
| [cat12_improvement_plan.md](cat12_improvement_plan.md) | Ранний план Cat1/Cat2 улучшений (до MMR). |
| [cat12_plan_status.md](cat12_plan_status.md) | Статус полного плана Cat1/Cat2. |
| [cat12_changes_research.md](cat12_changes_research.md) | Исследование изменений по категориям 2, 5, 6. Multi-agent, complexity, effectiveness. |

---

## Диагностика и changelog

| Файл | Что внутри |
|------|-----------|
| [locomo_diagnostic_20260410.md](locomo_diagnostic_20260410.md) | Найден критический баг: DB corruption — все запуски писали в общую БД. Isolation fix. |
| [changelog_v094_entity_scoped_retrieval.md](changelog_v094_entity_scoped_retrieval.md) | Changelog v0.9.4: entity-scoped retrieval, что изменилось, что добавлено. |
| [fixes_11052026.md](fixes_11052026.md) | OpenAI embeddings support — статус, конфиг, баг asyncio event loop. |
| [impact_analysis_pattern_facts.md](impact_analysis_pattern_facts.md) | Анализ влияния pattern facts на LOCOMO conv-0. |
| [locomo_analysis.md](locomo_analysis.md) | Dataset anatomy: 1986 QA pairs, 10 convs. Категории, distribution вопросов. |

---

## Архитектурные исследования

| Файл | Что внутри |
|------|-----------|
| [raw_first_memory_architecture_20260413.md](raw_first_memory_architecture_20260413.md) | Финальная архитектурная позиция: raw-first memory substrate, typed materialization, deterministic raw-mode operators, competitor comparison, MIT-style analysis + Harvard-style critique. |
| [architecture_synthesis.md](architecture_synthesis.md) | Синтез архитектуры: что выжило после критики. Финальные принципы дизайна. |
| [approach_evolution_20260410.md](approach_evolution_20260410.md) | Эволюция подходов: entity scoping → pattern memory. История решений. |
| [analysis_approach_critique.md](analysis_approach_critique.md) | Критика подходов: почему большинство не сработает. Что может. |
| [implementation_plan_v2.md](implementation_plan_v2.md) | Plan v2: entity-scoped retrieval для LOCOMO. (Предшественник текущего плана.) |
| [ma_scenario_analysis.md](ma_scenario_analysis.md) | Multi-agent scenario analysis для entity-scoped retrieval. |

---

## Экспериментальные идеи (исследовались, не реализованы)

| Файл | Что внутри |
|------|-----------|
| [pattern_memory_architecture.md](pattern_memory_architecture.md) | Pattern Memory: ассоциативный retrieval на основе co-occurrence. Биологическая аналогия. |
| [strand_data_structure.md](strand_data_structure.md) | Strand: DNA-inspired binary co-occurrence структура. Технический дизайн. |
| [strand_stress_test_critique.md](strand_stress_test_critique.md) | Стресс-тест критика Strand: где ломается. |
| [biological_mechanisms_deep_dive.md](biological_mechanisms_deep_dive.md) | Биологические механизмы памяти: что организмы реально делают. Hippocampus, engram. |
| [cognitive_memory_model.md](cognitive_memory_model.md) | ai-knot как психологическая модель памяти. Теоретическая база. |

---

## Конкуренты

| Файл | Что внутри |
|------|-----------|
| [competitors/00_comparison_table.md](competitors/00_comparison_table.md) | **Главная таблица сравнения.** Retrieval техники всех конкурентов. Уникальность ai-knot. |
| [competitors/01_mem0.md](competitors/01_mem0.md) | Mem0: семантический search + get_all + graph memory. API паттерны. |
| [competitors/02_letta.md](competitors/02_letta.md) | Letta (MemGPT): archival memory, 3-tier architecture. Только semantic search. |
| [competitors/03_zep_graphiti.md](competitors/03_zep_graphiti.md) | Zep/Graphiti: temporal knowledge graph, entity nodes, BFS traversal. |
| [competitors/04_cognee.md](competitors/04_cognee.md) | Cognee: knowledge graph + vector hybrid. |
| [competitors/05_supermemory.md](competitors/05_supermemory.md) | Supermemory: simple vector store. |
| [competitors/06_hindsight.md](competitors/06_hindsight.md) | Hindsight/Vectorize: TEMPR, spreading activation. |

---

## Академические статьи

| Файл | Что внутри |
|------|-----------|
| [papers/00_index.md](papers/00_index.md) | Главный индекс статей с оценкой применимости. |
| [papers/01_magma_multi_graph_memory.md](papers/01_magma_multi_graph_memory.md) | MAGMA: multi-graph agentic memory (arXiv 2601.03236, Jan 2026) |
| [papers/02_synapse_spreading_activation.md](papers/02_synapse_spreading_activation.md) | SYNAPSE: episodic-semantic via spreading activation (arXiv 2601.02744) |
| [papers/03_hypermem_hypergraph.md](papers/03_hypermem_hypergraph.md) | HyperMem: hypergraph для long-term conversations (arXiv 2604.08256, Apr 2026) |
| [papers/04_hindsight_tempr.md](papers/04_hindsight_tempr.md) | Hindsight TEMPR: retains, recalls, reflects (arXiv 2512.12818) |
| [papers/05_evermemos_engram.md](papers/05_evermemos_engram.md) | EverMemOS: self-organizing memory OS (arXiv 2601.02163) |
| [papers/06_zep_graphiti_temporal_kg.md](papers/06_zep_graphiti_temporal_kg.md) | Zep temporal KG (arXiv 2501.13956) |
| [papers/07_timem_temporal_tree.md](papers/07_timem_temporal_tree.md) | TiMem: temporal-hierarchical consolidation (arXiv 2601.02845) |
| [papers/08_amem_zettelkasten.md](papers/08_amem_zettelkasten.md) | A-MEM: Zettelkasten для LLM agents (arXiv 2502.12110, NeurIPS 2025) |
| [papers/09_memmachine_ground_truth.md](papers/09_memmachine_ground_truth.md) | MemMachine: ground-truth preserving (arXiv 2604.04853, Apr 2026) |
| [papers/10_self_rag_corrective_rag.md](papers/10_self_rag_corrective_rag.md) | Self-RAG + CRAG: corrective retrieval augmented generation |
| [papers/11_surveys_benchmarks.md](papers/11_surveys_benchmarks.md) | Surveys + benchmarks: LoCoMo, LOCOMO10, LongMemEval |

---

## Публикации / маркетинг

| Файл | Что внутри |
|------|-----------|
| [paper_plan.md](paper_plan.md) | Plan академической статьи: entity-scoped retrieval для LLM agent memory. |
| [marketing_paper_plan.md](marketing_paper_plan.md) | Маркетинговый план публикации. |
