# Benchmark Suite

Evaluation suite for ai-knot and competing memory backends.
Entry point: `python -m tests.eval.benchmark.runner`.

---

## Runner CLI

```
python -m tests.eval.benchmark.runner [OPTIONS]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `auto` | `basic` / `extended` / `auto` — selects backend set (see Modes) |
| `--backends` | _(from mode)_ | Override: comma-separated backend names; bypasses `--mode` |
| `--scenarios` | `all` | Comma-separated scenario prefixes (`s1,s4`) or `all` / `legacy` |
| `--language` | `en` | Fixture language: `en`, `ru`, or `both` |
| `--mock-judge` | off | **Offline mode**: MockJudge + StubProvider — zero Ollama calls |
| `--quick` | off | Reduce S6 concurrency 50→20 for CI / low-resource envs |
| `--fast` | off | Mini run: S1+S4+S7, backends=ai_knot+baseline+qdrant, mock judge |
| `--multi-agent` | off | Run **only** multi-agent scenarios (S8 through S25) |
| `--skip-multi-agent` | off | Skip the MA tail appended to every standard run |
| `--output` | `benchmark_report.md` | Path for the Markdown report |
| `--raw-output` | `benchmark_raw.json` | Path for the raw JSON results |
| `--long-run` | off | Run each MA scenario in a timed loop for `--duration` seconds. Reports aggregate metrics with mean±stdev |
| `--duration` | `60` | Seconds per scenario in `--long-run` mode |
| `--jsonl-output` | `benchmark_live.jsonl` | Incremental JSONL output — each scenario writes a line immediately on completion for live analysis |
| `--ma-storage` | `sqlite` | Comma-separated storage backends for MA: `sqlite`, `yaml`, `postgres`. Example: `--ma-storage sqlite,yaml` |
| `--ma-postgres-dsn` | _(none)_ | PostgreSQL DSN for MA postgres storage |
| `--ma-category` | `all` | Filter MA scenarios — `protocol` (CAS, sync, concurrency) or `retrieval` (ranking, trust, assembly) |

### Common invocations

```bash
# Auto-detect services and run full suite
python -m tests.eval.benchmark.runner

# Offline run — no Ollama required, fully deterministic
python -m tests.eval.benchmark.runner --mock-judge

# Offline, only non-LLM backends (guaranteed zero network calls)
python -m tests.eval.benchmark.runner --mock-judge --backends baseline,ai_knot_no_llm

# Specific scenarios only
python -m tests.eval.benchmark.runner --scenarios s1,s3,s5 --mock-judge

# Multi-agent scenarios only
python -m tests.eval.benchmark.runner --multi-agent --mock-judge

# Russian fixtures
python -m tests.eval.benchmark.runner --language ru --mock-judge

# Both languages sequentially
python -m tests.eval.benchmark.runner --language both --mock-judge

# Legacy S1–S7 scenarios
python -m tests.eval.benchmark.runner --scenarios legacy --mock-judge
```

### Output Formats

- `benchmark_report.md` — Markdown summary with protocol/retrieval split tables
- `benchmark_raw.json` — Schema v2: each metric is `{"mean": X, "stdev": Y}`, with `schema_version: 2` at root
- `benchmark_live.jsonl` — One JSON line per completed scenario (live streaming), includes `language` field

---

## Execution flow

```
CLI parse
  │
  ├─ mock_judge? ─── No ──→ check_ollama_available() → exit if down
  │                                │
  │                 Yes ──→ skip check
  │
  ├─ build judge:   mock_judge → MockJudge()  |  else → OllamaJudge()
  ├─ build provider: mock_judge → StubProvider() | else → OllamaProvider()
  │
  ├─ multi_agent flag? ──→ _run_multi_agent() → write reports → exit
  │
  ├─ resolve backends (--backends override or _build_backends_for_mode(mode))
  ├─ resolve scenarios (_build_scenarios(scenarios_arg))
  ├─ resolve language bundles (_build_bundles(language))
  │
  ├─ for each bundle (language):
  │     asyncio.gather(*[_run_backend(b, scenarios, judge) for b in backends])
  │
  └─ unless --skip-multi-agent:
        _run_multi_agent_inline(judge, extra_ma_backends)
              │
              ├─ always: AiKnotMultiAgentBackend
              └─ extended mode AND not mock_judge: + Mem0MultiAgentBackend

write benchmark_report.md + benchmark_raw.json
```

---

## Modes

### `auto` (default)
Resolves to `extended` when all three conditions hold:
- Qdrant running at `localhost:6333`
- `qdrant-client` installed
- `mem0ai` installed

Otherwise resolves to `basic`.

### `basic`
| Backend | LLM calls |
|---|---|
| `baseline` | none |
| `ai_knot_no_llm` | none |
| `ai_knot` _(skipped when `--mock-judge`)_ | Ollama extraction |
| `qdrant` (emulator) | Ollama embeddings |
| `qdrant_extraction` | Ollama extraction |
| `mem0` (emulator) | Ollama extraction + embeddings |
| `memvid` | none |

### `extended`
Same base as `basic` (minus emulators) plus:
- `qdrant_real` — real Qdrant server + Ollama embeddings
- `mem0_real` — real mem0ai library + Ollama

### `--mock-judge` effect on backend selection

When `--mock-judge` is set and no `--backends` override is given:
- `ai_knot` (LLM extraction) is **not added** to the base set
- `Mem0MultiAgentBackend` is **not added** to the MA tail
- All LLM-free backends run normally

---

## Backends reference

| Name | Class | LLM | Notes |
|---|---|---|---|
| `baseline` | `BaselineBackend` | none | FIFO ring buffer |
| `ai_knot_no_llm` | `AiKnotNoLlmBackend` | none | `kb.add()` direct, BM25 retrieval |
| `ai_knot` | `AiKnotBackend(use_add=False)` | Ollama extraction | `kb.learn_async()` → Extractor |
| `qdrant` | `QdrantEmulator` | Ollama embeddings | In-process Qdrant emulator |
| `qdrant_extraction` | `QdrantWithExtractionBackend` | Ollama extraction + embeddings | Qdrant + Extractor |
| `qdrant_real` | `QdrantRealBackend` | Ollama embeddings | Real Qdrant server required |
| `mem0` | `Mem0Emulator` | Ollama extraction + embeddings | In-process mem0 emulator |
| `mem0_real` | `Mem0RealBackend` | Ollama (via mem0ai) | Real mem0ai + Ollama required |
| `memvid` | `MemvidBackend` | none | BM25 + semantic, no Ollama calls |
| `ai_knot_multi_agent` | `AiKnotMultiAgentBackend` | none | `kb.add()` + SharedMemoryPool |

---

## Judge

| Class | LLM | Scores returned |
|---|---|---|
| `OllamaJudge` | `qwen2.5:7b` via `localhost:11434` | `relevance`, `completeness`, `faithfulness` (each 1–5, 3 runs averaged) |
| `MockJudge` | none | fixed: relevance=4.0, completeness=3.5, faithfulness=4.5 |

`OllamaJudge` fires `3 metrics × 3 runs = 9` concurrent HTTP calls per scenario invocation.

---

## Scenarios

Scenarios are grouped in three sets:

| Set | Selector | Backends required |
|---|---|---|
| `_ALL` | `--scenarios all` (default) | `MemoryBackend` |
| `_LEGACY` | `--scenarios legacy` | `MemoryBackend` |
| `_MA` | appended to all runs; `--multi-agent` for standalone | `MultiAgentMemoryBackend` |

---

### Default set (`_ALL`)

#### S1 — MRR & Precision@k (`s1_mrr`)

**Pain point:** retrieval quality — "my agent can't find the right fact."

Each query has one ground-truth relevant fact. A hit is ATC token-containment ≥ 0.5
(deterministic). When Ollama is available, also computes `semantic_mrr` via cosine ≥ 0.65.

| Metric | Type | Description |
|---|---|---|
| `lexical_mrr` | deterministic | Mean Reciprocal Rank via ATC containment |
| `semantic_mrr` | Ollama or fallback | MRR via cosine similarity |
| `p_at_1` | deterministic | Precision@1 |
| `p_at_3` | deterministic | Precision@3 |
| `p_at_5` | deterministic | Precision@5 |

---

#### S2 — Semantic Gap / Paraphrase Recall (`s2_semantic_gap`)

**Pain point:** "BM25 doesn't find a fact if the query is phrased differently."

Inserts verbatim facts, queries with paraphrases. Measures the gap between
lexical (ATC ≥ 0.45) and semantic (cosine ≥ 0.65) recall at top-3.

| Metric | Type | Description |
|---|---|---|
| `lexical_recall_at3` | deterministic | Fraction of paraphrase queries with lexical hit in top-3 |
| `semantic_recall_at3` | Ollama or 0.0 | Fraction with semantic hit in top-3 |
| `semantic_gap` | derived | `semantic_recall - lexical_recall` (>0 = dense retrieval helps) |

---

#### S3 — Staleness Resistance (`s3_staleness`)

**Pain point (#1):** "system returns an outdated fact instead of the current one."

Inserts 5 topics × 5 versions (25 facts total, interleaved). Queries each topic
and checks whether top-1 is the latest version (v5).

| Metric | Type | Description |
|---|---|---|
| `latest_state_accuracy` | deterministic | Fraction of queries where top-1 is v5 |
| `consolidation_ratio` | deterministic | `1 - stored_count / 25` (memory compression) |
| `overconsolidation_rate` | deterministic | `max(0, consolidation_ratio - latest_state_accuracy)` — compression that lost data |

---

#### S4 — Memory Compression F1 (`s4_compression_f1`)

**Pain point (#2):** "memory grows unboundedly" / "agent sees 50 copies of the same fact."

Sub-test A: 50 paraphrases of one rule → should collapse (high `dedup_ratio`).
Sub-test B: 20 genuinely distinct rules → should NOT merge (high `retention_ratio`).

| Metric | Type | Description |
|---|---|---|
| `dedup_ratio` | deterministic | `1 - unique_stored / 50` — paraphrase collapse rate |
| `retention_ratio` | deterministic | `distinct_rules_found / 20` — unique fact preservation |
| `compression_f1` | deterministic | Harmonic mean of dedup and retention |

---

#### S5 — Noise Tolerance (`s5_noise`)

**Pain point (#2):** "noise in context degrades LLM answer quality."

Inserts 200 noise facts + 5 signal facts. Measures whether signal facts surface
in top-3 for their corresponding targeted queries.

| Metric | Type | Description |
|---|---|---|
| `signal_recall_at3` | deterministic | Fraction of signal queries with signal hit in top-3 (ATC ≥ 0.45) |
| `contamination_at3` | deterministic | Fraction of retrieved items that are noise |
| `snr` | deterministic | `signal_recall / max(contamination, 0.01)` |

---

#### S6 — Context Economy / Token Efficiency (`s6_token_economy`)

**Pain point (#4):** "memory fills the entire context window."

Inserts a 15-fact user profile, retrieves top-5 for each query. Measures
tokens injected vs the full profile and signal density.

| Metric | Type | Description |
|---|---|---|
| `token_compression` | deterministic | `1 - retrieved_tokens / raw_tokens` |
| `p_at_3_lexical` | deterministic | Fraction of queries with relevant fact in top-3 (ATC-based) |
| `quality_per_token` | deterministic | `p@3 / (1 - token_compression)` — signal per token |

---

#### S7 — Grounding Rate / Hallucination Resistance (`s7_grounding`)

**Pain point (#3):** "LLM extraction adds details that were never stated."

After inserting facts, retrieves all stored content via broad queries. For each
retrieved text, computes max ATC overlap against all original inserted facts.
Low grounding (<0.3) = hallucination risk.

| Metric | Type | Description |
|---|---|---|
| `mean_grounding` | deterministic | Average max-ATC across all retrieved texts |
| `hallucination_rate` | deterministic | Fraction of retrieved texts with max-ATC < 0.3 |

---

#### S8 — Latency & Throughput at Scale (`s8_throughput`)

**Pain point:** "memory system is too slow for real-time agents."

Inserts 200 semantically distinct facts, runs 5 warmup queries, then measures
p50/p95/p99 latency and throughput over 20 concurrent queries (5 at a time).

| Metric | Type | Description |
|---|---|---|
| `p50_ms` | deterministic | Median retrieve latency (ms) |
| `p95_ms` | deterministic | 95th-percentile latency (ms) |
| `p99_ms` | deterministic | 99th-percentile latency (ms) |
| `throughput` | deterministic | Queries per second (20 queries / wall time) |

---

#### S9 — Scale Sensitivity (`s9_scale`)

Tests how retrieval quality and latency degrade as corpus grows.

Inserts 5 signal facts + N noise facts (N ∈ {0, 50, 200, 500, 1000}).
Measures MRR (lexical ATC ≥ 0.5) and p95 latency at each scale point.

| Metric | Type | Description |
|---|---|---|
| `mrr_at_{N}` | deterministic | Lexical MRR at each corpus size |
| `p95_ms_at_{N}` | deterministic | 95th-percentile latency at each corpus size |
| `mrr_degradation` | deterministic | Fractional MRR loss from N=0 to N=1000 |

---

#### S16 — Explicit Update Semantics (`s16_update_correctness`)

Verifies that `MemoryOp` signals emitted by the extractor are correctly honoured
during slot resolution in `KnowledgeBase.learn()`.

Only runs against `AiKnotBackend` (requires `_kb` attribute); returns skipped for all other backends.

| Metric | Type | Description |
|---|---|---|
| `delete_correctness` | deterministic | 1.0 if DELETE closes the slot without inserting |
| `noop_correctness` | deterministic | 1.0 if NOOP neither inserts nor mutates existing |
| `update_correctness` | deterministic | 1.0 if UPDATE forces supersede over reinforce |

---

#### S-LoCoMo — Long-Context Memory QA (`s_locomo`)

Evaluates recall against the [LoCoMo10 dataset](https://github.com/snap-research/locomo)
(10 conversations, ~199 QA pairs each). Dataset is downloaded on first run and cached
in `$TMPDIR/ai_knot_locomo10.json`. Set `LOCOMO_FILE` env var or `--locomo-file` to use
a local copy.

Protocol: insert each session turn via `backend.insert()`, retrieve top-5, compute
best-doc token F1 against the gold answer.

| Metric | Type | Description |
|---|---|---|
| `overall_f1` | deterministic | Mean best-doc token F1 over all QA pairs |
| `single_hop_f1` | deterministic | Category 1 questions |
| `multi_hop_f1` | deterministic | Category 2 questions |
| `temporal_f1` | deterministic | Category 3 questions |
| `adversarial_f1` | deterministic | Category 4 questions (if present in dataset) |

---

### Multi-agent set (`_MA`)

All MA scenarios require `MultiAgentMemoryBackend` and run as a tail after every standard
run (unless `--skip-multi-agent`). Use `--multi-agent` for standalone execution.

Default backend: `AiKnotMultiAgentBackend` (no LLM).
Optional addition: `Mem0MultiAgentBackend` (needs Ollama; excluded under `--mock-judge`).

Scenarios fall into two categories, selectable via `--ma-category`:

| Category | Scenarios | Focus |
|---|---|---|
| **Protocol Correctness** | S10, S11, S13, S17, S20, S25 | CAS, sync, concurrency |
| **Retrieval & Behavior** | S8, S9, S12, S14, S15, S16, S18, S19, S21, S22, S23, S24 | Ranking, trust, assembly |

---

#### S8 — Multi-Team Knowledge Commons (`s8_ma_isolation`)

Cross-domain overlap with 4 teams. Verifies that each agent's private KB
is invisible to the other while shared pool merges correctly.

Sub-tests: A) self-recall — each agent retrieves its own domain.
B) cross-contamination — each agent should NOT retrieve the other's facts.

| Metric | Type | Description |
|---|---|---|
| `self_recall` | deterministic | Fraction of self-domain queries returning ≥1 relevant result |
| `isolation_score` | deterministic | `1 - cross_contamination_rate` (1.0 = perfect isolation) |

---

#### S9 — Competing Documentation Sources (`s9_ma_pool_publish`)

Conflicting facts from multiple agents with CAS supersession.
Agent A inserts N facts into its private namespace and publishes to the shared pool.
Agent B (empty private KB) queries the pool and must find them.

| Metric | Type | Description |
|---|---|---|
| `pool_recall` | deterministic | Fraction of expected facts found by Agent B |
| `publish_count` | informational | Number of facts published |
| `relevance` | judge | Judge score on pool retrieval quality (1–5) |

---

#### S10 — MESI CAS (`s10_ma_mesi_cas`)

Slot supersession correctness. Agent A publishes salary v1 ($95k) for Jordan Lee.
Agent B publishes salary v2 ($140k) for the same `(entity, attribute)` slot. The pool
must retain exactly one active version (the latest).

| Metric | Type | Description |
|---|---|---|
| `cas_correctness` | deterministic | 1.0 if exactly 1 active fact for the entity+attribute |
| `latest_surfaced` | deterministic | 1.0 if retrieved fact contains the v2 keyword |

---

#### S11 — Progressive Knowledge Catchup (`s11_ma_mesi_sync`)

Incremental sync delta. Verifies token-efficient incremental synchronisation:
`sync_dirty()` must return only facts that changed since the last sync, not the full pool.

Flow: Agent A publishes 5+1 facts → Agent B syncs (gets all 6) → Agent A updates
1 slot → Agent B syncs again (should get only 1 changed fact).

Reference: arXiv 2603.15183 reports 95% token savings with MESI lazy invalidation.

| Metric | Type | Description |
|---|---|---|
| `initial_sync_completeness` | deterministic | Fraction of initial facts received in first sync |
| `incremental_efficiency` | deterministic | `1 - (facts in second sync / pool_size)` — 1.0 = only dirty facts |

---

#### S12 — Priority Triage Under Load (`s12_topic_gating`)

Dynamic utility threshold. Verifies topic channel routing and utility-threshold gating.
Agent A inserts 9 high-importance (0.8) + 3 low-importance (0.15) facts. Publishes with
threshold=0.3. Expected: 9 high-importance facts published; 3 low-importance filtered.

| Metric | Type | Description |
|---|---|---|
| `channel_precision` | deterministic | Fraction of channel-filtered queries returning ≥1 result |
| `gating_filter_rate` | deterministic | 1.0 if published_count == 9 (low-importance filtered out) |
| `pool_recall` | deterministic | Fraction of the 3 channel queries returning ≥1 result |

---

#### S13 — Concurrent Writers (`s13_concurrent_writers`)

Thread-safe CAS under contention. 4 agents each publish a different salary value for
the same `(entity, attribute)` slot via `asyncio.gather`. Verifies that the shared pool
maintains a correct version chain: exactly one active fact remains and all superseded
versions are properly closed.

| Metric | Type | Description |
|---|---|---|
| `no_lost_updates` | deterministic | 1.0 if exactly 1 active fact for the slot |
| `version_chain_integrity` | deterministic | 1.0 if total slot versions (active + invalid) == 4 |

---

#### S14 — Trust Drift & Recovery (`s14_trust_drift`)

Auto-trust after invalidation. Agent A publishes 8 structured facts. Agent B immediately
supersedes each one. Verifies that auto-trust penalises Agent A (high quick-invalidation
rate → trust floor).

Trust formula (Marsh 1994):
```
trust = min(1, used / published) × (1 − quick_invalidation_rate)
```

| Metric | Type | Description |
|---|---|---|
| `trust_floor_reached` | deterministic | 1.0 if Agent A's trust ≤ 0.25 after max invalidation |

---

#### S15 — Cross-Team Signal Contamination (`s15_topic_leakage`)

Shared terminology isolation. Agent A publishes to `"devops"`, Agent B publishes to
`"frontend"`. Agent C queries each channel. Verifies zero cross-channel leakage in
retrieval results.

| Metric | Type | Description |
|---|---|---|
| `channel_isolation` | deterministic | `1.0 - cross_channel_leakage_rate` |
| `devops_recall` | deterministic | 1.0 if devops query returned ≥1 devops-domain result |
| `frontend_recall` | deterministic | 1.0 if frontend query returned ≥1 frontend-domain result |

---

#### S16 — Knowledge Relay (`s16_ma_relay`)

Layered chain publishing. Agents form a relay chain where each agent publishes
knowledge that the next agent in the chain consumes and augments.

---

#### S17 — Self-Correction via Sync (`s17_ma_self_correction`)

CAS-triggered correction. When a sync reveals that a previously published fact has
been superseded, the originating agent must update its local state accordingly.

---

#### S18 — Trust Calibration (`s18_ma_trust_calibration`)

Reliable vs unreliable agent gradient. Multiple agents with varying reliability
publish facts; trust scores must reflect actual accuracy over time.

---

#### S19 — Incident Reconstruction (`s19_ma_incident_recon`)

Evidence recall with red herrings. Agents collaboratively reconstruct an incident
timeline from fragmented observations, filtering irrelevant noise.

---

#### S20 — Belief Revision (`s20_ma_belief_revision`)

Multi-round contradiction resolution. Agents receive contradictory information
across multiple rounds and must converge on the correct canonical fact.

---

#### S21 — Distributed Product Knowledge Assembly (`s21_ma_product_assembly`)

5 specialists each contribute domain-specific product knowledge. The shared pool
must assemble a coherent, non-redundant product description.

---

#### S22 — Temporal Staleness Detection (`s22_ma_temporal_staleness`)

Version freshness. Agents must detect and prefer temporally recent facts over
stale entries when the shared pool contains multiple versions.

---

#### S23 — Adversarial Noise Injection (`s23_ma_adversarial_noise`)

Trust-weighted suppression. One agent injects noisy or incorrect facts. The system
must use trust signals to suppress low-quality contributions in retrieval.

---

#### S24 — Multi-Round Onboarding (`s24_ma_onboarding`)

KB absorption. A new agent joins a mature shared pool and must efficiently absorb
the existing knowledge base through incremental sync rounds.

---

#### S25 — Knowledge Conflict Resolution at Scale (`s25_ma_conflict_at_scale`)

40 conflicting facts across multiple agents must be resolved down to 10 canonical
facts via CAS and trust-weighted consensus.

---

### Legacy set (`_LEGACY`)

Selectable via `--scenarios legacy`. Not included in the default run.

| ID | Name | Key metrics |
|---|---|---|
| `s1_profile_retrieval` | Profile Retrieval | `relevance`, `completeness` (judge), `token_reduction` |
| `s2_avoid_repeats` | Avoid Repeats | `recall`, `semantic_recall`, `novelty` |
| `s3_feedback_learning` | Feedback Learning | `completeness` (judge), `rule_coverage`, `semantic_coverage` |
| `s4_deduplication` | Deduplication | `dedup_ratio`, `retention_ratio` |
| `s5_decay` | Temporal Decay | `relevance` (judge), `retention_delta` |
| `s6_load` | Load & Reliability | `relevance` (judge), `p95_latency_ms`, `error_rate` |
| `s7_consolidation` | Temporal Consolidation | `consolidation_ratio`, `semantic_latest_recall`, `faithfulness` (judge) |

Legacy scenarios use `LanguageBundle` fixtures and accept `--language`. Several use
Ollama embeddings or judge calls; use `--mock-judge` to run offline.

---

## LLM dependency matrix

| Scenario | Judge call? | Embedding call? | Notes |
|---|---|---|---|
| s1_mrr | no | `semantic_mrr` only | Falls back to lexical if Ollama down |
| s2_semantic_gap | no | `semantic_recall_at3` only | Falls back to 0.0 |
| s3_staleness | no | `semantic_freshness` only | Falls back to 0.0 |
| s4_compression_f1 | no | no | Fully deterministic |
| s5_noise | no | no | Fully deterministic |
| s6_token_economy | no | no | Fully deterministic |
| s7_grounding | no | no | Fully deterministic |
| s8_throughput | no | no | Fully deterministic |
| s9_scale | no | no | Fully deterministic |
| s16_update_correctness | no | no | Fully deterministic |
| s_locomo | no | no | Downloads dataset on first run |
| s8_ma_isolation | no | no | Fully deterministic |
| s9_ma_pool_publish | judge | no | `relevance` score |
| s10_ma_mesi_cas | no | no | Fully deterministic |
| s11_ma_mesi_sync | no | no | Fully deterministic |
| s12_topic_gating | no | no | Fully deterministic |
| s13_concurrent_writers | no | no | Fully deterministic |
| s14_trust_drift | no | no | Fully deterministic |
| s15_topic_leakage | no | no | Fully deterministic |
| s1_profile_retrieval (legacy) | judge | no | `relevance`, `completeness` |
| s2_avoid_repeats (legacy) | judge + embeddings | yes | `semantic_recall` |
| s3_feedback_learning (legacy) | judge + embeddings | yes | `semantic_coverage` |
| s4_deduplication (legacy) | no | no | Deterministic |
| s5_decay (legacy) | judge | no | `relevance` |
| s6_load (legacy) | judge | no | `relevance` |
| s7_consolidation (legacy) | judge + embeddings | yes | `semantic_latest_recall`, `faithfulness` |

**Guaranteed zero LLM calls** (offline-safe regardless of backend):
```bash
python -m tests.eval.benchmark.runner --mock-judge --backends baseline,ai_knot_no_llm
```
This runs `s1_mrr` through `s_locomo` (all 11 default scenarios) + all 18 MA scenarios
with `AiKnotMultiAgentBackend`, using `MockJudge` and `StubProvider`.

---

## Output files

| File | Format | Contents |
|---|---|---|
| `benchmark_report.md` | Markdown | Per-backend table with all scenario scores |
| `benchmark_raw.json` | JSON | Full nested results: `backends["{name}:{lang}"][scenario_id]` |

Both files are written after all backends finish. Existing files are overwritten silently.
