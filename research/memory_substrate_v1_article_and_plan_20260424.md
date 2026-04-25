# ai-knot Memory Substrate v1

## Черновик научной статьи и инженерный план переделок

Дата: 2026-04-24  
Статус: research / architecture proposal  
Ограничения: без prompt changes, без LOCOMO-specific словарей, без judge hacks, без brute-force context flooding.

---

## Название

**From Retrieval Logs to Multi-Projection Agent Memory: Evidence Ribbons, Mention Graphs, and Temporal Ledgers for Long-Term Conversational Recall**

Рабочий русский вариант:

**От поиска по логу к многопроекционной памяти агента: evidence ribbons, mention graph и temporal ledger для долгосрочной разговорной памяти**

---

## Abstract

Long-term conversational memory is often implemented as retrieval over raw dialogue turns or extracted facts. This design is attractive because it is simple, auditable, and cheap, but it fails under realistic multi-session workloads where answers depend on entity continuity, temporal normalization, aggregation across sessions, and evidence packaging for downstream language models. ai-knot's LOCOMO experiments show this failure mode clearly: ranking-only improvements, Personalized PageRank, HyDE-lite, answer-sheet prepending, and self-consistency voting do not move Cat1 enough to reach product-grade recall. The bottleneck is not a single ranker; it is the memory representation itself.

We propose **Memory Substrate v1**, a multi-projection architecture for agent memory. Raw episodes remain the immutable authority, while derived projections serve different retrieval and reasoning needs: a Mention Graph for entity continuity, an Event Ledger for normalized temporal facts, clean ClauseFrame atomic claims, Evidence Ribbons for bounded parent-context retrieval, and an LLM Context Pack that exposes normalized evidence without leaking a final answer. This design borrows validated ideas from temporal knowledge-graph memory, hierarchical retrieval, contextual retrieval, and production memory layers, but keeps ai-knot's constraints: deterministic core, no benchmark-specific patterns, and no prompt changes.

The expected outcome is not just higher LOCOMO accuracy. The product goal is a memory layer that can answer real user questions such as "what did I promise?", "when did I do that?", "what changed?", "what tools did we try?", and "what did we discuss with Sarah?" across months of conversation.

---

## 1. Problem Statement

The current ai-knot pipeline is already more sophisticated than naive RAG. It has raw episodes, deterministic materialization, atomic claims, support bundles, query contracts, raw episode fallback, hybrid BM25 plus embeddings, and SET-aware widening.

However, the canonical `p1-1b-2conv` run still sits at:

| Category | Accuracy |
|---|---:|
| Cat1 single-hop / aggregation factual | 30.23% |
| Cat2 temporal | 50.79% |
| Cat3 inference | 69.23% |
| Cat4 open-ended | 80.70% |
| Cat1-4 aggregate | 62.66% |

The failure pattern is structural:

1. **Entity gates lose continuity.** Retrieval starts from exact entity mentions in raw text. Turns with "she", "we", "the kids", "my family", or omitted names can be excluded before ranking.
2. **Atomic claims are sparse and noisy.** In the canonical 2-conv DB there are 788 raw episodes but only 295 atomic claims. The top subjects include noise such as `That must`, `I'd`, and `Your faith in me`.
3. **Temporal reasoning is computed but not always surfaced.** `time_resolve()` can resolve relative dates, but downstream evidence rendering can still show the misleading session date as the dominant text.
4. **Aggregation questions are treated like ranked retrieval.** Cat1 in LOCOMO is often not a single fact lookup; it requires collecting sets across sessions.
5. **Downstream LLMs read evidence, not internal trace.** If structured results are not rendered into the context consumed by the answer model, operator-level wins are invisible.

This explains why prior work in the project plateaued:

| Attempt | Result | Interpretation |
|---|---:|---|
| PPR entity-only / entity+token | no uplift | graph ranking does not solve missing/vocab-mismatched candidates |
| HyDE-lite | about +1 Q | expansion is weak if candidate pool and materializer are weak |
| Answer sheet prepend | +2/30 wrong Cat1 | sheet quality limited by noisy/incomplete claims |
| Self-consistency | +0/30 | voting cannot recover absent evidence |
| A+B stacked | worse than A alone | noisy union can destroy precision |

The conclusion is hard but useful: **55% Cat1 is not reachable by another ranker alone.**

---

## 2. Research Hypothesis

The central hypothesis:

> Long-term conversational memory requires multiple derived projections over the same raw authority. A single flat retrieval plane cannot simultaneously optimize entity continuity, temporal truth, aggregation recall, precision, and LLM usability.

The proposed architecture:

```text
Raw Episodes
  -> Mention Graph
  -> Event Ledger
  -> ClauseFrame Atomic Claims
  -> Session Capsules
  -> Evidence Ribbons
  -> LLM Context Pack
```

Each projection has a separate job:

| Projection | Job | Failure it addresses |
|---|---|---|
| Raw Episodes | source of truth | auditability, evidence |
| Mention Graph | entity continuity | pronouns, speaker continuation, group aliases |
| Event Ledger | temporal truth | yesterday / last week / valid time |
| ClauseFrame Claims | clean normalized facts | sparse/noisy materializer |
| Session Capsules | coarse recall | aggregation and long-range topic recall |
| Evidence Ribbons | bounded parent context | lost context without full-session flooding |
| LLM Context Pack | readable evidence | structured reasoning hidden from downstream LLM |

This is a product architecture, not a benchmark patch. LOCOMO is useful here because it measures real agent-memory capabilities: long-range recall, temporal consistency, aggregation, and inference.

---

## 3. Related Work, GitHub Landscape, and What We Borrow

Research snapshot: public sources checked on 2026-04-24. Do not cite May 2026 changes without re-checking them on or after their actual publication date.

The earlier related-work list was too small. By early 2026, the active field has moved beyond "vector store plus extracted facts". Strong systems now treat memory as an organized lifecycle: typed stores, temporal graphs, agent-managed write paths, offline consolidation, entity linking, multimodal traces, and benchmark-specific evaluation harnesses. This does not weaken the Memory Substrate v1 thesis; it makes the thesis more urgent. The competitive question is no longer "retrieval or graph?" but "which memory projections are explicit, rebuildable, inspectable, and cheap enough to run in product?"

### 3.1 Benchmarks: LOCOMO, LongMemEval, BEAM

LOCOMO remains the closest benchmark to ai-knot's current failure mode because it tests multi-session conversational recall, temporal questions, aggregation, and inference. Source: https://arxiv.org/abs/2402.17753

LongMemEval is now routinely paired with LOCOMO in memory-system repos, and Mem0's README also reports BEAM results at 1M and 10M tokens. Source: https://github.com/mem0ai/mem0

What ai-knot should learn:

- LOCOMO is necessary but insufficient; a credible memory paper should also discuss LongMemEval or a product scenario suite.
- Cat1 is not just single-hop QA. It behaves like evidence aggregation over a user/session/entity slice.
- Cat2 needs valid-time/event-time projection, not stronger semantic similarity alone.

### 3.2 Production memory systems

| System | Current direction | What changed by 2026 | What ai-knot should borrow | What not to copy blindly |
|---|---|---|---|---|
| Mem0 | production memory layer | April 2026 README reports a new ADD-only extraction algorithm, first-class agent facts, entity linking, and semantic+BM25+entity fusion; self-reported LoCoMo 91.6 | entity linking as a retrieval feature; add-only provenance discipline; open evaluation harness | treating self-reported benchmark numbers as externally verified; overwriting ai-knot's raw-authority model |
| Zep / Graphiti | temporal context graph | context graphs preserve provenance, fact validity windows, and hybrid retrieval across semantic, keyword, and graph traversal | explicit historical truth, relationship evolution, episode-backed graph edges | graph DB as the first milestone; heavyweight ontology before materializer quality |
| Letta | stateful agent runtime | memory blocks + archival memory + agentic insertion/search; memory is part of the agent loop | clear distinction between pinned working memory and archival retrieval | relying on agent tool loops for every recall path; opaque write decisions |
| LangMem / LangGraph | memory primitives for agents | hot-path memory tools plus background extraction/consolidation, integrated with LangGraph stores | background memory manager as an optional write path; API-level separation between store/search/manage | making downstream LLM behavior the only source of memory correctness |
| Cognee | knowledge engine / graph memory | vector + graph + LLM reasoning modes, temporal retriever, natural-language-to-Cypher, provenance options | multiple retrievers behind one recall API; graph-context formatting; temporal retrieval mode | LLM-to-Cypher in the critical path for low-latency product recall |
| Supermemory | hosted memory API / MCP | cross-session semantic retrieval and pattern-like user memory, designed for integrations | simple add/search product surface and MCP ergonomics | insufficient public detail for claiming deep temporal or evidence semantics |
| MemOS | memory operating system | MemCube abstraction, multimodal/tool/persona memory, scheduling, feedback/correction, multi-agent sharing | memory lifecycle language: scheduling, migration, correction, composition | broad OS scope before ai-knot has robust per-question projections |

Sources: https://arxiv.org/abs/2504.19413, https://github.com/mem0ai/mem0, https://github.com/getzep/graphiti, https://docs.letta.com/guides/core-concepts/memory/archival-memory, https://github.com/langchain-ai/langmem, https://docs.cognee.ai/core-concepts/main-operations/search, https://supermemory.ai/docs/supermemory-mcp/technology, https://arxiv.org/abs/2507.03724, https://github.com/MemTensor/MemOS

Interpretation:

- Mem0's April 2026 direction is especially relevant because it converges on entity linking and multi-signal retrieval, exactly where ai-knot's candidate-pool failures point.
- Graphiti is the strongest evidence that temporal validity and provenance should be first-class memory fields, not text decoration.
- Letta and LangMem show that product memory is becoming agent-facing infrastructure, but ai-knot should keep deterministic retrieval/rendering gates so results remain reproducible.
- Cognee and MemOS show a broader "memory operating system" trend: memory is not one index; it is orchestration over several representations.

### 3.3 New research systems after the original draft

| Paper / repo | Key idea | Relevance to Memory Substrate v1 |
|---|---|---|
| MIRIX | six memory types: Core, Episodic, Semantic, Procedural, Resource Memory, Knowledge Vault; multi-agent memory control; multimodal screen-memory use case | validates typed memory planes; suggests future `ResourceMemory` and `ProceduralMemory`, but v1 can stay text-first |
| LightMem | sensory / short-term / long-term stages, topic grouping, offline sleep-time updates, LoCoMo and LongMemEval evaluation scripts | supports Session Capsules and offline consolidation; warns against online-only extraction cost |
| StructMem | hierarchical memory with event-level bindings and cross-event connections, released under the LightMem project | close to Evidence Ribbons and Session Capsules; the important idea is preserving event binding, not just summary text |
| A-MEM | Zettelkasten-style dynamic indexing/linking; memory evolution updates older notes as new memories arrive | supports rebuildable linking and memory evolution; ai-knot should make updates traceable to raw evidence |
| EverMemOS | MemCells, MemScenes, Foresight signals, reconstructive recollection | supports multi-level projections and context assembly; "foresight" belongs after raw/evidence gates are stable |
| Hindsight / TEMPR | four logical networks for world facts, agent experiences, entity summaries, evolving beliefs; retain/recall/reflect operations | directly reinforces multi-plane memory and evidence vs inference separation |

Sources: https://arxiv.org/abs/2507.07957, https://arxiv.org/abs/2510.18866, https://github.com/zjunlp/LightMem, https://arxiv.org/abs/2502.12110, https://arxiv.org/abs/2601.02163, https://arxiv.org/abs/2512.12818

Interpretation:

- The strongest analogue is no longer plain Mem0 or Zep alone. It is the cluster of systems that explicitly separate memory types and lifecycle stages.
- ai-knot's differentiator should be **rebuildable, evidence-grounded projections over immutable raw episodes**, not merely "we also have graph memory".
- The plan should keep raw as authority but accept that modern competitors already have entity linking, temporal graphs, lifecycle operations, and background consolidation.

### 3.4 Classical retrieval work still matters

RAPTOR builds a hierarchy of summaries and retrieves across multiple abstraction levels. Source: https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html

Anthropic Contextual Retrieval and Jina Late Chunking both show that chunks are weak when encoded without parent context. Sources: https://www.anthropic.com/research/contextual-retrieval and https://arxiv.org/abs/2409.04701

Lost in the Middle shows that long contexts are not reliably used by LLMs; evidence order and placement matter. Source: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long

Microsoft GraphRAG treats broad corpus questions as query-focused summarization rather than simple nearest-neighbor retrieval. Source: https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/

What ai-knot should borrow:

- Coarse-to-fine retrieval, but with deterministic Session Capsules before LLM-generated abstractions.
- Parent-aware evidence packaging, but as bounded Evidence Ribbons rather than full-session flooding.
- Compact normalized headers before raw evidence, so downstream LLMs see the resolved entity/time/fact first.
- Map/reduce-style assembly for aggregation questions, but grounded in raw episode pointers.

### 3.5 Revised competitive claim

The original claim "rankers are not enough" is still correct, but incomplete. The more defensible 2026 claim is:

> Agent memory is converging on organized, typed, temporal, and lifecycle-aware substrates. ai-knot should compete by making those projections rebuildable from immutable raw evidence, deterministic enough to debug, and directly visible in the LLM context pack.

This gives ai-knot a sharper position:

- Against Mem0: not just entity-linked retrieval, but raw-backed rebuildable projections and deterministic operators.
- Against Graphiti/Cognee: temporal and graph-like projections without requiring graph infrastructure as the first product dependency.
- Against Letta/LangMem: agent-friendly memory APIs without making agent tool loops the only correctness path.
- Against LightMem/EverMemOS/Hindsight: similar multi-plane memory thesis, but scoped to a practical LOCOMO/product retrofit with explicit kill gates.

### 3.6 Benchmark matrix and evaluation fit

The paper should not rely on LOCOMO alone. LOCOMO is a good diagnostic for the current ai-knot bottleneck, but 2026 memory systems are increasingly evaluated across scale, personalization, dynamic state, and agentic action. The right evaluation story is a matrix:

| Benchmark / suite | What it stresses | Why it matters for ai-knot | Blind spot | How to use it |
|---|---|---|---|---|
| LOCOMO | long multi-session conversation QA; single-hop, temporal, inference, open-ended questions | closest match to current Cat1/Cat2 failures | can be solved partly by stronger reader/context on small instances | primary regression suite for v1 |
| LongMemEval | information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention across 500 questions | maps directly to Event Ledger, update semantics, and evidence abstention | mostly chat-assistant memory, not full agent workflows | second benchmark once v1 stabilizes |
| BEAM | 100K to 10M-token memory stress; 10 ability types including temporal reasoning, event ordering, contradiction resolution | tests whether projections survive production-scale history growth | newer benchmark; reported scores may depend heavily on harness choices | scale stress test for Session Capsules and Evidence Ribbons |
| Agent Memory Benchmark / AMB | open harness across LoCoMo, LongMemEval, PersonaMem, LifeBench, MemBench; tracks accuracy, speed, cost | forces product-style comparison, not accuracy-only comparison | vendor-originated harness; must inspect prompts/models | use as external reproducibility target, not sole authority |
| MemoryArena | memory used in interdependent multi-session agentic tasks | tests whether memory guides later action, not only QA recall | young benchmark; may require agent/tool integration | future product-level validation |
| MemGround | gamified dynamic state tracking, temporal association, hierarchical reasoning | exposes failures in sustained state and event ordering | young benchmark; not yet a standard leaderboard | future stress test for Event Ledger and evolving projections |
| Product scenarios | reminders, project decisions, profile recall, multi-agent handoff, "what changed?" | closest to actual ai-knot users | requires hand-curated gold answers | mandatory release gate before product claims |

Sources: https://arxiv.org/abs/2410.10813, https://arxiv.org/abs/2510.27246, https://github.com/vectorize-io/agent-memory-benchmark, https://arxiv.org/abs/2602.16313, https://arxiv.org/abs/2604.14158

Evaluation implication:

- Report LOCOMO as the fast iteration benchmark.
- Report LongMemEval or AMB only after the context-pack interface is stable.
- Treat BEAM as the architecture stress test: if v1 works only at LOCOMO size, it is not a memory substrate yet.
- Track latency, token budget, extraction cost, and reproducibility beside accuracy.

### 3.7 Confidence levels for competitor claims

Competitor numbers are not equal evidence. The analysis should label every external score with a confidence tier:

| Tier | Evidence type | Examples | How to cite |
|---|---|---|---|
| A | conference/peer-reviewed or mature benchmark paper plus public dataset/code | LongMemEval paper, BEAM paper | cite as benchmark evidence |
| B | arXiv paper plus public benchmark spec or repo when available | MemoryArena, MemGround, LightMem, MIRIX | cite as emerging research evidence |
| C | open-source repo with reproducible scripts and raw result files | Mem0 memory-benchmarks, AMB if run locally | cite as reproducible only after local run |
| D | product docs or vendor blog with methodology | Mem0 docs, Hindsight blog, Supermemory docs | cite as self-reported product evidence |
| E | marketing page or leaderboard without enough methodology | vendor scoreboards | cite only as market signal, not scientific evidence |

Rules for the paper:

- Use "reported" for external benchmark numbers until ai-knot reproduces them locally.
- Never compare scores if answerer, judge, extraction model, top-k, context budget, or oracle mode differ.
- Separate retrieval recall from answer accuracy. A system can retrieve the right evidence and still fail because of the reader.
- Publish ai-knot's harness settings: dataset version, model versions, prompt hashes, token budget, latency, and extraction cost.
- Add a "claim confidence" column to any competitor score table.

This protects the paper from the weakest criticism: cherry-picking self-reported leaderboards.

### 3.8 Architectural taxonomy

The analysis should compare systems by memory design axes, not by brand names:

| Axis | Common options | ai-knot v1 target |
|---|---|---|
| Source of truth | raw log, extracted facts, graph edges, summaries | raw episodes as immutable authority |
| Write path | online LLM extraction, deterministic extraction, offline consolidation, agent-managed writes | deterministic core plus optional offline rebuild |
| Memory planes | episodic, semantic, procedural, temporal, belief, resource/tool memory | episodic raw, semantic claims, temporal ledger, mention graph, capsules |
| Time model | timestamp only, event time, valid time, bitemporal validity | event_time, observed_at, valid_from, valid_until where needed |
| Entity model | string matching, entity extraction, entity linking, coreference graph | Mention Graph with confidence tiers |
| Retrieval | vector, BM25, graph traversal, temporal filter, fusion, agentic search | hybrid raw/claims/mentions/events/capsules with deterministic fusion |
| Evidence interface | hidden prompt context, citations, raw snippets, normalized context pack | LLM Context Pack with normalized evidence and raw pointers |
| Update semantics | overwrite, delete, soft-delete, append-only, invalidate/version | append raw, rebuild projections, invalidate derived rows |
| Scale strategy | bigger context, wider top-k, hierarchy, graph, lifecycle scheduling | capsules and ribbons before context flooding |
| Debuggability | opaque agent memory, partial traces, full provenance | every projection row points back to raw episodes |

This taxonomy makes the thesis falsifiable. If a future competitor matches ai-knot on all axes with lower cost and better reproducibility, the claimed differentiation disappears.

### 3.9 Gap analysis against ai-knot

| System / cluster | What it currently does better | Gap exposed in ai-knot | v1 answer | Gate |
|---|---|---|---|---|
| Mem0 | entity linking, ADD-only extraction, fused semantic/BM25/entity retrieval, public memory-benchmark repo | current candidate pool misses implicit entities and agent-generated facts | Mention Graph, improved materialization, scorecard | entity_gate_miss_rate down >= 40% |
| Graphiti / Zep | temporal context graph, validity windows, provenance-backed graph edges | current time resolution is not always visible in answer context | Event Ledger and LLM Context Pack | temporal off-by-one errors down >= 50% |
| Letta / LangMem | agent-facing memory APIs, hot-path tools, background memory management | ai-knot API exposes retrieval more than memory lifecycle | later memory manager API, after v1 projections | no hidden answer leakage |
| Cognee | graph + vector + temporal retrieval modes behind one interface | ai-knot lacks a unified multi-retriever abstraction | Query compiler and operator routing | no benchmark-category branching |
| LightMem / StructMem | staged memory and event bindings with offline consolidation | ai-knot has raw episodes but weak higher-level grouping | Session Capsules and Evidence Ribbons | aggregation recall improves without bloat |
| Hindsight / TEMPR | multiple memory networks and explicit retain/recall/reflect framing | ai-knot needs clearer separation between evidence, inference, and evolving belief | raw authority plus projection confidence | derived truth never overrides raw |
| MemOS / EverMemOS | memory lifecycle language: schedule, consolidate, migrate, correct | ai-knot plan may look too narrow against "memory OS" systems | explicitly scope v1, postpone OS breadth | text evidence/time/entity gates pass first |
| BEAM/Huge-context baselines | expose collapse at 10M tokens and when context stuffing fails | LOCOMO alone may hide scale brittleness | add scale simulation after v1 | accuracy/latency tracked with corpus growth |

Strategic read:

- The fastest route to credible improvement is not a full graph system. It is closing the Mem0/Graphiti/LightMem gaps with smaller, inspectable projections.
- The strongest paper framing is "minimal memory substrate" rather than "new universal memory OS".
- The strongest product framing is "evidence you can debug" rather than "memory that magically remembers".

### 3.10 Threats to the thesis

The article should explicitly name what could make the hypothesis wrong:

| Threat | Why plausible | How to test | Design response |
|---|---|---|---|
| Representation is not the bottleneck; extraction quality is | current claims are sparse/noisy, so projections may amplify bad materialization | run Mention Graph/Evidence Ribbons over raw-only candidates before ClauseFrame V7 | sequence M1-M3 before M5 |
| Raw authority is too expensive at scale | raw-backed evidence can increase storage, indexing, and context cost | measure evidence_token_budget and latency as corpus grows | use ribbons/capsules, not full-session dumps |
| LLM-managed memory beats deterministic projections | modern memory systems use powerful extractors and background agents | compare deterministic vs LLM-managed write paths on same harness | allow optional offline memory manager while keeping provenance |
| LOCOMO gains do not transfer to product | benchmark questions may not match reminders, project memory, or multi-agent handoff | product scenario suite with hand-audited answers | make product scenarios release gates |
| Mention Graph creates false continuity | pronoun/coreference inference can attach facts to the wrong person | audit false_coref_rate by confidence tier | inferred mentions expand recall but never stand alone as truth |
| Event Ledger over-resolves vague time | "last year" and "recently" are often intervals, not exact dates | track granularity and resolution_method | render uncertainty in context pack |
| Context Pack leaks answers | normalized evidence may become an answer sheet | red-team context pack for final-answer leakage | only evidence fields, no answer sentence |
| Projections become stale or inconsistent | rebuildable indexes can drift from raw after migrations | projection checksum and rebuild tests | all derived rows carry source ids and version |
| Benchmark contamination creeps in | LOCOMO nouns or category-specific branching can sneak into tests | static scan for benchmark terms; generic examples only | enforce "no LOCOMO phrase" gate |
| The market moves faster than v1 | competitors may ship stronger typed memory before ai-knot v1 lands | monthly competitor refresh | keep v1 narrow and measurable |

This threats section is important. It turns the article from a manifesto into a research plan: each major claim has a failure mode, a test, and a design response.

---

## 4. Proposed Architecture

### 4.1 Raw Episodes: immutable authority

Raw turns remain the source of truth. Derived projections are indexes, caches, or views. They must be rebuildable from raw episodes.

Invariant:

```text
No derived memory can override raw evidence.
No final answer can be stored as truth without evidence pointers.
```

### 4.2 Mention Graph

Current retrieval is too dependent on explicit name strings. The Mention Graph attaches entity references to episodes even when the raw text does not contain the entity name.

Schema:

```text
episode_mentions(
  agent_id,
  episode_id,
  entity,
  mention_type,
  confidence,
  source,
  created_at
)
```

Mention types:

| Type | Confidence | Meaning |
|---|---:|---|
| explicit | 1.00 | entity name appears directly |
| speaker | 0.90 | episode speaker is the entity |
| continuation | 0.65 | local session continuation from prior explicit mention |
| pronoun | 0.55 | local pronoun resolution |
| group_alias | 0.45 | "kids", "family", "team" linked to a person |

Retrieval rule:

```text
candidate episodes = explicit entity hits
                   ∪ mention-linked episodes above threshold
                   ∪ local evidence ribbon neighbors
```

Product benefit:

- "What did Alice say about her kids?" can retrieve turns that say "they loved dinosaurs" without repeating "Alice".

Metric hypothesis:

- `entity_gate_miss_rate` decreases by at least 40%.
- Cat1 improves by +3..6 Q on canonical 2-conv without cat4 regression.

Risk:

- False coreference can attach facts to the wrong person.

Mitigation:

- Confidence tiers.
- Use inferred mentions for recall expansion, not as standalone truth.
- Keep raw evidence visible.

### 4.3 Event Ledger

Time should not live only as text. The Event Ledger stores event time, observation time, and validity time separately.

Schema:

```text
event_ledger(
  agent_id,
  event_id,
  entity,
  relation,
  value,
  event_time,
  event_time_granularity,
  valid_from,
  valid_until,
  observed_at,
  source_episode_id,
  confidence,
  resolution_method
)
```

Examples:

```text
"yesterday" relative to 2023-05-07 -> event_time=2023-05-06
"last Friday" relative to session date -> calendar-resolved date
"last year" -> interval/year granularity
```

Product benefit:

- Reminders, timelines, "when did I...", "what changed since..." become reliable.

Metric hypothesis:

- Temporal off-by-one class decreases by at least 50%.
- Cat2 moves from about 51% toward 62-70%.

Risk:

- Over-resolving vague time expressions.

Mitigation:

- Store granularity and resolution method.
- Render uncertainty explicitly as evidence, not as final answer.

### 4.4 ClauseFrame Materializer V7

The current materializer should evolve from pattern patches to a small deterministic compiler:

```text
Sentence
  -> cleanup
  -> speaker injection
  -> temporal phrase extraction
  -> subject resolution
  -> verb canonicalization
  -> object span extraction
  -> polarity / modality detection
  -> deictic and noise guard
  -> AtomicClaim / EventLedger row
```

Internal representation:

```python
ClauseFrame(
    subject: str,
    verb: str,
    canonical_relation: str,
    object_text: str,
    modifiers: dict[str, str],
    time_expr: str | None,
    polarity: str,
    confidence: float,
    source_span: tuple[int, int],
)
```

Generic frames:

| Frame | Example | Claim |
|---|---|---|
| speaker action object | "I read Charlotte's Web" | read( speaker, book ) |
| speaker preference object | "I love pottery" | likes( speaker, pottery ) |
| speaker acquisition object | "I bought new shoes" | acquired/bought( speaker, shoes ) |
| speaker activity with group | "I went camping with my family" | activity( speaker, camping, with=family ) |
| entity state value | "Caroline is single" | state( Caroline, single ) |
| entity transition value | "I moved to Sweden" | moved_to( speaker, Sweden ) |

Acceptance gates:

```text
claim_coverage >= baseline + 20%
noisy_subject_rate <= baseline / 2
relation_reachability = 100%
no LOCOMO phrases in tests
cat4 >= baseline - 1pp
cat5 >= baseline - 1pp
```

Risk:

- Regex zoo returns under a new name.

Mitigation:

- ClauseFrame pipeline must be organized by compiler stages, not by one-off benchmark patterns.
- Every new extraction rule needs generic tests.

### 4.5 Evidence Ribbon Retrieval

Full parent-session retrieval is too blunt. Evidence Ribbons are bounded parent context:

```text
ribbon = child hit
       + useful local neighbors
       + mention-linked continuation turns
       + same topic/action/object turns
```

Algorithm:

1. Retrieve child hits from raw, claims, event ledger, mentions, and capsules.
2. Group by session.
3. For each group, build candidate ribbon turns.
4. Score ribbons by:
   - child hit score
   - mention confidence
   - temporal proximity
   - topic/action overlap
   - redundancy penalty
5. Render top 2-4 ribbons under budget.

Product benefit:

- Users get coherent evidence around an event or topic without dumping the whole session.

Metric hypothesis:

- Cat1 set recall improves by +4..7 Q.
- Evidence token count stays bounded.
- Cat4 and Cat5 do not regress materially.

### 4.6 Session Capsules

Session Capsules are coarse retrieval units. They should be built only after Mention Graph and Materializer V7 cleanup, otherwise they compress noise.

Deterministic capsule example:

```text
Session 2023-07-19
Entities: Caroline, Melanie
Actions: joined LGBTQ activist group; discussed regular meetings; planned events
Artifacts: group name "Connected LGBTQ Activists"
Dates: 2023-07-19
Open threads: activism, community support
Evidence: episode ids...
```

Use:

- index for coarse retrieval
- route to relevant sessions
- render raw/ribbons as authority

Metric hypothesis:

- Helps Cat1 aggregation and Cat3 inference.
- Should not be allowed to override raw evidence.

### 4.7 LLM Context Pack

`QueryAnswer.llm_context` should become the primary context product for downstream LLMs.

It must not include a final answer. It should include normalized evidence:

```text
# Normalized Evidence
- Entity: Caroline
  Event: went to LGBTQ support group
  Resolved date: 2023-05-06
  Evidence: ep_123

# Evidence Ribbons
[1] [2023-05-07] Caroline: I went to a LGBTQ support group yesterday...
```

Why this matters:

- Keeps product API honest.
- Makes structured reasoning visible.
- Reduces Lost-in-the-Middle risk by putting compact evidence first.

Acceptance gate:

- Cat2 improves without answer leakage.
- Review must confirm no final answer is being injected.

---

## 5. Experimental Program

### 5.1 Baseline

Use `p1-1b-2conv` as canonical baseline:

```text
cat1 = 13/43 = 30.23%
cat2 = 32/63 = 50.79%
cat3 = 9/13 = 69.23%
cat4 = 92/114 = 80.70%
cat1-4 = 146/233 = 62.66%
```

### 5.2 Product scorecard

Do not rely only on benchmark score. Track:

| Metric | Purpose |
|---|---|
| claim_coverage | memory extraction recall |
| noisy_subject_rate | materializer precision |
| entity_gate_miss_rate | candidate-pool failure |
| mention_continuity_rate | coreference/continuation coverage |
| relative_time_resolution_rate | temporal normalization |
| evidence_density | context quality |
| evidence_token_budget | cost/latency control |
| structured_context_visibility | whether downstream LLM sees normalized evidence |
| benchmark_claim_confidence | whether external comparisons are reproduced or self-reported |
| reproducibility_manifest_complete | dataset/model/prompt/token settings captured |
| scale_slope | accuracy/latency degradation as corpus size grows |
| product_scenario_pass_rate | transfer from LOCOMO to real ai-knot workflows |

### 5.3 Kill gates

Every phase needs a stop-loss condition:

| Phase | Continue only if |
|---|---|
| Mention Graph | entity_gate_miss_rate down >= 40%, no cat5 regression |
| Evidence Ribbons | Cat1 up, cat4 >= baseline - 1pp |
| Event Ledger | off-by-one errors down >= 50% |
| ClauseFrame V7 | coverage up >= 20%, noisy subjects halved |
| Session Capsules | improves aggregation without increasing evidence bloat |
| LLM Context Pack | no final-answer leakage, cat2 improves |

### 5.4 Evaluation sequence

1. Offline diagnostic preflight.
2. Unit tests on generic non-LOCOMO examples.
3. Canonical 2-conv LOCOMO run.
4. Per-category regression audit.
5. Full 10-conv smoke or staged full run.
6. Product scenario tests: reminders, profile recall, project decisions, multi-agent handoff.
7. LongMemEval/AMB pilot once `llm_context` is stable.
8. BEAM-style scale simulation after Session Capsules and Evidence Ribbons.
9. Reproducibility report with claim confidence labels for every external comparison.

---

## 6. Engineering Plan

### Milestone M0: Observability

Files likely touched:

- `scripts/`
- `tests/`
- `aiknotbench/`
- maybe `src/ai_knot/query_runtime.py` trace fields

Deliverables:

- `memory_scorecard.py`
- JSON report per run
- reproducibility manifest: dataset version, model versions, prompt hashes, token budget, latency, extraction cost
- CI or local gate script

No product behavior change.

### Milestone M1: LLM Context Pack

Files likely touched:

- `src/ai_knot/query_types.py`
- `src/ai_knot/query_runtime.py`
- `src/ai_knot/_mcp_tools.py`
- `npm/src/types.ts`
- `aiknotbench/src/aiknot.ts`

Deliverable:

- `QueryAnswer.llm_context`
- bench adapter consumes `llm_context`, not raw-only `evidence_text`

Guard:

- context contains normalized evidence, not final answers.

### Milestone M2: Mention Graph

Files likely touched:

- new migration: `src/ai_knot/migrations/v4_episode_mentions.py`
- `src/ai_knot/materialization.py` or new `mentions.py`
- `src/ai_knot/storage/sqlite_storage.py`
- `src/ai_knot/query_runtime.py`

Deliverables:

- `episode_mentions` table
- mention builder
- mention-expanded raw search

Guard:

- inferred mention confidence threshold.
- explicit mentions always outrank inferred mentions.

### Milestone M3: Evidence Ribbons

Files likely touched:

- `src/ai_knot/query_runtime.py`
- `src/ai_knot/storage/sqlite_storage.py`
- new module: `src/ai_knot/evidence_ribbons.py`

Deliverables:

- child hit grouping
- bounded ribbon renderer
- density-aware context budget

Guard:

- no full-session dump as default.

### Milestone M4: Event Ledger

Files likely touched:

- migration: `src/ai_knot/migrations/v5_event_ledger.py`
- `src/ai_knot/materialization.py`
- `src/ai_knot/query_operators.py`
- `src/ai_knot/query_runtime.py`

Deliverables:

- event ledger rows
- relative time resolver
- temporal evidence projection in `llm_context`

Guard:

- store granularity and resolution method.

### Milestone M5: ClauseFrame V7

Files likely touched:

- `src/ai_knot/materialization.py`
- possible new modules:
  - `src/ai_knot/clause_frame.py`
  - `src/ai_knot/time_resolver.py`
  - `src/ai_knot/mention_resolution.py`
- `tests/test_materializer_frames_v7.py`

Deliverables:

- compiler-like extraction pipeline
- subject sanitizer
- generic tests

Guard:

- no LOCOMO nouns/phrases.

### Milestone M6: Generic Query Compiler

Files likely touched:

- `src/ai_knot/query_contract.py`
- `src/ai_knot/relation_vocab.py`
- `tests/test_query_contract.py`

Deliverables:

- structural aggregation detector
- temporal-first routing
- multihop relation plan skeleton

Guard:

- no benchmark category branching.

### Milestone M7: Session Capsules

Files likely touched:

- new table / migration
- new module: `src/ai_knot/session_capsules.py`
- `src/ai_knot/support_retrieval.py`

Deliverables:

- deterministic capsule builder
- coarse-to-fine retrieval

Guard:

- capsules are routing hints, not truth.

---

## 7. Expected Impact

Non-additive, but plausible:

| Component | Cat1 effect | Cat2 effect | Risk |
|---|---:|---:|---|
| LLM Context Pack | +0..2 Q | +2..5 Q | answer leakage if poorly designed |
| Mention Graph | +3..6 Q | +0..2 Q | false coref |
| Evidence Ribbons | +4..7 Q | +1..2 Q | context bloat |
| Event Ledger | +0..1 Q | +4..7 Q | over-resolving vague time |
| ClauseFrame V7 | +4..8 Q | +2..5 Q | regex zoo / noise |
| Session Capsules | +2..5 Q | +0..2 Q | compressed noise |

Target after v1:

```text
cat1: 50-58%
cat2: 62-70%
cat3: >=65%
cat4: >=78%
cat1-4: 68-72%
```

---

## 8. What Not To Do

1. Do not change answer or judge prompts.
2. Do not add LOCOMO-specific noun lists.
3. Do not increase `top_k` as the main strategy.
4. Do not ship answer sheets on current noisy claims.
5. Do not implement full GraphRAG before Mention Graph and Event Ledger.
6. Do not use self-consistency to compensate for missing evidence.
7. Do not allow derived projections to become ungrounded truth.
8. Do not chase self-reported benchmark scoreboards without reproducible local gates.
9. Do not make agentic search loops the only correctness path for recall.
10. Do not expand into a broad memory OS or multimodal memory before text evidence, time, and entity continuity are stable.

---

## 9. Thesis for a Scientific Review Board

The defensible thesis is:

> Agent memory should not be a single vector store or a single fact table. It should be a set of rebuildable projections over immutable conversational evidence. Each projection solves a different cognitive and retrieval problem: entity continuity, temporal normalization, aggregation, abstraction, and evidence presentation. LOCOMO improves when these product capabilities improve, not because the system was tuned to the benchmark.

This is stronger than "we improved retrieval". It is a coherent memory architecture.

---

## 10. First Implementation Recommendation

Start with:

```text
M0 Scorecard
M1 LLM Context Pack
M2 Mention Graph
M3 Evidence Ribbons
```

Reason:

- They address candidate-pool and evidence-rendering failures directly.
- They are less risky than Materializer V7.
- They create observability before deep extraction refactors.
- They make later Event Ledger and ClauseFrame V7 easier to validate.

Only after M1-M3 should the project attempt M5 ClauseFrame V7. Otherwise, the team risks another large materializer refactor without knowing whether retrieval/rendering still hides the win.
