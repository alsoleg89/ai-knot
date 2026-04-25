# Pattern Memory Architecture — Associative Retrieval for LLM Agent Memory

## Date: 2026-04-10
## Status: Research / Design phase

---

## 1. Origin: Biological Analogy

### What unites DNA, ant colonies, and 3nm processors?

**Information is not the data — information is the RELATIONSHIPS between data points.**

- **DNA**: A gene is meaningless without its regulatory context. The gene + promoter + enhancer + epigenetic marks = actual information. One nucleotide means nothing. A PATTERN of nucleotides = protein.
- **Ant colonies**: One pheromone dot is noise. A GRADIENT of pheromone across space = route. No single ant knows the route. The route exists only as a pattern BETWEEN ants.
- **3nm processors**: One memory address is just a number. The PATTERN of access across addresses = information (locality, stride, branching). Prefetcher uses patterns, not individual addresses.

### Application to memory retrieval

Current ai-knot: stores 700 facts, searches them as **independent objects**. BM25 scores each fact separately. "Melanie goes swimming with kids" is an isolated string.

But the answer to "What activities does Melanie do?" is NOT in any single fact. It's a **PATTERN BETWEEN** facts. No individual fact contains the answer. The answer emerges from relationships.

This is how DNA works: no single gene encodes an "organism". The organism emerges from the INTERACTION of genes.

---

## 2. Core Insight: Token Co-occurrence as Associative Memory

### Where relationships live

In **token co-occurrence** across facts.

If in a corpus of 700 facts:
- "melanie" + "pottery" co-occur in 15 facts
- "melanie" + "camping" co-occur in 8 facts
- "melanie" + "swimming" co-occur in 2 facts
- "melanie" + "running" co-occur in 4 facts

...this is a **relationship graph** that already contains the answer to "What activities does Melanie do?" — before the question is even asked. Like DNA contains the plan for an organism before expression begins.

### Two layers of memory (as in the brain)

**Episodic memory** — specific facts: "Melanie goes swimming with kids on 8 May"
**Semantic memory** — association patterns: `melanie ↔ {pottery(15), camping(8), swimming(2), kids(20)}`

ai-knot already conceptually distinguishes these types (`MemoryType.EPISODIC` / `SEMANTIC`). But the implementation is identical — a flat list of strings. Semantic memory is currently just a fact with type="semantic". No real semantic structure.

Pattern memory IS the real semantic layer — compressed, associative, pattern-based.

---

## 3. What Is a "Mini-Aggregation" Concretely

For each significant token in the corpus — a compressed profile of its associations:

```
melanie → pottery:15, kids:20, camping:8, painting:12, 
          swimming:2, running:4, music:3, beach:4
          [350 facts, 88-byte bitvector]

pottery → melanie:15, kids:8, clay:6, bowl:5, 
          workshop:4, class:3
          [77 facts, 88-byte bitvector]
```

Each profile = **compressed knowledge** about what this token means **in this specific corpus**. Not in English generally, but in THIS agent's memory.

This is like a neuroscience engram — a pattern of synaptic connections that encodes a concept. "Melanie" is not a string — it's a constellation of associations.

---

## 4. Compact Binary Structure

### Size calculation (700 facts, ~500 unique significant tokens)

**Co-occurrence matrix** (upper triangle, symmetric):
- 500 × 499 / 2 = 124,750 pairs
- 4-bit counters (0-15): **62 KB**
- 1-byte counters (0-255): **122 KB**

**Posting bitvectors** (token → which facts contain it):
- 700 facts = 88 bytes per bitvector
- 500 tokens × 88 bytes = **44 KB**

**Token table** (hash → id mapping):
- 500 × 8 bytes = **4 KB**

**Total: ~110 KB** — the entire associative graph of all relationships.

```
L1 cache:  32-64 KB  ← posting bitvectors (44KB) fit here
L2 cache: 256-512 KB ← entire structure (110KB) fits here
RAM:       ...       ← pattern lookup never goes here
```

**The agent's entire "semantic memory" fits in CPU L2 cache.**

### Scaling properties

| Corpus size | Unique tokens | Matrix (4-bit) | Bitvectors | Total |
|-------------|---------------|-----------------|------------|-------|
| 700 facts | ~500 | 62 KB | 44 KB | ~110 KB |
| 2,000 facts | ~1,000 | 250 KB | 250 KB | ~500 KB |
| 5,000 facts | ~2,000 | 1 MB | 1.25 MB | ~2.3 MB |
| 10,000 facts | ~3,000 | 2.25 MB | 3.75 MB | ~6 MB |

At 10K facts: still fits in L3 cache (typically 8-32 MB). Beyond that: sparse matrix formats or top-K truncation per token.

---

## 5. Query Pipeline

```
Query: "What activities does Melanie partake in?"

Step 1: Token lookup (nanoseconds)
  query_tokens = {melanie, activities, partake}
  token_ids = hash(melanie)→42, hash(activities)→17, ...

Step 2: Co-occurrence expansion (microseconds)
  row[42] (melanie) → pottery:15, kids:20, camping:8, swimming:2, ...
  row[17] (activities) → pottery:5, camping:3, swimming:1, ...
  merged expansion = {pottery:0.4, camping:0.2, swimming:0.05, kids:0.5, ...}

Step 3: Candidate selection via bitvectors (microseconds)
  bv[melanie] OR bv[pottery] OR bv[camping] OR bv[swimming]
  = bitvector with ~200 set bits → 200 candidate facts
  (bitwise OR — one CPU instruction per 64 facts)

Step 4: BM25 on 200 candidates with expanded query (milliseconds)
  → ranked top_k=60

Steps 1-3: microseconds. Pattern lookup, not search.
Like an L1 cache hit instead of RAM access.
```

---

## 6. Difference from Existing Techniques

### vs. Standard Inverted Index
- Inverted index: token → [fact_ids] (posting list)
- Pattern memory: token → [co-occurring tokens with weights] + [fact_ids]
- The CO-OCCURRENCE LAYER is what's new
- Standard inverted index answers: "find facts containing X"
- Co-occurrence layer answers: "what is X associated with in this corpus?"
- The first is retrieval. The second is UNDERSTANDING.

### vs. Pseudo-Relevance Feedback (PRF)
- PRF: expand query from TOP-K retrieved facts (biased by initial results)
- Pattern memory: expand query from ALL facts via co-occurrence (corpus-wide)
- PRF creates echo chambers (if top-K is all pottery, PRF expands with pottery terms)
- Pattern memory sees "melanie" + "swimming" co-occurrence even if no swimming fact is in top-K
- PRF = ant following ITS OWN trail (echo chamber)
- Pattern memory = ant seeing ALL ants' trails (collective intelligence)

### vs. LLM Query Expansion
- LLM: "activities" → "hobbies, pastimes, recreation" (from training data)
- Pattern memory: "melanie" → "pottery, camping, swimming" (from THIS corpus)
- LLM knows language. Pattern memory knows **these specific data**.
- For aggregation, you need data associations, not language associations.

### vs. Knowledge Graph
- KG: explicit entity-relation-entity triples (requires NER, relation extraction)
- Pattern memory: implicit token-token associations (requires only tokenization)
- KG is top-down (impose structure). Pattern memory is bottom-up (structure emerges).
- KG needs maintenance. Pattern memory auto-updates with each add/learn.

### vs. memvid
| | memvid | ai-knot + pattern memory |
|--|--------|--------------------------|
| Storage | Raw sessions in video codec | Facts + 110KB binary patterns |
| Model | Filing cabinet | Associative memory |
| On query | Search through texts (Tantivy) | Activate patterns → retrieve details |
| Coverage | 16% (10 sessions of 54) | Pattern-guided: all associated facts |
| Aggregation | OR-fallback, diversify post-hoc | Native: co-occurrence == aggregation |
| Compression | Video codec (compresses text) | Co-occurrence matrix (compresses KNOWLEDGE) |

memvid compresses **text**. ai-knot compresses **relationships between facts**. Text is data. Relationships are memory.

---

## 7. Lifecycle

```
INGEST (learn/add):
  1. Store fact in facts.db
  2. Tokenize fact → significant tokens
  3. Update co-occurrence counts in patterns.bin (incremental)
  4. Set bits in posting bitvectors
  → O(tokens²) per fact, amortized

RECALL:
  1. Pattern lookup → expanded query + candidate bitvector (μs)
  2. BM25 on candidates only (ms)
  → No full-corpus scan needed

FORGET:
  1. Remove/expire fact in facts.db
  2. Decrement co-occurrence counts
  3. Clear bit in posting bitvectors
  → Ebbinghaus decay applies to PATTERNS too
     (rarely co-occurring tokens fade, frequently co-occurring strengthen)
```

Patterns are not static — they **live**. New facts strengthen associations. Forgotten facts weaken them. Like synapses: use it or lose it.

---

## 8. Intent-Dependent Expansion (Reading Frames)

Same co-occurrence structure, different "reading frame" depending on query intent:

**AGGREGATION** ("What activities does Melanie partake in?"):
- High expansion: take top-20 co-occurring tokens per query token
- Low threshold: include tokens with count >= 2
- Goal: maximize coverage, find ALL associated concepts

**POINT** ("What is Melanie's salary?"):
- Low expansion: take top-5 co-occurring tokens
- High threshold: include tokens with count >= 5
- Goal: maximize precision, find the specific fact

**TEMPORAL** ("When did Melanie go camping?"):
- Expand from content tokens, but boost tokens that are dates or temporal markers
- Co-occurrence structure tells you: "camping" co-occurs with "june", "july", "summer"

**MULTI-HOP** ("Where does Alex's wife work?"):
- Round 1: expand "alex" + "wife" → find "maria" in co-occurrence
- Round 2: expand "maria" + "work" → find "google" in co-occurrence
- The hop happens IN THE PATTERN STRUCTURE, not in retrieved facts

This is like DNA reading frames: same sequence, different interpretation depending on the regulatory context (= query intent).

---

## 9. Biological Parallel (Complete)

| Biology | ai-knot pattern memory |
|---------|------------------------|
| Nucleotide | Token |
| Gene | Fact |
| Regulatory network | Token co-occurrence graph |
| Reading frame | Intent-dependent expansion weights |
| Protein expression | Query expansion → BM25 → retrieved facts |
| Epigenetics (context changes reading) | Intent classifier changes which associations activate |
| Hebbian learning ("fire together, wire together") | Co-occurrence counts: facts together, tokens associate |
| Synaptic pruning | Ebbinghaus decay on co-occurrence counts |
| Engram (memory trace) | Co-occurrence profile of a token |
| Episodic memory | Individual facts in facts.db |
| Semantic memory | Pattern structure in patterns.bin |

---

## 10. Competitive Uniqueness

### Techniques NO competitor uses:
1. **Persistent binary co-occurrence structure** as a retrieval accelerator — ZERO competitors
2. **Intent-dependent expansion from corpus statistics** — ZERO competitors
3. **Bitvector candidate pre-filtering** before BM25 — ZERO competitors (Tantivy has bitvectors internally, but not for co-occurrence-based candidate generation)
4. **Hebbian-style associative memory** in a production memory system — only Hindsight has spreading activation, but on a curated graph, not on emergent token co-occurrence

### Overlap check:
- memvid: ~0% (completely different paradigm)
- Mem0: ~0% (they use vector similarity, no co-occurrence)
- Zep: ~5% (they have graph edges, but curated, not emergent)
- Hindsight: ~10% (spreading activation is related, but mechanism is different)
- MAGMA: ~5% (multi-graph routing is related conceptually)

---

## 11. Open Questions

1. **Multi-word entities**: "Harry Potter" is two tokens. Co-occurrence captures "harry" + "potter" co-occur, but not as a unit. Options: bigram detection, phrase mining.

2. **Common token pollution**: "said", "also", "really" co-occur with everything. Solution: IDF-weighted co-occurrence (down-weight common tokens).

3. **Sparse updates**: when a single fact is added, updating the full matrix is O(tokens²). For incremental adds, only update affected rows/columns. Need delta-update protocol.

4. **Memory budget**: at 10K+ facts, structure grows to 6MB+. Need sparse representation or top-K truncation per token row.

5. **Multilingual**: tokenization differs by language. Co-occurrence is language-agnostic in principle but depends on tokenizer quality. Need to test with Russian/Chinese corpora.

6. **Benchmark**: need to measure actual impact on LOCOMO before/after pattern memory. Hypothesis: Cat 1 (aggregation) improves 25% → 50%+, Cat 2 (temporal) improves 40% → 55%+.

---

## 12. Implementation Plan (Draft)

### Phase 0: Fix DB isolation (prerequisite bug fix)
Each benchmark run must use its own SQLite. Without this, all measurements are meaningless.

### Phase 1: PatternMemory class
- `PatternMemory` — standalone class, no KnowledgeBase dependency
- `build(facts: list[Fact]) -> PatternMemory` — construct from fact list
- `expand(query_tokens: set[str], intent: Intent, top_k: int) -> dict[str, float]` — return expansion terms with weights
- `candidates(tokens: set[str]) -> set[int]` — return candidate fact indices via bitvector OR
- Binary serialization: `save(path)` / `load(path)`

### Phase 2: Integration into KnowledgeBase._execute_recall()
- Build PatternMemory lazily on first recall (cache for subsequent calls)
- Invalidate on add/learn (rebuild or incremental update)
- Use expansion + candidates before BM25 search
- Intent-dependent expansion parameters

### Phase 3: Benchmark
- LOCOMO with clean DB + pattern memory
- Ablation: pattern memory ON/OFF
- Per-category scores
- Latency measurement (pattern lookup vs. full BM25)

### Phase 4: Incremental updates
- Delta-update protocol for add() / learn() / forget()
- Avoid full rebuild on every fact change

### Phase 5: Optimization
- Binary format specification
- Cache-aligned layout
- SIMD-friendly bitvector operations
- Sparse representation for large corpora (10K+ facts)
