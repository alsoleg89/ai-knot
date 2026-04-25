# Biological Mechanisms — What Organisms ACTUALLY Do With Memory

## Date: 2026-04-11
## Goal: Find a real mechanism, not a metaphor

---

## 1. What the Critic Killed and Why It Matters

The critic showed that Strand (co-occurrence matrix) is:
- A worse version of embeddings (which already capture co-occurrence)
- Directionless in dialogue (can't tell "does X" from "talks about X")
- Redundant with BM25 inverted index

Biology metaphors were post-hoc. DNA → 2-bit encoding is not prediction, it's coincidence.

**So: can we find a biological mechanism that is NOT just a metaphor, that actually
PREDICTS a useful computational property that current IR techniques miss?**

---

## 2. Mechanisms Considered

### 2.1 Olfaction (smell recognition)
Pattern of receptor activation → smell identification.
This is: input vector → pattern matching → classification.
**= Embeddings. Already done. Rejected.**

### 2.2 Immune system (clonal selection)
Antibody matches antigen → clone → specialize → remember.
This is: find match → create variants → keep best → store.
**= Relevance feedback + fine-tuning. Close to PRF. Rejected.**

### 2.3 Place cells / grid cells (spatial navigation)
Hexagonal coordinate grid + event-location binding.
This is: embedding space + positional encoding.
**= Dense retrieval with coordinates. Already done. Rejected.**

### 2.4 Sparse Distributed Memory (Kanerva, 1988)
Binary address space, distributed storage, Hamming distance retrieval.
Mathematically elegant, actually models human long-term memory.
**= Locality-sensitive hashing. Implementable but unclear advantage over BM25+embeddings. Deferred.**

### 2.5 Reconsolidation (each recall modifies the memory)
Every time you recall a fact, it becomes labile again and is re-stored
in a potentially modified form. This is why eyewitness testimony degrades.
**Interesting for rescripting but doesn't help retrieval. Deferred to future.**

### 2.6 Emotional tagging (amygdala)
Emotional significance modulates memory strength.
**= ai-knot's importance field. Already done.**

---

## 3. The One That Survives: Complementary Learning Systems

**McClelland, McNaughton & O'Reilly (1995). "Why There Are Complementary
Learning Systems in the Hippocampus and Neocortex." Psychological Review.**

Validated by: Kumaran, Hassabis & McClelland (2016). "What Learning Systems
do Intelligent Agents Need?" Trends in Cognitive Sciences.

This is not a metaphor. It's one of the most validated theories in
computational neuroscience, with direct experimental evidence.

### 3.1 The Theory

The brain has TWO memory systems with DIFFERENT properties:

**HIPPOCAMPUS (fast learner):**
- Stores specific episodes quickly (one-shot learning)
- High fidelity — remembers exact details
- Pattern completion — given partial cue, reconstructs full episode
- BUT: catastrophic interference — new episodes overwrite old ones
- Capacity limited, decays fast

**NEOCORTEX (slow learner):**
- Extracts general patterns from many episodes over time
- Low fidelity — stores gist, not details
- Generalization — can handle novel situations
- BUT: learns slowly — needs many exposures (consolidation)
- High capacity, stable

### 3.2 How They Work Together

```
AWAKE (encoding):
  Experience → Hippocampus (fast store)
  New episode stored immediately with full detail

SLEEP (consolidation):
  Hippocampus REPLAYS episodes to Neocortex
  Neocortex extracts PATTERNS across episodes
  Pattern: "Melanie does outdoor activities" (from camping, hiking, beach episodes)
  Individual episodes decay in hippocampus
  Patterns persist in neocortex

RECALL:
  Query → activates BOTH systems
  Neocortex: provides pattern/schema ("Melanie does many activities")
  Hippocampus: provides specific details ("went camping in June, swam in May")
  Combined: full answer with both breadth and depth
```

### 3.3 Why This Is NOT Just a Metaphor

CLS theory makes a SPECIFIC PREDICTION about memory retrieval:

> A system with ONLY episodic storage (individual facts) will fail at
> AGGREGATION queries because no single episode contains the answer.
> A system with ONLY semantic storage (patterns) will fail at SPECIFIC
> queries because patterns lose detail.
> The optimal system has BOTH and searches BOTH.

This is EXACTLY ai-knot's problem:
- ai-knot has only episodic storage (individual facts)
- Aggregation queries fail (Cat 1: 25%)
- Specific queries work better (Cat 4: 64%)
- CLS predicts: add a pattern store → aggregation improves

This is a **testable prediction from neuroscience theory**, not a post-hoc analogy.

---

## 4. Application to ai-knot: Dual-Store Architecture

### 4.1 Fast Store = Hippocampus (existing)

Individual facts stored by `kb.add()` and `kb.learn()`.
- Exact content, timestamps, entity fields
- Ebbinghaus decay (existing)
- BM25 search for specific queries

No changes needed. This is already built.

### 4.2 Slow Store = Neocortex (NEW)

**Consolidated pattern facts**, built periodically from groups of related facts.

A pattern fact is NOT a summary (no LLM, no text generation).
A pattern fact is a BAG-OF-TOKENS extracted from a group of related facts.

Example: from individual facts about Melanie's activities:
```
Fact 1: "Melanie creates pottery as a creative outlet"
Fact 2: "Melanie and family went camping in the forest"
Fact 3: "Melanie: I'm off to go swimming with the kids"
Fact 4: "Melanie runs for destressing and clear mind"
Fact 5: "Melanie took her kids to a museum with dinosaur exhibit"
Fact 6: "Melanie and her family went on a camping trip"
```

Consolidated pattern fact:
```
content: "melanie activities pottery camping swimming running museum kids family forest dinosaur destressing creative"
type: MemoryType.CONSOLIDATED
importance: 0.8 (average of source facts)
source_facts: [id1, id2, id3, id4, id5, id6]  # provenance
```

This is NOT summarization. It's TOKEN UNION — the bag-of-words from all facts
in the group, deduplicated, stopwords removed.

### 4.3 Consolidation Process (="sleep")

```python
def consolidate(facts: list[Fact]) -> list[Fact]:
    """
    Group related facts, create pattern facts from each group.
    No LLM. Pure token-level operations.
    """
    # 1. Build token vectors for each fact (TF-IDF or just token sets)
    # 2. Cluster by cosine similarity (or simpler: by shared top tokens)
    # 3. For each cluster > 3 facts: create pattern fact from token union
    # 4. Return pattern facts (to be stored alongside originals)
```

Clustering options (simplest first):
- **By entity**: group all facts mentioning "melanie" → Melanie pattern
  (But this is entity-scoping, which was criticized as format-specific)
- **By top tokens**: group facts sharing 3+ significant tokens
  (Format-agnostic, language-agnostic)
- **By TF-IDF cosine**: proper clustering, most accurate
  (More compute but still no LLM)

**Recommended: by top tokens** — simplest, no matrix math, O(n²) but
n < 1000 so it's milliseconds.

### 4.4 Recall from Dual Store

```python
def recall(query, top_k):
    all_facts = storage.load(agent_id)
    patterns = [f for f in all_facts if f.type == CONSOLIDATED]
    episodes = [f for f in all_facts if f.type != CONSOLIDATED]
    
    if intent == AGGREGATION:
        # Search patterns first (broad coverage)
        pattern_hits = bm25.search(query, patterns, top_k=5)
        # Then search episodes for details
        episode_hits = bm25.search(query, episodes, top_k=top_k-5)
        # Merge: patterns provide coverage, episodes provide detail
        return merge(pattern_hits, episode_hits)
    else:
        # Non-aggregation: search episodes directly (existing behavior)
        return bm25.search(query, episodes, top_k=top_k)
```

### 4.5 Properties

**Why this works for aggregation:**
- Pattern fact "melanie pottery camping swimming running museum..." contains
  ALL activity tokens in ONE document
- BM25 matches this pattern fact with HIGH score for "What activities does Melanie?"
- The pattern fact IS the aggregation, pre-computed
- Individual facts then provide specific details

**Why it doesn't hurt precision:**
- For point queries (ENTITY_LOOKUP, etc.), search skips patterns
- Patterns are additional facts, not replacements
- Existing behavior preserved for non-aggregation

**Why it's not "session storage with extra steps":**
- Sessions group by TIME. Patterns group by TOPIC.
- A session contains 18 turns about various topics
- A pattern contains tokens from 6 facts across 6 different sessions about one topic
- Sessions = temporal grouping. Patterns = semantic grouping.

**Why it's not "extraction is lossy":**
- Pattern facts are built from EXISTING stored facts (both raw and extracted)
- They don't replace originals — they ADD a compressed representation
- If extraction missed "swimming" but raw turn has it, pattern still captures it
- Zero information loss — pattern is additive

**Why it's not "just embeddings":**
- Embeddings compress into a dense vector (loses token-level info)
- Pattern fact keeps ACTUAL TOKENS (BM25 can match them)
- BM25 + pattern fact = lexical matching on aggregated tokens
- Embeddings + individual facts = semantic matching on individual items
- These are COMPLEMENTARY, not competing (just like hippocampus + neocortex)

---

## 5. Consolidation Details

### 5.1 When to consolidate

Option A: After every N facts added (e.g., every 50 facts)
Option B: On first recall after facts have changed
Option C: Explicit `kb.consolidate()` call

**Recommended: Option B** — lazy consolidation on first recall after changes.
Like sleep: you don't consolidate while awake (ingesting), you consolidate
when you need to recall.

### 5.2 Clustering algorithm (simple version)

```python
def _cluster_for_consolidation(facts: list[Fact]) -> list[list[Fact]]:
    """Group facts by shared significant tokens."""
    from collections import defaultdict
    
    # Each fact → set of significant tokens (stopwords removed)
    fact_tokens = {f.id: set(tokenize(f.content)) for f in facts}
    
    # Build inverted index: token → [fact_ids]
    token_to_facts = defaultdict(set)
    for fid, tokens in fact_tokens.items():
        for t in tokens:
            token_to_facts[t].add(fid)
    
    # Find clusters: facts sharing 3+ significant tokens
    # Simple greedy: start from most-connected fact, grow cluster
    used = set()
    clusters = []
    
    for f in sorted(facts, key=lambda f: len(fact_tokens[f.id]), reverse=True):
        if f.id in used:
            continue
        cluster = {f.id}
        cluster_tokens = set(fact_tokens[f.id])
        
        for other in facts:
            if other.id in used or other.id in cluster:
                continue
            shared = cluster_tokens & fact_tokens[other.id]
            if len(shared) >= 3:
                cluster.add(other.id)
                cluster_tokens |= fact_tokens[other.id]
        
        if len(cluster) >= 3:
            clusters.append([f for f in facts if f.id in cluster])
            used |= cluster
    
    return clusters
```

### 5.3 Pattern fact creation

```python
def _create_pattern_fact(cluster: list[Fact]) -> Fact:
    """Create a consolidated pattern fact from a cluster of related facts."""
    # Union of all significant tokens
    all_tokens = set()
    for f in cluster:
        all_tokens |= set(tokenize(f.content))
    
    # Remove very common tokens (IDF filter)
    # Keep only tokens that appear in < 50% of ALL facts (not just cluster)
    
    content = " ".join(sorted(all_tokens))
    importance = sum(f.importance for f in cluster) / len(cluster)
    
    return Fact(
        content=content,
        type=MemoryType.CONSOLIDATED,  # or new type
        importance=importance,
        tags=["consolidated"],
        source_snippets=[f.id for f in cluster[:5]],  # provenance
    )
```

### 5.4 Expected pattern count

For 700 facts with ~500 unique tokens:
- ~20-50 clusters of 3+ facts
- ~20-50 pattern facts
- Each pattern fact: ~30-80 tokens
- Total pattern storage: ~5-15 KB of text
- Negligible overhead

---

## 6. CLS Prediction vs ai-knot Reality

| CLS Prediction | ai-knot Status |
|---|---|
| Episodic-only fails at aggregation | Confirmed: Cat 1 = 25% |
| Semantic-only fails at specifics | N/A (don't have semantic store yet) |
| Both together > either alone | **Testable hypothesis** |
| Fast store decays, slow store persists | Ebbinghaus decay exists, need consolidation |
| Consolidation happens during "downtime" | Lazy consolidation on first recall |

The theory PREDICTS that adding consolidated patterns will improve Cat 1
(aggregation) without hurting Cat 2-4. This is a testable, falsifiable prediction.

---

## 7. Relationship to Other Ideas in Research

### vs. Strand (pattern_memory_architecture.md)
Strand stores token-level co-occurrence in a binary matrix.
CLS stores cluster-level patterns as text facts.
Strand is a data structure. CLS is an architecture.
CLS is simpler (no binary format) and more general (patterns > pairwise co-occurrence).
**CLS subsumes the useful part of Strand and discards the problematic parts.**

### vs. Entity-Scoped Retrieval (implementation_plan_v2.md)
Entity-scoping narrows search to entity-relevant facts.
CLS creates pattern facts that aggregate entity-relevant information.
These are complementary: entity-scoping + pattern facts = narrower search on richer data.
**But pattern facts don't REQUIRE entity-scoping to work.**

### vs. Overfetch + MMR (architecture_synthesis.md)
Overfetch + MMR is a retrieval-time diversity technique.
CLS is a storage-time consolidation technique.
These are complementary: pattern facts + MMR = even better coverage.
**But CLS pattern facts may REPLACE the need for MMR entirely** (because the
pattern fact already IS the diverse aggregation).

### vs. memvid session storage
memvid stores full sessions (~2500 chars each, temporal grouping).
CLS stores pattern facts (~100-200 tokens each, semantic grouping).
memvid has 54 sessions. CLS would have ~30-50 patterns.
Retrieval: memvid searches sessions. CLS searches patterns + individual facts.
**CLS patterns are semantically organized; memvid sessions are temporally organized.
For aggregation (which needs SEMANTIC grouping), CLS should outperform.**

---

## 8. What This Means for the Cognitive Model

```
Level 1: ENCODING (learn/add)     → Fast store (hippocampus)   EXISTING
Level 2: FORGETTING (Ebbinghaus)  → Both stores                EXISTING  
Level 3: CONSOLIDATION (sleep)    → Fast → Slow store          NEW
Level 4: RECALL (search)          → Search both stores         MODIFY
Level 5: RECONSOLIDATION (future) → Recall modifies patterns   FUTURE
```

Level 3 (consolidation) is the MECHANISTICALLY GROUNDED addition from CLS theory.
It's not a metaphor — it's a direct implementation of a validated neuroscience model.

---

## 9. Critique Survival Check

| Critique | Does CLS survive? |
|---|---|
| #1 Reinventing embeddings | YES — pattern facts work with BM25, not vectors |
| #2 Co-occurrence ≠ relationship | PARTIALLY — clustering by shared tokens is undirected, but pattern facts group by topic, not by speaker pair |
| #3 BM25 already knows this | YES — pattern facts create NEW documents that BM25 couldn't find before (aggregated tokens in one place) |
| #4 L1 cache claims | N/A — no binary format, just text facts |
| #5 O(N²) scaling | YES — ~50 pattern facts regardless of corpus size |
| #6 Bio analogy post-hoc | NO — CLS is a validated neuroscience theory with testable predictions |
| #7 Rescripting vaporware | N/A — removed from v1 |
| #8 Nothing measured | STILL TRUE — must measure |

**CLS survives 6/7 critiques.** Only #8 remains: must measure.

---

## 10. Implementation Priority

```
Step 1: DONE   — Fix DB isolation
Step 2: NEXT   — Run LOCOMO on clean DB → baseline measurement  
Step 3: IF <55% — Add consolidation (pattern facts) to _execute_recall()
Step 4: MEASURE — Run LOCOMO with pattern facts → compare
Step 5: IF pattern facts help — consider adding MMR on top
Step 6: IF not   — try overfetch + MMR without patterns
```

Consolidation implementation: ~100 lines of Python.
No new dependencies. No binary format. No embeddings. No LLM.
Pure token-level operations on existing facts.

---

## 11. One-Sentence Summary

**Complementary Learning Systems theory (McClelland 1995) predicts that adding
consolidated pattern facts (token-union clusters, no LLM) alongside individual
facts will improve aggregation retrieval — this is a testable, falsifiable
prediction from validated neuroscience, not a metaphor.**
