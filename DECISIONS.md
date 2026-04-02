# Architecture Decisions

Non-obvious choices made during development, and why.

---

## 1. BM25 retrieval instead of embeddings

The first prototype used cosine similarity over sentence-transformers embeddings.
It worked fine for natural-language facts but failed on short procedural entries:
"uses pytest", "prefers snake_case", "deploy with Docker Compose" — these are
essentially keyword strings, and semantic search kept missing exact tool names.

v0.1–0.4 used raw TF-IDF. v0.5 upgraded to **Okapi BM25** (Robertson & Zaragoza,
2009) which handles term saturation and document length normalization better than
raw TF-IDF. BM25 scores are p95-clipped and normalized to [0, 1] before blending:

```
score = 0.6 × bm25 + 0.2 × retention + 0.2 × importance
```

The weights came from manual testing across ~200 facts. Flipping bm25 and
retention weights made recent-but-irrelevant facts rank too high.

Evaluated on a 30-case golden dataset: MRR=0.87, P@5=0.87, nDCG=0.88.

We'll add optional embedding support in a future version, but it will be additive —
BM25 stays as the default because it requires zero external dependencies
and works predictably at the fact counts ai-knot is designed for (~10k max).

---

## 2. Power-law decay instead of exponential / LRU / TTL

LRU evicts the least recently used fact regardless of importance.
TTL expires facts after a fixed time regardless of how often they're accessed.
Neither model captures reinforcement — the idea that using a fact makes it
stick longer.

v0.1–0.4 used exponential decay (`exp(−t/S)`). v0.5 switched to a
**power-law curve** (Wixted & Ebbesen, 1997):

```
retention(t) = (1 + t / (9 × stability))^(-1)
stability = 336h × importance × (1 + ln(1 + access_count))
```

Why power-law over exponential? Wixted & Ebbesen (1997) showed that human
forgetting follows a power function, not an exponential one (R²=98.9% vs
96.3% across multiple datasets). The key difference: power-law has a
**heavier tail** — important facts persist over months instead of
vanishing after days. This matches how agents should treat knowledge:
core user preferences should fade slowly, even without frequent access.

A fact accessed 5 times has roughly 2.8× the stability of a fact accessed once.
A high-importance fact (`importance=1.0`) decays half as fast as a default one.

The 336h base (two weeks) came from empirical testing: most agent conversations
have relevant context that stays useful for 1–3 weeks. Below that, the decay
was too aggressive and pruned facts users expected to persist.

One known trade-off: occasionally useful facts do get pruned. That's intentional.
Context rot from accumulating stale facts is worse than occasional amnesia.

---

## 4. ATC verification instead of blind trust

LLM-extracted facts are not always grounded in the source text. Models
occasionally hallucinate plausible-sounding facts that were never mentioned.
Blindly storing these pollutes the knowledge base.

v0.5 adds **Asymmetric Token Containment** (ATC), inspired by Broder (1997):

```
ATC(snippet, source) = |tokens(snippet) ∩ tokens(source)| / |tokens(snippet)|
```

Unlike symmetric Jaccard, ATC is asymmetric: a short fact fully contained in
a long source text scores 1.0. Facts with ATC < 0.6 are flagged
`supported=False` — they're preserved for manual review but won't contaminate
downstream retrieval.

Why 0.6? Below that threshold, empirically, extracted facts start including
information synthesized by the model rather than grounded in the conversation.
The threshold is configurable per deployment.

---

## 5. Inverted index for BM25 instead of brute-force scan

BM25 scoring requires computing term frequencies for every document on every
query. With a naive approach this is O(N × Q) where N = number of facts and
Q = number of query terms. For small knowledge bases (<100 facts) this is fine,
but it scales poorly.

v0.5 adds an **inverted index** (`InvertedIndex` class) built from posting lists.
Each unique term maps to a list of (doc_index, term_frequency) pairs. At query
time, only documents containing at least one query term are scored — complexity
drops to O(Q × avg_postings).

The index is built once per `search()` call. For a typical knowledge base of
500 facts with 10 tokens each, the index adds ~40KB of memory and reduces
search time by ~5× compared to brute-force.

Trade-off: the index is rebuilt on every search, not persisted. This is
intentional — the fact list can change between calls (learn → recall cycle),
so a stale index would be a correctness risk. At the target scale (~10k facts)
the rebuild cost is negligible (<1ms).

---

## 3. Full-replace `save()` instead of upsert

`StorageBackend.save(agent_id, facts)` replaces the entire fact list atomically.
The obvious alternative was an upsert: only write changed facts.

We chose full-replace for two reasons:

First, it makes the storage protocol trivially correct. With upsert you need
to handle insert vs update vs delete — three separate operations that have to
be kept in sync. Full-replace is one operation: write the current state.
New backends can't get this wrong.

Second, YAML files are human-readable and Git-trackable. A full-replace
produces a clean diff. Partial upserts produce noisy diffs with interleaved
changes that are hard to review.

The downside is write amplification: saving 1 changed fact writes all N facts.
At the scale ai-knot targets (hundreds, not millions of facts per agent) this
is not a bottleneck. If it becomes one, the storage backend can implement
internal diffing transparently — the protocol doesn't need to change.

---

## 6. Type-aware decay exponents instead of uniform forgetting

v0.5 used a single decay exponent (`-1`) for all memory types. This means
semantic facts ("user prefers Python") and episodic facts ("discussed deployment
on Monday") forgot at the same rate — which doesn't match how memory works.

Tulving (1972) distinguished episodic and semantic memory as fundamentally
different systems. FSRS (Ye 2022-2024) showed adaptive decay scheduling
improves retention prediction. v0.6 applies per-type exponents:

```
decay_exp = { semantic: 0.8, procedural: 1.0, episodic: 1.3 }
retention(t) = (1 + t / (9 × stability))^(-decay_exp)
```

Semantic facts (core preferences, tool choices) decay 20% slower.
Episodic facts (events, meetings) decay 30% faster. This means after 30 days,
a semantic fact retains ~63% vs ~47% for episodic — matching the intuition
that "user prefers pytest" should outlast "discussed CI on Tuesday".

---

## 7. Per-agent trust instead of flat provenance discount

v0.5 applied a flat 0.8× discount to all facts from other agents. This treats
every agent as equally (un)trustworthy — a monitoring agent's alerts get the
same discount as an untested third-party agent's suggestions.

v0.6 adds a **per-agent trust matrix** (Marsh 1994) to `SharedMemoryPool`:

```python
pool.update_trust("monitoring-agent", +0.1)   # reliable source
pool.update_trust("experimental-bot", -0.15)  # often wrong
```

Trust scores are clamped to [0.1, 1.0] and applied during `recall()` as
score multipliers. New agents start at the default 0.8. This lets the system
learn which agents produce reliable knowledge over time.

---

## 8. Stemmed Jaccard instead of raw token overlap

The `extractor._jaccard_similarity()` function originally used `str.split()`
for tokenization, while the retriever used the shared `tokenize()` (with
stemming). This created inconsistencies: "caching" and "cached" were different
tokens in the extractor but the same stem (`cach`) in the retriever.

v0.6 aligns both to use `tokenize()` (Broder 1997). This ensures deduplication
and retrieval agree on what counts as a "matching" term.

---

## 9. LLM auto-tagging instead of manual tag entry

BM25F gives tags 2× weight (`_W_TAGS=2.0`), but `Fact.tags` was always empty
after `learn()`. Users had to manually supply tags via `add(tags=[...])` — most
didn't bother. The entire BM25F tags field was dead weight.

v0.6 adds `"tags"` to the extraction system prompt. The LLM generates 1-3
domain tags per fact during the same call it already makes for content, type,
and importance. **Zero extra LLM calls.** When the LLM omits tags (older models,
edge cases), `_parse_fact()` falls back to an empty list.

This follows the **base + enhanced** pattern: `add()` accepts user-supplied tags,
`learn()` generates them automatically.

---

## 10. Opt-in LLM query expansion instead of multilingual stemmer

The English stemmer in `tokenizer.py` doesn't help for Russian, Chinese, or
other languages. Instead of building a multilingual stemmer (which would add
dependencies and complexity), v0.6 offers LLM-based query expansion.

`KnowledgeBase(llm_recall=True)` expands queries with LLM-generated synonyms
before BM25 search. This covers vocabulary gaps across any language without
touching the tokenizer. Results are cached (LRU, 128 entries).

The feature is opt-in because it adds an LLM round-trip to every `recall()`.
Default behavior (`llm_recall=False`) is unchanged — zero latency added.

## 11. Cyrillic stemmer + weighted LLM expansion (revising #10)

Benchmarking against mem0 and qdrant showed that LLM expansion alone does not
work for non-Latin scripts. The root cause: expansion only changes the query
side, but documents are indexed with unstemmed tokens. BM25 requires symmetric
normalization at both index and query time (Robertson & Zaragoza, 2009).

v0.8 adds a zero-dependency Cyrillic (Russian) stemmer in `tokenizer.py` using
a Snowball-lite algorithm. The tokenizer auto-detects script via Unicode block
check and dispatches to the appropriate stemmer. English rules are unchanged.

Additionally, LLM expansion now uses weighted tokens via `expansion_weights`
(new terms get weight 0.4, original query terms keep 1.0). This prevents
expansion from diluting the original query signal — a known issue with blind
query expansion (Xu & Croft, 1996). PRF and LLM expansions are merged.

Other changes in this revision:
- `now` parameter on all recall/decay methods (clock injection for testing)
- Configurable RRF weights via `rrf_weights` parameter
- Multilingual expansion prompt with Russian example
