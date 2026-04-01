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
