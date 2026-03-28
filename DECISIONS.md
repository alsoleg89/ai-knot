# Architecture Decisions

Non-obvious choices made during development, and why.

---

## 1. TF-IDF retrieval instead of embeddings

The first prototype used cosine similarity over sentence-transformers embeddings.
It worked fine for natural-language facts but failed on short procedural entries:
"uses pytest", "prefers snake_case", "deploy with Docker Compose" — these are
essentially keyword strings, and semantic search kept missing exact tool names.

TF-IDF handles this better because keyword overlap is exactly what matters for
procedure recall. The retriever computes a hybrid score:

```
score = 0.6 × tfidf + 0.2 × retention + 0.2 × importance
```

The weights came from manual testing across ~200 facts. Flipping tfidf and
retention weights made recent-but-irrelevant facts rank too high.

We'll add optional embedding support in v0.3, but it will be additive —
TF-IDF stays as the default because it requires zero external dependencies
and works predictably at the fact counts agentmemo is designed for (~10k max).

---

## 2. Ebbinghaus decay instead of LRU or TTL

LRU evicts the least recently used fact regardless of importance.
TTL expires facts after a fixed time regardless of how often they're accessed.
Neither model captures reinforcement — the idea that using a fact makes it
stick longer.

Ebbinghaus does:

```
retention(t) = exp(−t / stability)
stability = 336h × importance × (1 + ln(1 + access_count))
```

A fact accessed 5 times has roughly 2.8× the stability of a fact accessed once.
A high-importance fact (`importance=1.0`) decays half as fast as a default one.

The 336h base (two weeks) came from empirical testing: most agent conversations
have relevant context that stays useful for 1–3 weeks. Below that, the decay
was too aggressive and pruned facts users expected to persist.

One known trade-off: occasionally useful facts do get pruned. That's intentional.
Context rot from accumulating stale facts is worse than occasional amnesia.

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
At the scale agentmemo targets (hundreds, not millions of facts per agent) this
is not a bottleneck. If it becomes one, the storage backend can implement
internal diffing transparently — the protocol doesn't need to change.
