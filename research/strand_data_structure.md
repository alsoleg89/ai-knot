# Strand — DNA-Inspired Binary Co-occurrence Structure

## Date: 2026-04-10
## Status: Research complete, ready for implementation

---

## 1. Naming

**Strand** — as in DNA strand. A compact binary encoding of all token-to-token
associations in an agent's memory. Like a DNA strand encodes an organism's
blueprint, a Strand encodes the agent's associative knowledge.

---

## 2. Optimal Structure: 2-bit Quantized Half-Triangle Bitmap

### Why 2 bits (not 1, 4, or 8)?

| Encoding | Size (N=500) | Fits in | Info |
|----------|-------------|---------|------|
| 1-bit boolean | 15.3 KB | L1 | No strength — pair seen 10× same as 1× |
| **2-bit quantized** | **30.6 KB** | **L1** | **4 levels: none/weak/moderate/strong** |
| 4-bit nibble | 61.2 KB | L2 | 16 levels — overkill, pushes out of L1 |
| 1-byte count | 122 KB | L2 | Full counts but 4× larger |

2 bits is the DNA-optimal encoding: 4 nucleotides, 4 strength levels.
Captures the essential ranking signal while staying in L1 cache.

Quantization: `00`=none (0), `01`=weak (1-2), `10`=moderate (3-7), `11`=strong (8+).
Thresholds tunable.

### Structure layout

```
[MPH Index]     token string → integer id 0..N-1     ~4 KB
[Strand Matrix] half-triangle, 2-bit per pair         ~30.6 KB (N=500)
[Generation]    per-token freshness counter            500 bytes
[Posting BVs]   token → fact bitvector (88 bytes each) ~44 KB (N=500, 700 facts)

Total at N=500:  ~79 KB  → fits in L1+L2
Total at N=1000: ~257 KB → fits in L2
Total at N=2000: ~1.0 MB → fits in L3
```

### Half-triangle indexing

Symmetric matrix — only store upper triangle. For token indices i < j:

```
offset(i, j) = i * (2*N - i - 1) // 2 + (j - i - 1)
```

124,750 entries at N=500. Packed as 2-bit fields in uint8 array:
byte_idx = offset // 4, bit_shift = (offset % 4) * 2.

---

## 3. Operations and Complexity

### Point query: "Does A co-occur with B?"
```
i = MPH(A), j = MPH(B)
if i > j: swap
bits = strand[offset(i,j)]  → 2-bit value
```
**O(1), single cache line hit. ~5 nanoseconds.**

### Row scan: "All tokens co-occurring with A"
```
i = MPH(A)
scan all j ≠ i: extract 2-bit at offset(min(i,j), max(i,j))
collect non-zero entries with their strength
```
**O(N/32) with bit scanning. ~500 ns at N=500.**
Returns top-K associated tokens sorted by strength.

### Intersection: "Tokens co-occurring with both A and B"
```
row_A = extract_row(A)  → N-bit vector
row_B = extract_row(B)  → N-bit vector
result = row_A AND row_B → bitwise AND
```
**O(N/32), SIMD-accelerable. 2 cache line reads + 1 AND.**

### Candidate facts: "Facts containing any of these tokens"
```
bv = zeroes(num_facts)
for token in expanded_tokens:
    bv |= posting_bitvector[token]
candidates = popcount(bv)  → iterate set bits
```
**O(K × fact_bytes) where K = expanded tokens. Microseconds.**

### Update: "Fact added/removed"
```
tokens = tokenize(new_fact)
for (a, b) in all_pairs(tokens):
    current = strand[offset(a,b)]
    strand[offset(a,b)] = min(3, current + quantize(1))
    # (or decrement on removal)
for token in tokens:
    posting_bv[token] |= (1 << fact_idx)
```
**O(t²) per fact where t = tokens in fact. ~225 ops for 15-token fact.**

---

## 4. Temporal Decay (Pheromone Evaporation)

### Generation counter approach

Instead of decaying every pair individually (O(N²) per decay tick),
store a per-token "generation" counter (1 byte each, 500 bytes total).

```
generation[token] = current_epoch when token was last updated
```

At query time: if `current_epoch - generation[token] > threshold`,
treat that token's co-occurrence values as one strength level weaker.

```
raw_strength = strand[offset(i,j)]  → 0-3
staleness = current_epoch - max(generation[i], generation[j])
effective_strength = max(0, raw_strength - staleness // decay_window)
```

**Cost: 0 for updates, O(1) per query pair.** Decay is lazy — only evaluated when read.
Like ant pheromone: evaporation happens continuously, but you only observe it when an ant walks that path.

### Periodic compaction

Every M facts added, re-quantize the entire strand from current fact set.
This "resets" the pheromone map, ensuring accuracy.

At N=500: full rebuild = scan 700 facts × 15 tokens × 15 tokens = 157K pair updates.
**~1 millisecond.** Can afford to do this every 100 facts.

---

## 5. Comparison with Alternatives (Research Results)

### Bloom filters
- At N=500, 10% density: 15.2 KB (comparable to 1-bit bitmap)
- BUT: 1% false positive rate, no strength info, no row scan
- **Verdict: dominated by bitmap at this scale**

### CSR sparse matrix
- At N=500, 10% density: 37.5 KB (larger than 2-bit bitmap)
- Lookup: O(log d) binary search instead of O(1)
- **Verdict: only wins at N > 5000 with very sparse matrices**

### Compressed inverted index (Lucene/Tantivy style)
- Delta-encoded posting lists: 23.3 KB at N=500
- Good density but O(d) scan for pairwise lookup
- **Verdict: useful for posting lists but not co-occurrence matrix**

### Trie structures
- Integer token IDs have no shared prefixes → no compression benefit
- Pointer overhead exceeds bitmap size at N < 2000
- **Verdict: not suitable for pairwise co-occurrence**

### Winner: 2-bit half-triangle bitmap (Strand)
- Best density-speed tradeoff for N = 500-2000
- O(1) point lookup, O(N/32) row scan, SIMD-accelerable intersection
- 30.6 KB at N=500 — entire associative memory in L1 cache

---

## 6. Scaling Analysis

| Facts | Tokens (est.) | Strand matrix | Posting BVs | Total | Fits in |
|-------|---------------|---------------|-------------|-------|---------|
| 500 | 400 | 20 KB | 25 KB | ~49 KB | L1 |
| 700 | 500 | 31 KB | 44 KB | ~79 KB | L1/L2 |
| 2,000 | 1,000 | 122 KB | 250 KB | ~380 KB | L2 |
| 5,000 | 2,000 | 488 KB | 1.25 MB | ~1.7 MB | L3 |
| 10,000 | 3,000 | 1.1 MB | 3.75 MB | ~4.9 MB | L3 |
| 50,000 | 5,000 | 3.1 MB | 31 MB | ~34 MB | RAM |

At 50K facts: need sparse representation. Options:
- Top-K per token (keep only K strongest co-occurrences per row) → CSR
- Hierarchical: frequent tokens in L1 bitmap, rare tokens in CSR
- Tiered: hot tokens (top 500) in Strand, cold tokens in posting-only mode

For LOCOMO benchmark (700 facts, ~500 tokens): **79 KB total. L1/L2 cache.**

---

## 7. Integration with BM25 Pipeline

### Current pipeline:
```
recall(query)
  → load all facts
  → decay
  → intent classify
  → BM25 search (ALL facts, full scan)
  → entity-boost, entity-hop, dedup
  → return top_k
```

### With Strand:
```
recall(query)
  → load all facts
  → decay
  → intent classify
  → Strand expansion (μs):
  │   → tokenize query
  │   → look up co-occurrence rows
  │   → expand query with associated tokens (intent-dependent depth)
  │   → bitvector candidate selection
  → BM25 search (CANDIDATES ONLY, not full scan)
  → dedup
  → return top_k
```

Strand replaces entity-boost and entity-hop with a more general mechanism.
Entity-hop is a special case of co-occurrence hop: "alex::wife" → "maria"
is captured by co-occurrence of "alex" + "wife" + "maria" in the same facts.

---

## 8. File Format (Draft)

```
STRAND v1 binary format:

Header (32 bytes):
  magic:      4 bytes  "STRD"
  version:    2 bytes  0x0001
  num_tokens: 2 bytes  N
  num_facts:  4 bytes  F
  matrix_off: 4 bytes  offset to strand matrix
  posting_off:4 bytes  offset to posting bitvectors
  gen_off:    4 bytes  offset to generation vector
  reserved:   8 bytes

Token table (variable):
  For each token: [hash: 4 bytes] [string_len: 2 bytes] [string: variable]
  Sorted by hash for binary search (or MPH if we implement it)

Strand matrix (N*(N+1)/2 * 2 bits, padded to 8-byte boundary):
  Half-triangle, row-major, 2-bit packed

Posting bitvectors (N * ceil(F/8) bytes):
  One bitvector per token, packed

Generation vector (N bytes):
  One byte per token, epoch counter

Footer:
  checksum: 4 bytes (CRC32 of everything above)
```

For N=500, F=700: header(32) + tokens(~8KB) + matrix(31KB) + posting(44KB) + gen(500) + footer(4) ≈ **84 KB**

---

## 9. Why "Strand" as a Name

1. **DNA strand**: the structure that encodes biological information
2. **Thread/strand**: a continuous sequence (the packed binary array)
3. **Strand of associations**: each token's co-occurrence row is a "strand" of connected concepts
4. Compact, evocative, unique. No competitor uses this term.

Possible class names:
- `Strand` — the data structure
- `StrandMemory` — the integration with KnowledgeBase
- `strand.bin` — the binary file
- "Strand-based retrieval" — the technique

---

## 10. Next Steps

1. Implement `Strand` class in Python (pure Python first, optimize later)
2. Integrate into `KnowledgeBase._execute_recall()` as query expansion
3. Benchmark on LOCOMO with clean DB
4. If promising: optimize with numpy/bitarray for SIMD-like operations
5. Design incremental update protocol
6. Binary serialization
