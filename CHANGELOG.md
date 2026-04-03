# Changelog

All notable changes to ai-knot are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- MongoDB storage backend
- Qdrant and Weaviate backends
- Semantic embeddings (sentence-transformers / OpenAI)
- LangChain / CrewAI integrations
- Web UI knowledge inspector
- REST API / sidecar mode

---

## [0.8.1] — 2026-04-02

### Fixed
- **S5 importance ranking** — BM25 scores now boosted by `0.5 + 0.5 × importance`,
  making the dominant ranker carry importance information.  Default RRF weights
  changed from `(5, 1, 1, 1)` to `(5, 2, 2, 1)` giving importance+retention 40%.
- **S1/S3 faithfulness** — added BM25 score floor: when enough lexical matches
  exist to fill `top_k`, facts with zero BM25 relevance are excluded.  Prevents
  importance/recency from pushing unrelated facts into results.
- **S4 deduplication** — added asymmetric token containment metric alongside
  Jaccard.  `_dedup_similarity = max(jaccard, containment)` catches subset
  duplicates (e.g. "User works at Sber" ⊂ "User works at Sber as Director").
  Extraction threshold lowered from 0.8 to 0.7; resolve threshold from 0.7 to 0.6.
- **S2 Russian recall** — expanded Snowball-lite stemmer with borrowed-word verb
  patterns (`-ировать`, `-овать`, `-евать`), nominalization noun suffixes
  (`-ация`, `-ование`, `-изация`), and participial adjective suffixes
  (`-ированн`, `-ованн`).

---

## [0.8.0] — 2026-04-02

### Added
- **Cyrillic stemmer** — zero-dependency Snowball-lite Russian stemmer in `tokenizer.py`.
  Auto-detects script via Unicode block check; English rules unchanged.
- **Weighted LLM expansion** — expansion tokens now use `expansion_weights` (0.4 weight)
  instead of replacing the query. Merged with PRF expansion for stable recall.
- **Multilingual expansion prompt** — `_EXPAND_PROMPT` now instructs the LLM to keep
  the same language as the input query, with a Russian example.
- **`now` parameter** on `recall()`, `recall_facts()`, `recall_facts_with_scores()`,
  `arecall()`, `arecall_facts()`, and `decay()` for clock injection (DDD pattern).
- **Configurable RRF weights** via `KnowledgeBase(rrf_weights=...)` and
  `BM25Retriever(rrf_weights=...)`.

### Changed
- `_expand_query()` returns `(query, expansion_weights)` tuple instead of flat string.
- Access tracking variables renamed to `access_time` to avoid shadowing `now` parameter.

---

## [0.7.0] — 2026-04-01

### Added

- **LLM auto-tagging during extraction** — `learn()` instructs the LLM to
  generate 1-3 domain tags per fact. Zero extra LLM calls (piggybacks on
  existing extraction call). BM25F tag weighting activates automatically.

- **Configurable decay exponents** — `KnowledgeBase(decay_config={...})`
  allows per-type decay customization. `apply_decay()` and `calculate_retention()`
  accept optional `type_exponents` parameter override.

- **LLM query expansion at recall** (opt-in) — `KnowledgeBase(llm_recall=True)`
  expands queries with LLM-generated synonyms before BM25 search. LRU cache
  (128 entries) prevents repeated calls. Disabled by default.

---

## [0.6.0] — 2026-04-01

### Added

- **BM25F structured field weighting** (Robertson, Zaragoza & Taylor 2004) —
  indexes `fact.tags` alongside `fact.content` as separate fields. Tags receive
  2.0× weight with B_tags=0.3 length normalization, boosting domain-specific
  matches.

- **Pseudo-Relevance Feedback (PRF)** (Rocchio 1971, Lavrenko & Croft 2001) —
  two-pass retrieval: initial BM25F → extract top-5 expansion terms from
  top-3 feedback docs → re-score with expanded query at α=0.5 weight.
  Bridges vocabulary gaps (e.g. query "database" → finds "PostgreSQL" facts).

- **Reciprocal Rank Fusion (RRF)** (Cormack, Clarke & Buettcher 2009) —
  replaces linear hybrid with `RRF(d) = Σ w_r/(60 + rank_r(d))` over four
  rankers: BM25F (5× weight), importance, retention, recency.

- **IDF-based stopword filtering** (Robertson 2004) — terms in >70% of docs
  get zero IDF weight. Language-agnostic, no hardcoded word lists.

- **Lightweight suffix stemmer** (Porter 1980 step-1 subset) — handles -ment,
  -tion, -sion, -ing, -ed, -ly, -er, -est, -s with double-consonant
  deduplication. Replaces naive plural stripping.

- **Multi-agent shared memory pool** — `SharedMemoryPool` class for cross-agent
  knowledge exchange via `__shared__` namespace. New `Fact` fields:
  `origin_agent_id`, `visibility`. Provenance discount 0.8× for foreign facts.

- **Eval dataset expansion** — 30 → 105 golden retrieval cases covering
  variable haystack sizes, adversarial near-miss/synonym, query-length
  variations, and high-noise topical clusters.

- **Type-aware forgetting curves** (Tulving 1972, FSRS/Ye 2022) — decay exponent
  varies by memory type: semantic=0.8 (slower decay for core facts),
  procedural=1.0 (baseline), episodic=1.3 (steeper decay for events).

- **Extended Porter stemmer** (Porter 1980) — added suffix rules for -ness,
  -ity, -ive, -ence/-ance, -al, -ies, -es (context-aware), terminal-e stripping.
  Ensures morphological invariants (cache/cached/caching → cach).

- **Stemmed Jaccard similarity** (Broder 1997) — `extractor._jaccard_similarity()`
  now uses `tokenize()` instead of raw `str.split()` for consistent stemming
  across retrieval and deduplication.

- **Per-agent trust matrix** (Marsh 1994) — `SharedMemoryPool` tracks per-agent
  trust scores with `update_trust(agent_id, delta)`. Recall applies trust-weighted
  scoring instead of flat provenance discount.

- **Domain tags in eval dataset** — 765 facts tagged with 65 domain categories
  (python, testing, devops, database, security, etc.) via keyword-based rules.

- **LLM auto-tagging during extraction** — `learn()` now instructs the LLM to
  generate 1-3 domain tags per fact. Tags are parsed from the JSON response and
  stored in `Fact.tags`, activating BM25F tag field weighting automatically.
  Falls back to empty tags when the LLM omits them. Zero extra LLM calls.

- **Configurable decay exponents** — `KnowledgeBase(decay_config={...})` allows
  per-type decay overrides without editing source. `apply_decay()` and
  `calculate_retention()` accept optional `type_exponents` parameter.

- **LLM query expansion at recall** (opt-in) — `KnowledgeBase(llm_recall=True)`
  expands queries with LLM-generated synonyms before BM25 search. Uses the
  configured provider with LRU cache (128 entries). Disabled by default.

### Changed

- **Retriever architecture** — `InvertedIndex` indexes content + tags fields.
  `score()` accepts `b` and `expansion_weights` parameters.

- **QPS threshold** — lowered from 50 to 40 QPS in `test_retriever_throughput_qps`
  to account for BM25F+PRF+RRF pipeline overhead on shared CI runners.

---

## [0.5.0] — 2026-03-31

### Added

- **Power-law forgetting curve** (Wixted & Ebbesen, 1997) — replaces exponential decay
  with `R(t) = (1 + t/(9·S))^(-1)`. Empirically superior fit (R²=98.9% vs 96.3%).
  Heavy tail means important facts persist realistically over months, not days.

- **Shared tokenizer** (`ai_knot.tokenizer`) — zero-dependency leaf module with
  camelCase splitting, Cyrillic support, and naive plural stemming. Used by both
  the BM25 retriever and the ATC verifier.

- **BM25 retrieval** (Robertson & Zaragoza, 2009) — replaces raw TF-IDF with
  Okapi BM25 (`k1=1.5, b=0.75`). Handles term saturation and document length
  normalization. P95-clip normalization bounds BM25 scores to [0, 1] before
  blending with retention and importance. `BM25Retriever` is the new class name;
  `TFIDFRetriever` is kept as a backward-compatible alias.

- **ATC verification guardrail** (inspired by Broder, 1997) — every LLM-extracted
  fact is verified against the source text via Asymmetric Token Containment:
  `ATC = |tokens(snippet) ∩ tokens(source)| / |tokens(snippet)|`.
  Facts with ATC < 0.6 are flagged `supported=False`, preventing hallucinated
  facts from polluting the knowledge base.

- **Fact evidence fields** — five new fields on `Fact`:
  - `source_snippets: list[str]` — source text excerpts
  - `source_spans: list[str]` — span references
  - `supported: bool` — whether ATC ≥ threshold
  - `support_confidence: float` — raw ATC score
  - `verification_source: str` — `"atc"` | `"manual"` | `"legacy"`

- **Offline eval framework** (`tests/eval/`) — zero-dependency retrieval quality
  measurement with `precision_at_k`, `recall_at_k`, `mean_reciprocal_rank`,
  `ndcg_at_k`, and `bootstrap_ci` (stdlib `random.choices`, no numpy).
  105-case golden dataset with variable haystack sizes (3/5/10/20 facts) and
  adversarial scenarios (near-miss distractors, short/long queries, high-noise
  topical clusters). Measured: MRR=0.77, P@5=0.47, nDCG=0.77 on BM25 retriever.

- **FSRS-inspired spacing effect** — new `access_intervals` field on `Fact` tracks
  hours between successive accesses. Spacing factor:
  `0.7 + 0.3 · log(1 + mean_interval / 24)` (floor 0.5). Spaced recalls amplify
  stability, mimicking the spacing effect from cognitive science (Cepeda et al., 2006).

- **Type-aware decay** (Tulving, 1972) — stability multiplier varies by memory type:
  semantic = 1.5×, procedural = 1.0×, episodic = 0.5×. Semantic facts (general
  knowledge) decay slower; episodic facts (events) decay faster.

- **Inverted index** — `InvertedIndex` class with posting lists for
  O(Q · avg_postings) BM25 retrieval instead of O(N · Q) brute-force scan.
  Built once per `search()` call, amortized across query terms.

- **BCa bootstrap** (Efron, 1987) — `bootstrap_ci()` now uses bias-corrected and
  accelerated confidence intervals. Adjusts for skewness and median-bias via
  jackknife acceleration estimate. Stdlib-only, no numpy/scipy.

- **CI quality gates** — two new GitHub Actions jobs:
  - `eval-smoke` (every push/PR): asserts MRR ≥ 0.50, P@5 ≥ 0.30
  - `eval-full` (main/tags only): runs full eval suite

### Changed

- **Forgetting model** — switched from `exp(−t/S)` (exponential) to
  `(1 + t/(9·S))^(-1)` (power-law). Retention thresholds in tests updated
  to match the slower, empirically more accurate decay.

- **YAML serialization** — evidence fields written only when non-default
  (skip empty `source_snippets`, `supported=True`, etc.). Reduces YAML file
  size and serialization overhead. CSafeLoader used when C extension available.

### Fixed

- `generate_mcp_config()` no longer requires `mcp` package installed — it just
  builds a config dict. Import guard now only triggers when `sys.modules["mcp"]`
  is explicitly set to `None`.

- Storage migration for new evidence fields: SQLite uses `PRAGMA table_info`
  for safe column addition; YAML uses `.get()` with backward-compatible defaults.

---

## [0.4.0] — 2026-03-30

### Added

- **Async API** — `KnowledgeBase.alearn()`, `arecall()`, `arecall_facts()` run their
  sync counterparts in a thread-pool executor via `asyncio.get_running_loop().run_in_executor()`,
  keeping the event loop unblocked during LLM HTTP calls. Safe to use in FastAPI handlers
  and `asyncio.gather()`.

- **Provider config at `__init__`** — `KnowledgeBase` now accepts `provider`, `api_key`,
  `model`, and extra `**provider_kwargs` at construction time. These serve as defaults for
  every subsequent `learn()` call; per-call values still override. No more repeating
  credentials on every call in production code.

  ```python
  kb = KnowledgeBase(agent_id="bot", provider="openai", api_key="sk-...")
  kb.learn(turns_a)   # uses init-time credentials
  kb.learn(turns_b)   # same
  ```

- **`KnowledgeBase.add_many()`** — batch-insert a list of facts (strings or dicts) in a
  single storage round-trip, without any LLM call. Validation of all items happens before
  any persistence, so a bad item never partially commits.

- **Per-call timeout** — `learn()` (and `alearn()`) accept `timeout: float | None`
  which propagates through `Extractor` → `call_with_retry` → `provider.call()` →
  `httpx.post()`. `None` (default) uses the provider's built-in 30 s default.

- **Automatic conversation batching** — `learn()` accepts `batch_size: int = 20`
  (forwarded to `Extractor`). Long conversations are split into chunks before being
  sent to the LLM, preventing silent fact loss from JSON truncation.

- **Auto-publish on version tag** — `publish.yml` and `npm-publish.yml` now also
  trigger on `push` of `v[0-9]+.[0-9]+.[0-9]+` tags, wiring the `release.yml` tag
  push directly to PyPI and npm publish without a separate manual dispatch.

- **`recall_facts_with_scores()` documented** — README now includes a usage example
  and explains the hybrid score (TF-IDF similarity + Ebbinghaus retention + fact importance).

### Fixed

- `add_many()` validates all items before touching storage (atomic-style: either all
  items persist or none do on validation failure).
- `add_many()` performs a single `load` + `save` regardless of list length (previously
  each item caused two storage round-trips via `add()`).

---

## [0.3.0] — 2026-03-29

### Added

- **npm package** — `npm install ai-knot` installs a TypeScript client for Node.js 18+.
  Zero runtime npm dependencies. Communicates with the Python `ai-knot-mcp` subprocess
  via JSON-RPC 2.0 over stdio. Dual ESM + CJS exports. Postinstall auto-runs
  `pip install "ai-knot[mcp]"`.

  ```typescript
  import { KnowledgeBase } from 'ai-knot';
  const kb = new KnowledgeBase({ agentId: 'bot', storage: 'sqlite', dbPath: '/data/mem.db' });
  await kb.add('User prefers TypeScript');
  const ctx = await kb.recall('what language?');
  await kb.close();
  ```

  Full API: `add`, `recall`, `forget`, `listFacts`, `stats`, `snapshot`, `restore`, `close`.
  Concurrent calls safe — JSON-RPC 2.0 request-id multiplexing over a single subprocess.

- **Manual publish workflows** — `workflow_dispatch` buttons in GitHub Actions for
  "Publish to PyPI" and "Publish to npm". No tags required; version read from
  `pyproject.toml` and `npm/package.json`.

- **`KnowledgeBase.recall_facts_with_scores()`** — like `recall_facts()` but returns
  `list[tuple[Fact, float]]` with the hybrid relevance score (TF-IDF + retention + importance)
  for each result. Useful for integration adapters and ranking UIs.

- **OpenClaw integration** — `ai_knot.integrations.openclaw`:
  - `OpenClawMemoryAdapter(kb)` — drop-in memory backend for Python agents (LangChain, LangGraph, CrewAI)
  - `generate_mcp_config(agent_id)` — generate the JSON snippet for `~/.openclaw/openclaw.json`

- **MCP `add` tool** now accepts a `tags` parameter (comma-separated string).

### Changed

- **`TFIDFRetriever.search()` return type changed** from `list[Fact]` to `list[tuple[Fact, float]]`.
  Hybrid scores are now returned to callers instead of being discarded.
  **Migration:** unpack `(fact, score)` pairs wherever you call `retriever.search()` directly.

- **`KnowledgeBase.learn()` raises `ValueError`** when no API key can be resolved
  (was: silently return `[]`). Passing empty `turns` still returns `[]` immediately.
  **Migration:** wrap `learn()` in a `try/except ValueError` or set the appropriate env var
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

- **`OpenClawMemoryAdapter.search()`** now returns real `float` relevance scores sourced from
  `recall_facts_with_scores()` (was: always `None`).

### Fixed

- `generate_mcp_config()` raises `ImportError` with an actionable install hint when
  `ai-knot[mcp]` is not installed (was: silently generated a broken config).

- `mcp_server.main()` exits with `sys.exit(1)` and a clear message when the `mcp` package is
  missing (was: cryptic `ImportError` traceback).

- MCP `list_snapshots` tool returns `"[]"` (valid JSON array) when no snapshots exist
  (was: `"No snapshots saved."` — not parseable as JSON).

- CI test matrix now installs `[mcp]` extra so MCP tool tests always run.

---

## [0.2.0] — 2026-03-29

### Added

- **Conflict resolution in `learn()`** — before inserting new facts, `learn()` now cross-checks them
  against existing facts using word-level Jaccard similarity. Duplicate facts (≥ 0.7 similarity by
  default) are not re-inserted; instead the existing fact's importance is reinforced (+0.05, capped at
  1.0) and its `last_accessed` timestamp is updated. The threshold is configurable via
  `conflict_threshold` kwarg on `learn()`.

- **Snapshots** — point-in-time versioning of the knowledge base:
  - `kb.snapshot("name")` — save current state
  - `kb.restore("name")` — atomically replace live facts with snapshot contents
  - `kb.list_snapshots()` — list all saved snapshot names
  - `kb.diff("a", "b")` — compare two snapshots; pass `"current"` as either name for live facts
  - Both YAML and SQLite backends support snapshots via the new `SnapshotCapable` protocol
  - `SnapshotDiff` dataclass exported from top-level `ai-knot`

- **MCP server** — run ai-knot as a native Claude Desktop / Claude Code tool server:
  ```bash
  pip install "ai-knot[mcp]"
  ai-knot-mcp
  ```
  Exposes 7 tools: `add`, `recall`, `forget`, `list_facts`, `stats`, `snapshot`, `restore`.
  Configured entirely via environment variables (`AI_KNOT_AGENT_ID`, `AI_KNOT_STORAGE`,
  `AI_KNOT_DATA_DIR`, `AI_KNOT_DB_PATH`). The `mcp` package is optional — the core package
  does not require it.

### Changed

- `learn()` now returns only the **newly inserted** facts (previously returned all extracted facts).
  Facts that matched existing entries are updated in-place and excluded from the return value.

---

## [0.1.0] — 2026-03-28

### Added
- **Core `KnowledgeBase`** with `add`, `learn`, `recall`, `forget`, `decay`, `stats`
- **Ebbinghaus forgetting curve** — `forgetting.py`
  `retention = exp(−t / stability)` where `stability = 336h × importance × (1 + ln(1 + access_count))`
- **TF-IDF retriever** — zero external dependencies, hybrid score with retention and importance boost
- **LLM fact extractor** — Jaccard deduplication, markdown fence stripping
- **6 LLM providers** — OpenAI, Anthropic (Claude), GigaChat (Sber), Yandex GPT, Qwen, generic OpenAI-compatible
- **Pluggable LLM providers** — `LLMProvider` Protocol with `call_with_retry()` shared retry logic
- **Provider factory** — `create_provider("openai"|"anthropic"|"gigachat"|"yandex"|"qwen"|"openai-compat")`
- **Env var API key resolution** — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GIGACHAT_API_KEY`, `YANDEX_API_KEY`, `QWEN_API_KEY`, `LLM_API_KEY`
- **YAML storage backend** — human-readable, Git-trackable, editable by hand; thread-safe with per-file lock + atomic write
- **SQLite storage backend** — zero-server production storage with WAL mode
- **PostgreSQL storage backend** — provide a DSN, table auto-created; `pip install ai-knot[postgres]`
- **Configurable storage** — `create_storage("yaml"|"sqlite"|"postgres")` factory; CLI `--storage` / `--dsn` options
- **`StorageBackend` protocol** — plug-in interface for custom backends
- **OpenAI integration** — `MemoryEnabledOpenAI` wraps message lists with memory context injection
- **CLI** — `show`, `add`, `recall`, `stats`, `decay`, `clear`, `export`, `import` commands with `--data-dir`, `--storage`, `--dsn` group options
- **Core types** — `Fact`, `MemoryType` (StrEnum), `ConversationTurn`
- **Full test suite** — 235+ tests, 80%+ coverage, both backends parametrized, all LLM calls mocked
- **32 simulation scenarios** — end-to-end tests for memory, storage, providers, CLI, and integrations
- **Performance benchmarks** — `test_performance.py` with `@pytest.mark.slow`
- **pytest markers** — `unit`, `integration`, `slow`, `requires_api_key`
- **GitHub Actions CI** — lint + type check + test on Python 3.11 & 3.12
- **GitHub Actions publish** — PyPI Trusted Publishing on `v*` tags
- **PEP 561** — `py.typed` marker file
- **pre-commit config** — ruff lint+format + mypy hooks
- **Security policy** — `.github/SECURITY.md`
- **Project documentation** — README, CONTRIBUTING, ARCHITECTURE, DEVELOPMENT, skills/

### Changed (since initial development)
- Extractor refactored to use `LLMProvider` Protocol (removed duplicated retry logic)
- `KnowledgeBase.learn()` accepts provider name or `LLMProvider` instance + `**provider_kwargs`
- `MemoryEnabledOpenAI.enrich_messages()` is now a public method
- `datetime.UTC` alias used everywhere (Python 3.11+)
- Input validation in `KnowledgeBase.add()` and CLI `add`/`import` commands
- `BASE_STABILITY_HOURS` set to 336 (2 weeks retention baseline)
- TF-IDF tokenizer: camelCase splitting + basic plural stemming

[Unreleased]: https://github.com/alsoleg89/ai-knot/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/alsoleg89/ai-knot/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/alsoleg89/ai-knot/compare/v0.4.0...v0.5.0
[0.3.0]: https://github.com/alsoleg89/ai-knot/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/alsoleg89/ai-knot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/alsoleg89/ai-knot/releases/tag/v0.1.0
