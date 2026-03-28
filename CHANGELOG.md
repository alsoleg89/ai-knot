# Changelog

All notable changes to agentmemo are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- **Configurable storage** — `create_storage("yaml"|"sqlite"|"postgres")` factory; CLI `--storage` / `--dsn` options
- **PostgreSQL storage backend** — provide a DSN, table auto-created; `pip install agentmemo[postgres]`
- **Pluggable LLM providers** — `LLMProvider` Protocol with `call_with_retry()` shared retry logic
- **6 LLM providers** — OpenAI, Anthropic (Claude), GigaChat (Sber), Yandex GPT, Qwen, generic OpenAI-compatible
- **Provider factory** — `create_provider("openai"|"anthropic"|"gigachat"|"yandex"|"qwen"|"openai-compat")`
- **Env var API key resolution** — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GIGACHAT_API_KEY`, `YANDEX_API_KEY`, `QWEN_API_KEY`, `LLM_API_KEY`
- **Thread-safe YAML storage** — per-file lock + atomic write (tempfile → fsync → rename)
- **SQLite WAL mode** — better concurrent read/write performance
- **Error path tests** — `test_extractor_errors.py`, `test_cli_errors.py`, `test_concurrent.py`
- **Provider tests** — `test_providers.py` (retry, factory, all 6 providers mocked)
- **Storage factory tests** — `test_storage_factory.py`
- **Performance benchmarks** — `test_performance.py` with `@pytest.mark.slow`
- **pytest markers** — `unit`, `integration`, `slow`, `requires_api_key`
- **pre-commit config** — ruff lint+format + mypy hooks
- **PEP 561** — `py.typed` marker file

### Changed
- Extractor refactored to use `LLMProvider` Protocol (removed duplicated retry logic)
- `KnowledgeBase.learn()` accepts provider name or `LLMProvider` instance + `**provider_kwargs`
- CLI options `--data-dir`, `--storage`, `--dsn` moved to group level (inherited by all subcommands)
- `MemoryType` uses `StrEnum` instead of `(str, Enum)`
- `datetime.UTC` alias used everywhere (Python 3.11+)
- Input validation in `KnowledgeBase.add()` and CLI `add` command
- Import validation in CLI `import` command (YAML structure, per-fact fields)
- `BASE_STABILITY_HOURS` increased from 168 → 336 (2 weeks retention baseline)
- TF-IDF tokenizer: camelCase splitting + basic plural stemming

### Planned
- MongoDB storage backend
- Qdrant and Weaviate backends
- Semantic embeddings (sentence-transformers / OpenAI)
- MCP server support
- LangChain / CrewAI integrations
- Web UI knowledge inspector
- REST API / sidecar mode

---

## [0.1.0] — 2026-03-28

### Added
- **Core `KnowledgeBase`** with `add`, `learn`, `recall`, `forget`, `decay`, `stats`
- **Ebbinghaus forgetting curve** — `forgetting.py`  
  `retention = exp(−t / stability)` where `stability = 168h × importance × (1 + ln(1 + access_count))`
- **TF-IDF retriever** — zero external dependencies, hybrid score with retention and importance boost
- **LLM fact extractor** — OpenAI and Anthropic providers via `httpx`; Jaccard deduplication
- **YAML storage backend** — human-readable, Git-trackable, editable by hand
- **SQLite storage backend** — zero-server production storage
- **`StorageBackend` protocol** — plug-in interface for custom backends
- **OpenAI integration** — `MemoryEnabledOpenAI` wraps message lists with memory context injection
- **CLI** — `show`, `add`, `recall`, `stats`, `decay`, `clear`, `export`, `import` commands
- **Core types** — `Fact`, `MemoryType`, `ConversationTurn`
- **Full test suite** — 80 %+ coverage, both backends parametrized, all LLM calls mocked
- **GitHub Actions CI** — lint + type check + test on Python 3.11 & 3.12
- **GitHub Actions publish** — PyPI Trusted Publishing on `v*` tags
- **Project documentation** — README, CONTRIBUTING, ARCHITECTURE, DEVELOPMENT, skills/

[Unreleased]: https://github.com/alsoleg89/Agentmemo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alsoleg89/Agentmemo/releases/tag/v0.1.0
