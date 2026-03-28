# Changelog

All notable changes to agentmemo are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

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
  `retention = exp(−t / stability)` where `stability = 336h × importance × (1 + ln(1 + access_count))`
- **TF-IDF retriever** — zero external dependencies, hybrid score with retention and importance boost
- **LLM fact extractor** — Jaccard deduplication, markdown fence stripping
- **6 LLM providers** — OpenAI, Anthropic (Claude), GigaChat (Sber), Yandex GPT, Qwen, generic OpenAI-compatible
- **Pluggable LLM providers** — `LLMProvider` Protocol with `call_with_retry()` shared retry logic
- **Provider factory** — `create_provider("openai"|"anthropic"|"gigachat"|"yandex"|"qwen"|"openai-compat")`
- **Env var API key resolution** — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GIGACHAT_API_KEY`, `YANDEX_API_KEY`, `QWEN_API_KEY`, `LLM_API_KEY`
- **YAML storage backend** — human-readable, Git-trackable, editable by hand; thread-safe with per-file lock + atomic write
- **SQLite storage backend** — zero-server production storage with WAL mode
- **PostgreSQL storage backend** — provide a DSN, table auto-created; `pip install agentmemo[postgres]`
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

[Unreleased]: https://github.com/alsoleg89/agentmemo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alsoleg89/agentmemo/releases/tag/v0.1.0
