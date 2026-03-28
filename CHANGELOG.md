# Changelog

All notable changes to agentmemo are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- PostgreSQL + pgvector storage backend
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
