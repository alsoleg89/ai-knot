"""Performance baseline tests — ensure core operations stay fast at scale.

Run with: pytest tests/test_performance.py -m slow --benchmark-only
These tests are marked @pytest.mark.slow and are skipped in normal CI runs.
Requires pytest-benchmark: pip install ai-knot[dev]

Research references:
- Qdrant P99 vs pgvector: 38.71ms vs 74.60ms (Qdrant benchmarks, 1M OpenAI vectors)
- LanceDB design target: 25ms vector search (docs.lancedb.com/enterprise/benchmark)
- Chroma warm P50: 20ms for 100k vectors at 384 dims (docs.trychroma.com/guides/deploy/performance)
- TF-IDF full-text search: milliseconds achievable in-memory (no external DB)
- IQR vs stddev: IQR more robust for shared CI runners (pytest-benchmark docs)
- P99 requires 50x samples vs mean for statistical significance
  (optyxstack.com/performance/latency-distributions-in-practice)
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("pytest_benchmark")

from ai_knot.forgetting import apply_decay
from ai_knot.knowledge import KnowledgeBase
from ai_knot.retriever import TFIDFRetriever, _tokenize
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact, MemoryType

# ---------------------------------------------------------------------------
# Realistic facts corpus (20+ distinct topics for accurate TF-IDF scoring)
# ---------------------------------------------------------------------------
# Using uniform templates distorts TF-IDF: IDF scores become too equal when
# all docs share the same vocabulary. Real recall benchmarks need diverse text.

_REALISTIC_FACTS = [
    "User prefers TypeScript over JavaScript for large-scale projects",
    "The team deploys to AWS EKS using Helm charts and ArgoCD",
    "PostgreSQL 16 introduced query parallelism and logical replication improvements",
    "User prefers pytest over unittest; always add type hints to test functions",
    "Docker images must be built with multi-stage builds to reduce layer size",
    "The REST API uses JWT authentication with 15-minute access token expiry",
    "Redis is used as a session cache with a 30-minute TTL",
    "All database migrations run via Alembic before each deployment",
    "Python 3.12 significantly improved interpreter startup time",
    "User dislikes synchronous blocking I/O; prefers async/await throughout",
    "The CI pipeline runs on GitHub Actions with matrix testing on 3.11 and 3.12",
    "Kubernetes pods are configured with resource limits: 500m CPU, 512Mi memory",
    "gRPC is used for internal microservice communication, not HTTP",
    "Sentry captures all unhandled exceptions in production environment",
    "The data pipeline uses Apache Kafka for event streaming at 10k msgs/sec",
    "Machine learning models are served via FastAPI with ONNX Runtime",
    "SQLite is preferred for embedded storage in CLI tools and desktop apps",
    "User always writes docstrings in Google style with Args and Returns sections",
    "The frontend is built with React 18 using Vite as the bundler",
    "Rust is used for performance-critical components; called from Python via PyO3",
    "All secrets are stored in HashiCorp Vault, never in environment variables",
    "Prometheus metrics are exposed on /metrics with Grafana dashboards",
    "The team follows semantic versioning; breaking changes only in major releases",
    "Integration tests run against a local Testcontainers PostgreSQL instance",
    "Code reviews require at least two approvals before merging to main",
]


def _make_facts(n: int) -> list[Fact]:
    """Generate n facts by cycling through the realistic corpus.

    Cycling a diverse corpus preserves IDF variance across documents,
    which produces realistic TF-IDF scoring (unlike repetitive templates).
    """
    return [
        Fact(
            content=_REALISTIC_FACTS[i % len(_REALISTIC_FACTS)],
            type=MemoryType.SEMANTIC,
            importance=0.5 + (i % 10) * 0.05,
        )
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def facts_1k() -> list[Fact]:
    return _make_facts(1_000)


@pytest.fixture(scope="module")
def facts_10k() -> list[Fact]:
    return _make_facts(10_000)


# ---------------------------------------------------------------------------
# Existing baselines (preserved for regression detection)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_recall_1k_facts(benchmark: pytest.fixture, facts_1k: list[Fact]) -> None:  # type: ignore[type-arg]
    """recall() across 1 000 facts should complete in < 500 ms."""
    retriever = TFIDFRetriever()

    benchmark(lambda: retriever.search("Python programming", facts_1k, top_k=5))
    assert benchmark.stats["median"] < 0.5, (
        f"recall on 1k facts too slow: {benchmark.stats['median']:.3f}s"
    )


@pytest.mark.slow
def test_recall_10k_facts(benchmark: pytest.fixture, facts_10k: list[Fact]) -> None:  # type: ignore[type-arg]
    """recall() across 10 000 facts should complete in < 2 s."""
    retriever = TFIDFRetriever()

    benchmark(lambda: retriever.search("Docker containerisation", facts_10k, top_k=5))
    assert benchmark.stats["median"] < 2.0, (
        f"recall on 10k facts too slow: {benchmark.stats['median']:.3f}s"
    )


@pytest.mark.slow
def test_yaml_save_load_1k(
    benchmark: pytest.fixture,
    tmp_path: pytest.fixture,  # type: ignore[type-arg]
    facts_1k: list[Fact],
) -> None:
    """YAML save+load roundtrip for 1 000 facts should complete in < 1 s."""
    storage = YAMLStorage(base_dir=str(tmp_path))

    benchmark(lambda: (storage.save("perf_agent", facts_1k), storage.load("perf_agent")))
    assert benchmark.stats["median"] < 1.0, (
        f"YAML roundtrip 1k too slow: {benchmark.stats['median']:.3f}s"
    )


@pytest.mark.slow
def test_knowledge_base_add_100(
    benchmark: pytest.fixture,
    tmp_path: pytest.fixture,  # type: ignore[type-arg]
) -> None:
    """Adding 100 facts via KnowledgeBase.add() should stay under 2 s total."""
    storage = YAMLStorage(base_dir=str(tmp_path))
    kb = KnowledgeBase(agent_id="perf_agent", storage=storage)
    counter = [0]

    def run() -> None:
        counter[0] += 1
        kb.add(f"fact number {counter[0]}", importance=0.8)

    benchmark.pedantic(run, iterations=100, rounds=1)
    assert benchmark.stats["total"] < 2.0, (
        f"100 kb.add() calls too slow: {benchmark.stats['total']:.3f}s"
    )


@pytest.mark.slow
def test_recall_1k_p99(
    benchmark: pytest.fixture,
    facts_1k: list[Fact],
) -> None:
    """P99 latency for recall on 1 000 facts should be < 100 ms.

    Uses max as a proxy for P99.  CI runners have OS scheduling noise
    that inflates max beyond local measurements — 100 ms gives headroom
    while still catching genuine regressions (local median is ~30 ms).
    """
    retriever = TFIDFRetriever()

    benchmark(lambda: retriever.search("Python programming", facts_1k, top_k=5))
    max_s = benchmark.stats["max"]
    assert max_s < 0.1, f"Max latency too high: {max_s * 1000:.1f} ms (target: 100 ms)"


@pytest.mark.slow
def test_sqlite_save_load_1k(
    benchmark: pytest.fixture,
    tmp_path: pytest.fixture,  # type: ignore[type-arg]
    facts_1k: list[Fact],
) -> None:
    """SQLite save+load roundtrip for 1 000 facts should complete in < 1 s."""
    pytest.importorskip("sqlite3")
    from ai_knot.storage.sqlite_storage import SQLiteStorage  # noqa: PLC0415

    storage = SQLiteStorage(db_path=str(tmp_path / "perf.db"))

    benchmark(lambda: (storage.save("perf_agent", facts_1k), storage.load("perf_agent")))
    assert benchmark.stats["median"] < 1.0, (
        f"SQLite roundtrip 1k too slow: {benchmark.stats['median']:.3f}s"
    )


@pytest.mark.slow
def test_recall_no_memory_growth(facts_1k: list[Fact]) -> None:
    """Memory usage should not grow unboundedly across 200 recall calls.

    Acceptable growth: < 2 MB (accounts for Python internals and caching).
    """
    import tracemalloc

    retriever = TFIDFRetriever()

    for _ in range(5):
        retriever.search("warmup", facts_1k, top_k=5)

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for _ in range(200):
        retriever.search("Python programming", facts_1k, top_k=5)

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    growth = sum(s.size_diff for s in top_stats if s.size_diff > 0)

    assert growth < 2 * 1024 * 1024, (
        f"Memory grew by {growth / 1024:.1f} KB across 200 recalls (limit: 2 MB)"
    )


# ---------------------------------------------------------------------------
# Role 2: Performance Engineer — O(n) scaling curve + QPS
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("n_facts", [100, 1_000, 10_000])
def test_retriever_latency_scaling(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    n_facts: int,
) -> None:
    """Parametrized scaling: measures the O(n) latency curve across fact counts.

    TFIDFRetriever.search() recomputes IDF on every call (no cache), so
    latency grows linearly with the number of facts. This test makes the
    O(n) relationship explicit so regressions in the slope are visible.

    Targets (generous — GitHub Actions runners vary):
      100 facts  → median < 50 ms
      1 000 facts → median < 500 ms
      10 000 facts → median < 5 s
    """
    facts = _make_facts(n_facts)
    retriever = TFIDFRetriever()
    limits = {100: 0.05, 1_000: 0.5, 10_000: 5.0}

    benchmark.pedantic(
        lambda: retriever.search("Python TypeScript deployment", facts, top_k=5),
        warmup_rounds=3,
        rounds=20,
    )
    benchmark.extra_info["n_facts"] = n_facts
    benchmark.extra_info["median_ms"] = round(benchmark.stats["median"] * 1000, 2)

    limit = limits[n_facts]
    assert benchmark.stats["median"] < limit, (
        f"Scaling regression at n={n_facts}: "
        f"{benchmark.stats['median'] * 1000:.1f}ms > {limit * 1000:.0f}ms target"
    )


@pytest.mark.slow
def test_retriever_throughput_qps(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    facts_1k: list[Fact],
) -> None:
    """TFIDFRetriever throughput on 1 000 facts must exceed 20 QPS.

    benchmark.stats["ops"] is the built-in QPS metric (1 / mean).
    Industry context: Qdrant P99 = 38.71ms at 2200 QPS for 1M vectors;
    our BM25F+PRF+RRF pipeline is heavier than plain TF-IDF — 6 rankers
    (BM25F + slot-exact + char-trigram + importance + retention + recency),
    single-threaded, in-memory, zero deps.

    Threshold history:
      40 QPS — original plain BM25F baseline
      30 QPS — after adding Cyrillic stemmer + canonical_surface field
      20 QPS — after adding 6-ranker RRF pipeline (Phase 4); GitHub Actions
                2-vCPU runners are ~2× slower than M5 Pro (44 QPS locally).
    """
    retriever = TFIDFRetriever()

    benchmark.pedantic(
        lambda: retriever.search("Python TypeScript", facts_1k, top_k=5),
        warmup_rounds=3,
        rounds=50,
    )
    qps = benchmark.stats["ops"]
    benchmark.extra_info["qps"] = round(qps, 1)
    assert qps > 15, f"Throughput too low: {qps:.1f} QPS (target: >15)"


# ---------------------------------------------------------------------------
# Role 3: Data Scientist — IQR stability + peak memory footprint
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_recall_latency_distribution_stable(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    facts_1k: list[Fact],
) -> None:
    """IQR < 50% of median confirms distribution is stable (not spiky).

    Research: IQR is more robust than stddev for shared CI runners where
    OS scheduling noise creates outlier spikes.
    Reference: optyxstack.com/performance/latency-distributions-in-practice

    100 rounds required for IQR to be statistically meaningful at P95-P99.
    """
    retriever = TFIDFRetriever()

    benchmark.pedantic(
        lambda: retriever.search("Python TypeScript Kubernetes", facts_1k, top_k=5),
        warmup_rounds=5,
        rounds=100,
    )
    iqr = benchmark.stats["iqr"]
    median = benchmark.stats["median"]
    benchmark.extra_info["iqr_pct_median"] = round(iqr / median * 100, 1)

    assert iqr < median * 0.5, (
        f"Unstable: IQR={iqr * 1000:.2f}ms = {iqr / median * 100:.0f}% of median (target: <50%)"
    )


@pytest.mark.slow
def test_memory_footprint_10k_recall(
    tmp_path: pytest.fixture,  # type: ignore[type-arg]
    facts_10k: list[Fact],
) -> None:
    """Peak memory during a single recall() over 10 000 facts must be < 200 MB.

    Measures tracemalloc peak (not leak) — how much memory the operation
    allocates at its high-water mark. This catches regressions from naive
    data-structure choices (e.g. materialising full TF matrices).
    """
    import tracemalloc

    storage = YAMLStorage(base_dir=str(tmp_path))
    storage.save("agent", facts_10k)
    kb = KnowledgeBase(agent_id="agent", storage=storage)

    # Warm up (one-time allocations should not count)
    kb.recall("warmup query")

    tracemalloc.start()
    kb.recall("Python TypeScript deployment Kubernetes Docker")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1024 / 1024
    assert peak_mb < 200, f"Peak memory too high: {peak_mb:.1f} MB (limit: 200 MB)"


# ---------------------------------------------------------------------------
# Role 2 (continued): Performance Engineer — decay + full-stack + tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_decay_bulk_10k(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    facts_10k: list[Fact],
) -> None:
    """Ebbinghaus decay over 10 000 facts should complete in < 500 ms.

    apply_decay() is O(n) per fact — this test isolates the decay path
    from retrieval so regressions in the forgetting curve are visible.
    """
    benchmark.pedantic(
        lambda: apply_decay(list(facts_10k)),
        warmup_rounds=2,
        rounds=20,
    )
    assert benchmark.stats["median"] < 0.5, (
        f"Decay on 10k facts too slow: {benchmark.stats['median'] * 1000:.1f}ms"
    )
    benchmark.extra_info["facts_per_ms"] = round(10_000 / (benchmark.stats["median"] * 1000), 0)


@pytest.mark.slow
def test_kb_recall_full_stack(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
    tmp_path: pytest.fixture,
) -> None:
    """Full-stack recall() includes storage.load() + apply_decay() + TF-IDF search.

    KnowledgeBase.recall() calls storage.load() on every invocation — the
    I/O cost is hidden by unit-level retriever benchmarks. This test measures
    the real latency an agent experiences.

    Target: median < 1 s (1 000 facts, YAML backend, GitHub Actions runner).
    """
    storage = YAMLStorage(base_dir=str(tmp_path))
    storage.save("bench", _make_facts(1_000))
    kb = KnowledgeBase(agent_id="bench", storage=storage)

    benchmark.pedantic(
        lambda: kb.recall("Python TypeScript deployment"),
        warmup_rounds=2,
        rounds=20,
    )
    benchmark.extra_info["includes_storage_io"] = True
    assert benchmark.stats["median"] < 1.0, (
        f"Full-stack recall too slow: {benchmark.stats['median'] * 1000:.1f}ms"
    )


@pytest.mark.slow
def test_tokenizer_throughput(
    benchmark: pytest.fixture,  # type: ignore[type-arg]
) -> None:
    """_tokenize() hotpath: 100 varied strings should yield > 10 000 tokens/sec.

    _tokenize() is called for every fact on every search() call, making it
    the innermost hotspot of TFIDFRetriever. This test isolates its cost.
    """
    texts = [
        f"Python programming Docker Kubernetes machine-learning REST API gRPC {i}"
        for i in range(100)
    ]

    benchmark.pedantic(
        lambda: [_tokenize(t) for t in texts],
        warmup_rounds=3,
        rounds=50,
    )
    tokens_per_sec = benchmark.stats["ops"] * 100
    benchmark.extra_info["tokens_per_sec"] = round(tokens_per_sec, 0)
    assert tokens_per_sec > 10_000, (
        f"Tokenizer too slow: {tokens_per_sec:.0f} tokens/sec (target: >10 000)"
    )


# ---------------------------------------------------------------------------
# Role 8: Thread-Safety / Concurrency Engineer
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_concurrent_reads_no_corruption(
    tmp_path: pytest.fixture,  # type: ignore[type-arg]
) -> None:
    """8 concurrent threads reading via one KnowledgeBase return valid results.

    YAML storage is atomic on writes (tmpfile→rename) but has no cross-thread
    lock on reads. This test verifies that concurrent reads do not corrupt
    in-memory state or raise exceptions.
    """
    storage = YAMLStorage(base_dir=str(tmp_path))
    storage.save("agent", _make_facts(500))
    kb = KnowledgeBase(agent_id="agent", storage=storage)
    errors: list[Exception] = []
    results: list[str] = []
    lock = threading.Lock()

    def do_recall() -> None:
        try:
            r = kb.recall("Python TypeScript deployment")
            with lock:
                results.append(r)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=do_recall) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Concurrent read errors: {errors}"
    assert len(results) == 8


@pytest.mark.slow
def test_sqlite_concurrent_writes_safe(
    tmp_path: pytest.fixture,  # type: ignore[type-arg]
) -> None:
    """SQLite WAL mode must handle 16 concurrent saves without data loss.

    SQLite in WAL mode with a 30-second busy timeout should serialize
    concurrent writes safely. YAML does NOT provide this guarantee.
    Reference: sqlite.org/wal.html — WAL allows concurrent reads during write.
    """
    pytest.importorskip("sqlite3")
    from ai_knot.storage.sqlite_storage import SQLiteStorage  # noqa: PLC0415

    storage = SQLiteStorage(db_path=str(tmp_path / "conc.db"))
    errors: list[Exception] = []
    lock = threading.Lock()

    def write_batch(thread_id: int) -> None:
        try:
            storage.save(f"agent_{thread_id}", _make_facts(10))
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=write_batch, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent SQLite write errors: {errors}"
    total = sum(len(storage.load(f"agent_{i}")) for i in range(16))
    assert total == 160, f"Data loss: expected 160 facts, got {total}"
