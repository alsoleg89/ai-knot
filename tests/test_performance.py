"""Performance baseline tests — ensure core operations stay fast at scale.

Run with: pytest tests/test_performance.py --benchmark-only
These tests are marked @pytest.mark.slow and are skipped in normal CI runs.
Requires pytest-benchmark: pip install agentmemo[dev]
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")

from agentmemo.knowledge import KnowledgeBase
from agentmemo.retriever import TFIDFRetriever
from agentmemo.storage.yaml_storage import YAMLStorage
from agentmemo.types import Fact, MemoryType


def _make_facts(n: int) -> list[Fact]:
    topics = [
        "Python programming language",
        "Docker containerisation",
        "PostgreSQL database",
        "machine learning model",
        "REST API design",
    ]
    return [
        Fact(
            content=f"{topics[i % len(topics)]} fact number {i}",
            type=MemoryType.SEMANTIC,
            importance=0.5 + (i % 5) * 0.1,
        )
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def facts_1k() -> list[Fact]:
    return _make_facts(1_000)


@pytest.fixture(scope="module")
def facts_10k() -> list[Fact]:
    return _make_facts(10_000)


@pytest.fixture(scope="module")
def retriever() -> TFIDFRetriever:
    return TFIDFRetriever()


@pytest.mark.slow
def test_recall_1k_facts(benchmark: pytest.fixture, facts_1k: list[Fact]) -> None:  # type: ignore[type-arg]
    """recall() across 1 000 facts should complete in < 500 ms."""
    retriever = TFIDFRetriever()

    def run() -> None:
        retriever.search("Python programming", facts_1k, top_k=5)

    benchmark(run)
    # benchmark automatically asserts; we also check the median explicitly.
    assert benchmark.stats["median"] < 0.5, (
        f"recall on 1k facts too slow: {benchmark.stats['median']:.3f}s"
    )


@pytest.mark.slow
def test_recall_10k_facts(benchmark: pytest.fixture, facts_10k: list[Fact]) -> None:  # type: ignore[type-arg]
    """recall() across 10 000 facts should complete in < 2 s."""
    retriever = TFIDFRetriever()

    def run() -> None:
        retriever.search("Docker containerisation", facts_10k, top_k=5)

    benchmark(run)
    assert benchmark.stats["median"] < 2.0, (
        f"recall on 10k facts too slow: {benchmark.stats['median']:.3f}s"
    )


@pytest.mark.slow
def test_yaml_save_load_1k(
    benchmark: pytest.fixture,
    tmp_path: pytest.fixture,
    facts_1k: list[Fact],  # type: ignore[type-arg]
) -> None:
    """YAML save+load roundtrip for 1 000 facts should complete in < 1 s."""
    storage = YAMLStorage(base_dir=str(tmp_path))

    def run() -> None:
        storage.save("perf_agent", facts_1k)
        storage.load("perf_agent")

    benchmark(run)
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
