"""Abstract base for benchmark backends and shared result types."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class InsertResult:
    """Outcome of inserting one text chunk into a backend."""

    facts_stored: int  # total facts in store after this insert
    facts_extracted: int  # facts returned by extraction (before dedup/filter)
    insert_ms: float  # wall time for the insert operation


@dataclass
class RetrievalResult:
    """Outcome of a single retrieval query."""

    texts: list[str]  # retrieved fact texts in ranked order
    scores: list[float]  # backend-native scores (BM25, cosine, etc.)
    retrieve_ms: float


@dataclass
class ScenarioResult:
    """Results from one scenario run on one backend."""

    scenario_id: str  # e.g. "s4_deduplication"
    backend_name: str
    # metric -> [run1, run2, run3] (3 judge runs for median + stdev)
    judge_scores: dict[str, list[float]]
    insert_result: InsertResult | None
    retrieval_result: RetrievalResult | None
    notes: str = ""


@dataclass
class BenchmarkMetrics:
    """Aggregated results for a single backend across all scenarios."""

    backend_name: str
    scenario_results: list[ScenarioResult] = field(default_factory=list)

    def median_score(self, scenario_id: str, metric: str) -> float:
        """Return median judge score for a scenario + metric pair."""
        for r in self.scenario_results:
            if r.scenario_id == scenario_id:
                scores = r.judge_scores.get(metric, [])
                if scores:
                    s = sorted(scores)
                    n = len(s)
                    mid = n // 2
                    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
        return 0.0

    def stdev_score(self, scenario_id: str, metric: str) -> float:
        """Return standard deviation of judge scores for a scenario + metric."""
        import statistics

        for r in self.scenario_results:
            if r.scenario_id == scenario_id:
                scores = r.judge_scores.get(metric, [])
                if len(scores) > 1:
                    return statistics.stdev(scores)
        return 0.0


class MemoryBackend(abc.ABC):
    """Abstract base class for all benchmark memory backends."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""
        ...

    @abc.abstractmethod
    async def insert(self, text: str) -> InsertResult:
        """Store a text chunk (with optional extraction / embedding)."""
        ...

    @abc.abstractmethod
    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        """Return top-k ranked results for a query."""
        ...

    @abc.abstractmethod
    async def reset(self) -> None:
        """Clear all stored state. Called between scenarios."""
        ...

    async def tick_decay(self, *, hours: float) -> None:  # noqa: B027
        """Simulate passage of time by `hours` and apply forgetting.

        Default no-op — only ai-knot backend overrides this.
        Intentionally not abstract: other backends silently skip decay.
        """

    async def count_stored(self) -> int | None:  # noqa: B027
        """Return the exact number of facts currently stored, or None if unsupported.

        Used by S4 deduplication to measure stored count directly instead of
        inferring it from retrieve() results (which only return top-k).
        Backends that can cheaply count stored facts should override this.
        """
        return None
