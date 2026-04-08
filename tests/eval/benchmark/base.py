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
    extraction_tokens: int = 0  # approximate tokens spent on LLM extraction (0 for no-LLM backends)


@dataclass
class RetrievalResult:
    """Outcome of a single retrieval query."""

    texts: list[str]  # retrieved fact texts in ranked order
    scores: list[float]  # backend-native scores (BM25, cosine, etc.)
    retrieve_ms: float
    evidence_texts: list[str] = field(default_factory=list)  # enriched with source_snippets


@dataclass
class LongRunStats:
    """Structured stats from a --long-run timed scenario execution."""

    iterations: int
    wall_time_s: float
    avg_iter_s: float
    metric_stdev: dict[str, float]  # metric name -> stdev across iterations


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
    language: str = "en"  # fixture language used for this run ("en" | "ru")
    long_run_stats: LongRunStats | None = None  # populated in --long-run mode


@dataclass
class BenchmarkMetrics:
    """Aggregated results for a single backend across all scenarios."""

    backend_name: str
    language: str = "en"  # fixture language used for this run ("en" | "ru")
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

    async def reset_session(self) -> None:  # noqa: B027
        """Clear per-session novelty state without wiping the KB.

        Called by S2 before each query pair so recall is measured on a fresh
        session (no excluded_ids bias), while novelty is still tested within
        r1 → r2 of the same query. Default no-op; override in backends that
        track session-level deduplication.
        """

    async def count_stored(self) -> int | None:  # noqa: B027
        """Return the exact number of facts currently stored, or None if unsupported.

        Used by S4 deduplication to measure stored count directly instead of
        inferring it from retrieve() results (which only return top-k).
        Backends that can cheaply count stored facts should override this.
        """
        return None


class MultiAgentMemoryBackend(abc.ABC):
    """Abstract base for backends that support multi-agent knowledge sharing.

    Provides isolated private namespaces per agent plus a shared pool for
    cross-agent retrieval.  Used by S8–S11 multi-agent benchmark scenarios.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""
        ...

    @abc.abstractmethod
    async def reset(self) -> None:
        """Clear all state (private namespaces + shared pool)."""
        ...

    @abc.abstractmethod
    async def insert_for_agent(self, agent_id: str, text: str) -> InsertResult:
        """Store a text chunk in the agent's private namespace."""
        ...

    @abc.abstractmethod
    async def add_structured(
        self, agent_id: str, content: str, *, entity: str, attribute: str
    ) -> None:
        """Add a fact with entity+attribute addressing to an agent's private namespace."""
        ...

    @abc.abstractmethod
    async def retrieve_for_agent(
        self, agent_id: str, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        """Retrieve from an agent's private namespace only."""
        ...

    @abc.abstractmethod
    async def publish_to_pool(self, agent_id: str, *, utility_threshold: float = 0.0) -> int:
        """Publish all of agent's active facts to the shared pool.

        Args:
            utility_threshold: Minimum utility score (state_confidence × importance)
                for a fact to be published. Facts below this threshold are filtered.
                Default 0.0 publishes everything.

        Returns:
            Number of facts published.
        """
        ...

    @abc.abstractmethod
    async def pool_retrieve(
        self, requesting_agent_id: str, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        """Retrieve from the shared pool with provenance discounting."""
        ...

    @abc.abstractmethod
    async def pool_count_active_for_entity(self, entity: str, attribute: str) -> int:
        """Count active facts in the shared pool for a given entity+attribute pair.

        Used by S10 to verify MESI CAS prevents duplicates.
        """
        ...

    @abc.abstractmethod
    async def sync_dirty(self, agent_id: str) -> list[str]:
        """Return text of facts changed in the pool since the last sync for agent_id.

        Implements MESI lazy invalidation: only returns facts with version >
        the last-seen version for this agent, from agents other than agent_id.
        Used by S11 to verify token-efficient incremental sync.
        """
        ...

    async def insert_for_agent_with_meta(
        self,
        agent_id: str,
        text: str,
        *,
        topic_channel: str = "",
        importance: float = 0.5,
    ) -> InsertResult:
        """Insert with topic_channel and importance metadata.

        Default: delegates to insert_for_agent (ignores metadata).
        Override in backends that support topic routing and utility gating.
        """
        return await self.insert_for_agent(agent_id, text)

    async def pool_retrieve_for_channel(
        self, agent_id: str, query: str, *, top_k: int = 5, topic_channel: str = ""
    ) -> RetrievalResult:
        """Retrieve from pool filtered by topic_channel.

        Default: delegates to pool_retrieve (no channel filter).
        Override in backends that support topic channel routing.
        """
        return await self.pool_retrieve(agent_id, query, top_k=top_k)
