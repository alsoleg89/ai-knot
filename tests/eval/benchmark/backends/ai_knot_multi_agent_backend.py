"""ai-knot multi-agent backend for benchmark scenarios S8–S20.

Wraps multiple KnowledgeBase instances (one per agent) sharing a single
StorageBackend and a SharedMemoryPool for cross-agent knowledge exchange.

Supports three storage backends selectable at construction time:
  - "sqlite"   — SQLite file (default; fast, no extra deps)
  - "yaml"     — YAML files (human-readable, Git-friendly)
  - "postgres" — PostgreSQL (production-grade; requires psycopg and a running server)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from datetime import UTC, datetime

from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.storage import StorageBackend, create_storage
from ai_knot.types import SlotDelta
from tests.eval.benchmark.base import InsertResult, MultiAgentMemoryBackend, RetrievalResult

_POOL_AGENT_IDS = ("agent_a", "agent_b", "agent_c", "agent_d")
# Shared-pool namespace used internally by SharedMemoryPool.
_SHARED_NAMESPACE = "__shared__"


class AiKnotMultiAgentBackend(MultiAgentMemoryBackend):
    """ai-knot backend with SharedMemoryPool for multi-agent scenarios.

    Creates up to four isolated KnowledgeBase instances sharing one
    StorageBackend and a SharedMemoryPool.  Each agent has its own private
    namespace; the pool provides cross-agent retrieval with MESI coherence.

    Args:
        storage_type: One of "sqlite" (default), "yaml", "postgres".
        postgres_dsn: DSN for PostgreSQL (required when storage_type="postgres").
            Falls back to ``AI_KNOT_DSN`` env var.
    """

    def __init__(
        self,
        storage_type: str = "sqlite",
        *,
        postgres_dsn: str = "",
    ) -> None:
        self._storage_type = storage_type
        self._postgres_dsn = postgres_dsn or os.environ.get("AI_KNOT_DSN", "")
        self._tmp_dir: str = ""
        self._storage: StorageBackend | None = None
        self._kbs: dict[str, KnowledgeBase] = {}
        self._pool: SharedMemoryPool | None = None

    @property
    def name(self) -> str:
        suffix = f"_{self._storage_type}" if self._storage_type != "sqlite" else ""
        return f"ai_knot_multi_agent{suffix}"

    async def reset(self) -> None:
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = ""

        if self._storage_type == "postgres":
            storage = create_storage("postgres", dsn=self._postgres_dsn or None)
            # Clear only benchmark-owned namespaces: static agents, shared pool,
            # and any dynamic agents created in previous runs.
            # SAFETY: never clear unknown namespaces — the DSN may point to a
            # shared database with real ai-knot data.
            benchmark_ns = {_SHARED_NAMESPACE, *_POOL_AGENT_IDS}
            benchmark_ns.update(k for k in self._kbs if k not in _POOL_AGENT_IDS)
            # Also discover agents that match the benchmark naming convention
            # (agent_X) but were created by other benchmark processes.
            for ns in storage.list_agents():  # type: ignore[attr-defined]
                if ns.startswith("agent_") or ns == _SHARED_NAMESPACE:
                    benchmark_ns.add(ns)
            for ns in benchmark_ns:
                storage.save(ns, [])  # type: ignore[attr-defined]
        else:
            self._tmp_dir = tempfile.mkdtemp(prefix=f"aiknot_ma_{self._storage_type}_")
            storage = create_storage(self._storage_type, base_dir=self._tmp_dir)

        self._storage = storage
        self._kbs = {}
        self._pool = SharedMemoryPool(storage=self._storage)
        for agent_id in _POOL_AGENT_IDS:
            self._pool.register(agent_id)
            self._kbs[agent_id] = KnowledgeBase(agent_id, storage=self._storage)

    def _kb(self, agent_id: str) -> KnowledgeBase:
        assert self._storage is not None, "call reset() first"
        if agent_id not in self._kbs:
            assert self._pool is not None
            self._pool.register(agent_id)
            self._kbs[agent_id] = KnowledgeBase(agent_id, storage=self._storage)
        return self._kbs[agent_id]

    async def insert_for_agent(self, agent_id: str, text: str) -> InsertResult:
        t0 = time.perf_counter()
        kb = self._kb(agent_id)
        kb.add(text)
        stored = sum(1 for f in kb.list_facts() if f.is_active())
        return InsertResult(
            facts_stored=stored,
            facts_extracted=1,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    def _patch_fact_by_id(self, kb: KnowledgeBase, fact_id: str, **attrs: object) -> None:
        facts = kb.list_facts()
        for f in facts:
            if f.id == fact_id:
                for k, v in attrs.items():
                    setattr(f, k, v)
                break
        kb.replace_facts(facts)

    async def add_structured(
        self, agent_id: str, content: str, *, entity: str, attribute: str
    ) -> None:
        kb = self._kb(agent_id)
        # Supersede any existing active fact for the same slot within this agent's KB.
        now = datetime.now(UTC)
        facts = kb.list_facts()
        for f in facts:
            if (
                f.is_active(now)
                and f.entity.lower().strip() == entity.lower().strip()
                and f.attribute.lower().strip() == attribute.lower().strip()
            ):
                f.valid_until = now
        kb.replace_facts(facts)
        fact = kb.add(content)
        self._patch_fact_by_id(kb, fact.id, entity=entity, attribute=attribute)

    async def retrieve_for_agent(
        self, agent_id: str, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        kb = self._kb(agent_id)
        pairs = kb.recall_facts_with_scores(query, top_k=top_k)
        return RetrievalResult(
            texts=[f.source_verbatim or f.content for f, _ in pairs],
            scores=[s for _, s in pairs],
            retrieve_ms=(time.perf_counter() - t0) * 1000,
        )

    async def publish_to_pool(self, agent_id: str, *, utility_threshold: float = 0.0) -> int:
        assert self._pool is not None, "call reset() first"
        kb = self._kb(agent_id)
        fact_ids = [f.id for f in kb.list_facts() if f.is_active()]
        if not fact_ids:
            return 0
        published = self._pool.publish(
            agent_id, fact_ids, kb=kb, utility_threshold=utility_threshold
        )
        return len(published)

    async def pool_retrieve(
        self, requesting_agent_id: str, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        assert self._pool is not None, "call reset() first"
        t0 = time.perf_counter()
        pairs = self._pool.recall(query, requesting_agent_id, top_k=top_k)
        return RetrievalResult(
            texts=[f.source_verbatim or f.content for f, _ in pairs],
            scores=[s for _, s in pairs],
            retrieve_ms=(time.perf_counter() - t0) * 1000,
        )

    async def pool_count_active_for_entity(self, entity: str, attribute: str) -> int:
        assert self._pool is not None, "call reset() first"
        entity_lower = entity.lower().strip()
        attribute_lower = attribute.lower().strip()
        return sum(
            1
            for f in self._pool.list_shared_facts()
            if f.entity.lower().strip() == entity_lower
            and f.attribute.lower().strip() == attribute_lower
            and f.is_active()
        )

    async def sync_dirty(self, agent_id: str) -> list[str]:
        assert self._pool is not None, "call reset() first"
        deltas: list[SlotDelta] = self._pool.sync_slot_deltas(agent_id)
        return [d.prompt_surface or d.content for d in deltas]

    async def insert_for_agent_with_meta(
        self,
        agent_id: str,
        text: str,
        *,
        topic_channel: str = "",
        importance: float = 0.5,
    ) -> InsertResult:
        t0 = time.perf_counter()
        kb = self._kb(agent_id)
        fact = kb.add(text)
        if topic_channel or importance != 0.5:
            self._patch_fact_by_id(kb, fact.id, topic_channel=topic_channel, importance=importance)
        stored = sum(1 for f in kb.list_facts() if f.is_active())
        return InsertResult(
            facts_stored=stored,
            facts_extracted=1,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    async def pool_retrieve_for_channel(
        self, agent_id: str, query: str, *, top_k: int = 5, topic_channel: str = ""
    ) -> RetrievalResult:
        assert self._pool is not None, "call reset() first"
        t0 = time.perf_counter()
        pairs = self._pool.recall(query, agent_id, top_k=top_k, topic_channel=topic_channel)
        return RetrievalResult(
            texts=[f.source_verbatim or f.content for f, _ in pairs],
            scores=[s for _, s in pairs],
            retrieve_ms=(time.perf_counter() - t0) * 1000,
        )

    def __del__(self) -> None:
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
