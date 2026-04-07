"""ai-knot multi-agent backend for benchmark scenarios S8–S11.

Wraps multiple KnowledgeBase instances (one per agent) sharing a single
SQLiteStorage and a SharedMemoryPool for cross-agent knowledge exchange.
"""

from __future__ import annotations

import shutil
import tempfile
import time

from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import SlotDelta
from tests.eval.benchmark.base import InsertResult, MultiAgentMemoryBackend, RetrievalResult

_POOL_AGENT_IDS = ("agent_a", "agent_b", "agent_c")


class AiKnotMultiAgentBackend(MultiAgentMemoryBackend):
    """ai-knot backend with SharedMemoryPool for multi-agent scenarios.

    Creates up to three isolated KnowledgeBase instances sharing one
    SQLiteStorage and a SharedMemoryPool.  Each agent has its own private
    namespace; the pool provides cross-agent retrieval with MESI coherence.
    """

    def __init__(self) -> None:
        self._tmp_dir: str = ""
        self._storage: SQLiteStorage | None = None
        self._kbs: dict[str, KnowledgeBase] = {}
        self._pool: SharedMemoryPool | None = None

    @property
    def name(self) -> str:
        return "ai_knot_multi_agent"

    async def reset(self) -> None:
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = tempfile.mkdtemp(prefix="aiknot_ma_bench_")
        db_path = f"{self._tmp_dir}/ma_bench.db"
        self._storage = SQLiteStorage(db_path)
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

    async def add_structured(
        self, agent_id: str, content: str, *, entity: str, attribute: str
    ) -> None:
        kb = self._kb(agent_id)
        fact = kb.add(content)
        # Patch entity+attribute on the saved fact.
        facts = kb.list_facts()
        for f in facts:
            if f.id == fact.id:
                f.entity = entity
                f.attribute = attribute
                break
        kb.replace_facts(facts)

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

    async def publish_to_pool(self, agent_id: str) -> int:
        assert self._pool is not None, "call reset() first"
        kb = self._kb(agent_id)
        fact_ids = [f.id for f in kb.list_facts() if f.is_active()]
        if not fact_ids:
            return 0
        published = self._pool.publish(agent_id, fact_ids, kb=kb)
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

    def __del__(self) -> None:
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
