"""mem0 multi-agent backend for benchmark scenarios S8–S12.

Uses mem0's native agent_id isolation for per-agent namespaces.
The shared pool is a synthetic agent_id="__pool__".

Requires: Ollama running at http://localhost:11434 (same as mem0_real).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time

from tests.eval.benchmark.backends._mem0_utils import (
    MEM0_SEM as _SEM,
)
from tests.eval.benchmark.backends._mem0_utils import (
    build_ollama_mem0_config,
    mem0_chroma_path,
)
from tests.eval.benchmark.backends._mem0_utils import (
    normalize_mem0_results as _normalize,
)
from tests.eval.benchmark.base import InsertResult, MultiAgentMemoryBackend, RetrievalResult

_POOL_AGENT_ID = "__pool__"
_DEFAULT_CHROMA_PATH = mem0_chroma_path("ma_bench")


class Mem0MultiAgentBackend(MultiAgentMemoryBackend):
    """mem0 backend for multi-agent benchmark scenarios S8–S12.

    Per-agent isolation: mem0 native agent_id parameter.
    Shared pool: synthetic agent_id="__pool__".
    Dirty sync: count-watermark over get_all() — assumes stable insertion order
    from Chroma. Reliable for benchmark (append-only pool); not for production.
    """

    def __init__(self, chroma_path: str = _DEFAULT_CHROMA_PATH) -> None:
        self._chroma_path = chroma_path
        self._memory: object | None = None
        self._last_seen: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "mem0_multi_agent"

    async def _mem(self) -> object:
        if self._memory is None:
            try:
                from mem0 import Memory  # type: ignore[import-untyped]
            except ImportError as e:
                raise ImportError("mem0ai not installed. Run: pip install -e '.[benchmark]'") from e
            self._memory = Memory.from_config(
                build_ollama_mem0_config(
                    collection_name="mem0_ma_bench", chroma_path=self._chroma_path
                )
            )
        return self._memory

    async def reset(self) -> None:
        memory = await self._mem()
        with contextlib.suppress(Exception):
            memory.reset()  # type: ignore[union-attr]
        self._last_seen.clear()

    # ------------------------------------------------------------------ inserts

    async def insert_for_agent(self, agent_id: str, text: str) -> InsertResult:
        return await self.insert_for_agent_with_meta(agent_id, text)

    async def insert_for_agent_with_meta(
        self,
        agent_id: str,
        text: str,
        *,
        topic_channel: str = "",
        importance: float = 0.5,
    ) -> InsertResult:
        memory = await self._mem()
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        metadata: dict[str, object] = {"importance": importance}
        if topic_channel:
            metadata["topic_channel"] = topic_channel
        async with _SEM:
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(
                            memory.add,  # type: ignore[union-attr]
                            [{"role": "user", "content": text}],
                            agent_id=agent_id,
                            metadata=metadata,
                        ),
                    ),
                    timeout=120.0,
                )
                facts_extracted = len(_normalize(result))
            except Exception:
                facts_extracted = 0
        return InsertResult(
            facts_stored=facts_extracted,
            facts_extracted=facts_extracted,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    async def add_structured(
        self, agent_id: str, content: str, *, entity: str, attribute: str
    ) -> None:
        """Store verbatim with entity/attribute metadata (infer=False skips LLM extraction)."""
        memory = await self._mem()
        loop = asyncio.get_running_loop()
        async with _SEM:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(
                            memory.add,  # type: ignore[union-attr]
                            [{"role": "user", "content": content}],
                            agent_id=agent_id,
                            metadata={"entity": entity, "attribute": attribute},
                            infer=False,
                        ),
                    ),
                    timeout=120.0,
                )

    # ---------------------------------------------------------------- retrieval

    async def retrieve_for_agent(
        self, agent_id: str, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        memory = await self._mem()
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    functools.partial(
                        memory.search,  # type: ignore[union-attr]
                        query,
                        agent_id=agent_id,
                        limit=top_k,
                    ),
                ),
                timeout=60.0,
            )
            hits = _normalize(results)
            texts = [r.get("memory", str(r)) for r in hits]
            scores = [float(r.get("score", 0.0)) for r in hits]
        except Exception:
            texts, scores = [], []
        return RetrievalResult(
            texts=texts, scores=scores, retrieve_ms=(time.perf_counter() - t0) * 1000
        )

    # -------------------------------------------------------------------- pool

    async def publish_to_pool(self, agent_id: str, *, utility_threshold: float = 0.0) -> int:
        memory = await self._mem()
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(
                None,
                functools.partial(memory.get_all, agent_id=agent_id, limit=10000),  # type: ignore[union-attr]
            )
            facts = _normalize(raw)
        except Exception:
            return 0

        published = 0
        for fact in facts:
            meta = fact.get("metadata") or {}
            if float(meta.get("importance", 0.5)) < utility_threshold:
                continue
            text = fact.get("memory", "")
            if not text:
                continue
            async with _SEM:
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            functools.partial(
                                memory.add,  # type: ignore[union-attr]
                                [{"role": "user", "content": text}],
                                agent_id=_POOL_AGENT_ID,
                                metadata=meta,
                                infer=False,
                            ),
                        ),
                        timeout=60.0,
                    )
                    published += 1
                except Exception:
                    pass
        return published

    async def pool_retrieve(
        self, requesting_agent_id: str, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        memory = await self._mem()
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    functools.partial(
                        memory.search,  # type: ignore[union-attr]
                        query,
                        agent_id=_POOL_AGENT_ID,
                        limit=top_k,
                    ),
                ),
                timeout=60.0,
            )
            hits = _normalize(results)
            texts = [r.get("memory", str(r)) for r in hits]
            scores = [float(r.get("score", 0.0)) for r in hits]
        except Exception:
            texts, scores = [], []
        return RetrievalResult(
            texts=texts, scores=scores, retrieve_ms=(time.perf_counter() - t0) * 1000
        )

    async def pool_retrieve_for_channel(
        self, agent_id: str, query: str, *, top_k: int = 5, topic_channel: str = ""
    ) -> RetrievalResult:
        memory = await self._mem()
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        # Over-fetch when filtering by channel (mem0 has no server-side metadata filter).
        fetch_limit = top_k * 3 if topic_channel else top_k
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    functools.partial(
                        memory.search,  # type: ignore[union-attr]
                        query,
                        agent_id=_POOL_AGENT_ID,
                        limit=fetch_limit,
                    ),
                ),
                timeout=60.0,
            )
            hits = _normalize(results)
            if topic_channel:
                hits = [
                    r
                    for r in hits
                    if (r.get("metadata") or {}).get("topic_channel") == topic_channel
                ]
            hits = hits[:top_k]
            texts = [r.get("memory", str(r)) for r in hits]
            scores = [float(r.get("score", 0.0)) for r in hits]
        except Exception:
            texts, scores = [], []
        return RetrievalResult(
            texts=texts,
            scores=scores,
            retrieve_ms=(time.perf_counter() - t0) * 1000,
        )

    async def pool_count_active_for_entity(self, entity: str, attribute: str) -> int:
        memory = await self._mem()
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(
                None,
                functools.partial(memory.get_all, agent_id=_POOL_AGENT_ID, limit=10000),  # type: ignore[union-attr]
            )
            facts = _normalize(raw)
        except Exception:
            return 0
        entity_l, attribute_l = entity.lower().strip(), attribute.lower().strip()
        return sum(
            1
            for f in facts
            if (f.get("metadata") or {}).get("entity", "").lower().strip() == entity_l
            and (f.get("metadata") or {}).get("attribute", "").lower().strip() == attribute_l
        )

    async def sync_dirty(self, agent_id: str) -> list[str]:
        """Return pool facts not yet seen by agent_id (count-watermark approach)."""
        memory = await self._mem()
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(
                None,
                functools.partial(memory.get_all, agent_id=_POOL_AGENT_ID, limit=10000),  # type: ignore[union-attr]
            )
            all_pool = _normalize(raw)
        except Exception:
            return []
        last = self._last_seen.get(agent_id, 0)
        new_facts = all_pool[last:]
        self._last_seen[agent_id] = len(all_pool)
        return [f.get("memory", "") for f in new_facts if f.get("memory")]
