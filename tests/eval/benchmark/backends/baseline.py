"""Baseline backend: no memory, raw context dump.

Stores all inserted texts verbatim and returns the most recent N on retrieval.
No extraction, no deduplication, no ranking — just a FIFO buffer.
Represents the current state of most production agents that dump raw context.
"""

from __future__ import annotations

import time

from tests.eval.benchmark.base import InsertResult, MemoryBackend, RetrievalResult


class BaselineBackend(MemoryBackend):
    """No memory — returns the most recently inserted texts (FIFO window)."""

    def __init__(self) -> None:
        self._texts: list[str] = []

    @property
    def name(self) -> str:
        return "baseline"

    async def insert(self, text: str) -> InsertResult:
        t0 = time.perf_counter()
        self._texts.append(text)
        return InsertResult(
            facts_stored=len(self._texts),
            facts_extracted=1,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        t0 = time.perf_counter()
        results = list(reversed(self._texts))[:top_k]
        return RetrievalResult(
            texts=results,
            scores=[1.0] * len(results),
            retrieve_ms=(time.perf_counter() - t0) * 1000,
        )

    async def reset(self) -> None:
        self._texts.clear()
