"""mem0 emulator: LLM extraction + dense vector retrieval.

Mimics mem0's architecture:
  insert → LLM extraction (ai-knot Extractor) → embed each extracted fact → store
  retrieve → embed query → cosine similarity search

Key difference from QdrantEmulator: LLM extraction before storing.
Key difference from AiKnotBackend: dense cosine retrieval instead of BM25.

No mem0 library required — pure Python emulation.

M5 Pro optimization: Extractor.extract() is wrapped in run_in_executor so the
asyncio event loop isn't blocked during the blocking LLM HTTP call.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from ai_knot.extractor import Extractor
from ai_knot.providers.base import LLMProvider
from ai_knot.types import ConversationTurn
from tests.eval.benchmark.backends.qdrant_emulator import (
    QdrantEmulator,
    embed_text,
)
from tests.eval.benchmark.base import InsertResult


class Mem0Emulator(QdrantEmulator):
    """LLM extraction + cosine similarity retrieval.

    Inherits all embedding and retrieval logic from QdrantEmulator.
    Overrides insert() to run text through ai-knot's Extractor first.
    """

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__()
        self._extractor = Extractor(provider)

    @property
    def name(self) -> str:
        return "mem0_emulator"

    async def insert(self, text: str) -> InsertResult:
        t0 = time.perf_counter()

        # Wrap sync Extractor in executor — frees event loop for other coroutines.
        # On M5 Pro 14-core, the thread pool uses idle perf cores during extraction.
        turns = [ConversationTurn(role="user", content=text)]
        loop = asyncio.get_running_loop()
        facts = await loop.run_in_executor(None, self._extractor.extract, turns)

        # If extraction returns nothing, store raw text as fallback
        fact_texts = [f.content for f in facts] if facts else [text]

        for ft in fact_texts:
            emb: list[float] = []
            if self._ollama_ok:
                try:
                    emb = await embed_text(ft)
                except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout):
                    self._ollama_ok = False
            self._store.append((ft, emb))

        return InsertResult(
            facts_stored=len(self._store),
            facts_extracted=len(facts),
            insert_ms=(time.perf_counter() - t0) * 1000,
        )
