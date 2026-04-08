"""QdrantEmulator + LLM Extractor backend for the benchmark.

Track B control: same Extractor as ai_knot (qwen2.5:7b), but stores extracted
facts in QdrantEmulator (dense cosine retrieval) instead of ai_knot's BM25+CAS.

Purpose: isolates the contribution of ai_knot's storage algorithm (BM25, temporal
CAS, dedup) from LLM extraction quality. Comparing ai_knot vs qdrant_extraction
shows what the storage layer adds on top of identical LLM extraction.

Install: pip install ai-knot[ollama]  (for OllamaProvider)
Requires: Ollama running at http://localhost:11434
"""

from __future__ import annotations

import time

from ai_knot.extractor import Extractor
from ai_knot.types import ConversationTurn
from tests.eval.benchmark.backends.qdrant_emulator import QdrantEmulator
from tests.eval.benchmark.base import InsertResult, MemoryBackend, RetrievalResult


class QdrantWithExtractionBackend(MemoryBackend):
    """QdrantEmulator + LLM Extractor (same as ai_knot).

    insert(): runs Extractor.extract() on the input text, stores each extracted
              fact as a separate embedding in QdrantEmulator.
    retrieve(): delegates to QdrantEmulator (cosine similarity over qwen2.5 embeddings).
    """

    def __init__(self, provider: object) -> None:
        self._qdrant = QdrantEmulator()
        self._extractor = Extractor(provider)  # type: ignore[arg-type]
        self._total_stored = 0

    @property
    def name(self) -> str:
        return "qdrant_extraction"

    async def reset(self) -> None:
        await self._qdrant.reset()
        self._total_stored = 0

    async def insert(self, text: str) -> InsertResult:
        t0 = time.perf_counter()
        turns = [ConversationTurn(role="user", content=text)]
        facts = self._extractor.extract(turns)

        facts_extracted = len(facts)
        if facts:
            for fact in facts:
                await self._qdrant.insert(fact.content)
            self._total_stored += facts_extracted
        else:
            # Fallback: store the raw text if extraction yields nothing
            await self._qdrant.insert(text)
            self._total_stored += 1

        return InsertResult(
            facts_stored=self._total_stored,
            facts_extracted=facts_extracted,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        return await self._qdrant.retrieve(query, top_k=top_k)

    async def count_stored(self) -> int | None:
        return self._total_stored
