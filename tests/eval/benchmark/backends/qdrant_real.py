"""Real Qdrant backend: AsyncQdrantClient + Ollama embeddings (llama3.2:3b, 3072-dim).

Ported from ContentOs/scripts/ai_knot_eval/backends/qdrant_backend.py.
Changes from ContentOs version:
  - Yandex text-search-doc (256-dim) → Ollama llama3.2:3b (3072-dim)
  - No hash-embedding fallback (Ollama is always local)
  - insert(text) interface instead of insert_facts(list[str])
  - Collection name: "ai_knot_bench" (won't conflict with ContentOs)
  - Semaphore: 5 → 3 (GPU memory limited on Ollama)
"""

from __future__ import annotations

import contextlib
import time

from tests.eval.benchmark.backends.qdrant_emulator import embed_text
from tests.eval.benchmark.base import InsertResult, MemoryBackend, RetrievalResult

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "ai_knot_bench"
VECTOR_SIZE = 3072  # llama3.2:3b via Ollama


class QdrantRealBackend(MemoryBackend):
    """Real Qdrant vector search via AsyncQdrantClient + Ollama embeddings.

    No LLM extraction: raw text stored and retrieved directly.
    """

    def __init__(self) -> None:
        self._client: object | None = None
        self._point_id_counter = 0

    @property
    def name(self) -> str:
        return "qdrant_real"

    async def _get_client(self) -> object:
        if self._client is None:
            await self._setup()
        return self._client  # type: ignore[return-value]

    async def _setup(self) -> None:
        try:
            from qdrant_client import AsyncQdrantClient  # type: ignore[import-untyped]
            from qdrant_client.models import Distance, VectorParams  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "qdrant-client not installed. Run: pip install -e '.[benchmark]'"
            ) from e

        self._client = AsyncQdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False
        )
        with contextlib.suppress(Exception):
            await self._client.create_collection(  # type: ignore[union-attr]
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    async def reset(self) -> None:
        client = await self._get_client()
        with contextlib.suppress(Exception):
            from qdrant_client.models import Distance, VectorParams  # type: ignore[import-untyped]

            await client.delete_collection(COLLECTION_NAME)  # type: ignore[union-attr]
            await client.create_collection(  # type: ignore[union-attr]
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        self._point_id_counter = 0

    async def insert(self, text: str) -> InsertResult:
        client = await self._get_client()
        t0 = time.perf_counter()

        try:
            from qdrant_client.models import PointStruct  # type: ignore[import-untyped]

            vector = await embed_text(text)
            self._point_id_counter += 1
            await client.upsert(  # type: ignore[union-attr]
                collection_name=COLLECTION_NAME,
                points=[PointStruct(
                    id=self._point_id_counter,
                    vector=vector,
                    payload={"text": text},
                )],
            )
        except Exception:
            pass

        return InsertResult(
            facts_stored=self._point_id_counter,
            facts_extracted=1,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        client = await self._get_client()
        t0 = time.perf_counter()

        try:
            q_vec = await embed_text(query)
            results = await client.search(  # type: ignore[union-attr]
                collection_name=COLLECTION_NAME,
                query_vector=q_vec,
                limit=top_k,
            )
            texts = [r.payload.get("text", "") for r in results if r.payload]
            scores = [r.score for r in results]
        except Exception:
            texts, scores = [], []

        return RetrievalResult(
            texts=texts,
            scores=scores,
            retrieve_ms=(time.perf_counter() - t0) * 1000,
        )

    async def count_stored(self) -> int | None:
        client = await self._get_client()
        try:
            info = await client.get_collection(COLLECTION_NAME)  # type: ignore[union-attr]
            return info.points_count  # type: ignore[no-any-return]
        except Exception:
            return None
