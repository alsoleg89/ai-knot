"""Real Qdrant backend: AsyncQdrantClient + Ollama embeddings.

Ported from ContentOs/scripts/ai_knot_eval/backends/qdrant_backend.py.
Changes from ContentOs version:
  - Yandex text-search-doc (256-dim) → Ollama qwen2.5:7b (auto-detected dim)
  - No hash-embedding fallback (Ollama is always local)
  - insert(text) interface instead of insert_facts(list[str])
  - Per-instance unique collection name prevents reset() collisions when
    multiple backends run in parallel (asyncio.gather in runner.py)
  - Vector size auto-detected from a probe embed call — no hardcoded dimension
"""

from __future__ import annotations

import contextlib
import time
import uuid

import httpx

from tests.eval.benchmark.backends.qdrant_emulator import embed_text
from tests.eval.benchmark.base import InsertResult, MemoryBackend, RetrievalResult

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
_QDRANT_BASE = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
# Reuse a single HTTP client for all Qdrant REST calls — avoids per-call TCP handshake.
# search() was dropped in qdrant-client 1.7+; the REST endpoint still exists in server 1.8.0.
_HTTP = httpx.AsyncClient(timeout=30.0)


class QdrantRealBackend(MemoryBackend):
    """Real Qdrant vector search via AsyncQdrantClient + Ollama embeddings.

    No LLM extraction: raw text stored and retrieved directly.
    Each instance gets a unique collection name so parallel benchmark runs
    don't clobber each other's data via reset().
    """

    def __init__(self) -> None:
        self._client: object | None = None
        self._point_id_counter = 0
        self._collection = f"ai_knot_bench_{uuid.uuid4().hex[:8]}"
        self._vector_size: int | None = None  # auto-detected on first setup

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
        # Auto-detect embedding dimension from the active model — no hardcoded size.
        # Avoids breakage when switching models (3072 for llama3.2:3b, 3584 for qwen2.5:7b, etc.)
        probe = await embed_text("_")
        self._vector_size = len(probe)
        with contextlib.suppress(Exception):
            await self._client.create_collection(  # type: ignore[union-attr]
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    async def reset(self) -> None:
        client = await self._get_client()
        with contextlib.suppress(Exception):
            from qdrant_client.models import Distance, VectorParams  # type: ignore[import-untyped]

            await client.delete_collection(self._collection)  # type: ignore[union-attr]
            await client.create_collection(  # type: ignore[union-attr]
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
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
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=self._point_id_counter,
                        vector=vector,
                        payload={"text": text},
                    )
                ],
            )
        except Exception:
            pass

        return InsertResult(
            facts_stored=self._point_id_counter,
            facts_extracted=1,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        await self._get_client()  # ensure collection exists before timing
        t0 = time.perf_counter()

        try:
            q_vec = await embed_text(query)
            resp = await _HTTP.post(
                f"{_QDRANT_BASE}/collections/{self._collection}/points/search",
                json={"vector": q_vec, "limit": top_k, "with_payload": True},
            )
            resp.raise_for_status()
            hits = resp.json().get("result", [])
            texts = [h["payload"].get("text", "") for h in hits if h.get("payload")]
            scores = [h["score"] for h in hits]
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
            info = await client.get_collection(self._collection)  # type: ignore[union-attr]
            return info.points_count  # type: ignore[no-any-return]
        except Exception:
            return None
