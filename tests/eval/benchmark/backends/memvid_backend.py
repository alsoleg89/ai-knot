"""Memvid backend for the benchmark suite.

Wraps memvid 0.1.x (MemvidEncoder + MemvidRetriever) with a lazy-rebuild
pattern: the video index is rebuilt only when retrieve() is called after
one or more new inserts.  This avoids an O(n) rebuild per insert while
still presenting a consistent view at query time.

Install: pip install memvid
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from tests.eval.benchmark.base import InsertResult, MemoryBackend, RetrievalResult


class MemvidBackend(MemoryBackend):
    """Memvid video-encoded memory backend.

    Stores text chunks in a compact video file (MP4 + JSON index) using
    memvid's BM25 + semantic search.  The video is rebuilt lazily before
    each retrieval so sequential inserts are cheap.
    """

    def __init__(self) -> None:
        self._tmp_dir: str = ""
        self._chunks: list[str] = []
        self._dirty: bool = False
        self._retriever: object | None = None  # MemvidRetriever after build

    @property
    def name(self) -> str:
        return "memvid"

    async def reset(self) -> None:
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = tempfile.mkdtemp(prefix="memvid_bench_")
        self._chunks = []
        self._dirty = False
        self._retriever = None

    async def insert(self, text: str) -> InsertResult:
        t0 = time.perf_counter()
        self._chunks.append(text)
        self._dirty = True
        return InsertResult(
            facts_stored=len(self._chunks),
            facts_extracted=1,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    def _rebuild(self) -> None:
        """Rebuild the video index from all current chunks."""
        from memvid import MemvidEncoder, MemvidRetriever

        enc = MemvidEncoder()
        if self._chunks:
            enc.add_chunks(self._chunks)
        mp4_path = str(Path(self._tmp_dir) / "mem.mp4")
        idx_path = str(Path(self._tmp_dir) / "mem.json")
        enc.build_video(mp4_path, idx_path, show_progress=False, allow_fallback=True)
        self._retriever = MemvidRetriever(mp4_path, idx_path)
        self._dirty = False

    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        t0 = time.perf_counter()

        if not self._chunks:
            return RetrievalResult(texts=[], scores=[], retrieve_ms=0.0)

        if self._dirty or self._retriever is None:
            self._rebuild()

        assert self._retriever is not None
        from memvid import MemvidRetriever

        retriever: MemvidRetriever = self._retriever  # type: ignore[assignment]
        try:
            hits = retriever.search_with_metadata(query, top_k=top_k)
            texts = [h.get("text", h.get("chunk", "")) for h in hits]
            scores = [float(h.get("score", 0.0)) for h in hits]
        except Exception:
            # Fallback to basic search if metadata call fails
            texts = retriever.search(query, top_k=top_k)
            scores = [1.0 - i * 0.1 for i in range(len(texts))]

        return RetrievalResult(
            texts=texts,
            scores=scores,
            retrieve_ms=(time.perf_counter() - t0) * 1000,
        )

    async def count_stored(self) -> int | None:
        return len(self._chunks)

    def __del__(self) -> None:
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
