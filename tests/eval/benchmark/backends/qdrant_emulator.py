"""Qdrant emulator: pure Python in-memory cosine similarity vector store.

No external Qdrant server or Docker required.
Embeddings via Ollama's OpenAI-compat endpoint (qwen2.5:7b, 3584-dim).
Falls back to TF-IDF cosine if Ollama is not reachable.

Key difference from ai-knot: dense vector retrieval vs BM25/TF-IDF.
Key difference from Mem0Emulator: raw text is stored directly (no LLM extraction).
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter

import httpx

from tests.eval.benchmark.base import InsertResult, MemoryBackend, RetrievalResult

EMBED_URL = "http://localhost:11434/v1/embeddings"
EMBED_MODEL = "qwen2.5:7b"
_SEM = asyncio.Semaphore(1)  # serialize embed calls — qwen2.5:7b fills GPU, no benefit to parallel
_HTTP = httpx.AsyncClient(timeout=120.0)  # reuse connection pool across all embed calls


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tfidf_cosine(query: str, texts: list[str], top_k: int) -> list[tuple[str, float]]:
    """Minimal bag-of-words TF-IDF cosine — no stemming, no deps."""
    if not texts:
        return []

    def bow(t: str) -> Counter[str]:
        return Counter(t.lower().split())

    all_docs = texts + [query]
    bows = [bow(d) for d in all_docs]
    n = len(texts)
    df: Counter[str] = Counter()
    for b in bows[:n]:
        df.update(b.keys())
    vocab = list(df.keys())

    def vec(b: Counter[str]) -> list[float]:
        total = max(sum(b.values()), 1)
        return [(b.get(w, 0) / total) * math.log((n + 1) / (df.get(w, 0) + 1)) for w in vocab]

    q_vec = vec(bows[-1])
    ranked = [(t, _cosine(q_vec, vec(bows[i]))) for i, t in enumerate(texts)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Async embed helper
# ---------------------------------------------------------------------------


async def embed_text(text: str) -> list[float]:
    """Call Ollama embeddings API for a single text. Dimension depends on model."""
    async with _SEM:
        resp = await _HTTP.post(
            EMBED_URL,
            headers={"Authorization": "Bearer ollama", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return list(data["data"][0]["embedding"])


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed via a single Ollama HTTP call — much faster than N individual calls.

    Ollama supports input: list[str] for /v1/embeddings. On M5 Pro Metal a single
    batch call of 200 texts is ~10× faster than 200 sequential single calls because
    the GPU processes them in one forward pass and network overhead is paid once.
    """
    if not texts:
        return []
    async with _SEM:
        resp = await _HTTP.post(
            EMBED_URL,
            headers={"Authorization": "Bearer ollama", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        # Response: {"data": [{"embedding": [...], "index": 0}, ...]}
        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        return [list(item["embedding"]) for item in sorted_items]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class QdrantEmulator(MemoryBackend):
    """In-memory cosine similarity store backed by Ollama embeddings.

    When Ollama is unavailable, automatically falls back to TF-IDF cosine
    so the benchmark suite can still run without Ollama for framework testing.
    """

    def __init__(self) -> None:
        # List of (original_text, embedding_or_empty)
        self._store: list[tuple[str, list[float]]] = []
        self._ollama_ok: bool = True

    @property
    def name(self) -> str:
        return "qdrant_emulator"

    async def reset(self) -> None:
        self._store.clear()
        self._ollama_ok = True

    async def insert(self, text: str) -> InsertResult:
        t0 = time.perf_counter()
        emb: list[float] = []
        if self._ollama_ok:
            try:
                emb = await embed_text(text)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout):
                self._ollama_ok = False
        self._store.append((text, emb))
        return InsertResult(
            facts_stored=len(self._store),
            facts_extracted=1,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        t0 = time.perf_counter()
        texts = [t for t, _ in self._store]

        if not self._ollama_ok or any(len(e) == 0 for _, e in self._store):
            ranked = _tfidf_cosine(query, texts, top_k)
            return RetrievalResult(
                texts=[t for t, _ in ranked],
                scores=[s for _, s in ranked],
                retrieve_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            q_emb = await embed_text(query)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout):
            self._ollama_ok = False
            ranked = _tfidf_cosine(query, texts, top_k)
            return RetrievalResult(
                texts=[t for t, _ in ranked],
                scores=[s for _, s in ranked],
                retrieve_ms=(time.perf_counter() - t0) * 1000,
            )

        scored = [(t, _cosine(q_emb, e)) for t, e in self._store]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        return RetrievalResult(
            texts=[t for t, _ in top],
            scores=[s for _, s in top],
            retrieve_ms=(time.perf_counter() - t0) * 1000,
        )
