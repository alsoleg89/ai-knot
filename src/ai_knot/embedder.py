"""Lightweight async embedder using Ollama's OpenAI-compatible /v1/embeddings endpoint.

Uses httpx (already a core dependency) — no extra packages required.
Gracefully degrades: returns empty lists when Ollama is unreachable so that
callers (e.g. learn_async semantic dedup) can fall back to lexical-only mode.
"""

from __future__ import annotations

import asyncio
import logging
import math

import httpx

logger = logging.getLogger(__name__)

# Module-level shared client and semaphore — reuses the connection pool and
# prevents saturating the GPU with concurrent embedding requests.
_HTTP: httpx.AsyncClient | None = None
_SEM = asyncio.Semaphore(1)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_TIMEOUT = 30.0


def _get_client() -> httpx.AsyncClient:
    global _HTTP
    if _HTTP is None:
        _HTTP = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
    return _HTTP


async def embed_texts(
    texts: list[str],
    *,
    base_url: str = _DEFAULT_BASE_URL,
    model: str = _DEFAULT_MODEL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[list[float]]:
    """Embed *texts* via Ollama's /v1/embeddings endpoint.

    Returns a list of float vectors, one per input text.
    Returns an empty list (not raises) on any connection or HTTP error so
    callers can fall back to lexical-only logic without special-casing.

    Args:
        texts: Strings to embed.  Empty input returns [] immediately.
        base_url: Base URL of the Ollama server (default: localhost:11434).
        model: Embedding model name (default: nomic-embed-text).
        timeout: Per-request timeout in seconds.
    """
    if not texts:
        return []
    url = f"{base_url.rstrip('/')}/v1/embeddings"
    client = _get_client()
    try:
        async with _SEM:
            resp = await client.post(
                url,
                headers={"Authorization": "Bearer ollama", "Content-Type": "application/json"},
                json={"model": model, "input": texts},
                timeout=timeout,
            )
            resp.raise_for_status()
        data = resp.json()
        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        return [list(item["embedding"]) for item in sorted_items]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, KeyError) as exc:
        logger.debug("embed_texts: Ollama unavailable (%s) — skipping semantic dedup", exc)
        return []


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
