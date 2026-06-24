"""Async embedder for any OpenAI-compatible /v1/embeddings endpoint.

Uses httpx (already a core dependency) — no extra packages required.
Gracefully degrades: returns empty lists when the endpoint is unreachable so
callers (e.g. learn_async semantic dedup) can fall back to lexical-only mode.
"""

from __future__ import annotations

import logging
import math

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_TIMEOUT = 30.0


async def embed_texts(
    texts: list[str],
    *,
    base_url: str = _DEFAULT_BASE_URL,
    model: str = _DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[list[float]]:
    """Embed *texts* via an OpenAI-compatible /v1/embeddings endpoint.

    Returns a list of float vectors, one per input text.
    Returns an empty list (not raises) on any connection or HTTP error so
    callers can fall back to lexical-only logic without special-casing.

    A fresh ``httpx.AsyncClient`` is created per call so this function is safe
    to use from any event loop, including ones created by ``asyncio.run()`` in
    a thread pool (the previous module-level singleton caused
    ``RuntimeError: Future attached to a different loop`` in that context).

    Args:
        texts: Strings to embed.  Empty input returns [] immediately.
        base_url: Base URL of the embedding server (default: localhost:11434 for Ollama).
            Use ``https://api.openai.com`` for OpenAI embeddings.
        model: Embedding model name (default: nomic-embed-text).
            Use ``text-embedding-3-small`` or ``text-embedding-3-large`` for OpenAI.
        api_key: Bearer token for the endpoint.  Defaults to ``"ollama"`` (no-op
            sentinel accepted by Ollama).  Pass an OpenAI key when using OpenAI.
        timeout: Per-request timeout in seconds.
    """
    if not texts:
        return []
    url = f"{base_url.rstrip('/')}/v1/embeddings"
    bearer = api_key or "ollama"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
                json={"model": model, "input": texts},
                timeout=timeout,
            )
            resp.raise_for_status()
        data = resp.json()
        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        return [list(item["embedding"]) for item in sorted_items]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, KeyError) as exc:
        logger.debug("embed_texts: embedding endpoint unavailable (%s) — skipping", exc)
        return []


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
