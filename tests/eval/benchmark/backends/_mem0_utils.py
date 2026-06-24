"""Shared utilities for mem0-based benchmark backends."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

# Single semaphore for all mem0 backends — Chroma uses SQLite which is not
# safe for concurrent writes even across separate collection names.
MEM0_SEM = asyncio.Semaphore(1)


def mem0_chroma_path(suffix: str) -> str:
    return str(Path(tempfile.gettempdir()) / f"mem0_{suffix}_chroma")


def normalize_mem0_results(response: object) -> list[dict[str, object]]:
    """Normalize mem0 API response across versions.

    ≤0.1.x returns {"results": [...]}, ≥0.1.y returns [...] directly.
    """
    if isinstance(response, list):
        return response  # type: ignore[return-value]
    if isinstance(response, dict):
        return response.get("results", [])  # type: ignore[return-value]
    return []


def build_ollama_mem0_config(*, collection_name: str, chroma_path: str) -> dict[str, object]:
    """Return a mem0 Memory.from_config()-compatible dict using local Ollama."""
    return {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "qwen2.5:7b",
                "ollama_base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "qwen2.5:7b",
                "ollama_base_url": "http://localhost:11434",
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": collection_name,
                "path": chroma_path,
            },
        },
    }
