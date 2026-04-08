"""Helper to check if Ollama is available before running the benchmark."""

from __future__ import annotations


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Return True if Ollama is running and reachable."""
    import httpx

    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False
