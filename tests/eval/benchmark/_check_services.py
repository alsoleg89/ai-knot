"""Service availability checks for extended benchmark mode."""

from __future__ import annotations


def check_qdrant_available(host: str = "localhost", port: int = 6333) -> bool:
    """Return True if Qdrant is running and reachable."""
    import httpx

    try:
        r = httpx.get(f"http://{host}:{port}/healthz", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def check_mem0_available() -> bool:
    """Return True if mem0ai is installed and Ollama is reachable."""
    try:
        import mem0  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False

    from tests.eval.benchmark._check_ollama import check_ollama_available

    return check_ollama_available()


def check_qdrant_client_installed() -> bool:
    """Return True if qdrant-client package is installed."""
    try:
        import qdrant_client  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False
