"""Tests for ai_knot.embedder.

Regression coverage for the event-loop isolation bug:
  Module-level asyncio.Semaphore / AsyncClient singletons caused
  ``RuntimeError: Future attached to a different loop`` when embed_texts()
  was called via asyncio.run() inside a ThreadPoolExecutor (the path taken
  by KnowledgeBase._embed_for_recall under MCP server's running loop).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock, patch


def _fake_response(vectors: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}
    return resp


def _make_mock_client(vectors: list[list[float]]) -> MagicMock:
    """Return a mock AsyncClient whose post() returns the given vectors."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_fake_response(vectors))
    return mock_client


class TestEmbedTexts:
    """Unit tests for embed_texts()."""

    def test_empty_input_returns_empty(self) -> None:
        from ai_knot.embedder import embed_texts

        result = asyncio.run(embed_texts([]))
        assert result == []

    def test_returns_vectors_on_success(self) -> None:
        from ai_knot.embedder import embed_texts

        expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client = _make_mock_client(expected)

        with patch("ai_knot.embedder.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(embed_texts(["hello", "world"]))

        assert result == expected

    def test_sends_bearer_token(self) -> None:
        from ai_knot.embedder import embed_texts

        mock_client = _make_mock_client([[0.1]])

        with patch("ai_knot.embedder.httpx.AsyncClient", return_value=mock_client):
            asyncio.run(
                embed_texts(["hello"], api_key="sk-test-key", base_url="https://api.openai.com")
            )

        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-test-key"

    def test_ollama_default_bearer(self) -> None:
        from ai_knot.embedder import embed_texts

        mock_client = _make_mock_client([[0.1]])

        with patch("ai_knot.embedder.httpx.AsyncClient", return_value=mock_client):
            asyncio.run(embed_texts(["hello"]))

        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer ollama"

    def test_openai_url_construction(self) -> None:
        from ai_knot.embedder import embed_texts

        mock_client = _make_mock_client([[0.1]])

        with patch("ai_knot.embedder.httpx.AsyncClient", return_value=mock_client):
            asyncio.run(embed_texts(["hello"], base_url="https://api.openai.com", api_key="sk-x"))

        url = mock_client.post.call_args.args[0]
        assert url == "https://api.openai.com/v1/embeddings"

    def test_connect_error_returns_empty(self) -> None:
        import httpx

        from ai_knot.embedder import embed_texts

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("ai_knot.embedder.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(embed_texts(["hello"]))

        assert result == []

    def test_http_status_error_returns_empty(self) -> None:
        import httpx

        from ai_knot.embedder import embed_texts

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock()
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("ai_knot.embedder.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(embed_texts(["hello"]))

        assert result == []


class TestEmbedTextsEventLoopIsolation:
    """Regression tests for the 'Future attached to a different loop' bug.

    Before the fix, embed_texts() used a module-level asyncio.Semaphore and
    AsyncClient.  These are bound to the event loop that was active when they
    were created.  Calling embed_texts() via asyncio.run() inside a
    ThreadPoolExecutor (the path taken by KnowledgeBase._embed_for_recall
    when the MCP server's loop is already running) created a *new* event loop
    in the thread, causing:

        RuntimeError: Task <...> got Future <asyncio.locks.Semaphore ...>
        attached to a different loop

    The fix removes the singletons: each call creates its own AsyncClient as
    a context manager, making the function safe to use from any event loop.
    """

    def test_callable_from_thread_pool_asyncio_run(self) -> None:
        """embed_texts() must not raise when called via asyncio.run() in a thread.

        Simulates the exact call site in KnowledgeBase._embed_for_recall:
          - outer event loop is running (MCP server context)
          - embed_texts is dispatched through ThreadPoolExecutor + asyncio.run()
        """
        from ai_knot.embedder import embed_texts

        expected = [[0.1, 0.2]]
        mock_client = _make_mock_client(expected)

        async def outer() -> list[list[float]]:
            """Simulate a running event loop (e.g. FastMCP server)."""
            loop = asyncio.get_running_loop()
            with (
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
                patch("ai_knot.embedder.httpx.AsyncClient", return_value=mock_client),
            ):
                result: list[list[float]] = await loop.run_in_executor(
                    pool,
                    lambda: asyncio.run(embed_texts(["hello"])),
                )
            return result

        result = asyncio.run(outer())
        assert result == expected

    def test_sequential_calls_in_different_loops(self) -> None:
        """Three sequential asyncio.run() calls each create their own event loop — no errors.

        This verifies that embed_texts() has no module-level asyncio objects
        (Semaphore, AsyncClient) that would be invalidated when a new loop is
        created.  Each asyncio.run() creates a distinct loop; if a singleton
        were bound to the first loop, the second call would raise
        ``RuntimeError: Future attached to a different loop``.

        Tests are sequential (not concurrent) to avoid race conditions from
        patching the same module attribute in multiple threads simultaneously.
        """
        from ai_knot.embedder import embed_texts

        mock_client = _make_mock_client([[0.1, 0.2]])

        with patch("ai_knot.embedder.httpx.AsyncClient", return_value=mock_client):
            for _ in range(3):
                result = asyncio.run(embed_texts(["test"]))
                assert result == [[0.1, 0.2]]
