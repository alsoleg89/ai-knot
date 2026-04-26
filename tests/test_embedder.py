"""Unit tests for ai_knot.embedder — async OpenAI-compatible embedder."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from ai_knot.embedder import cosine, embed_texts

# ---- cosine -----------------------------------------------------------------


class TestCosine:
    def test_identical_vectors_score_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_norm_returns_zero(self) -> None:
        # Either argument with zero norm short-circuits to 0.0 to avoid div-by-zero.
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine([1.0, 1.0], [0.0, 0.0]) == 0.0


# ---- embed_texts ------------------------------------------------------------


@pytest.mark.real_embedder  # opt out of the conftest stub for these tests
class TestEmbedTextsErrorPaths:
    """The embedder must degrade gracefully (return []) on any failure mode."""

    def test_empty_input_returns_empty_immediately(self) -> None:
        assert asyncio.run(embed_texts([])) == []

    def test_connect_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Point at a definitely-unreachable address; httpx raises ConnectError
        # which the embedder catches and converts to [].
        result = asyncio.run(embed_texts(["some text"], base_url="http://127.0.0.1:1", timeout=0.5))
        assert result == []

    def test_http_status_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _BadResponse:
            status_code = 500

            def raise_for_status(self) -> None:
                raise httpx.HTTPStatusError(
                    "boom",
                    request=httpx.Request("POST", "http://x"),
                    response=self,  # type: ignore[arg-type]
                )

            def json(self) -> dict[str, Any]:
                return {}

        class _StubClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> _StubClient:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def post(self, *args: Any, **kwargs: Any) -> _BadResponse:
                return _BadResponse()

        monkeypatch.setattr("ai_knot.embedder.httpx.AsyncClient", _StubClient)
        result = asyncio.run(embed_texts(["hello"], base_url="http://stubbed"))
        assert result == []

    def test_malformed_response_missing_data_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _MalformedResponse:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                return {"unexpected": "shape"}

        class _StubClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> _StubClient:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def post(self, *args: Any, **kwargs: Any) -> _MalformedResponse:
                return _MalformedResponse()

        monkeypatch.setattr("ai_knot.embedder.httpx.AsyncClient", _StubClient)
        result = asyncio.run(embed_texts(["hello"], base_url="http://stubbed"))
        assert result == []

    def test_successful_response_parsed_in_index_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _OkResponse:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                # Intentionally out-of-order; embedder must sort by `index`.
                return {
                    "data": [
                        {"index": 1, "embedding": [0.4, 0.5]},
                        {"index": 0, "embedding": [0.1, 0.2]},
                    ]
                }

        class _StubClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> _StubClient:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def post(self, *args: Any, **kwargs: Any) -> _OkResponse:
                return _OkResponse()

        monkeypatch.setattr("ai_knot.embedder.httpx.AsyncClient", _StubClient)
        result = asyncio.run(embed_texts(["a", "b"], base_url="http://stubbed"))
        assert result == [[0.1, 0.2], [0.4, 0.5]]
