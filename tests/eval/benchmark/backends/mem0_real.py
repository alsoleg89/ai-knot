"""Real mem0 backend: mem0ai library + Ollama LLM/embedder + Chroma vector store.

Ported from ContentOs/scripts/ai_knot_eval/backends/mem0_backend.py.
Changes from ContentOs version:
  - Yandex GPT + Yandex embedder → Ollama llama3.2:3b (LLM + embedder)
  - No API key required (Ollama is local)
  - insert(text) interface instead of insert_facts(list[str])
  - collection_name: "mem0_bench", path: /tmp/mem0_bench_chroma
  - Kept Semaphore(1) for Chroma writes (SQLite not thread-safe)
  - Kept role=user fix (mem0 extraction ignores assistant messages)
  - Kept response_format=None patch (Ollama may not support JSON mode)
"""

from __future__ import annotations

import asyncio
import functools
import tempfile
import time
from pathlib import Path

from tests.eval.benchmark.base import InsertResult, MemoryBackend, RetrievalResult

_DEFAULT_CHROMA_PATH = str(Path(tempfile.gettempdir()) / "mem0_bench_chroma")
_SEM = asyncio.Semaphore(1)  # Chroma/SQLite not safe for concurrent writes


class Mem0RealBackend(MemoryBackend):
    """Real mem0 with Ollama LLM/embedder and Chroma vector store."""

    def __init__(self, chroma_path: str = _DEFAULT_CHROMA_PATH) -> None:
        self._chroma_path = chroma_path
        self._memory: object | None = None
        self._user_id = "ai_knot_bench"
        self._stored_count = 0

    @property
    def name(self) -> str:
        return "mem0_real"

    def _build_config(self) -> dict[str, object]:
        return {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3.2:3b",
                    "ollama_base_url": "http://localhost:11434",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "llama3.2:3b",
                    "ollama_base_url": "http://localhost:11434",
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "mem0_bench",
                    "path": self._chroma_path,
                },
            },
        }

    async def _get_memory(self) -> object:
        if self._memory is None:
            await self._setup()
        return self._memory  # type: ignore[return-value]

    async def _setup(self) -> None:
        try:
            from mem0 import Memory  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "mem0ai not installed. Run: pip install -e '.[benchmark]'"
            ) from e

        config = self._build_config()
        self._memory = Memory.from_config(config)

        # Patch out response_format — Ollama may not support JSON mode.
        # mem0 v1.0.9+ always passes response_format={"type":"json_object"} internally.
        _orig = self._memory.llm.generate_response  # type: ignore[union-attr]

        def _generate_no_response_format(  # noqa: ANN202
            messages: object, response_format: object = None, **kwargs: object
        ) -> object:
            return _orig(messages, response_format=None, **kwargs)

        self._memory.llm.generate_response = _generate_no_response_format  # type: ignore[union-attr]

    async def reset(self) -> None:
        memory = await self._get_memory()
        import contextlib

        with contextlib.suppress(Exception):
            memory.delete_all(user_id=self._user_id)  # type: ignore[union-attr]
        self._stored_count = 0

    async def insert(self, text: str) -> InsertResult:
        memory = await self._get_memory()
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()

        async with _SEM:
            try:
                # role=user is REQUIRED — mem0's extraction prompt explicitly ignores
                # assistant/system messages. role=assistant → 0 facts stored.
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(
                            memory.add,  # type: ignore[union-attr]
                            [{"role": "user", "content": text}],
                            user_id=self._user_id,
                        ),
                    ),
                    timeout=60.0,
                )
                facts_extracted = len(result.get("results", [])) if result else 0
                self._stored_count += facts_extracted
            except Exception:
                facts_extracted = 0

        return InsertResult(
            facts_stored=self._stored_count,
            facts_extracted=facts_extracted,
            insert_ms=(time.perf_counter() - t0) * 1000,
        )

    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        memory = await self._get_memory()
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    functools.partial(
                        memory.search,  # type: ignore[union-attr]
                        query,
                        user_id=self._user_id,
                        limit=top_k,
                    ),
                ),
                timeout=30.0,
            )
            texts = [r.get("memory", str(r)) for r in (results or [])]
            scores = [r.get("score", 0.0) for r in (results or [])]
        except Exception:
            texts, scores = [], []

        return RetrievalResult(
            texts=texts,
            scores=scores,
            retrieve_ms=(time.perf_counter() - t0) * 1000,
        )
