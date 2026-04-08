"""ai-knot backend: BM25/TF-IDF retrieval with LLM extraction.

Wraps the production KnowledgeBase using SQLiteStorage in a temp dir.
Supports tick_decay() to simulate time passage for S5 decay scenario.

M5 Pro optimizations:
- SQLite WAL mode: dramatically better concurrent read performance on APFS/NVMe
- Extractor wrapped in run_in_executor so the asyncio event loop isn't blocked
  during LLM HTTP calls
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
from datetime import UTC, datetime, timedelta

from ai_knot.knowledge import KnowledgeBase
from ai_knot.providers.base import LLMProvider
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import ConversationTurn
from tests.eval.benchmark._eval_utils import estimate_extraction_tokens
from tests.eval.benchmark.base import InsertResult, MemoryBackend, RetrievalResult


class AiKnotBackend(MemoryBackend):
    """Production ai-knot KnowledgeBase with BM25+TF-IDF retrieval.

    Args:
        provider: LLM provider for fact extraction. Pass StubProvider for
            mock runs where extraction is not needed.
        use_add: When True, bypass LLM extraction and use kb.add() directly.
            Useful for load testing or when the provider is a stub.
    """

    def __init__(self, provider: LLMProvider, *, use_add: bool = False) -> None:
        self._provider = provider
        self._use_add = use_add
        self._tmp_dir: str = ""
        self._kb: KnowledgeBase | None = None
        self._session_seen: set[str] = set()

    @property
    def name(self) -> str:
        return "ai_knot"

    async def reset(self) -> None:
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = tempfile.mkdtemp(prefix="aiknot_bench_")
        db_path = f"{self._tmp_dir}/bench.db"

        # Enable WAL mode before SQLiteStorage opens the DB — APFS/NVMe on M5 Pro
        # benefits significantly from WAL for concurrent reads during S6.
        _conn = sqlite3.connect(db_path)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.commit()
        _conn.close()

        storage = SQLiteStorage(db_path)
        self._kb = KnowledgeBase("bench", storage=storage, provider=self._provider)
        self._session_seen = set()

    async def insert(self, text: str) -> InsertResult:
        assert self._kb is not None, "call reset() before insert()"
        t0 = time.perf_counter()

        if self._use_add:
            self._kb.add(text)
            stored = len(self._kb.list_facts())
            return InsertResult(
                facts_stored=stored,
                facts_extracted=1,
                insert_ms=(time.perf_counter() - t0) * 1000,
                extraction_tokens=0,
            )

        turns = [ConversationTurn(role="user", content=text)]
        new_facts = await self._kb.learn_async(turns)
        stored = len(self._kb.list_facts())
        return InsertResult(
            facts_stored=stored,
            facts_extracted=len(new_facts),
            insert_ms=(time.perf_counter() - t0) * 1000,
            extraction_tokens=estimate_extraction_tokens(text),
        )

    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        assert self._kb is not None, "call reset() before retrieve()"
        t0 = time.perf_counter()
        pairs = self._kb.recall_facts_with_scores(
            query, top_k=top_k, excluded_ids=self._session_seen
        )
        self._session_seen.update(f.id for f, _ in pairs)
        return RetrievalResult(
            texts=[f.answer_surface for f, _ in pairs],
            scores=[s for _, s in pairs],
            retrieve_ms=(time.perf_counter() - t0) * 1000,
            evidence_texts=[f.evidence_surface for f, _ in pairs],
        )

    async def count_stored(self) -> int | None:
        """Return count of currently valid facts — used by S4 for accurate dedup measurement."""
        if self._kb is None:
            return 0
        return sum(1 for f in self._kb.list_facts() if f.is_active())

    async def reset_session(self) -> None:
        """Clear session_seen so the next retrieval starts with no exclusions."""
        self._session_seen = set()

    async def tick_decay(self, *, hours: float) -> None:
        """Simulate passage of `hours` by backdating last_accessed on all facts."""
        assert self._kb is not None, "call reset() before tick_decay()"
        facts = self._kb.list_facts()
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        for f in facts:
            f.last_accessed = cutoff
        self._kb.replace_facts(facts)
        self._kb.decay()

    def __del__(self) -> None:
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


class AiKnotNoLlmBackend(AiKnotBackend):
    """ai-knot with direct kb.add() — no LLM extraction.

    Stores facts verbatim via add() instead of going through the Extractor.
    This isolates the contribution of BM25 retrieval from LLM extraction,
    making it a clean control condition between raw-storage backends (baseline,
    qdrant) and the full extraction pipeline (ai_knot).
    """

    def __init__(self) -> None:
        from tests.eval.benchmark._stub_provider import StubProvider

        super().__init__(StubProvider(), use_add=True)

    @property
    def name(self) -> str:
        return "ai_knot_no_llm"
