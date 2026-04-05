"""LLM judge for benchmark evaluation.

Scores retrieved text on three dimensions (1-5 scale):
  relevance    — does the retrieved content answer the query?
  completeness — does it cover all key aspects?
  faithfulness — is the retrieved text accurate and non-hallucinated?

OllamaJudge runs JUDGE_RUNS independent calls per metric and returns the raw
list so callers can compute median/stdev themselves. On M5 Pro Metal the 3 runs
per metric are fired concurrently via asyncio.gather (Ollama handles parallel
inference). The synchronous score() wrapper is kept for backward compatibility.
"""

from __future__ import annotations

import abc
import asyncio
import re
import statistics

import httpx

OLLAMA_BASE_URL = "http://localhost:11434/v1"
JUDGE_MODEL = "qwen2.5:7b"
JUDGE_RUNS = 3

_SYSTEM_PROMPT = """You are an evaluation judge for a memory retrieval system.
Given a query and retrieved text snippets, score the retrieval on the requested
dimension from 1 (very poor) to 5 (excellent).

Return ONLY a single integer: 1, 2, 3, 4, or 5. Nothing else."""

_USER_TEMPLATE = """Query: {query}

Retrieved text:
{context}

Score the {metric} of this retrieval (1=very poor, 5=excellent):"""

ALL_METRICS = ("relevance", "completeness", "faithfulness")


def _parse_score(raw: str) -> float:
    """Extract a 1-5 score from raw LLM output.

    Handles common LLM response styles:
      "4"  /  "4."  /  "Score: 4"  /  "The retrieval scores 4 out of 5."
    Falls back to neutral 3.0 (not 1.0) when no digit is found, so parse
    failures don't systematically bias rankings downward.
    """
    # Fast path: first token is already a bare digit
    try:
        val = float(raw.strip().split()[0].rstrip(".,:"))
        if 1.0 <= val <= 5.0:
            return val
    except (ValueError, IndexError):
        pass
    # Regex fallback: find the first standalone 1-5 digit in the response
    m = re.search(r"\b([1-5])\b", raw)
    if m:
        return float(m.group(1))
    return 3.0  # neutral — no score found


def score_stats(scores: list[float]) -> tuple[float, float]:
    """Return (median, stdev) for a list of scores."""
    med = float(statistics.median(scores))
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return med, std


class BaseJudge(abc.ABC):
    """Abstract judge interface."""

    @abc.abstractmethod
    async def score_all_async(
        self,
        query: str,
        retrieved_texts: list[str],
    ) -> dict[str, list[float]]:
        """Score all metrics; return {metric: [raw_run1, raw_run2, raw_run3]}."""
        ...

    def score_all(
        self,
        query: str,
        retrieved_texts: list[str],
    ) -> dict[str, list[float]]:
        """Synchronous wrapper around score_all_async (blocks — avoid in hot paths)."""
        try:
            asyncio.get_running_loop()
            # We're already inside an event loop (e.g. scenario coroutine).
            # Run in a thread pool executor to avoid nested-loop error.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.score_all_async(query, retrieved_texts))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.score_all_async(query, retrieved_texts))


class OllamaJudge(BaseJudge):
    """Async Ollama judge — fires JUDGE_RUNS calls per metric concurrently.

    On M5 Pro Metal, Ollama handles 3 parallel inference requests efficiently.
    """

    def __init__(
        self,
        *,
        model: str = JUDGE_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        runs: int = JUDGE_RUNS,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._runs = runs
        self._timeout = timeout

    async def _call_async(self, user_content: str) -> float:
        """Single async LLM call; returns parsed score."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": "Bearer ollama", "Content-Type": "application/json"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            raw = str(resp.json()["choices"][0]["message"]["content"])
            return _parse_score(raw)

    async def _score_metric_async(self, query: str, context: str, metric: str) -> list[float]:
        """Fire JUDGE_RUNS calls concurrently and return raw scores."""
        user_content = _USER_TEMPLATE.format(query=query, context=context, metric=metric)
        # M5 Pro Metal: run all judge calls concurrently
        scores = await asyncio.gather(
            *[self._call_async(user_content) for _ in range(self._runs)],
            return_exceptions=True,
        )
        result: list[float] = []
        for s in scores:
            result.append(s if isinstance(s, float) else 1.0)
        return result

    async def score_all_async(
        self,
        query: str,
        retrieved_texts: list[str],
    ) -> dict[str, list[float]]:
        context = "\n---\n".join(retrieved_texts) if retrieved_texts else "(no results)"
        # Run all 3 metrics concurrently — 9 Ollama calls total, fired in parallel
        results = await asyncio.gather(
            *[self._score_metric_async(query, context, m) for m in ALL_METRICS]
        )
        return dict(zip(ALL_METRICS, results, strict=False))


class MockJudge(BaseJudge):
    """Deterministic judge for offline CI. Returns fixed scores, zero variance."""

    _FIXED: dict[str, float] = {
        "relevance": 4.0,
        "completeness": 3.5,
        "faithfulness": 4.5,
    }

    async def score_all_async(
        self,
        query: str,
        retrieved_texts: list[str],
    ) -> dict[str, list[float]]:
        return {m: [self._FIXED.get(m, 3.0)] * JUDGE_RUNS for m in ALL_METRICS}
