"""Shared embedding and scoring utilities for professional benchmark scenarios (S1–S8).

Embedding: Ollama qwen2.5:7b (same model as qdrant_emulator).
Fallback: ATC token containment when Ollama is unavailable.
"""

from __future__ import annotations

from ai_knot.embedder import cosine as _cosine
from ai_knot.embedder import embed_texts as _embed_texts
from ai_knot.tokenizer import tokenize

_EMBED_MODEL = "qwen2.5:7b"


async def maybe_embed_batch(texts: list[str]) -> list[list[float]] | None:
    """Batch-embed via Ollama; return None if Ollama is unreachable."""
    result = await _embed_texts(texts, model=_EMBED_MODEL, timeout=60.0)
    return result if result else None


def atc_score(snippet: str, source: str) -> float:
    """Asymmetric Token Containment: fraction of snippet tokens found in source.

    Returns 1.0 if snippet is empty (vacuously supported).
    """
    s_tokens = set(tokenize(snippet))
    if not s_tokens:
        return 1.0
    src_tokens = set(tokenize(source))
    return len(s_tokens & src_tokens) / len(s_tokens)


def best_atc_against(candidate: str, targets: list[str]) -> float:
    """Max ATC score from any target against the candidate."""
    if not targets:
        return 0.0
    return max(atc_score(t, candidate) for t in targets)


def hit_rank_lexical(
    relevant_text: str,
    retrieved: list[str],
    *,
    threshold: float = 0.5,
) -> int | None:
    """Return 1-based rank of the first retrieved text that 'hits' the relevant text.

    Hit criterion (ATC-based, deterministic):
      max(atc(relevant→retrieved), atc(retrieved→relevant)) >= threshold
    Returns None if no hit found.
    """
    for i, r in enumerate(retrieved):
        score = max(atc_score(relevant_text, r), atc_score(r, relevant_text))
        if score >= threshold:
            return i + 1
    return None


def hit_rank_semantic(
    relevant_emb: list[float],
    retrieved_embs: list[list[float]],
    *,
    threshold: float = 0.65,
) -> int | None:
    """Return 1-based rank of first retrieved embedding with cosine >= threshold."""
    for i, emb in enumerate(retrieved_embs):
        if _cosine(relevant_emb, emb) >= threshold:
            return i + 1
    return None


def mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank from a list of 1-based ranks (None = not found)."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def precision_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of queries where the relevant fact is in top-k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def percentile(data: list[float], p: float) -> float:
    """p-th percentile of data using linear interpolation."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def estimate_extraction_tokens(text: str) -> int:
    """Approximate input tokens consumed by LLM extraction.

    Uses word count × 2 to account for system prompt overhead.
    This is a heuristic — actual token counts depend on the model and tokenizer.
    """
    return len(text.split()) * 2


def has_domain_hit(texts: list[str], keywords: set[str]) -> bool:
    """Return True if any text in *texts* contains at least one keyword from *keywords*.

    Strips common punctuation before matching so "kubernetes," matches "kubernetes".
    Used by MA isolation/leakage scenarios to detect domain overlap.
    """
    return any({w.lower().strip(".,;:()") for w in text.split()} & keywords for text in texts)
