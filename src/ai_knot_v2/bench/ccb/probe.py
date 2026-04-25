"""Extraction-sufficiency probe — bench/ only, never imported in core/ops/store/api/.

Validates that a rendered atom pack carries enough information for a reader to
answer a given question (I(y*_Q; π_R(Q, render(W_Q))) ≥ (1−ε)·H(y*_Q)).
"""

from __future__ import annotations

from typing import Any, Protocol

from ai_knot_v2.bench.ccb.render import render_pack_eswp
from ai_knot_v2.core.atom import MemoryAtom


class _ReaderProtocol(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between predicted and gold answer strings."""
    pred_toks = set(pred.lower().split())
    gold_toks = set(gold.lower().split())
    if not gold_toks:
        return 0.0
    prec = len(pred_toks & gold_toks) / max(1, len(pred_toks))
    rec = len(pred_toks & gold_toks) / len(gold_toks)
    denom = prec + rec
    return 2 * prec * rec / denom if denom > 0 else 0.0


def validate_extraction_sufficiency(
    atoms: list[MemoryAtom],
    query: str,
    reader: Any,
    expected_answer: str | None = None,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Call reader on rendered pack; return (is_sufficient, score).

    reader must expose .complete(system: str, user: str) -> str.
    This function must remain in bench/ — never call from core/.
    """
    rendered = render_pack_eswp(atoms, query)
    response: str = reader.complete(
        system="Answer the question from the context below. Be concise and direct.",
        user=f"Context:\n{rendered}\n\nQuestion: {query}",
    )

    if expected_answer is not None:
        score = token_f1(response, expected_answer)
    else:
        q_words = {w.lower() for w in query.split() if len(w) > 3}
        r_words = {w.lower() for w in response.split()}
        score = len(q_words & r_words) / max(1, len(q_words))

    return score >= threshold, score
