"""Quality gate tests — run as part of eval-smoke CI job.

These tests assert minimum retrieval quality thresholds on the golden dataset.
A failure here means BM25 retrieval quality has regressed below acceptable levels.

Thresholds (intentionally lenient — minimum sanity, not performance targets):
  MRR  >= 0.50  (random baseline ≈ 0.17 for top-1 of 5)
  P@5  >= 0.30
"""

from __future__ import annotations

from tests.eval.runner import run_eval


def test_mrr_above_threshold() -> None:
    """MRR must be >= 0.50 on the golden dataset."""
    result = run_eval()
    mrr = result["mrr"]["mean"]
    assert mrr >= 0.50, f"MRR {mrr:.3f} below 0.50 threshold — retrieval quality regression"


def test_precision_above_threshold() -> None:
    """Precision@5 must be >= 0.30 on the golden dataset."""
    result = run_eval()
    p5 = result["precision"]["mean"]
    assert p5 >= 0.30, f"P@5 {p5:.3f} below 0.30 threshold — retrieval quality regression"
