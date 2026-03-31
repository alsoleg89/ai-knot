"""Eval runner — runs retrieval cases and reports metrics."""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot.retriever import BM25Retriever
from ai_knot.types import Fact, MemoryType
from tests.eval.datasets import RETRIEVAL_DATASET, RetrievalCase
from tests.eval.metrics import (
    bootstrap_ci,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _make_fact(d: dict[str, object]) -> Fact:
    return Fact(
        id=str(d["id"]),
        content=str(d["content"]),
        type=MemoryType.SEMANTIC,
        importance=float(d.get("importance", 0.8)),  # type: ignore[arg-type]
        access_count=int(d.get("access_count", 0)),  # type: ignore[arg-type]
        retention_score=1.0,
        last_accessed=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )


def run_case(case: RetrievalCase, retriever: BM25Retriever, k: int = 5) -> dict[str, float]:
    facts = [_make_fact(d) for d in case.facts]
    relevant = set(case.relevant_ids)
    results = retriever.search(case.query, facts, top_k=k)
    retrieved_ids = [f.id for f, _ in results]
    return {
        "precision": precision_at_k(retrieved_ids, relevant, k),
        "recall": recall_at_k(retrieved_ids, relevant, k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant),
        "ndcg": ndcg_at_k(retrieved_ids, relevant, k),
    }


def run_eval(
    dataset: list[RetrievalCase] = RETRIEVAL_DATASET, k: int = 5
) -> dict[str, dict[str, float]]:
    retriever = BM25Retriever()
    all_metrics: dict[str, list[float]] = {"precision": [], "recall": [], "mrr": [], "ndcg": []}
    for case in dataset:
        result = run_case(case, retriever, k=k)
        for metric, val in result.items():
            all_metrics[metric].append(val)

    summary: dict[str, dict[str, float]] = {}
    for metric, scores in all_metrics.items():
        mean = sum(scores) / len(scores) if scores else 0.0
        lo, hi = bootstrap_ci(scores)
        summary[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(run_eval(), indent=2))
