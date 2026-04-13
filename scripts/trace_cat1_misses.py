"""Classify recall pipeline failures for WRONG Cat1 (or Cat2) QA answers.

Uses existing benchmark run DBs — no re-ingestion, no LLM calls (except
embeddings for recall which are fast local Ollama calls).

Usage
-----
    .venv/bin/python scripts/trace_cat1_misses.py \
        --run-ids post-freeze-0,post-freeze-1,post-freeze-2

    # Cat2, different run set
    .venv/bin/python scripts/trace_cat1_misses.py \
        --run-ids post-freeze-0,post-freeze-1,post-freeze-2 --category 2

    # Single run
    .venv/bin/python scripts/trace_cat1_misses.py --run-ids post-freeze-0

Output
------
Each WRONG answer is classified into one of:
  extraction_miss      — gold fact not found in KB at all (extraction failed)
  pool_miss            — fact exists in KB but recall never considered it
  select_topk_drop     — in pool but dropped by greedy IDF-coverage select
  mmr_drop             — selected but MMR diversity dedup removed it
  answered_wrong_judge — recall found gold, but judge scored it WRONG anyway
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

# Ensure the project root is importable.
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from ai_knot.knowledge import KnowledgeBase  # noqa: E402
from ai_knot.storage.sqlite_storage import SQLiteStorage  # noqa: E402
from ai_knot.types import Fact  # noqa: E402

RUNS_DIR = _ROOT / "aiknotbench" / "data" / "runs"
SAMPLES_DIR = _ROOT / "scripts" / "trace_samples"

FUZZY_THRESHOLD_STRICT = 0.70
FUZZY_THRESHOLD_LENIENT = 0.50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fuzzy_match(gold: str, facts: list[Fact]) -> tuple[Fact | None, float]:
    """Return the fact whose content best matches gold_answer, with score."""
    gold_l = gold.lower().strip()
    best_fact: Fact | None = None
    best_score = 0.0
    for f in facts:
        score = difflib.SequenceMatcher(None, gold_l, f.content.lower()).ratio()
        if score > best_score:
            best_score = score
            best_fact = f
    return best_fact, best_score


def _classify(
    gold_fid: str,
    trace: dict[str, Any],
    result_ids: set[str],
) -> str:
    """Return bucket name given the gold fact ID and trace dict."""
    stage1 = trace.get("stage1_candidates", {})
    all_pool: set[str] = (
        set(stage1.get("from_bm25", []))
        | set(stage1.get("from_rare_tokens", []))
        | set(stage1.get("from_entity_hop", []))
        | set(stage1.get("from_dense", []))
    )
    if gold_fid not in all_pool:
        return "pool_miss"

    selected_ids = set(trace.get("stage3_rrf", {}).get("selected_ids", []))
    if gold_fid not in selected_ids:
        return "select_topk_drop"

    if gold_fid not in result_ids:
        return "mmr_drop"

    return "answered_wrong_judge"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-ids",
        required=True,
        help="Comma-separated list of run IDs (directories under aiknotbench/data/runs/)",
    )
    parser.add_argument("--category", type=int, default=1, help="QA category to trace (default: 1)")
    parser.add_argument("--top-k", type=int, default=60, help="top_k for recall (default: 60)")
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=FUZZY_THRESHOLD_STRICT,
        help=f"Min fuzzy ratio to count as extraction_miss (default: {FUZZY_THRESHOLD_STRICT})",
    )
    parser.add_argument(
        "--dump-samples", type=int, default=2, help="Samples per bucket to dump (default: 2)"
    )
    args = parser.parse_args()

    run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]

    # Collect WRONG QA records across all runs.
    wrongs: list[dict[str, Any]] = []
    for rid in run_ids:
        log_path = RUNS_DIR / rid / "log.jsonl"
        if not log_path.exists():
            print(f"WARNING: {log_path} not found — skipping", file=sys.stderr)
            continue
        with open(log_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("category") == args.category and rec.get("verdict") == "WRONG":
                    rec["_run_id"] = rid
                    wrongs.append(rec)

    if not wrongs:
        print(f"No WRONG answers found for category {args.category} in {run_ids}", file=sys.stderr)
        sys.exit(0)

    print(f"\nCat{args.category} WRONG answers: {len(wrongs)} (across {run_ids})")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_order = [
        "extraction_miss",
        "pool_miss",
        "select_topk_drop",
        "mmr_drop",
        "answered_wrong_judge",
    ]

    for rec in wrongs:
        rid = rec["_run_id"]
        conv_idx = rec["convIdx"]
        question = rec["question"]
        gold_answer = rec.get("goldAnswer", "")
        qa_idx = rec.get("qaIdx", "?")

        db_path = RUNS_DIR / rid / "knot.db"
        if not db_path.exists():
            print(f"  WARNING: DB not found at {db_path}", file=sys.stderr)
            continue

        storage = SQLiteStorage(db_path=str(db_path))
        kb = KnowledgeBase(agent_id=f"conv-{conv_idx}", storage=storage)

        all_facts = kb.list_facts()

        # Fuzzy-match gold answer against all facts.
        best_fact, best_score = _fuzzy_match(gold_answer, all_facts)

        if best_score < args.fuzzy_threshold:
            bucket = "extraction_miss"
            sample: dict[str, Any] = {
                "run_id": rid,
                "conv_idx": conv_idx,
                "qa_idx": qa_idx,
                "question": question,
                "gold_answer": gold_answer,
                "best_fact_content": best_fact.content if best_fact else None,
                "best_score": round(best_score, 3),
                "bucket": bucket,
            }
        else:
            assert best_fact is not None
            pairs, trace = kb.recall_facts_with_trace(question, top_k=args.top_k)
            result_ids = {f.id for f, _ in pairs}
            bucket = _classify(best_fact.id, trace, result_ids)
            sample = {
                "run_id": rid,
                "conv_idx": conv_idx,
                "qa_idx": qa_idx,
                "question": question,
                "gold_answer": gold_answer,
                "gold_fact_id": best_fact.id,
                "gold_fact_content": best_fact.content,
                "best_score": round(best_score, 3),
                "bucket": bucket,
                "trace_summary": {
                    "pool_size": trace.get("stage1_candidates", {}).get("total"),
                    "selected_count": len(trace.get("stage3_rrf", {}).get("selected_ids", [])),
                    "pre_mmr_count": len(trace.get("stage4b_mmr", {}).get("pre_mmr_ids", [])),
                    "post_mmr_count": len(trace.get("stage4b_mmr", {}).get("post_mmr_ids", [])),
                    "gold_in_pool": best_fact.id
                    in (
                        set(trace.get("stage1_candidates", {}).get("from_bm25", []))
                        | set(trace.get("stage1_candidates", {}).get("from_rare_tokens", []))
                        | set(trace.get("stage1_candidates", {}).get("from_entity_hop", []))
                        | set(trace.get("stage1_candidates", {}).get("from_dense", []))
                    ),
                    "gold_in_selected": best_fact.id
                    in trace.get("stage3_rrf", {}).get("selected_ids", []),
                    "gold_in_output": best_fact.id in result_ids,
                },
            }
            if bucket not in ("extraction_miss",) and len(buckets[bucket]) < args.dump_samples:
                # Include full trace in samples for deep inspection.
                sample["full_trace"] = trace

        buckets[bucket].append(sample)

    # ---- Summary -------------------------------------------------------
    total = len(wrongs)
    print()
    max_label = max(len(b) for b in bucket_order)
    for b in bucket_order:
        count = len(buckets[b])
        pct = 100 * count / total if total else 0
        label_map = {
            "extraction_miss": "fact not in KB (extraction failed)",
            "pool_miss": "fact in KB but not in recall pool",
            "select_topk_drop": "in pool, dropped by greedy select",
            "mmr_drop": "selected, dropped by MMR diversity",
            "answered_wrong_judge": "recall found gold, judge marked WRONG",
        }
        print(f"  {b:<{max_label}}  {count:3d}  ({pct:5.1f}%)   {label_map[b]}")

    # ---- Dump samples --------------------------------------------------
    dumped: list[str] = []
    for b in bucket_order:
        for i, s in enumerate(buckets[b][: args.dump_samples]):
            path = SAMPLES_DIR / f"{b}_{i}.json"
            with open(path, "w") as f:
                json.dump(s, f, indent=2, default=str, ensure_ascii=False)
            dumped.append(str(path.relative_to(_ROOT)))

    if dumped:
        print(f"\nSamples dumped ({len(dumped)} files):")
        for p in dumped:
            print(f"  {p}")

    # ---- Recommendation ------------------------------------------------
    dominant = max(bucket_order, key=lambda b: len(buckets[b]))
    print(f"\nLargest bucket: {dominant!r} ({len(buckets[dominant])} / {total})")
    recs = {
        "extraction_miss": "Focus: LLM extractor (prompt, chunk size, multi-turn context).",
        "pool_miss": "Focus: widen recall channels (dense weight, rare-token threshold).",
        "select_topk_drop": "Focus: _select_topk IDF-coverage criteria.",
        "mmr_drop": "Focus: MMR (lambda_, token normalisation, date-strip).",
        "answered_wrong_judge": "Judge issue — recall works, LLM answer format is off.",
    }
    print(f"Recommendation: {recs.get(dominant, '')}")


if __name__ == "__main__":
    main()
