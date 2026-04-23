"""Offline replay of cat1 WRONG questions against an existing knot.db.

Purpose: measure retrieval-layer changes without running the full bench. For
each cat1 WRONG question in a baseline run, re-execute the query pipeline
against the same DB and record whether the rendered evidence text contains
the gold tokens. Gold-in-context is a tight proxy for CORRECT on
retrieval-bottleneck questions (if the fact never reaches the LLM, the
answer cannot be right).

Usage
-----
    # Baseline (current code)
    .venv/bin/python scripts/cat1_retrieval_replay.py \\
        --db /tmp/replay.db \\
        --run aiknotbench/data/runs/p1-1b-2conv

    # Same DB, after enabling new retrieval flag
    AIKNOT_CLAIMS_FIRST_PROMOTION=1 .venv/bin/python \\
        scripts/cat1_retrieval_replay.py --db /tmp/replay.db \\
        --run aiknotbench/data/runs/p1-1b-2conv --label with-promo

The script prints per-question gold-in-context status and a summary table.
Writes a JSON report at `{run}/replay_{label}.json` for diffing between
runs.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from ai_knot.query_runtime import execute_query  # noqa: E402
from ai_knot.storage.sqlite_storage import SQLiteStorage  # noqa: E402


def _gold_tokens(gold: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", gold)]


def _contains_gold(text: str, gold: str) -> tuple[bool, list[str]]:
    toks = _gold_tokens(gold)
    if not toks:
        return (True, [])
    low = text.lower()
    found = [t for t in toks if t in low]
    return (len(found) == len(toks), found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to knot.db")
    ap.add_argument("--run", required=True, help="Baseline run dir with log.jsonl")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--category", type=int, default=1)
    ap.add_argument("--verdict", default="WRONG", help="Filter on baseline verdict")
    ap.add_argument(
        "--include-correct",
        action="store_true",
        help="Also replay CORRECT Q (for regression check)",
    )
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run)
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        print(f"ERROR: {log_path} not found", file=sys.stderr)
        return 2

    # Load baseline rows
    rows: list[dict] = []
    with log_path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    target = [
        r
        for r in rows
        if r.get("category") == args.category
        and (args.include_correct or r.get("verdict") == args.verdict)
    ]
    print(f"Loaded {len(target)} target Q from {log_path} (cat={args.category}, verdict={args.verdict})")

    # Use a temp DB copy per query because execute_query's
    # _drain_dirty_keys + bundle rebuild mutate the DB. Re-using one storage
    # across 30 Q would make later Q see a DB modified by earlier Q.
    tmp_dir = tempfile.mkdtemp(prefix="replay_")
    tmp_db = os.path.join(tmp_dir, "q.db")

    report: list[dict] = []
    base_has = 0
    new_has = 0
    flipped_gain: list[dict] = []
    flipped_loss: list[dict] = []
    errors: list[dict] = []

    for r in target:
        q = r["question"]
        gold = r["goldAnswer"]
        conv_idx = r["convIdx"]
        qa_idx = r["qaIdx"]
        agent_id = f"conv-{conv_idx}"

        # Baseline gold-containment from logged context
        base_ctx = r.get("context", "") or ""
        base_full, _ = _contains_gold(base_ctx, gold)

        try:
            shutil.copy2(args.db, tmp_db)
            storage = SQLiteStorage(db_path=tmp_db)
            ans = execute_query(storage, agent_id, q, top_k=60)
            new_ctx = getattr(ans, "evidence_text", "") or ""
        except Exception as ex:  # noqa: BLE001
            errors.append({"conv": conv_idx, "qa": qa_idx, "error": str(ex)[:200]})
            continue

        new_full, _ = _contains_gold(new_ctx, gold)
        if base_full:
            base_has += 1
        if new_full:
            new_has += 1

        row = {
            "conv": conv_idx,
            "qa": qa_idx,
            "cat": r.get("category"),
            "baseline_verdict": r.get("verdict"),
            "question": q[:100],
            "gold": gold[:100],
            "baseline_gold_in_ctx": base_full,
            "new_gold_in_ctx": new_full,
            "base_ctx_len": len(base_ctx),
            "new_ctx_len": len(new_ctx),
        }
        report.append(row)
        if (not base_full) and new_full:
            flipped_gain.append(row)
        elif base_full and (not new_full):
            flipped_loss.append(row)

    out_path = run_dir / f"replay_{args.label}.json"
    out_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "n": len(report),
                "errors": len(errors),
                "baseline_gold_in_ctx": base_has,
                "new_gold_in_ctx": new_has,
                "flipped_gain": len(flipped_gain),
                "flipped_loss": len(flipped_loss),
                "flipped_gain_q": [f"[{r['conv']}:{r['qa']}]" for r in flipped_gain],
                "flipped_loss_q": [f"[{r['conv']}:{r['qa']}]" for r in flipped_loss],
                "rows": report,
                "error_rows": errors,
            },
            indent=2,
        )
    )

    print("\n=== Replay Summary ===")
    print(f"  Target Q replayed:       {len(report)}")
    print(f"  Errors:                  {len(errors)}")
    print(f"  Gold-in-ctx baseline:    {base_has}/{len(report)}")
    print(f"  Gold-in-ctx new:         {new_has}/{len(report)}")
    print(f"  Flipped (no → yes):      {len(flipped_gain)}")
    print(f"  Flipped (yes → no):      {len(flipped_loss)}")
    print(f"  Saved: {out_path}")
    if flipped_gain:
        print("  ↑ Gains:")
        for r in flipped_gain:
            print(f"    [{r['conv']}:{r['qa']}] {r['question'][:60]}")
    if flipped_loss:
        print("  ↓ Losses:")
        for r in flipped_loss:
            print(f"    [{r['conv']}:{r['qa']}] {r['question'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
