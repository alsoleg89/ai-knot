#!/usr/bin/env python3
"""Compare two bench runs or compute noise-floor stddev from multiple runs.

Usage:
  python3 scripts/compare_runs.py <run_a> <run_b>            # side-by-side delta
  python3 scripts/compare_runs.py --noise <run_1> [<run_2> ...]  # noise-floor

Loads report.json from data/runs/<run_id>/ for each run.
Prints a markdown delta table with Wilson 95 % CI.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "data" / "runs"
BASELINES_DIR = Path(__file__).parent.parent / "data" / "baselines"

CAT_LABELS = {
    "1": "single-hop",
    "2": "temporal",
    "3": "inference",
    "4": "open-domain",
    "5": "adversarial",
}


def load_report(run_id: str) -> dict:
    path = RUNS_DIR / run_id / "report.json"
    if not path.exists():
        sys.exit(f"ERROR: report.json not found for run '{run_id}' at {path}")
    return json.loads(path.read_text())


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % CI. Returns (lower, upper) as fractions."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def fmt_stat(stat: dict) -> str:
    acc = stat.get("accuracy", 0)
    correct = stat.get("correct", 0)
    total = stat.get("total", 0)
    lo, hi = wilson_ci(correct, total)
    ci_half = (hi - lo) / 2 * 100
    return f"{acc * 100:.1f}%  ({correct}/{total}, ±{ci_half:.1f}pp CI95)"


def compare(run_a: str, run_b: str) -> None:
    ra = load_report(run_a)
    rb = load_report(run_b)

    print(f"\n## Comparison: {run_a}  vs  {run_b}\n")
    print(f"  A judge/answer : {ra.get('judgeModel')} / {ra.get('answerModel')}")
    print(f"  B judge/answer : {rb.get('judgeModel')} / {rb.get('answerModel')}")
    print(f"  A finished     : {ra.get('finishedAt', '?')}")
    print(f"  B finished     : {rb.get('finishedAt', '?')}")

    sha_a = (ra.get("git") or {}).get("aiKnotSha", "unknown")
    sha_b = (rb.get("git") or {}).get("aiKnotSha", "unknown")
    print(f"  A git sha      : {sha_a[:8] if sha_a != 'unknown' else 'unknown'}")
    print(f"  B git sha      : {sha_b[:8] if sha_b != 'unknown' else 'unknown'}")
    print()

    header = f"  {'Category':<16}  {'Run A':>22}  {'Run B':>22}  {'Delta':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    cats_a = ra.get("byType", {})
    cats_b = rb.get("byType", {})
    all_cats = sorted(set(list(cats_a.keys()) + list(cats_b.keys())))

    for cat in all_cats:
        label = f"cat{cat} ({CAT_LABELS.get(cat, '?')})"
        sa = cats_a.get(cat, {"accuracy": 0, "correct": 0, "total": 0})
        sb = cats_b.get(cat, {"accuracy": 0, "correct": 0, "total": 0})
        delta = (sb["accuracy"] - sa["accuracy"]) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<16}  {fmt_stat(sa):>22}  {fmt_stat(sb):>22}  {sign}{delta:.1f}pp")

    print()
    sa14 = ra.get("categories1to4", {"accuracy": 0, "correct": 0, "total": 0})
    sb14 = rb.get("categories1to4", {"accuracy": 0, "correct": 0, "total": 0})
    delta14 = (sb14["accuracy"] - sa14["accuracy"]) * 100
    sign = "+" if delta14 >= 0 else ""
    print(f"  {'cat1-4 TOTAL':<16}  {fmt_stat(sa14):>22}  {fmt_stat(sb14):>22}  {sign}{delta14:.1f}pp")
    print()

    if delta14 < -2.0:
        print(f"  ⚠  cat1-4 dropped {abs(delta14):.1f}pp — exceeds 2pp warning threshold")
    elif delta14 > 0:
        print(f"  ✓  cat1-4 improved {delta14:.1f}pp")


def noise_floor(run_ids: list[str]) -> None:
    reports = [load_report(r) for r in run_ids]

    print(f"\n## Noise-floor: {len(reports)} replicates\n")

    cats = sorted(set().union(*[set(r.get("byType", {}).keys()) for r in reports]))
    per_cat_stddev: dict[str, float] = {}

    for cat in cats:
        accs = [r.get("byType", {}).get(cat, {}).get("accuracy", 0.0) for r in reports]
        mean = sum(accs) / len(accs)
        variance = sum((a - mean) ** 2 for a in accs) / max(1, len(accs) - 1)
        std = math.sqrt(variance)
        per_cat_stddev[cat] = std
        label = f"cat{cat} ({CAT_LABELS.get(cat, '?')})"
        print(f"  {label:<20}  accs={[f'{a*100:.1f}%' for a in accs]}  stddev={std*100:.2f}pp")

    cat14_accs = [r.get("categories1to4", {}).get("accuracy", 0.0) for r in reports]
    mean14 = sum(cat14_accs) / len(cat14_accs)
    var14 = sum((a - mean14) ** 2 for a in cat14_accs) / max(1, len(cat14_accs) - 1)
    std14 = math.sqrt(var14)
    print(f"\n  {'cat1-4 TOTAL':<20}  accs={[f'{a*100:.1f}%' for a in cat14_accs]}  stddev={std14*100:.2f}pp")

    out = {
        "stddev_cat1_4": round(std14, 6),
        "stddev_per_cat": {k: round(v, 6) for k, v in per_cat_stddev.items()},
        "replicates": len(run_ids),
        "source_runs": run_ids,
        "mean_cat1_4": round(mean14, 6),
    }

    out_path = BASELINES_DIR / "noise_floor_2conv.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Written to {out_path}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run IDs to compare (2 for delta, 3+ for --noise)")
    parser.add_argument("--noise", action="store_true", help="Compute noise-floor stddev across replicates")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="After comparison, promote run_b to latest_2conv.json if it beats run_a",
    )
    args = parser.parse_args()

    if args.noise:
        if len(args.runs) < 2:
            sys.exit("ERROR: --noise requires at least 2 run IDs")
        noise_floor(args.runs)
        return

    if len(args.runs) != 2:
        sys.exit("ERROR: provide exactly 2 run IDs for delta comparison (or use --noise)")

    compare(args.runs[0], args.runs[1])

    if args.promote:
        run_b = args.runs[1]
        rb = load_report(run_b)
        latest_path = BASELINES_DIR / "latest_2conv.json"
        current = json.loads(latest_path.read_text()) if latest_path.exists() else {}
        if rb.get("categories1to4", {}).get("accuracy", 0) > current.get("cat1_4_aggregate", 0):
            sha = (rb.get("git") or {}).get("aiKnotSha", "unknown")
            entry = {
                "label": run_b,
                "run_id": run_b,
                "commit_sha": sha[:8] if sha != "unknown" else "unknown",
                "cat1_4_aggregate": round(rb["categories1to4"]["accuracy"], 6),
                "per_cat": {
                    k: round(v["accuracy"], 6)
                    for k, v in rb.get("byType", {}).items()
                    if k in ("1", "2", "3", "4")
                },
                "settings": {
                    "judge_model": rb.get("judgeModel"),
                    "answer_model": rb.get("answerModel"),
                    "top_k": (rb.get("config") or {}).get("resolved", {}).get("top_k"),
                    "profile": (rb.get("config") or {}).get("resolved", {}).get("profile"),
                },
                "notes": "Auto-promoted by compare_runs.py --promote",
                "verified": True,
            }
            # Append to index
            index_path = BASELINES_DIR / "index.json"
            index = json.loads(index_path.read_text()) if index_path.exists() else []
            index.append(entry)
            index_path.write_text(json.dumps(index, indent=2))
            latest_path.write_text(json.dumps(entry, indent=2))
            print(f"  ✓ Promoted {run_b} ({entry['cat1_4_aggregate']*100:.1f}%) to latest_2conv.json")
        else:
            print("  — Not promoting: run_b did not beat current latest baseline")


if __name__ == "__main__":
    main()
