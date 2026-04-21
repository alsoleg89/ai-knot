#!/usr/bin/env bash
# bench_gate.sh — Run the 2-conv gate and compare against baseline.
#
# Usage:
#   cd aiknotbench
#   ./scripts/bench_gate.sh [--convs 0,1] [--top-k 60] [--allow-drift]
#
# --allow-drift: skip canonical-settings check (for local ollama runs).
#                Results are tagged as drift-run and NOT comparable to
#                canonical (gpt-4o-mini) baseline numbers.
#
# Exits 0 (PASS) or 1 (FAIL / regression detected).
# Uses noise-floor from data/baselines/noise_floor_2conv.json if present,
# otherwise falls back to a conservative floor of stddev=0.025.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BENCH_ROOT"

export AIKNOT_DEBUG_TRACE=1

# Load .env so that tsx (Node) gets OPENAI_API_KEY, DEFAULT_ANSWER_MODEL etc.
# Bun loads .env automatically; tsx/Node does not.
if [[ -f ".env" ]]; then
  set -a; source .env; set +a
fi

# Resolve ai-knot-mcp to the venv binary when it is not already on PATH.
if ! command -v ai-knot-mcp &>/dev/null; then
  REPO_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"
  VENV_MCP="${REPO_ROOT}/.venv/bin/ai-knot-mcp"
  if [[ -x "$VENV_MCP" ]]; then
    export AI_KNOT_COMMAND="$VENV_MCP"
  fi
fi

CONVS="0,1"
TOP_K="60"
ALLOW_DRIFT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --convs)   CONVS="$2";  shift 2 ;;
    --top-k)   TOP_K="$2";  shift 2 ;;
    --allow-drift) ALLOW_DRIFT="--allow-drift"; shift ;;
    *) shift ;;
  esac
done

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")"
RUN_ID="gate-${SHA}-2conv"
if [[ -n "$ALLOW_DRIFT" ]]; then
  RUN_ID="gate-${SHA}-2conv-drift"
fi

echo ""
echo "================================================================="
echo "  bench_gate.sh  run=${RUN_ID}  convs=${CONVS}  top_k=${TOP_K}${ALLOW_DRIFT:+  drift=yes}"
echo "================================================================="
echo ""

if [[ -n "$ALLOW_DRIFT" ]]; then
  echo "  NOTE: --allow-drift active — model deviations from canonical are allowed."
  echo "        Results are NOT comparable to gpt-4o-mini baseline numbers."
  echo ""
fi

# ---------------------------------------------------------------------------
# Run the bench
# ---------------------------------------------------------------------------
npx tsx src/index.ts run -r "$RUN_ID" --convs "$CONVS" --top-k "$TOP_K" --force ${ALLOW_DRIFT}

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
REPORT="data/runs/${RUN_ID}/report.json"
if [[ ! -f "$REPORT" ]]; then
  echo "ERROR: report.json not written — bench may have crashed" >&2
  exit 1
fi

# Extract cat1-4 accuracy and per-cat from report.json (jq required)
if ! command -v jq &>/dev/null; then
  echo "WARNING: jq not found; skipping numeric gate check" >&2
  echo "PASS (no gate) — install jq to enable numeric regression check"
  exit 0
fi

CURRENT=$(jq '.categories1to4.accuracy' "$REPORT")
CAT1=$(jq '.byType["1"].accuracy // 0' "$REPORT")
CAT2=$(jq '.byType["2"].accuracy // 0' "$REPORT")
CAT3=$(jq '.byType["3"].accuracy // 0' "$REPORT")
CAT4=$(jq '.byType["4"].accuracy // 0' "$REPORT")

# ---------------------------------------------------------------------------
# Load baseline
# ---------------------------------------------------------------------------
BASELINE_FILE="data/baselines/latest_2conv.json"
if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "WARNING: no baseline file at ${BASELINE_FILE}; skipping regression check"
  echo "PASS (no baseline) — run scripts/noise_floor.sh then update latest_2conv.json"
  exit 0
fi

BASELINE=$(jq '.cat1_4_aggregate' "$BASELINE_FILE")
B_CAT1=$(jq '.per_cat["1"] // 0' "$BASELINE_FILE")
B_CAT2=$(jq '.per_cat["2"] // 0' "$BASELINE_FILE")
B_CAT3=$(jq '.per_cat["3"] // 0' "$BASELINE_FILE")
B_CAT4=$(jq '.per_cat["4"] // 0' "$BASELINE_FILE")

BASELINE_LABEL=$(jq -r '.label' "$BASELINE_FILE")

# ---------------------------------------------------------------------------
# Load noise floor
# ---------------------------------------------------------------------------
NOISE_FILE="data/baselines/noise_floor_2conv.json"
NOISE_AGGREGATE=0.025   # conservative default (2.5pp)
N_CAT1=0.035; N_CAT2=0.035; N_CAT3=0.060; N_CAT4=0.040

if [[ -f "$NOISE_FILE" ]]; then
  NOISE_AGGREGATE=$(jq '.stddev_cat1_4 // 0.025' "$NOISE_FILE")
  N_CAT1=$(jq '.stddev_per_cat["1"] // 0.035' "$NOISE_FILE")
  N_CAT2=$(jq '.stddev_per_cat["2"] // 0.035' "$NOISE_FILE")
  N_CAT3=$(jq '.stddev_per_cat["3"] // 0.060' "$NOISE_FILE")
  N_CAT4=$(jq '.stddev_per_cat["4"] // 0.040' "$NOISE_FILE")
fi

# ---------------------------------------------------------------------------
# Compute thresholds: drop > 1.5σ + 2pp (aggregate) or 1.5σ + 3pp (per-cat)
# ---------------------------------------------------------------------------
python3 - <<PYEOF
import sys, json, math

current      = float("${CURRENT}")
baseline     = float("${BASELINE}")
noise_agg    = float("${NOISE_AGGREGATE}")

cats = {
    "1": (float("${CAT1}"), float("${B_CAT1}"), float("${N_CAT1}"), "single-hop"),
    "2": (float("${CAT2}"), float("${B_CAT2}"), float("${N_CAT2}"), "temporal"),
    "3": (float("${CAT3}"), float("${B_CAT3}"), float("${N_CAT3}"), "inference"),
    "4": (float("${CAT4}"), float("${B_CAT4}"), float("${N_CAT4}"), "open-domain"),
}

print()
print("  Baseline : ${BASELINE_LABEL}")
print(f"  Baseline cat1-4 = {baseline*100:.1f}%   Current = {current*100:.1f}%")
print()
print(f"  {'Category':<20}  {'Baseline':>8}  {'Current':>8}  {'Delta':>8}  {'Threshold':>12}")
print(f"  {'-'*20}  {'--------':>8}  {'-------':>8}  {'-----':>8}  {'----------':>12}")

failures = []

for cat, (cur, base, noise, label) in cats.items():
    delta   = (cur - base) * 100
    thr_drop = -(1.5 * noise + 0.03) * 100  # 1.5σ + 3pp
    sign    = "+" if delta >= 0 else ""
    flag    = " ⚠" if delta < thr_drop else ""
    print(f"  cat{cat} ({label:<15})  {base*100:>7.1f}%  {cur*100:>7.1f}%  {sign}{delta:>6.1f}pp  >{thr_drop:.1f}pp{flag}")
    if delta < thr_drop:
        failures.append(f"cat{cat} dropped {abs(delta):.1f}pp (threshold {abs(thr_drop):.1f}pp)")

# Aggregate gate
delta_agg = (current - baseline) * 100
thr_drop_agg = -(1.5 * noise_agg + 0.02) * 100
sign = "+" if delta_agg >= 0 else ""
flag_agg = " ⚠" if delta_agg < thr_drop_agg else ""
print()
print(f"  {'cat1-4 TOTAL':<20}  {baseline*100:>7.1f}%  {current*100:>7.1f}%  {sign}{delta_agg:>6.1f}pp  >{thr_drop_agg:.1f}pp{flag_agg}")
print()

if delta_agg < thr_drop_agg:
    failures.append(f"cat1-4 aggregate dropped {abs(delta_agg):.1f}pp (threshold {abs(thr_drop_agg):.1f}pp)")

if failures:
    print("FAIL — regression detected:")
    for f in failures:
        print(f"  - {f}")
    print()
    print("Action: revert the commit or open a PARK entry in DECISIONS.md")
    sys.exit(1)
else:
    if delta_agg > 0:
        print(f"PASS — cat1-4 improved {delta_agg:.1f}pp")
        print()
        print(f"If this is a new best, run:")
        print(f"  python3 scripts/compare_runs.py ${RUN_ID} <prev_run> --promote")
    else:
        print(f"PASS — within noise floor (delta {sign}{delta_agg:.1f}pp)")
    sys.exit(0)
PYEOF
RESULT=$?

if [[ $RESULT -ne 0 ]]; then
  echo ""
  echo "  Report : ${REPORT}"
  echo ""
  exit 1
fi

echo "  Report : ${REPORT}"
echo ""
