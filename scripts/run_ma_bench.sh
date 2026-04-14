#!/usr/bin/env bash
# Run MA benchmark scenarios (S8-S26) with a configurable timeout.
#
# Usage:
#   ./scripts/run_ma_bench.sh              # default 30s timeout
#   ./scripts/run_ma_bench.sh 60           # 60s timeout
#   ./scripts/run_ma_bench.sh 10 protocol  # 10s, protocol only (S10,S11,S13,S17,S20,S25)
#   ./scripts/run_ma_bench.sh 10 retrieval # 10s, retrieval only (S8,S9,S12,S14-S24,S26)
#
# Categories:
#   all       — all MA scenarios (default)
#   protocol  — CAS, sync, concurrency, self-correction, conflict resolution
#   retrieval — ranking, trust, assembly, freshness, adversarial, onboarding

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TIMEOUT="${1:-30}"
CATEGORY="${2:-all}"

echo "MA bench: category=${CATEGORY}  timeout=${TIMEOUT}s"
echo "─────────────────────────────────────────"

.venv/bin/python -m tests.eval.benchmark.runner \
  --multi-agent \
  --ma-category "${CATEGORY}" \
  --mock-judge \
  --quick
