#!/usr/bin/env bash
# Full pipeline on a local multi-GPU box. Run inside screen:
#   screen -S sycophancy
#   ./cluster/run_pilot.sh
#   # detach with Ctrl-A then D; reattach with: screen -r sycophancy
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/logs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pilot-$(date +%Y%m%d-%H%M%S).log"
echo "[$(date)] root=$ROOT results=$RESULTS_DIR gpus=$CUDA_VISIBLE_DEVICES log=$LOG"

{
  echo "[$(date)] === generate (vLLM) ==="
  uv run python scripts/model/generate.py --results-dir "$RESULTS_DIR" "$@"

  echo "[$(date)] === judge (SAGE rubric) ==="
  uv run python scripts/judge/score.py --results-dir "$RESULTS_DIR"

  echo "[$(date)] === analyze ==="
  uv run python scripts/analysis/analyze.py --results-dir "$RESULTS_DIR"

  echo "[$(date)] === report ==="
  uv run python scripts/analysis/build_report.py --results-dir "$RESULTS_DIR"

  echo "[$(date)] === done ==="
} 2>&1 | tee "$LOG"
