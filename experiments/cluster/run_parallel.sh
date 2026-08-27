#!/usr/bin/env bash
# Generate all three variants concurrently: 3 models x 2 GPUs = 6 of 8 GPUs.
# Each writes to its own results dir, then the shards are merged for judging.
#
#   screen -S sycophancy-parallel
#   ./cluster/run_parallel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# model_key:gpu_pair
PAIRS=(
  "qwen3-8b-think:0,1"
  "qwen3-8b-nothink:2,3"
  "qwen3-8b-base:4,5"
)

pids=()
for pair in "${PAIRS[@]}"; do
  model="${pair%%:*}"; gpus="${pair##*:}"
  shard="$RESULTS_DIR/shards/$model"
  mkdir -p "$shard"
  echo "[$(date)] launching $model on GPUs $gpus -> $shard"
  CUDA_VISIBLE_DEVICES="$gpus" \
    uv run python scripts/model/generate.py \
      --models "$model" --results-dir "$shard" "$@" \
      > "$LOG_DIR/gen-$model.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
if [[ $fail -ne 0 ]]; then
  echo "[$(date)] at least one generation job failed; see $LOG_DIR/gen-*.log" >&2
  exit 1
fi

echo "[$(date)] merging shards -> $RESULTS_DIR/generations.jsonl"
cat "$RESULTS_DIR"/shards/*/generations.jsonl > "$RESULTS_DIR/generations.jsonl"
wc -l < "$RESULTS_DIR/generations.jsonl"

echo "[$(date)] === judge ==="
uv run python scripts/judge/score.py --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_DIR/judge.log"
echo "[$(date)] === analyze ==="
uv run python scripts/analysis/analyze.py --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_DIR/analyze.log"
echo "[$(date)] === report ==="
uv run python scripts/analysis/build_report.py --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_DIR/analyze.log"
echo "[$(date)] done -> $RESULTS_DIR"
