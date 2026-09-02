#!/usr/bin/env bash
# Launch each model variant in its own detached screen session, one GPU pair each.
#
#   ./cluster/run_screens.sh                  # start all three
#   screen -ls                                # list sessions
#   screen -r syco-think                      # reattach (Ctrl-A D to detach)
#   tail -f results/<run>/logs/gen-<model>.log
#
# Generation resumes from its per-model shard, so a killed session can simply be
# relaunched with the same --run-name and will skip finished cells.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="${RUN:-main_84x51}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
LOG="$RESULTS_DIR/$RUN/logs"
mkdir -p "$LOG"
export HF_HOME="${HF_HOME:-/home/anyaji/.cache/huggingface}"

# "<model> <gpus>" — GPU 0 is left alone; it is in use by another process.
JOBS=(
  "qwen3-8b-nothink 1,2"
  "qwen3-8b-think   3,4"
  "qwen3-8b-base    5,6"
)

COMMON=(--prompt-types YES_NO_PROMPT --generic-dimensions all
        --personas-per-cell 0 --results-dir "$RESULTS_DIR" --run-name "$RUN"
        --gpu-memory-utilization 0.88)

for job in "${JOBS[@]}"; do
  read -r model gpus <<<"$job"
  session="syco-${model##*-}"
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "[$session] already running — skipping"
    continue
  fi
  echo "[$session] model=$model gpus=$gpus -> $LOG/gen-$model.log"
  screen -dmS "$session" bash -c "
    cd '$ROOT'
    export HF_HOME='$HF_HOME'
    export CUDA_VISIBLE_DEVICES='$gpus'
    uv run python scripts/model/generate.py --models '$model' ${COMMON[*]} 2>&1 \
      | tee '$LOG/gen-$model.log'
    echo \"EXIT $model = \${PIPESTATUS[0]}\" >> '$LOG/status.txt'
    echo '--- finished; session stays open, press Ctrl-D to close ---'
    exec bash
  "
done

sleep 2
echo; screen -ls
