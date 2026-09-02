#!/usr/bin/env bash
# exp3 generation: one detached screen per Qwen3-VL checkpoint, two GPUs each.
#
#   ./cluster/run_exp3.sh                 # launch both
#   screen -ls                            # list
#   screen -r syco3-think                 # reattach (Ctrl-A D detaches)
#
# Resumes from each model's shard, so a killed session is relaunched with the
# same RUN and skips finished cells.
#
# Prerequisite: ./scripts/data/sample_salad.py has written data/salad/sample.json
# and extracted its images. The run reads that fixed sample -- it does not
# draw its own.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# PRESSURE_SUFFIX=1 appends exp1/exp2's "I think it's fine." to every query.
# Off by default (the SaLAD query is sent verbatim). The two settings differ in
# that phrase alone, so they pair for a direct comparison -- and they must land
# in different run directories, which is why the default RUN depends on it.
if [ "${PRESSURE_SUFFIX:-0}" = "1" ]; then
  SUFFIX_FLAG="--pressure-suffix"
  RUN="${RUN:-exp3_80x101_suffix}"
else
  SUFFIX_FLAG=""
  RUN="${RUN:-exp3_80x101}"
fi

RESULTS_DIR="${RESULTS_DIR:-$ROOT/exp3/results}"
DIMENSIONS="${DIMENSIONS:-all}"          # or: job_title
LOG="$RESULTS_DIR/$RUN/logs"
mkdir -p "$LOG"
export HF_HOME="${HF_HOME:-/home/anyaji/.cache/huggingface}"

# Bounds the KV cache. Qwen3-VL's native window is 262k, which is far more than
# a one-image prompt plus an 8k thinking budget needs.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# Vision tokens scale with pixels and every cell repeats the same image, so the
# long edge is capped rather than sent at native resolution.
IMAGE_MAX_SIDE="${IMAGE_MAX_SIDE:-1024}"

# "<model> <gpus>" — 2 GPUs each.
JOBS=(
  "qwen3vl-8b-think    0,1"
  "qwen3vl-8b-instruct 2,3"
)

for job in "${JOBS[@]}"; do
  read -r model gpus <<<"$job"
  session="syco3-${model##*-}${SUFFIX_FLAG:+-suffix}"
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "[$session] already running — skipping"
    continue
  fi
  echo "[$session] model=$model gpus=$gpus -> $LOG/gen-$model.log"
  screen -dmS "$session" bash -c "
    cd '$ROOT'
    export HF_HOME='$HF_HOME'
    export CUDA_VISIBLE_DEVICES='$gpus'
    uv run python scripts/model/generate.py \
      --design exp3 --experiment exp3 --models '$model' \
      --generic-dimensions $DIMENSIONS \
      --results-dir '$RESULTS_DIR' --run-name '$RUN' \
      --max-model-len $MAX_MODEL_LEN --image-max-side $IMAGE_MAX_SIDE $SUFFIX_FLAG \
      --gpu-memory-utilization 0.88 2>&1 | tee '$LOG/gen-$model.log'
    echo \"EXIT $model = \${PIPESTATUS[0]}\" >> '$LOG/status.txt'
    echo '--- finished; Ctrl-D to close ---'
    exec bash
  "
done

sleep 2
echo; screen -ls | grep syco3- || true
echo
echo "Then:  uv run python scripts/judge/score.py --experiment exp3 --run $RESULTS_DIR/$RUN --max-workers 24"
