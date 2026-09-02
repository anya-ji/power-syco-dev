#!/usr/bin/env bash
# exp2 generation: one detached screen per model variant, two GPUs each.
#
#   ./cluster/run_exp2.sh                 # launch all three
#   screen -ls                            # list
#   screen -r syco2-think                 # reattach (Ctrl-A D detaches)
#
# The no-suffix replication (yes/no prompts only, SAGE prompt sent verbatim):
#
#   RUN=exp2_84x101_nosuffix PROMPT_TYPES=YES_NO_PROMPT USER_SUFFIX=none \
#     SESSION_PREFIX=syco2ns ./cluster/run_exp2.sh
#
# Resumes from each model's shard, so a killed session is relaunched with the
# same RUN and skips finished cells.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="${RUN:-exp2_245x101}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/exp2/results}"
DIMENSIONS="${DIMENSIONS:-all}"          # or: job_title
LOG="$RESULTS_DIR/$RUN/logs"
mkdir -p "$LOG"
export HF_HOME="${HF_HOME:-/home/anyaji/.cache/huggingface}"

# Three framings: direct ask, unsafe-option question, and buried premise with an
# innocuous question -- one per family, rather than near-duplicates.
PROMPT_TYPES="${PROMPT_TYPES:-YES_NO_PROMPT SHORT_UNSAFE_CONTEXT_PLUS_SAFE_Q Q_W_2_UNSAFE_PROMPT}"

# Phrase appended to every user turn. Default is exp1/exp2's pressure framing
# (" I think it's fine."); "none" sends the SAGE prompt verbatim, which strips
# the user's stated opinion and leaves the status manipulation as the only
# variable. Distinct sessions so a no-suffix run can sit alongside the original.
USER_SUFFIX="${USER_SUFFIX:-}"
SESSION_PREFIX="${SESSION_PREFIX:-syco2}"
# A plain string, not an array: this gets spliced into the `bash -c` command
# below, where a nested "${arr[@]}" expansion inside the outer double quotes
# breaks the string and the screen dies before it can log anything.
# An `if`, not `[[ ]] && ...`: under `set -e` a failing AND-list exits the
# script, so the default (empty USER_SUFFIX) invocation would die right here.
SUFFIX_ARG=""
if [[ -n "$USER_SUFFIX" ]]; then
  SUFFIX_ARG="--user-suffix '$USER_SUFFIX'"
fi

# "<model> <gpus>" — all 8 GPUs are usable; 2 per model.
JOBS=(
  "qwen3-8b-think   0,1"
  "qwen3-8b-nothink 2,3"
  "qwen3-8b-base    4,5"
)

for job in "${JOBS[@]}"; do
  read -r model gpus <<<"$job"
  session="${SESSION_PREFIX}-${model##*-}"
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
      --design exp2 --models '$model' \
      --prompt-types $PROMPT_TYPES \
      --anchor-prompt-type YES_NO_PROMPT \
      $SUFFIX_ARG \
      --generic-dimensions $DIMENSIONS \
      --results-dir '$RESULTS_DIR' --run-name '$RUN' \
      --gpu-memory-utilization 0.88 2>&1 | tee '$LOG/gen-$model.log'
    echo \"EXIT $model = \${PIPESTATUS[0]}\" >> '$LOG/status.txt'
    echo '--- finished; Ctrl-D to close ---'
    exec bash
  "
done

sleep 2
echo; screen -ls | grep "$SESSION_PREFIX-" || true
echo
echo "Then:  uv run python scripts/judge/score.py --experiment exp2 --run $RESULTS_DIR/$RUN --max-workers 24"
