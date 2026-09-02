#!/usr/bin/env bash
# exp2-solo generation: exp2's grid with the two sides uncrossed.
#
# Each cell dresses exactly one party -- all four model roles against a silent
# user, all four user roles against a silent model -- so an effect is measured
# against a partner that claims nothing, which the crossed design cannot do.
# Same 101 cells per prompt as exp2_245x101, same prompts, same decoding.
#
#   ./cluster/run_exp2solo.sh                      # wait for a card, then go
#   INHERIT_CONTROL= ./cluster/run_exp2solo.sh     # generate the control too
#   GPUS="0,1 2,3 4,5" ./cluster/run_exp2solo.sh   # one screen per model
#   screen -r syco2s-all                           # reattach (Ctrl-A D detaches)
#
# Resumes from each model's shard, so a killed session -- or a relaunch once
# more GPUs free up -- skips the cells already written.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="${RUN:-exp2solo_245x101}"
# The no-role control sends no system message, so it is the same experiment in
# both designs: it is copied from the crossed run rather than regenerated, which
# also puts every contrast-against-control in the two runs on one baseline.
# Set to "" to generate it here instead.
INHERIT_CONTROL="${INHERIT_CONTROL:-exp2_245x101}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/exp2/results}"
DIMENSIONS="${DIMENSIONS:-all}"          # or: job_title
LOG="$RESULTS_DIR/$RUN/logs"
mkdir -p "$LOG"
export HF_HOME="${HF_HOME:-/home/anyaji/.cache/huggingface}"

PROMPT_TYPES="${PROMPT_TYPES:-YES_NO_PROMPT SHORT_UNSAFE_CONTEXT_PLUS_SAFE_Q Q_W_2_UNSAFE_PROMPT}"
USER_SUFFIX="${USER_SUFFIX:-}"
SESSION_PREFIX="${SESSION_PREFIX:-syco2s}"
MODELS="${MODELS:-qwen3-8b-think qwen3-8b-nothink qwen3-8b-base}"

# See run_exp2.sh: a plain string, spliced in, because a nested array expansion
# inside the outer double quotes breaks the `bash -c` command string.
SUFFIX_ARG=""
if [[ -n "$USER_SUFFIX" ]]; then
  SUFFIX_ARG="--user-suffix '$USER_SUFFIX'"
fi
CONTROL_ARG=""
if [[ -n "$INHERIT_CONTROL" ]]; then
  CONTROL_ARG="--inherit-control '$INHERIT_CONTROL'"
fi

# nlp4 is shared and its cards fill and empty without warning, so pinning a GPU
# at submit time is how a long run dies on startup. Leave GPUS unset and the
# screen waits for a card to come free and takes it then; set it to a
# space-separated list of comma-separated groups (one group per model) to fan
# out across GPUs you already hold.
GPUS="${GPUS:-}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
MIN_FREE="${MIN_FREE:-24000}"            # MiB a card must have free to be taken
MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"       # relaunches after losing a GPU race
RETRY_SLEEP="${RETRY_SLEEP:-60}"         # seconds between attempts
read -r -a MODEL_LIST <<<"$MODELS"

# Fanned out only when GPUS names a group per model; otherwise every model runs
# sequentially in one screen, which is what a contended box allows.
if [[ -n "$GPUS" ]]; then
  read -r -a GPU_GROUPS <<<"$GPUS"
else
  GPU_GROUPS=("wait")
fi

JOBS=()
if [[ ${#GPU_GROUPS[@]} -eq ${#MODEL_LIST[@]} && ${#MODEL_LIST[@]} -gt 1 ]]; then
  for i in "${!MODEL_LIST[@]}"; do
    JOBS+=("${MODEL_LIST[$i]}|${GPU_GROUPS[$i]}|${MODEL_LIST[$i]##*-}")
  done
else
  JOBS=("$MODELS|${GPU_GROUPS[0]}|all")
fi

for job in "${JOBS[@]}"; do
  IFS='|' read -r models gpus tag <<<"$job"
  session="${SESSION_PREFIX}-${tag}"
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "[$session] already running — skipping"
    continue
  fi
  if [[ "$gpus" == "wait" ]]; then
    ngpu="$GPUS_PER_JOB"
    echo "[$session] models=$models gpus=(waiting for $ngpu free) -> $LOG/gen-$tag.log"
  else
    ngpu=$(( $(tr -cd ',' <<<"$gpus" | wc -c) + 1 ))
    echo "[$session] models=$models gpus=$gpus (tp=$ngpu) -> $LOG/gen-$tag.log"
  fi
  screen -dmS "$session" bash -c "
    cd '$ROOT'
    export HF_HOME='$HF_HOME'
    # Retry, because on a contended box the card that was free when
    # wait_for_gpus looked is often gone by the time vLLM allocates a minute
    # later. Each attempt re-picks a GPU, and generate.py resumes from the
    # shard, so a lost race costs the startup time and nothing else.
    for attempt in \$(seq 1 $MAX_ATTEMPTS); do
      gpus='$gpus'
      if [[ \"\$gpus\" == wait ]]; then
        gpus=\$(MIN_FREE='$MIN_FREE' ./cluster/wait_for_gpus.sh $ngpu 2>&1 | tee -a '$LOG/gen-$tag.log' | tail -1)
      fi
      export CUDA_VISIBLE_DEVICES=\"\$gpus\"
      # A fraction of each card's TOTAL memory, sized from what is actually free
      # on the card we got: 0.88 is right on an idle box and refuses to start on
      # a shared one. The 3 GiB margin leaves room for the other tenant.
      util=\$(nvidia-smi --query-gpu=index,memory.free,memory.total \
              --format=csv,noheader,nounits \
        | awk -F', *' -v want=\"\$gpus\" 'BEGIN{split(want,w,\",\");for(i in w)k[w[i]]=1;u=1}
                (\$1 in k){f=(\$2-3000)/\$3; if(f<u)u=f}
                END{if(u>0.88)u=0.88; if(u<0.30)u=0.30; printf \"%.2f\", u}')
      echo \"[$tag] attempt \$attempt/$MAX_ATTEMPTS gpus=\$gpus gpu-memory-utilization=\$util\" | tee -a '$LOG/gen-$tag.log'
      uv run python scripts/model/generate.py \
        --design exp2solo --models $models \
        --prompt-types $PROMPT_TYPES \
        --anchor-prompt-type YES_NO_PROMPT \
        $SUFFIX_ARG $CONTROL_ARG \
        --generic-dimensions $DIMENSIONS \
        --results-dir '$RESULTS_DIR' --run-name '$RUN' \
        --tensor-parallel-size $ngpu \
        --gpu-memory-utilization \"\$util\" 2>&1 | tee -a '$LOG/gen-$tag.log'
      rc=\${PIPESTATUS[0]}
      echo \"EXIT $tag attempt \$attempt = \$rc\" >> '$LOG/status.txt'
      [[ \$rc -eq 0 ]] && break
      # A fixed GPU list that failed will keep failing; only the waiting form
      # has anything new to try.
      [[ '$gpus' != wait ]] && break
      echo \"[$tag] attempt \$attempt failed (rc=\$rc); waiting ${RETRY_SLEEP}s\" | tee -a '$LOG/gen-$tag.log'
      sleep $RETRY_SLEEP
    done
    echo '--- finished; Ctrl-D to close ---'
    exec bash
  "
done

sleep 2
echo; screen -ls | grep "$SESSION_PREFIX-" || true
echo
if [[ -n "$INHERIT_CONTROL" ]]; then
  echo "Then:  uv run python scripts/data/import_control.py \\"
  echo "         --from $RESULTS_DIR/$INHERIT_CONTROL --to $RESULTS_DIR/$RUN"
fi
echo "       uv run python scripts/judge/score.py --experiment exp2 --run $RESULTS_DIR/$RUN --max-workers 24"
