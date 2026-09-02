#!/usr/bin/env bash
# Print a comma-separated list of N GPUs with at least MIN_FREE MiB free,
# waiting until that many exist. nlp4 is shared and its cards fill and empty
# without warning, so a long run that grabs whatever was free at submit time
# dies on startup as often as it starts.
#
#   ./cluster/wait_for_gpus.sh 2              # two cards, default headroom
#   MIN_FREE=30000 ./cluster/wait_for_gpus.sh 1
#
# Only the GPU list goes to stdout; progress notes go to stderr, so this can be
# used as `gpus=$(./cluster/wait_for_gpus.sh 2)`.
set -euo pipefail

WANT="${1:-1}"
# An 8B checkpoint in bf16 is ~16 GiB of weights; the rest is KV cache and
# activations. Below this vLLM either refuses to start or thrashes.
MIN_FREE="${MIN_FREE:-24000}"          # MiB
POLL="${POLL:-120}"                    # seconds between checks
TIMEOUT="${TIMEOUT:-0}"                # seconds; 0 = wait indefinitely

start=$(date +%s)
notified=0
while :; do
  free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -F', *' -v m="$MIN_FREE" '$2 >= m {print $1}' | head -n "$WANT" \
         | paste -sd,)
  if [[ -n "$free" && $(tr -cd ',' <<<"$free" | wc -c) -eq $((WANT - 1)) ]]; then
    echo "$free"
    exit 0
  fi
  if [[ $notified -eq 0 ]]; then
    echo "waiting for $WANT GPU(s) with >=${MIN_FREE}MiB free (checking every ${POLL}s)" >&2
    notified=1
  fi
  if [[ "$TIMEOUT" -gt 0 && $(( $(date +%s) - start )) -ge "$TIMEOUT" ]]; then
    echo "timed out after ${TIMEOUT}s waiting for $WANT GPU(s)" >&2
    exit 1
  fi
  sleep "$POLL"
done
