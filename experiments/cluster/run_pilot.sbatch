#!/bin/bash
# SCF cluster batch job. Submit from the login node:
#   sbatch cluster/run_pilot.sbatch
# Check `safe` first -- if the group is at its preemptive_high_suhr cap,
# uncomment the --qos=normal line rather than taking someone's slot.
#SBATCH --job-name=sycophancy
#SBATCH --partition=berkeleynlp
#SBATCH --gres=gpu:2
#SBATCH --output=/data/anyaji/logs/%x-%j.out
#SBATCH --error=/data/anyaji/logs/%x-%j.err
# #SBATCH --qos=normal

set -euo pipefail

# Caches and outputs live on node-local /data, never /account.
export HF_HOME=/data/$USER/.cache/huggingface
export TORCH_HOME=/data/$USER/.cache/torch
export UV_CACHE_DIR=/data/$USER/.cache/uv
export VLLM_CACHE_ROOT=/data/$USER/.cache/vllm

PROJECT="${PROJECT:-/data/$USER/sycophancy/experiments}"
RESULTS_DIR="${RESULTS_DIR:-/data/$USER/sycophancy/results}"

mkdir -p /data/$USER/logs "$RESULTS_DIR"
cd "$PROJECT"

# generate.py resumes from generations.jsonl and score.py from judged.jsonl,
# so a preempted job can simply be resubmitted.
RUN="${RUN:-sbatch_$SLURM_JOB_ID}"
uv run python scripts/model/generate.py --results-dir "$RESULTS_DIR" --run-name "$RUN" "$@"
uv run python scripts/judge/score.py        --run "$RESULTS_DIR/$RUN"
uv run python scripts/analysis/analyze.py   --run "$RESULTS_DIR/$RUN"
uv run python scripts/analysis/build_dashboard.py --run "$RESULTS_DIR/$RUN"
uv run python scripts/analysis/build_report.py    --run "$RESULTS_DIR/$RUN"
