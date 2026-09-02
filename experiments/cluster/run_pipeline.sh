#!/usr/bin/env bash
# Downstream pipeline for a finished run: judge -> analyse -> dashboard -> report.
# Generation is launched separately (cluster/run_screens.sh).
#
#   ./cluster/run_pipeline.sh                  # most recent run
#   ./cluster/run_pipeline.sh results/main_84x51
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ARG=()
[[ $# -gt 0 ]] && RUN_ARG=(--run "$1")

echo "[$(date)] === judge (SAGE rubric) ==="
uv run python scripts/judge/score.py "${RUN_ARG[@]}"

echo "[$(date)] === analyse ==="
uv run python scripts/analysis/analyze.py "${RUN_ARG[@]}"

echo "[$(date)] === dashboard ==="
uv run python scripts/analysis/build_dashboard.py "${RUN_ARG[@]}"

echo "[$(date)] === report ==="
uv run python scripts/analysis/build_report.py "${RUN_ARG[@]}" || \
  echo "(report needs pdflatex; .tex still written)"

echo "[$(date)] done"
echo
echo "View it:  uv run python scripts/analysis/serve_dashboard.py"
