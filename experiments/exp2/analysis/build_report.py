#!/usr/bin/env python3
"""Render and compile the exp2 report."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _common  # noqa: E402,F401

import pandas as pd  # noqa: E402

from sycophancy.analysis import load_judged  # noqa: E402
from sycophancy.artifacts import RunPaths, latest_run  # noqa: E402
from sycophancy.config import DEFAULT_MODELS, results_dir  # noqa: E402
from sycophancy.report import compile_pdf  # noqa: E402
from sycophancy.report_exp2 import build_latex  # noqa: E402


def _optional(path: Path) -> pd.DataFrame | None:
    """Tables the analysis step may not have written (e.g. --no-bootstrap)."""
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--primary-model", default="qwen3-8b-think")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    paths = RunPaths(args.run) if args.run else latest_run(results_dir("exp2"))
    print(f"Run directory: {paths.root}")
    df = load_judged(paths.judged)
    t = paths.tables
    stats = pd.read_csv(t / "condition_stats.csv")
    ef = pd.read_csv(t / "factorial_effects.csv")
    cells = pd.read_csv(t / "cell_means.csv")
    dim = pd.read_csv(t / "dimension_effects.csv")
    by_ptype = _optional(t / "effects_by_prompt_type.csv")
    mixed = _optional(t / "factorial_effects_mixed.csv")
    vs_ctrl = _optional(t / "vs_control.csv")
    asym = _optional(t / "asymmetry_contrasts.csv")
    manifest = (json.loads(paths.run_config.read_text())
                if paths.run_config.exists() else {})

    models = [m for m in DEFAULT_MODELS if m in set(df["model"])]
    models += [m for m in df["model"].unique() if m not in models]

    report_dir = paths.root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tex = build_latex(df, stats, ef, cells, dim, models, args.primary_model,
                      by_ptype=by_ptype, mixed=mixed, vs_ctrl=vs_ctrl,
                      manifest=manifest, asym=asym)
    tex_path = report_dir / "report.tex"
    tex_path.write_text(tex)
    print(f"Wrote LaTeX -> {tex_path}")
    if not args.no_pdf:
        compile_pdf(tex_path)


if __name__ == "__main__":
    main()
