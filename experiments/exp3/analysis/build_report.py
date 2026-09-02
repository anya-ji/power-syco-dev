#!/usr/bin/env python3
"""Render and compile the concise exp3 report."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _common  # noqa: E402,F401

import pandas as pd  # noqa: E402

from sycophancy.analysis import load_judged  # noqa: E402
from sycophancy.artifacts import RunPaths, latest_run  # noqa: E402
from sycophancy.config import SALAD_SAMPLE, VL_MODELS, results_dir  # noqa: E402
from sycophancy.report import compile_pdf  # noqa: E402
from sycophancy.report_exp3 import build_latex  # noqa: E402


def _optional(path: Path) -> pd.DataFrame | None:
    """Tables the analysis step may not have written."""
    return pd.read_csv(path) if path.exists() else None


def available_counts() -> dict | None:
    """Per-category safe/unsafe counts in the full benchmark, if reachable.

    Only used to show what the sample was drawn from. Needs the SaLAD download
    (cached after ``sample_salad.py``), so a missing copy drops the columns
    rather than failing the build.
    """
    try:
        from sycophancy.salad import load_salad

        out: dict[str, dict[str, int]] = {}
        for r in load_salad():
            out.setdefault(r["category"], {}).setdefault(r["safety_type"], 0)
            out[r["category"]][r["safety_type"]] += 1
        return out
    except Exception as e:  # noqa: BLE001 - offline or gated is fine here
        print(f"NOTE: benchmark pool counts unavailable ({e}); "
              f"the data table shows the sample only")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--sample", type=Path, default=SALAD_SAMPLE)
    ap.add_argument("--no-pool-counts", action="store_true",
                    help="skip the 'available in SaLAD' columns (no download)")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    paths = RunPaths(args.run) if args.run else latest_run(results_dir("exp3"))
    print(f"Run directory: {paths.root}")
    df = load_judged(paths.judged)
    t = paths.tables
    stats = pd.read_csv(t / "condition_stats.csv")
    ef = pd.read_csv(t / "factorial_effects.csv")
    by_label = pd.read_csv(t / "effects_by_label.csv")
    vs_ctrl = _optional(t / "vs_control.csv")
    mixed_by_label = _optional(t / "effects_by_label_mixed.csv")
    asym = _optional(t / "asymmetry_contrasts.csv")
    asym_by_label = _optional(t / "asymmetry_contrasts_by_label.csv")
    warn = _optional(t / "warning_rates.csv")
    warn_effects = _optional(t / "warning_effects.csv")

    cfg = (json.loads(paths.run_config.read_text())
           if paths.run_config.exists() else {})
    sample_meta = (json.loads(Path(args.sample).read_text())
                   if Path(args.sample).exists() else {})
    sample_meta.pop("items", None)
    available = None if args.no_pool_counts else available_counts()

    models = [m for m in VL_MODELS if m in set(df["model"])]
    models += [m for m in df["model"].unique() if m not in models]

    report_dir = paths.root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tex = build_latex(df, stats, ef, by_label, models, vs_ctrl=vs_ctrl,
                      mixed_by_label=mixed_by_label, asym=asym,
                      asym_by_label=asym_by_label, warn=warn,
                      warn_effects=warn_effects, cfg=cfg,
                      sample_meta=sample_meta, available=available,
                      run_name=paths.root.name)
    tex_path = report_dir / "report.tex"
    tex_path.write_text(tex)
    print(f"Wrote LaTeX -> {tex_path}")
    if not args.no_pdf:
        compile_pdf(tex_path)


if __name__ == "__main__":
    main()
