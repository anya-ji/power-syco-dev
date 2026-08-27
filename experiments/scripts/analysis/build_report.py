#!/usr/bin/env python3
"""Render the LaTeX report from judged results and compile it to PDF."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy.analysis import load_judged, write_tables
from sycophancy.config import CONDITIONS, DEFAULT_MODELS, DEFAULT_RESULTS_DIR
from sycophancy.judge import DEFAULT_JUDGE_MODEL
from sycophancy.report import build_latex, compile_pdf, copy_figures


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--judged", type=Path, default=None)
    ap.add_argument("--figures-dir", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--primary-model", default=DEFAULT_MODELS[0])
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--name", default="report", help="output basename")
    ap.add_argument("--no-pdf", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    judged = args.judged or args.results_dir / "judged.jsonl"
    figures_dir = args.figures_dir or args.results_dir / "figures"
    report_dir = args.results_dir / "report"

    df = load_judged(judged)
    tables = write_tables(df, args.results_dir, args.conditions)

    ordered = [m for m in DEFAULT_MODELS if m in set(df["model"])]
    ordered += [m for m in df["model"].unique() if m not in ordered]
    primary = args.primary_model if args.primary_model in ordered else ordered[0]

    copy_figures(figures_dir, report_dir)

    tex = build_latex(df, tables["stats"], tables, models=ordered, primary=primary,
                      conditions=args.conditions, judge_model=args.judge_model)
    tex_path = report_dir / f"{args.name}.tex"
    tex_path.write_text(tex)
    print(f"Wrote LaTeX -> {tex_path}")

    if not args.no_pdf:
        compile_pdf(tex_path)


if __name__ == "__main__":
    main()
