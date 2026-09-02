#!/usr/bin/env python3
"""Aggregate judged rows into stat tables and figures."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy import plots
from sycophancy.artifacts import RunPaths, latest_run
from sycophancy.analysis import load_judged, write_tables
from sycophancy.config import (
    CONDITIONS, DEFAULT_MODELS, DEFAULT_RESULTS_DIR, MODEL_VARIANTS, THINKING,
    DEFAULT_EXPERIMENT, EXPERIMENTS, results_dir,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=EXPERIMENTS,
                    help="which experiment directory to work in")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="override; defaults to <experiment>/results")
    ap.add_argument("--run", type=Path, default=None,
                    help="run directory (default: most recent under --results-dir)")
    ap.add_argument("--judged", type=Path, default=None,
                    help="default: <results-dir>/judged.jsonl")
    ap.add_argument("--figures-dir", type=Path, default=None,
                    help="default: <results-dir>/figures")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--primary-model", default=DEFAULT_MODELS[0],
                    help="model used for the single-model figures")
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = RunPaths(args.run) if args.run else latest_run(args.results_dir)
    print(f"Run directory: {paths.root}")
    judged = args.judged or paths.judged
    figures_dir = args.figures_dir or paths.figures

    df = load_judged(judged)
    print(f"Loaded {len(df)} judged rows from {judged}")
    print(f"  {df['safety_fact'].nunique()} safety facts, "
          f"{df['prompt'].nunique()} prompts, {df['model'].nunique()} models")

    tables = write_tables(df, paths.tables, args.conditions)
    stats = tables["stats"]

    ordered = [m for m in DEFAULT_MODELS if m in set(df["model"])]
    ordered += [m for m in df["model"].unique() if m not in ordered]

    for model in ordered:
        idx = stats[stats["model"] == model].set_index("condition")
        print(f"\n=== {model} ===")
        cols = [c for c in ["rate", "ci95", "all_pass_rate", "all_pass_rate_k5", "n"]
                if c in idx.columns]
        print(idx[cols].round(3).to_string())
        c = tables["contrasts"][model]
        print(f"  domain gap (high-low):    {c.domain_gap:+.3f}")
        print(f"  irrel. gap (high-low):    {c.irrel_gap:+.3f}")
        print(f"  relevance (dom-irr high): {c.relevance_gap:+.3f}")

    print("\n  all_pass_rate    = every persona for a fact passed (raw; cell sizes differ)")
    print("  all_pass_rate_k5 = same over 5 matched personas -- use this across conditions")

    diag = tables["diagnostics"]
    if not diag.empty:
        print("\n=== Generation / judge diagnostics ===")
        print(diag.round(4).to_string())
        # Truncation only destroys the answer in thinking mode (cut off before
        # </think> leaves nothing to judge). Elsewhere it merely clips a tail.
        if "truncated" in diag.columns:
            for model, rate in diag["truncated"].items():
                if rate < 0.05:
                    continue
                thinking = MODEL_VARIANTS.get(model) and MODEL_VARIANTS[model].mode == THINKING
                note = ("cut off before </think> leaves no answer, which the judge "
                        "fails -- raise --max-tokens" if thinking else
                        "responses are clipped; check whether verdicts hinge on the tail")
                print(f"  WARNING: {model} truncated {rate:.1%} -- {note}")

    consistency = tables.get("sample_consistency")
    if consistency is not None and not consistency.empty:
        print("\n=== Repeated-sample consistency ===")
        print(consistency.round(3).to_string())
        worst = consistency["unanimous_cells"].min()
        print(f"  {worst:.0%} of cells had all samples agree (lower = noisier). "
              "A condition gap below this noise floor is not a finding.")

    for model, personas in tables.get("by_persona", {}).items():
        if personas.groupby("condition").size().max() > 1:
            spread = (personas.groupby("condition")["pass_rate"]
                      .agg(lambda s: s.max() - s.min()))
            print(f"\n=== {model}: within-condition persona spread ===")
            print(spread.round(3).to_string())
            print("  (large spread => the effect is partly persona identity, not status)")

    if not args.no_figures:
        print(f"\nFigures -> {figures_dir}")
        plots.make_all(stats, tables, figures_dir, ordered, args.primary_model, df=df)

    print(f"\nTables -> {paths.tables}")


if __name__ == "__main__":
    main()
