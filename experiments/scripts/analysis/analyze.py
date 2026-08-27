#!/usr/bin/env python3
"""Aggregate judged rows into stat tables and figures."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy import plots
from sycophancy.analysis import load_judged, write_tables
from sycophancy.config import CONDITIONS, DEFAULT_MODELS, DEFAULT_RESULTS_DIR


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
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
    judged = args.judged or args.results_dir / "judged.jsonl"
    figures_dir = args.figures_dir or args.results_dir / "figures"

    df = load_judged(judged)
    print(f"Loaded {len(df)} judged rows from {judged}")
    print(f"  {df['safety_fact'].nunique()} safety facts, "
          f"{df['prompt'].nunique()} prompts, {df['model'].nunique()} models")

    tables = write_tables(df, args.results_dir, args.conditions)
    stats = tables["stats"]

    ordered = [m for m in DEFAULT_MODELS if m in set(df["model"])]
    ordered += [m for m in df["model"].unique() if m not in ordered]

    for model in ordered:
        idx = stats[stats["model"] == model].set_index("condition")
        print(f"\n=== {model} ===")
        print(idx[["rate", "ci95", "sage_score", "n"]].round(3).to_string())
        c = tables["contrasts"][model]
        print(f"  domain gap (high-low):    {c.domain_gap:+.3f}")
        print(f"  irrel. gap (high-low):    {c.irrel_gap:+.3f}")
        print(f"  relevance (dom-irr high): {c.relevance_gap:+.3f}")

    diag = tables["diagnostics"]
    if not diag.empty:
        print("\n=== Generation / judge diagnostics ===")
        print(diag.round(4).to_string())
        if "truncated" in diag.columns and diag["truncated"].max() > 0.01:
            print("  WARNING: non-trivial truncation. A thinking run cut off before "
                  "</think> has no answer and is scored 'fail' -- raise --max-tokens.")

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
        plots.make_all(stats, tables, figures_dir, ordered, args.primary_model)

    print(f"\nTables -> {args.results_dir}")


if __name__ == "__main__":
    main()
