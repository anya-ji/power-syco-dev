#!/usr/bin/env python3
"""Report SAGE-Eval composition and what each subsampling choice costs to run."""

import argparse
import json

import _bootstrap  # noqa: F401

from sycophancy.config import (
    CATEGORY_TO_DOMAIN, CONDITIONS, DATASET_ID, DATASET_SPLIT, DEFAULT_MODELS,
)
from sycophancy.dataset import describe, filter_rows, load_sage


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--split", default=DATASET_SPLIT)
    ap.add_argument("--n-models", type=int, default=len(DEFAULT_MODELS))
    ap.add_argument("--n-conditions", type=int, default=len(CONDITIONS))
    ap.add_argument("--personas-per-cell", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_sage(args.dataset, args.split)
    full = describe(rows)

    options = {
        "ALL (no filter)": rows,
        "augmentation=False (all prompt types)": filter_rows(rows, None, False),
        "YES_NO_PROMPT (incl. augmented)": filter_rows(rows, ["YES_NO_PROMPT"], True),
        "YES_NO_PROMPT + augmentation=False": filter_rows(rows, ["YES_NO_PROMPT"], False),
    }
    summaries = {name: describe(sel) for name, sel in options.items()}

    if args.json:
        print(json.dumps({"full": full, "options": summaries}, indent=2))
        return

    print(f"=== {args.dataset} [{args.split}] ===")
    print(f"rows={full['rows']}  safety_facts={full['safety_facts']}")
    print(f"categories: {full['categories']}")
    print(f"prompt_types: {full['prompt_types']}")
    print(f"augmentation_types: {full['augmentation_types']}")

    per_prompt = args.n_conditions * args.personas_per_cell
    print(f"\n=== Subsampling options ===")
    print(f"(grid = prompts x {args.n_conditions} conditions x "
          f"{args.personas_per_cell} persona(s) x {args.n_models} models)")
    for name, summary in summaries.items():
        n = summary["rows"]
        gens = n * per_prompt * args.n_models
        rpf = summary["rows_per_fact"]
        print(f"\n  {name}")
        print(f"    rows={n:<6} facts={summary['safety_facts']:>3}/{full['safety_facts']}"
              f"   rows/fact min={rpf['min']} mean={rpf['mean']} max={rpf['max']}")
        print(f"    -> {gens:,} generations + {gens:,} judge calls")
        if rpf["max"] == 1:
            print("       note: 1 row/fact, so the SAGE model-level score "
                  "reduces to the plain pass rate")

    print("\n=== Category -> domain bank coverage ===")
    cats = sorted(full["categories"])
    missing = [c for c in cats if c not in CATEGORY_TO_DOMAIN]
    for c in cats:
        print(f"  {c:<16} -> {CATEGORY_TO_DOMAIN.get(c, 'MISSING')}")
    print(f"  unmapped categories: {missing or 'none'}")


if __name__ == "__main__":
    main()
