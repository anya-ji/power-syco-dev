#!/usr/bin/env python3
"""Print one safety fact and its prompt variants, to see how SAGE-Eval is built.

Each fact fans out over 9 prompt types x (1 base + 6 augmentation types x 2),
i.e. up to 117 rows. Useful for deciding which filter to run and for seeing what
the SAGE model-level score actually demands, since that metric requires every
variant of a fact to pass.
"""

import argparse
import textwrap
from collections import Counter

import _bootstrap  # noqa: F401

from sycophancy.config import DATASET_ID, DATASET_SPLIT
from sycophancy.dataset import load_sage

# Roughly increasing context length / indirectness.
PROMPT_TYPE_ORDER = [
    "YES_NO_PROMPT",
    "SHORT_UNSAFE_CONTEXT_PLUS_SAFE_Q",
    "SAFE_Q_PLUS_UNSAFE_CONTEXT_PLUS_SAFE_Q",
    "RELEVANT_SAFE_Q_PLUS_UNSAFE_Q",
    "IRRELEVANT_SAFE_Q_PLUS_UNSAFE_Q",
    "Q_W_2_UNSAFE_PROMPT",
    "INSTRUCTION_W_DIFFERENT_ELEMENTS",
    "LONG_CONTEXT_50",
    "LONG_CONTEXT_100",
]
AUGMENTATION_ORDER = [
    "none", "TYPOS", "SPACING_PUNCTUATIONS",
    "TONE_HAPPINESS", "TONE_DEPRESSION", "TONE_URGENCY", "TONE_ANGER",
]


def wrap(text: str, width: int = 94) -> str:
    return textwrap.fill(text, width, subsequent_indent="        ")


def sort_key(rows: list[dict], key: str, order: list[str]):
    def rank(value):
        return order.index(value) if value in order else len(order)
    return sorted(rows, key=lambda r: (rank(r[key]), r["prompt"]))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="macadamia",
                    help="substring to find the fact by (default: macadamia)")
    ap.add_argument("--index", type=int, default=0,
                    help="which matching fact to show, when several match")
    ap.add_argument("--prompt-type", default=None,
                    help="show every augmentation of just this prompt type")
    ap.add_argument("--base-only", action="store_true",
                    help="show only the 9 unaugmented prompts")
    ap.add_argument("--list", action="store_true",
                    help="list matching facts and exit")
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--split", default=DATASET_SPLIT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_sage(args.dataset, args.split)

    facts = sorted({r["safety_fact"] for r in rows})
    matches = [f for f in facts if args.match.lower() in f.lower()]
    if not matches:
        raise SystemExit(f"no safety fact matching {args.match!r}")

    if args.list:
        for i, f in enumerate(matches):
            print(f"[{i}] {f[:150]}")
        return

    fact = matches[args.index]
    sub = [r for r in rows if r["safety_fact"] == fact]
    if args.prompt_type:
        sub = [r for r in sub if r["prompt_type"] == args.prompt_type]
    if args.base_only:
        sub = [r for r in sub if not r["augmentation"]]

    print("=" * 96)
    print("SAFETY FACT")
    print("=" * 96)
    print(wrap(fact, 96))
    print(f"\ncategory: {sorted({r['category'] for r in sub})}"
          f"   source: {sorted({r['source'] for r in sub})}")
    print(f"rows shown: {len(sub)}"
          f"   prompt types: {len({r['prompt_type'] for r in sub})}"
          f"   augmentation types: {len({r['augmentation_type'] for r in sub})}")

    if args.prompt_type or args.base_only:
        group_key, order = "augmentation_type", AUGMENTATION_ORDER
        if args.base_only:
            group_key, order = "prompt_type", PROMPT_TYPE_ORDER
    else:
        group_key, order = "prompt_type", PROMPT_TYPE_ORDER

    print("\n" + "=" * 96)
    print(f"PROMPTS, grouped by {group_key}")
    print("=" * 96)

    seen: Counter = Counter()
    for row in sort_key(sub, group_key, order):
        value = row[group_key]
        seen[value] += 1
        total = sum(1 for r in sub if r[group_key] == value)
        tag = value if total == 1 else f"{value} #{seen[value]}"
        if group_key == "prompt_type" and not row["augmentation"]:
            tag += "  (base)"
        print(f"\n[{tag}]\n  {wrap(row['prompt'])}")


if __name__ == "__main__":
    main()
