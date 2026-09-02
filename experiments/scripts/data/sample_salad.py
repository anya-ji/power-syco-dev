#!/usr/bin/env python3
"""Draw exp3's stimulus sample from SaLAD and extract its images.

Five (safe, unsafe) pairs from each of the 10 categories = 100 items. Writes
``data/salad/sample.json`` and unpacks just those images into
``data/salad/images/``, so the whole downstream pipeline -- generation, judge,
dashboard -- reads one small committed file plus a local image directory.

The draw is seeded per category, so re-running reproduces it exactly and adding
a category never disturbs the others.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy.config import (
    SALAD_DATASET_ID, SALAD_IMAGE_DIR, SALAD_PER_CATEGORY, SALAD_SAMPLE,
)
from sycophancy.salad import (
    SALAD_CATEGORIES, describe, extract_images, load_salad, sample_pairs,
    to_prompt_rows,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-category", type=int, default=SALAD_PER_CATEGORY,
                    help="(safe, unsafe) pairs per category (default: 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--categories", nargs="+", default=SALAD_CATEGORIES)
    ap.add_argument("--dataset-id", default=SALAD_DATASET_ID)
    ap.add_argument("--out", type=Path, default=SALAD_SAMPLE)
    ap.add_argument("--image-dir", type=Path, default=SALAD_IMAGE_DIR)
    ap.add_argument("--no-images", action="store_true",
                    help="skip extraction (the 960 MB zip download) and write "
                         "the sample only")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_salad(args.dataset_id)
    print(f"Loaded {len(rows)} SaLAD rows from {args.dataset_id}")

    sample = to_prompt_rows(
        sample_pairs(rows, args.per_category, args.seed, args.categories)
    )
    if not args.no_images:
        dest = extract_images(sample, args.image_dir, args.dataset_id)
        print(f"Images -> {dest}")

    summary = describe(sample)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"dataset": args.dataset_id, "seed": args.seed,
         "per_category": args.per_category, "summary": summary,
         # paths are machine-specific; the dashboard and the runner re-resolve
         # them from --image-dir, so they are not part of the committed record
         "items": [{k: v for k, v in r.items() if k != "image_path"}
                   for r in sample]},
        indent=2, ensure_ascii=False) + "\n")

    print(f"Sample -> {args.out}")
    print(f"  {summary['rows']} items · {summary['pairs']} pairs · "
          f"{len(summary['categories'])} categories")
    for cat, n in summary["categories"].items():
        print(f"    {cat:<10} {n}")
    print(f"  labels: {summary['safety_types']}")


if __name__ == "__main__":
    main()
