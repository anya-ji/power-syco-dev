#!/usr/bin/env python3
"""Build the single-file local HTML dashboard for a run."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy.artifacts import RunPaths, latest_run
from sycophancy.config import (
    DEFAULT_EXPERIMENT, DEFAULT_RESULTS_DIR, EXPERIMENTS, results_dir,
)
from sycophancy.dashboard import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=EXPERIMENTS,
                    help="which experiment directory to work in")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="override; defaults to <experiment>/results")
    ap.add_argument("--run", type=Path, default=None,
                    help="run directory (default: most recent under --results-dir)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output html (default: <run>/dashboard.html)")
    ap.add_argument("--max-chars", type=int, default=800,
                    help="preview length per text field; the payload embeds every "
                         "row, so this drives page weight (default: 800)")
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="exp3: where the stimulus images live; thumbnails are "
                         "embedded in the row detail (default: data/salad/images)")
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="cap rows embedded for the explorer, stratified by "
                         "model x condition; 0 = embed all. Aggregates always "
                         "use every row.")
    args = ap.parse_args()
    if args.results_dir is None:
        args.results_dir = results_dir(args.experiment)

    paths = RunPaths(args.run) if args.run else latest_run(args.results_dir)
    print(f"Run directory: {paths.root}")
    out = build(paths, args.out, args.max_chars,
                None if args.max_rows == 0 else args.max_rows, args.image_dir)
    print(f"Open with:  file://{out.resolve()}")


if __name__ == "__main__":
    main()
