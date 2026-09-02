#!/usr/bin/env python3
"""Build exp3's dashboard: the SaLAD stimuli, the role banks, and the 9-cell cross."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _common  # noqa: E402,F401

from sycophancy.config import (  # noqa: E402
    ALL_GENERIC_DIMENSIONS, SALAD_IMAGE_DIR, SALAD_SAMPLE,
)
from sycophancy.salad_dashboard import build  # noqa: E402

EXP3 = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=EXP3 / "dashboard" / "inputs.html")
    ap.add_argument("--sample", type=Path, default=SALAD_SAMPLE)
    ap.add_argument("--image-dir", type=Path, default=SALAD_IMAGE_DIR)
    ap.add_argument("--generic-dimensions", nargs="+", default=ALL_GENERIC_DIMENSIONS)
    args = ap.parse_args()
    out = build(args.out, args.sample, args.image_dir, args.generic_dimensions)
    print(f"Open with:  file://{out.resolve()}")


if __name__ == "__main__":
    main()
