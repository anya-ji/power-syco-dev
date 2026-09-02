#!/usr/bin/env python3
"""Build exp2's roles dashboard: user roles, model roles, and the 20-cell cross."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _common  # noqa: E402,F401

from sycophancy.config import ALL_GENERIC_DIMENSIONS  # noqa: E402
from sycophancy.roles_dashboard import build  # noqa: E402

EXP2 = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=EXP2 / "dashboard" / "roles.html")
    ap.add_argument("--generic-dimensions", nargs="+", default=ALL_GENERIC_DIMENSIONS)
    args = ap.parse_args()
    out = build(args.out, args.generic_dimensions)
    print(f"Open with:  file://{out.resolve()}")


if __name__ == "__main__":
    main()
