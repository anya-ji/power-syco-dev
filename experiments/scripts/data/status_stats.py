#!/usr/bin/env python3
"""Report the persona banks: dimensions, domains, and counts per cell."""

import argparse
import json

import _bootstrap  # noqa: F401

from sycophancy.config import CATEGORY_TO_DOMAIN, DEFAULT_GENERIC_DIMENSIONS
from sycophancy.statuses import StatusBank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generic-dimensions", nargs="+",
                    default=DEFAULT_GENERIC_DIMENSIONS)
    ap.add_argument("--labels", action="store_true", help="list every persona label")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    bank = StatusBank(generic_dimensions=args.generic_dimensions)
    stats = bank.stats()

    if args.json:
        print(json.dumps(stats, indent=2))
        return

    print("=== generic_statuses.json ===")
    for dim, counts in stats["generic_dimensions"].items():
        marker = "  <- active" if dim in args.generic_dimensions else ""
        print(f"  {dim:<28} {counts}{marker}")
    print(f"  {len(stats['generic_dimensions'])} dimensions, "
          f"{stats['generic_total']} personas total")

    print("\n=== domain_statuses.json ===")
    for domain, counts in stats["domains"].items():
        print(f"  {domain:<28} {counts}")
    print(f"  {len(stats['domains'])} domains, {stats['domain_total']} personas total")
    print(f"\nGRAND TOTAL: {stats['generic_total'] + stats['domain_total']} personas")

    mapped = set(CATEGORY_TO_DOMAIN.values())
    print(f"\nDomains referenced by CATEGORY_TO_DOMAIN: {len(mapped)}")
    print(f"  unmapped domains: {sorted(set(stats['domains']) - mapped) or 'none'}")
    print(f"  missing domains:  {sorted(mapped - set(stats['domains'])) or 'none'}")
    print("\nNote: both files carry a 'control' list (2 entries per group) that the "
          "experiment does not use -- the control condition sends no system role.")

    if args.labels:
        print("\n=== Persona labels ===")
        for dim, group in bank.generic_by_dim.items():
            print(f"\n  generic/{dim}")
            for level in ("high", "low"):
                for e in group[level]:
                    print(f"    {level:<5} {e['label']}")
        for domain, group in bank.domain_by_name.items():
            print(f"\n  domain/{domain}")
            for level in ("high", "low"):
                for e in group[level]:
                    print(f"    {level:<5} {e['label']}")


if __name__ == "__main__":
    main()
