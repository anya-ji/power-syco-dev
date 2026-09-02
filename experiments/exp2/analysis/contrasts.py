#!/usr/bin/env python3
"""Factorial contrasts for exp2, with cluster-bootstrap CIs over safety facts.

Each relevance block is a 2x2 (user level x model level), so the meaningful
quantities are the two main effects and their interaction:

  model effect  mean(model high) - mean(model low),  averaging over user level
  user effect   mean(user high)  - mean(user low),   averaging over model level
  interaction   (uH_mH - uH_mL) - (uL_mH - uL_mL)

Rows are clustered within safety facts (one fact contributes several prompts and
many cells), so CIs come from resampling facts, not rows.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _common  # noqa: E402,F401

MODELS = ["qwen3-8b-think", "qwen3-8b-nothink", "qwen3-8b-base"]
BLOCKS = ["domain", "irrel"]


def cell(pool, model, block, u, m):
    v = [r["verdict"] == "pass" for r in pool
         if r["model"] == model and r["condition"] == f"{block}_u{u}_m{m}"]
    return np.mean(v) if v else np.nan


def effects(pool, model, block):
    hh = cell(pool, model, block, "high", "high")
    hl = cell(pool, model, block, "high", "low")
    lh = cell(pool, model, block, "low", "high")
    ll = cell(pool, model, block, "low", "low")
    return {
        "model effect (mHigh-mLow)": (hh + lh) / 2 - (hl + ll) / 2,
        "user effect (uHigh-uLow)": (hh + hl) / 2 - (lh + ll) / 2,
        "interaction": (hh - hl) - (lh - ll),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "results/exp2_245x101/raw/judged.jsonl")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.judged.read_text().splitlines() if l.strip()]
    by_fact = collections.defaultdict(list)
    for r in rows:
        by_fact[r["safety_fact"]].append(r)
    facts = sorted(by_fact)
    pool = [r for f in facts for r in by_fact[f]]
    print(f"{len(rows):,} judged rows, {len(facts)} safety facts, "
          f"cluster bootstrap B={args.boot}\n")

    rng = random.Random(0)
    draws = [[r for f in rng.choices(facts, k=len(facts)) for r in by_fact[f]]
             for _ in range(args.boot)]

    print(f"{'model':<18}{'block':<8}{'effect':<26}{'est':>8}{'95% CI':>20}{'p':>8}")
    for model in MODELS:
        for block in BLOCKS:
            obs = effects(pool, model, block)
            boots = collections.defaultdict(list)
            for d in draws:
                e = effects(d, model, block)
                for k, v in e.items():
                    boots[k].append(v)
            for k, est in obs.items():
                a = np.array(boots[k])
                lo, hi = np.percentile(a, [2.5, 97.5])
                p = 2 * min((a <= 0).mean(), (a >= 0).mean())
                star = "  *" if p < 0.05 else ""
                print(f"{model:<18}{block:<8}{k:<26}{est:>+8.3f}"
                      f"   [{lo:>+.3f},{hi:>+.3f}]{p:>8.3f}{star}")
        print()


if __name__ == "__main__":
    main()
