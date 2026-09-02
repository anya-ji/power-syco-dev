"""Load, filter, and subsample SAGE-Eval.

SAGE-Eval ships 11,297 rows: 104 safety facts x 9 prompt types x
(1 base + 6 augmentations). Filtering is therefore the main lever on run size,
and which filter is used also decides how stringent the SAGE model-level score
can be -- that metric requires *every* variant of a safety fact to pass, so a
filter leaving one row per fact reduces it to a plain pass rate.
"""

from __future__ import annotations

import random
from collections import Counter

from .config import DATASET_ID, DATASET_SPLIT, ExperimentConfig

KEEP_FIELDS = ("prompt", "category", "safety_fact", "prompt_type",
               "augmentation", "augmentation_type", "source")


def load_sage(dataset_id: str = DATASET_ID, split: str = DATASET_SPLIT) -> list[dict]:
    """Load the full split. Requires HF_TOKEN -- the dataset is gated."""
    from datasets import load_dataset

    return [dict(r) for r in load_dataset(dataset_id, split=split)]


def filter_rows(
    rows: list[dict],
    prompt_types: list[str] | None = None,
    include_augmented: bool = False,
) -> list[dict]:
    """Restrict to the requested prompt types and (optionally) augmentations."""
    out = rows
    if prompt_types:
        wanted = set(prompt_types)
        out = [r for r in out if r["prompt_type"] in wanted]
    if not include_augmented:
        out = [r for r in out if not r["augmentation"]]
    return out


def restrict_to_facts_with(rows: list[dict], anchor_type: str) -> list[dict]:
    """Keep only rows whose safety fact also has an ``anchor_type`` prompt.

    Lets a multi-prompt-type run cover exactly the fact set of an earlier
    single-type run, so the two are comparable on the same facts.
    """
    facts = {r["safety_fact"] for r in rows if r["prompt_type"] == anchor_type}
    return [r for r in rows if r["safety_fact"] in facts]


def subsample(
    rows: list[dict], n: int | None, seed: int, stratify_by_fact: bool = True
) -> list[dict]:
    """Draw ``n`` rows reproducibly, or all of them when ``n`` is None.

    Sampling is by safety fact rather than by row: the SAGE metric is scored per
    fact, so splitting a fact's variants across the in/out boundary would make
    it unscoreable. The pilot sampled rows directly, which was harmless only
    because its filter left exactly one row per fact.
    """
    if n is None or n >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    if not stratify_by_fact:
        return rng.sample(rows, n)

    by_fact: dict[str, list[dict]] = {}
    for row in rows:
        by_fact.setdefault(row["safety_fact"], []).append(row)

    facts = sorted(by_fact)
    rng.shuffle(facts)

    chosen: list[dict] = []
    for fact in facts:
        group = by_fact[fact]
        if len(chosen) + len(group) > n:
            break
        chosen.extend(group)
    return chosen


def project(rows: list[dict]) -> list[dict]:
    """Keep only the fields the experiment uses."""
    return [{k: r[k] for k in KEEP_FIELDS if k in r} for r in rows]


def load_sample(cfg: ExperimentConfig) -> list[dict]:
    """Load, filter, and subsample in one step."""
    rows = load_sage(cfg.dataset_id, cfg.dataset_split)
    rows = filter_rows(rows, cfg.prompt_types, cfg.include_augmented)
    if cfg.anchor_prompt_type:
        rows = restrict_to_facts_with(rows, cfg.anchor_prompt_type)
    rows = subsample(rows, cfg.n_samples, cfg.seed)
    return project(rows)


def describe(rows: list[dict]) -> dict:
    """Summary stats for a row set -- used by the stats script and run manifests."""
    facts = Counter(r["safety_fact"] for r in rows)
    per_fact = list(facts.values()) or [0]
    return {
        "rows": len(rows),
        "safety_facts": len(facts),
        "categories": dict(Counter(r["category"] for r in rows).most_common()),
        "prompt_types": dict(Counter(r["prompt_type"] for r in rows).most_common()),
        "augmentation_types": dict(
            Counter(r.get("augmentation_type", "none") for r in rows).most_common()
        ),
        "rows_per_fact": {
            "min": min(per_fact),
            "max": max(per_fact),
            "mean": round(sum(per_fact) / len(per_fact), 2),
        },
    }
