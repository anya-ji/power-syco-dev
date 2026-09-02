"""Load, pair and subsample SaLAD -- exp3's stimulus set.

SaLAD (``Holly301/SaLAD``) is 2,013 real-world image+text samples across 10
everyday categories, each labelled ``safe`` or ``unsafe``. The risk lives in the
*joint* image-text context: "If I need to cook something like this, what should
I do?" is innocuous text, and only the photo of wild mushrooms makes it
dangerous. That is exactly what makes it the right swap for SAGE-Eval in exp3 --
SAGE's risk is stated in the text, so a text-only model can find it.

Two schema quirks the loader normalises away:

* the gold field is named differently per label -- ``safety warning`` on unsafe
  rows, ``safe suggestion`` on safe rows -- so both land in ``gold_note``;
* images ship as a single flat ``image.zip``, not as a HF image column, so the
  sampled files are extracted to a local directory and referenced by path.

Rows are emitted with SAGE's field names (``prompt``, ``safety_fact``) on top of
the SaLAD-specific ones, so the shared generation/judge/analysis code reads them
unchanged.
"""

from __future__ import annotations

import json
import random
import zipfile
from collections import Counter
from pathlib import Path

SALAD_DATASET_ID = "Holly301/SaLAD"
DATA_FILE = "data.json"
IMAGE_ZIP = "image.zip"

#: the 10 categories, in the order the paper lists them
SALAD_CATEGORIES = [
    "Activity", "Exercise", "Food", "Game", "Health",
    "Home", "Office", "Research", "Study", "Travel",
]

SAFETY_TYPES = ["safe", "unsafe"]

#: gold-reference field name per label -- SaLAD names them differently
GOLD_FIELD = {"unsafe": "safety warning", "safe": "safe suggestion"}


def _hf_file(name: str, dataset_id: str = SALAD_DATASET_ID) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(dataset_id, name, repo_type="dataset"))


def load_salad(dataset_id: str = SALAD_DATASET_ID) -> list[dict]:
    """Every row, with the two gold fields folded into ``gold_note``."""
    raw = json.loads(_hf_file(DATA_FILE, dataset_id).read_text())
    out = []
    for r in raw:
        label = r["safety type"]
        field = GOLD_FIELD[label]
        if field not in r:
            raise KeyError(f"row {r['id']} labelled {label!r} has no {field!r}")
        out.append({
            "salad_id": r["id"],
            "category": r["category"],
            "safety_type": label,
            "image": r["image"],
            "text": r["text"].strip(),
            "gold_note": " ".join(r[field].split()),
            "gold_field": field,
        })
    return out


def sample_pairs(
    rows: list[dict],
    per_category: int = 5,
    seed: int = 42,
    categories: list[str] | None = None,
) -> list[dict]:
    """Draw ``per_category`` (safe, unsafe) pairs from each category.

    SaLAD has no natural pairing -- every row has its own image, and safe and
    unsafe items are not two takes on one scene -- so a "pair" here is one safe
    and one unsafe item from the same category, drawn independently and matched
    by index. That is bookkeeping, not a matched design: it guarantees the
    sample is label-balanced *within* every category, so an oversensitivity
    result and a missed-hazard result rest on the same 5 categories' worth of
    scenes rather than on differently-composed halves.

    Seeded per category, so adding or dropping a category leaves the others'
    draws untouched.
    """
    wanted = list(categories or SALAD_CATEGORIES)
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["category"], r["safety_type"]), []).append(r)

    out: list[dict] = []
    for cat in wanted:
        picks = {}
        for label in SAFETY_TYPES:
            pool = sorted(by_key.get((cat, label), []), key=lambda r: r["salad_id"])
            if len(pool) < per_category:
                raise ValueError(
                    f"{cat}/{label} has {len(pool)} rows, need {per_category}"
                )
            picks[label] = random.Random(f"{seed}|{cat}|{label}").sample(
                pool, per_category
            )
        for i in range(per_category):
            for label in SAFETY_TYPES:
                item = dict(picks[label][i])
                item["pair_idx"] = i
                item["pair_id"] = f"{cat}-{i}"
                out.append(item)
    return out


def to_prompt_rows(rows: list[dict]) -> list[dict]:
    """Add the field names the shared pipeline expects.

    ``prompt`` and ``safety_fact`` are SAGE's names; keeping them means
    generation, the judge's resume logic and the analysis aggregates work on
    exp3 rows without a parallel code path. ``prompt_type`` carries the safety
    label so the existing per-prompt-type splits become safe-vs-unsafe splits.
    """
    return [
        {**r,
         "prompt": r["text"],
         "safety_fact": r["gold_note"],
         "prompt_type": r["safety_type"].upper(),
         "augmentation_type": None}
        for r in rows
    ]


def extract_images(rows: list[dict], dest: Path,
                   dataset_id: str = SALAD_DATASET_ID) -> Path:
    """Extract just the sampled images from ``image.zip`` into ``dest``.

    The zip is ~960 MB for 2,013 images; a 100-row sample needs 5% of it, so
    only the named members are unpacked. Already-extracted files are left alone,
    which makes a rerun free.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = [r["image"] for r in rows]
    missing = [n for n in names if not (dest / n).exists()]
    if missing:
        with zipfile.ZipFile(_hf_file(IMAGE_ZIP, dataset_id)) as z:
            present = set(z.namelist())
            absent = [n for n in missing if n not in present]
            if absent:
                raise KeyError(f"{len(absent)} image(s) not in {IMAGE_ZIP}: {absent[:5]}")
            for name in missing:
                with z.open(name) as src, (dest / name).open("wb") as f:
                    f.write(src.read())
    for r in rows:
        r["image_path"] = str((dest / r["image"]).resolve())
    return dest


def describe(rows: list[dict]) -> dict:
    """Summary stats for a sample -- mirrors ``dataset.describe``."""
    per_cat = Counter(r["category"] for r in rows)
    return {
        "rows": len(rows),
        "categories": dict(sorted(per_cat.items())),
        "safety_types": dict(Counter(r["safety_type"] for r in rows).most_common()),
        "pairs": len({(r["category"], r["pair_idx"]) for r in rows if "pair_idx" in r}),
        "images": len({r["image"] for r in rows}),
        "gold_note_chars": {
            "min": min((len(r["gold_note"]) for r in rows), default=0),
            "max": max((len(r["gold_note"]) for r in rows), default=0),
        },
    }


def salad_banks(generic_dimensions: list[str]):
    """The (user, model) role banks exp3 runs on.

    The generic bank is shared with exp1/exp2 -- a Nobel laureate is equally
    irrelevant to a wild-mushroom photo as to a macadamia-nut question -- so
    only the domain side and its mirror are SaLAD's own.
    """
    from .config import (
        SALAD_CATEGORY_TO_DOMAIN, SALAD_DOMAIN_STATUSES, SALAD_MODEL_STATUSES,
    )
    from .model_statuses import ModelStatusBank
    from .statuses import StatusBank

    user = StatusBank(
        domain_path=SALAD_DOMAIN_STATUSES,
        generic_dimensions=generic_dimensions,
        category_to_domain=SALAD_CATEGORY_TO_DOMAIN,
    )
    return user, ModelStatusBank(SALAD_MODEL_STATUSES)


def load_sample(path=None, image_dir=None) -> list[dict]:
    """Read the committed exp3 sample and re-resolve its image paths."""
    from .config import SALAD_IMAGE_DIR, SALAD_SAMPLE

    path = Path(path or SALAD_SAMPLE)
    image_dir = Path(image_dir or SALAD_IMAGE_DIR)
    items = json.loads(path.read_text())["items"]
    for r in items:
        r["image_path"] = str((image_dir / r["image"]).resolve())
    return items
