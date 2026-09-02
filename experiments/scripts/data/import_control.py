#!/usr/bin/env python3
"""Copy one condition's finished rows from another run into this one.

The no-role control is the same experiment wherever it appears: no system
message is sent, so the only thing that varies is the user turn. When two runs
share a prompt set and decoding settings, regenerating it is wasted GPU time and
re-judging it is wasted API calls -- and the two runs then carry *different*
control draws, which is worse than sharing one, because every contrast against
control is measured against a different baseline in each.

This copies the rows instead, after checking the two runs actually agree on
everything that could make the control differ. It refuses rather than warns: a
silently mismatched baseline is not recoverable from the output.

    uv run python scripts/data/import_control.py \
      --from exp2/results/exp2_245x101 --to exp2/results/exp2solo_245x101

Generations land in their own shard (``generations__inherited-<cond>.jsonl``) so
nothing collides with a model shard a live job is appending to, and the judged
rows are appended to ``judged.jsonl`` so ``score.py`` skips them on resume.
"""

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy.artifacts import RunPaths
from sycophancy.config import DEFAULT_RESULTS_DIR, ROOT  # noqa: F401
from sycophancy.generation import job_key

#: manifest fields that would make one run's control incomparable with another's
MUST_MATCH = ["models", "model_ids", "sampling", "dataset", "include_augmented"]


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _observed_suffix(rows: list[dict]) -> set[str]:
    """What was actually appended to each dataset prompt, read off the rows.

    The manifest is not enough: runs launched before ``--user-suffix`` existed
    have no such field, so comparing manifests would call an older run
    mismatched when it is not.
    """
    return {r["user_msg"][len(r["prompt"]):]
            for r in rows if "user_msg" in r and "prompt" in r}


def _prompt_key(item: dict) -> tuple:
    return (item.get("prompt"), item.get("safety_fact"), item.get("category"),
            item.get("prompt_type"), item.get("augmentation_type"))


def check(src: RunPaths, dst: RunPaths, condition: str) -> list[dict]:
    """Everything that must hold for the copy to be sound. Returns the rows."""
    problems = []

    # 1. The prompt sets must be identical *and* identically ordered: rows are
    #    addressed by prompt_idx, so a reordered sample would silently attach
    #    each control response to the wrong prompt.
    a = json.loads(src.sampled_prompts.read_text())
    b = json.loads(dst.sampled_prompts.read_text())
    if len(a) != len(b):
        problems.append(f"prompt counts differ: {len(a)} vs {len(b)}")
    elif [_prompt_key(x) for x in a] != [_prompt_key(x) for x in b]:
        n = sum(1 for x, y in zip(a, b) if _prompt_key(x) != _prompt_key(y))
        problems.append(f"prompt sets differ at {n} of {len(a)} indices")

    # 2. Decoding, checkpoints and dataset must match.
    ca = json.loads(src.run_config.read_text()) if src.run_config.exists() else {}
    cb = json.loads(dst.run_config.read_text()) if dst.run_config.exists() else {}
    for field in MUST_MATCH:
        if field in ca and field in cb and ca[field] != cb[field]:
            problems.append(f"{field} differs between the runs")

    rows = [r for f in src.all_generations() for r in _rows(f)
            if r.get("condition") == condition]
    if not rows:
        problems.append(f"no {condition!r} rows in {src.root}")
        return _fail(problems)

    # 3. The user turn must match, which for the control is the whole stimulus.
    src_suffix = _observed_suffix(rows)
    dst_rows = [r for f in dst.all_generations() for r in _rows(f)]
    dst_suffix = _observed_suffix(dst_rows)
    if len(src_suffix) > 1:
        problems.append(f"source control has inconsistent user suffixes: {src_suffix}")
    if dst_suffix and src_suffix and src_suffix != dst_suffix:
        problems.append(f"user suffix differs: source {src_suffix} vs target "
                        f"{dst_suffix}")

    # 4. A control that carried a system prompt is a different condition.
    systems = {r.get("system_prompt") for r in rows}
    if systems - {"(none)", None, ""}:
        problems.append(f"source {condition!r} rows carry a system prompt: {systems}")

    return _fail(problems) or rows


def _fail(problems: list[str]):
    if problems:
        for p in problems:
            print(f"  ✗ {p}", file=sys.stderr)
        raise SystemExit(
            "refusing to import: the two runs do not agree on something that "
            "changes what the control measures"
        )
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, required=True,
                    help="run directory to copy from")
    ap.add_argument("--to", dest="dst", type=Path, required=True,
                    help="run directory to copy into")
    ap.add_argument("--condition", default="control",
                    help="condition to copy (default: control)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = RunPaths(args.src), RunPaths(args.dst)
    print(f"From: {src.root}\nTo:   {dst.root}")
    rows = check(src, dst, args.condition)
    by_model = {}
    for r in rows:
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1
    print(f"  ✓ prompt sets, decoding and user turn all match")
    print(f"  {len(rows)} {args.condition!r} generations: {by_model}")

    judged_src = [r for r in _rows(src.judged) if r.get("condition") == args.condition]
    print(f"  {len(judged_src)} of them already judged")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    dst.raw.mkdir(parents=True, exist_ok=True)
    gen_out = dst.raw / f"generations__inherited-{args.condition}.jsonl"
    have = {job_key(r) for f in dst.all_generations() for r in _rows(f)}
    new_gen = [r for r in rows if job_key(r) not in have]
    with gen_out.open("a") as f:
        for r in new_gen:
            f.write(json.dumps(r) + "\n")
    print(f"\nGenerations -> {gen_out}  (+{len(new_gen)}, "
          f"{len(rows) - len(new_gen)} already present)")

    done = {job_key(r) for r in _rows(dst.judged)}
    new_judged = [r for r in judged_src if job_key(r) not in done]
    with dst.judged.open("a") as f:
        for r in new_judged:
            f.write(json.dumps(r) + "\n")
    print(f"Judged      -> {dst.judged}  (+{len(new_judged)}, "
          f"{len(judged_src) - len(new_judged)} already present)")

    # Provenance in the target's own manifest: a reader must be able to tell
    # that these rows were not produced by this run.
    if dst.run_config.exists():
        cfg = json.loads(dst.run_config.read_text())
        inherited = cfg.setdefault("inherited_conditions", {})
        inherited[args.condition] = {
            "from_run": src.root.name,
            "generations": len(rows),
            "judged": len(judged_src),
        }
        dst.run_config.write_text(json.dumps(cfg, indent=2))
        print(f"Manifest    -> recorded {args.condition!r} as inherited from "
              f"{src.root.name}")

    print(f"\nNext: scripts/judge/score.py --run {dst.root} "
          f"(the inherited rows are skipped on resume)")


if __name__ == "__main__":
    main()
