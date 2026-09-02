#!/usr/bin/env python3
"""Score generations with the SAGE-Eval judge rubric via gemini-3.7-flash.

Reads ``generations.jsonl``, writes ``judged.jsonl``. Resumes: rows already
judged are skipped, so an interrupted run costs nothing to restart.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy.artifacts import (
    RunPaths, dump_prompts, dump_samples, latest_run,
)
from sycophancy.config import (
    DEFAULT_EXPERIMENT, DEFAULT_RESULTS_DIR, EXPERIMENTS, SALAD_IMAGE_DIR,
    results_dir,
)
from sycophancy.generation import PROMPT_FIELDS, job_key, load_done
from sycophancy import judge_salad as salad_judge
from sycophancy.judge import (
    DEFAULT_JUDGE_MODEL, GeminiJudge, build_prompt, judge_rows, verify_rubric,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=EXPERIMENTS,
                    help="which experiment directory to work in")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="override; defaults to <experiment>/results")
    ap.add_argument("--run", type=Path, default=None,
                    help="run directory (default: the most recent under --results-dir)")
    ap.add_argument("--generations", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--thinking-level", default="low",
                    help="judge reasoning budget; 'none' to omit the field")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="judge at most N rows")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--verify-rubric", action="store_true",
                    help="check the embedded rubric against the live SAGE-Eval "
                         "dataset card, then exit")
    ap.add_argument("--rubric", choices=["sage", "salad"], default=None,
                    help="which rubric to judge with; default: salad for "
                         "--experiment exp3, sage otherwise")
    ap.add_argument("--image-dir", type=Path, default=SALAD_IMAGE_DIR,
                    help="salad rubric only: where the stimulus images live")
    ap.add_argument("--image-max-side", type=int, default=1024,
                    help="--judge-image only: downscale images to this long edge "
                         "before sending them to the judge")
    ap.add_argument("--judge-image", action="store_true",
                    help="salad rubric: show the judge the image and give the "
                         "safe-case rubric the query and reference. More "
                         "informative, but the paper's judge is text-only, so "
                         "the default stays faithful to it and comparable with "
                         "its published numbers")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.verify_rubric:
        ok, message = verify_rubric()
        print(message)
        raise SystemExit(0 if ok else 1)

    args.results_dir = args.results_dir or results_dir(args.experiment)
    paths = RunPaths(args.run) if args.run else latest_run(args.results_dir)
    print(f"Run directory: {paths.root}")
    gen_files = [args.generations] if args.generations else paths.all_generations()
    out_path = args.out or paths.judged

    if not gen_files:
        raise SystemExit(f"no generations under {paths.raw}; run scripts/model/generate.py")

    rows = []
    for gf in gen_files:
        n_before = len(rows)
        rows += [json.loads(l) for l in gf.read_text().splitlines() if l.strip()]
        print(f"  {gf.name}: {len(rows) - n_before} rows")
    print(f"Loaded {len(rows)} generations from {len(gen_files)} file(s)")

    if not args.no_resume:
        done = load_done(out_path)
        rows = [r for r in rows if job_key(r) not in done]
        if done:
            print(f"Resuming: {len(done)} already judged, {len(rows)} to go")
    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        print("Nothing to judge.")
        return

    empty = sum(1 for r in rows if not (r.get("response") or "").strip())
    if empty:
        print(f"NOTE: {empty} rows have an empty response (likely truncated "
              f"thinking); they are scored 'fail' without an API call.")

    rubric = args.rubric or ("salad" if args.experiment == "exp3" else "sage")
    judge_kwargs = dict(
        model=args.judge_model,
        temperature=args.temperature,
        thinking_level=None if args.thinking_level == "none" else args.thinking_level,
    )

    if rubric == "salad":
        # SaLAD scores the two labels with two different rubrics, and the judge
        # is shown the image -- the risk is in the scene, not the words.
        judge = salad_judge.SaladJudge(**judge_kwargs)
        judge_texts = [
            salad_judge.build_salad_prompt(
                r["safety_type"], r["prompt"], r["safety_fact"],
                r.get("response", ""), args.judge_image)
            for r in rows
        ]
        by_label = Counter(r["safety_type"] for r in rows)
        print(f"Rubric: salad {dict(by_label)} — "
              + (f"augmented, image attached (long edge <= "
                 f"{args.image_max_side}px from {args.image_dir})"
                 if args.judge_image else
                 "the paper's protocol, text only"))
    else:
        judge = GeminiJudge(**judge_kwargs)
        judge_texts = [
            build_prompt(r["prompt"], r.get("response", ""), r["safety_fact"])
            for r in rows
        ]

    # Save the exact judge prompts before spending any API calls.
    dump_prompts(paths.judge_prompts / f"{args.judge_model}.txt", rows, judge_texts,
                 PROMPT_FIELDS, f"RENDERED JUDGE PROMPTS — {args.judge_model}")
    dump_samples(paths.samples, rows, judge_texts, PROMPT_FIELDS, "judge")
    print(f"Judge prompts -> {paths.judge_prompts / (args.judge_model + '.txt')}")
    print(f"Judging {len(rows)} rows with {args.judge_model} "
          f"({args.max_workers} workers)")
    if rubric == "salad":
        n = salad_judge.judge_rows(rows, judge, out_path, args.image_dir,
                                   max_workers=args.max_workers,
                                   max_side=args.image_max_side,
                                   with_image=args.judge_image)
    else:
        n = judge_rows(rows, judge, out_path, max_workers=args.max_workers)
    print(f"\nWrote {n} rows -> {out_path}")
    print(f"Next: scripts/analysis/analyze.py --run {paths.root}")


if __name__ == "__main__":
    main()
