#!/usr/bin/env python3
"""Score generations with the SAGE-Eval judge rubric via gemini-3.7-flash.

Reads ``generations.jsonl``, writes ``judged.jsonl``. Resumes: rows already
judged are skipped, so an interrupted run costs nothing to restart.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy.config import DEFAULT_RESULTS_DIR
from sycophancy.generation import job_key, load_done
from sycophancy.judge import (
    DEFAULT_JUDGE_MODEL, GeminiJudge, judge_rows, verify_rubric,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--generations", type=Path, default=None,
                    help="default: <results-dir>/generations.jsonl")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <results-dir>/judged.jsonl")
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
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.verify_rubric:
        ok, message = verify_rubric()
        print(message)
        raise SystemExit(0 if ok else 1)

    gen_path = args.generations or args.results_dir / "generations.jsonl"
    out_path = args.out or args.results_dir / "judged.jsonl"

    if not gen_path.exists():
        raise SystemExit(f"no generations at {gen_path}; run scripts/model/generate.py")

    rows = [json.loads(line) for line in gen_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(rows)} generations from {gen_path}")

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

    judge = GeminiJudge(
        model=args.judge_model,
        temperature=args.temperature,
        thinking_level=None if args.thinking_level == "none" else args.thinking_level,
    )
    print(f"Judging {len(rows)} rows with {args.judge_model} "
          f"({args.max_workers} workers)")
    n = judge_rows(rows, judge, out_path, max_workers=args.max_workers)
    print(f"\nWrote {n} rows -> {out_path}")
    print("Next: scripts/analysis/analyze.py")


if __name__ == "__main__":
    main()
