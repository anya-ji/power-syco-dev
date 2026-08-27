#!/usr/bin/env python3
"""Generate model responses across the (prompt x condition x persona) grid with vLLM.

Appends to ``<results-dir>/generations.jsonl``. Rerunning resumes: cells already
present are skipped, so a preempted job picks up where it stopped and adding a
condition only generates the new cells.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy.config import (
    CONDITIONS, DEFAULT_GENERIC_DIMENSIONS, DEFAULT_MODELS, DEFAULT_PROMPT_TYPES,
    DEFAULT_RESULTS_DIR, MODEL_VARIANTS, SAMPLING_DEFAULTS, THINKING,
    ExperimentConfig, resolve_variants, sampling_for,
)
from sycophancy.dataset import describe, load_sample
from sycophancy.generation import run_experiment


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n-samples", type=int, default=None,
                    help="prompts to sample; omit to use every prompt after filtering")
    ap.add_argument("--seed", type=int, default=42, help="dataset subsampling seed")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    choices=list(MODEL_VARIANTS),
                    help=f"variants to run (default: all of {DEFAULT_MODELS})")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--prompt-types", nargs="+", default=DEFAULT_PROMPT_TYPES,
                    help="SAGE prompt types to keep; pass 'all' for every type")
    ap.add_argument("--include-augmented", action="store_true",
                    help="keep the 6 augmented variants of each prompt "
                         "(makes the SAGE model-level score meaningfully stringent)")
    ap.add_argument("--generic-dimensions", nargs="+",
                    default=DEFAULT_GENERIC_DIMENSIONS,
                    help="generic_statuses.json dimensions for irrel_* conditions; "
                         "'all' uses all four (4x more irrel_ personas, so analysis "
                         "averages dimensions equally)")
    ap.add_argument("--personas-per-cell", type=int, default=1,
                    help="personas per (condition, category); 0 = all 5, averaged")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    # Sampling defaults are per-mode, from the Qwen3-8B model card. These flags
    # override every variant at once; omit them to use each variant's own.
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override the per-variant token budget")
    ap.add_argument("--sampling-seed", type=int, default=0,
                    help="vLLM sampling seed; decoding is stochastic, so this "
                         "is what makes a rerun reproducible")
    ap.add_argument("--samples-per-cell", type=int, default=1, metavar="N",
                    help="draw N samples per cell and keep them all, to average "
                         "out sampling noise (default: 1)")
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    prompt_types = None if args.prompt_types == ["all"] else args.prompt_types
    personas = None if args.personas_per_cell == 0 else args.personas_per_cell

    variants = resolve_variants(
        args.models, tensor_parallel_size=args.tensor_parallel_size
    )
    sampling = {
        key: sampling_for(
            variant,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            min_p=args.min_p, max_tokens=args.max_tokens,
            seed=args.sampling_seed, n=args.samples_per_cell,
        )
        for key, variant in variants.items()
    }
    cfg = ExperimentConfig(
        n_samples=args.n_samples, seed=args.seed, models=args.models,
        conditions=args.conditions, generic_dimensions=args.generic_dimensions,
        personas_per_cell=personas, prompt_types=prompt_types,
        include_augmented=args.include_augmented,
        results_dir=args.results_dir, sampling=sampling, variants=variants,
    )

    # Qwen3 explicitly warns against greedy decoding with reasoning enabled.
    if any(
        sampling[m].temperature == 0 and MODEL_VARIANTS[m].mode == THINKING
        for m in cfg.models
    ):
        print("WARNING: temperature=0 with thinking mode. The Qwen3 model card "
              "advises against greedy decoding here -- it can cause endless "
              "repetition, which shows up as a high truncation rate.")

    print("Sampling (per-mode defaults from the Qwen3-8B model card):")
    for key in cfg.models:
        s = sampling[key]
        print(f"  {key:<18} temp={s.temperature} top_p={s.top_p} top_k={s.top_k} "
              f"min_p={s.min_p} max_tokens={s.max_tokens} n={s.n} seed={s.seed}")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    sampled = load_sample(cfg)
    summary = describe(sampled)
    print(f"Sampled {summary['rows']} prompts covering "
          f"{summary['safety_facts']} safety facts (seed {cfg.seed})")

    n_personas = personas or 5
    total = (summary["rows"] * len(cfg.conditions) * n_personas
             * len(cfg.models) * args.samples_per_cell)
    print(f"Grid: {summary['rows']} prompts x {len(cfg.conditions)} conditions "
          f"x ~{n_personas} persona(s) x {len(cfg.models)} models "
          f"x {args.samples_per_cell} sample(s) = ~{total} generations")

    (cfg.results_dir / "sampled_prompts.json").write_text(json.dumps(sampled, indent=2))
    (cfg.results_dir / "run_config.json").write_text(json.dumps({
        "n_samples": cfg.n_samples, "seed": cfg.seed, "models": cfg.models,
        "model_ids": {k: v.hf_id for k, v in variants.items()},
        "tensor_parallel_size": args.tensor_parallel_size,
        "conditions": cfg.conditions, "generic_dimensions": cfg.generic_dimensions,
        "personas_per_cell": cfg.personas_per_cell,
        "prompt_types": cfg.prompt_types, "include_augmented": cfg.include_augmented,
        "dataset": f"{cfg.dataset_id}:{cfg.dataset_split}",
        "sampling": {k: vars(v) for k, v in sampling.items()},
        "dataset_summary": summary,
    }, indent=2))

    engine_kwargs = {"gpu_memory_utilization": args.gpu_memory_utilization}
    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len

    out_path = run_experiment(cfg, sampled, resume=not args.no_resume, **engine_kwargs)
    print(f"\nDone -> {out_path}")
    print("Next: scripts/judge/score.py")


if __name__ == "__main__":
    main()
