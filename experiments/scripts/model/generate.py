#!/usr/bin/env python3
"""Generate model responses across the (prompt x condition x persona) grid with vLLM.

Appends to ``<results-dir>/generations.jsonl``. Rerunning resumes: cells already
present are skipped, so a preempted job picks up where it stopped and adding a
condition only generates the new cells.
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from sycophancy.artifacts import new_run, write_json, write_templates

from sycophancy.config import (
    CONDITIONS, DEFAULT_GENERIC_DIMENSIONS, DEFAULT_MODELS, DEFAULT_PROMPT_TYPES,
    DEFAULT_RESULTS_DIR, MODEL_VARIANTS, SALAD_DATASET_ID, SAMPLING_DEFAULTS,
    SYCOPHANCY_SUFFIX, THINKING,
    ExperimentConfig, resolve_dimensions, resolve_variants, sampling_for,
    DEFAULT_EXPERIMENT, EXPERIMENTS, results_dir, NO_SUFFIX, SYCOPHANCY_SUFFIX,
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
    ap.add_argument("--design", choices=["exp1", "exp2", "exp2solo", "exp3"],
                    default="exp1",
                    help="exp1 = user status only; exp2 = user x model status, "
                         "blocked by relevance; exp2solo = the same grid with "
                         "the sides uncrossed, one dressed side per cell; "
                         "exp3 = exp2's grid over SaLAD image+text stimuli")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="default: the design's full condition set")
    ap.add_argument("--inherit-control", metavar="RUN", default=None,
                    help="drop the no-role control from the grid because it is "
                         "being copied from RUN (a run directory name or path). "
                         "The control sends no system message, so two runs over "
                         "the same prompts and decoding measure the same thing "
                         "there; sharing one draw also puts every "
                         "contrast-against-control on a common baseline. Copy "
                         "the rows with scripts/data/import_control.py")
    ap.add_argument("--pairs-per-cell", type=int, default=0,
                    help="exp2 only: (user, model) pairs per cell; 0 = all 5")
    ap.add_argument("--prompt-types", nargs="+", default=DEFAULT_PROMPT_TYPES,
                    help="SAGE prompt types to keep; pass 'all' for every type")
    ap.add_argument("--anchor-prompt-type", default=None,
                    help="restrict to safety facts that also have a prompt of "
                         "this type, so a multi-type run covers the same facts "
                         "as an earlier single-type one")
    ap.add_argument("--user-suffix", default=None,
                    help="phrase appended to every user turn; pass 'none' to "
                         "send the SAGE prompt verbatim, which removes the "
                         "user's stated opinion and leaves only the status "
                         "manipulation (default: \"%s\")" % SYCOPHANCY_SUFFIX)
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
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=EXPERIMENTS,
                    help="which experiment directory to work in")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="override; defaults to <experiment>/results")
    ap.add_argument("--run-name", default=None,
                    help="run directory name (default: run_<timestamp>); "
                         "reuse a name to resume that run")
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
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="exp3 only: where sample_salad.py extracted the images")
    ap.add_argument("--image-max-side", type=int, default=1024,
                    help="exp3 only: downscale images to this long edge before "
                         "the model sees them (Qwen3-VL tokenises at native "
                         "resolution, so this bounds vision tokens per cell)")
    ap.add_argument("--pressure-suffix", action="store_true",
                    help="exp3 only: append exp1/exp2's \"I think it's fine.\" "
                         "to each query. Off by default, which sends the SaLAD "
                         "query verbatim; turning it on gives a paired run "
                         "differing in that phrase alone")
    ap.add_argument("--no-resume", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    prompt_types = None if args.prompt_types == ["all"] else args.prompt_types
    if args.design == "exp3":
        # SaLAD has no prompt-type axis. Its analogue is the safe/unsafe label,
        # which travels on each item, so recording SAGE's default here would be
        # a lie in the run manifest.
        prompt_types = None
    if args.conditions is None:
        from sycophancy.model_statuses import EXP2_CONDITIONS, EXP2_SOLO_CONDITIONS
        # exp3 reuses exp2's condition grid unchanged -- only the stimuli and
        # the model differ. exp2solo swaps in the uncrossed grid.
        args.conditions = {
            "exp2": EXP2_CONDITIONS, "exp3": EXP2_CONDITIONS,
            "exp2solo": EXP2_SOLO_CONDITIONS,
        }.get(args.design, CONDITIONS)
    if args.inherit_control:
        from sycophancy.model_statuses import CONTROL

        dropped = [c for c in args.conditions if c != CONTROL]
        if len(dropped) == len(args.conditions):
            print(f"NOTE: --inherit-control given but {CONTROL!r} is not in the "
                  f"condition set; nothing dropped.")
        args.conditions = dropped
    personas = None if args.personas_per_cell == 0 else args.personas_per_cell
    if args.user_suffix is None:
        user_suffix = NO_SUFFIX if args.design == "exp3" else SYCOPHANCY_SUFFIX
    else:
        user_suffix = NO_SUFFIX if args.user_suffix.lower() == "none" else args.user_suffix

    args.generic_dimensions = resolve_dimensions(args.generic_dimensions)
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
        anchor_prompt_type=args.anchor_prompt_type,
        user_suffix=user_suffix,
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

    print("Sampling (each checkpoint's own model-card defaults):")
    for key in cfg.models:
        s = sampling[key]
        print(f"  {key:<18} temp={s.temperature} top_p={s.top_p} top_k={s.top_k} "
              f"min_p={s.min_p} max_tokens={s.max_tokens} n={s.n} seed={s.seed}")

    # --experiment names the directory; --results-dir overrides it outright.
    args.results_dir = args.results_dir or results_dir(args.experiment)
    paths = new_run(args.results_dir, args.run_name)
    print(f"Run directory: {paths.root}")
    write_templates(paths, args.design, user_suffix)

    if args.design == "exp3":
        from sycophancy.salad import describe as salad_describe
        from sycophancy.salad import load_sample as load_salad_sample

        sampled = load_salad_sample(image_dir=args.image_dir)
        # The sample is fixed by sample_salad.py, so -n here is a smoke-test
        # truncation of it rather than a fresh draw.
        if cfg.n_samples:
            sampled = sampled[: cfg.n_samples]
        summary = salad_describe(sampled)
        print(f"Loaded {summary['rows']} SaLAD items "
              f"({summary['safety_types']}) across "
              f"{len(summary['categories'])} categories")
    else:
        sampled = load_sample(cfg)
        summary = describe(sampled)
        print(f"Sampled {summary['rows']} prompts covering "
              f"{summary['safety_facts']} safety facts (seed {cfg.seed})")

    # Count the grid exactly rather than estimating: cell size varies by
    # condition (irrel_* spans every configured generic dimension) and by
    # category (each has its own domain bank).
    from collections import Counter

    from sycophancy.generation import build_jobs
    from sycophancy.statuses import StatusBank

    if args.design == "exp3":
        from sycophancy.salad import salad_banks
        bank, model_bank = salad_banks(cfg.generic_dimensions)
    else:
        bank = StatusBank(generic_dimensions=cfg.generic_dimensions)
    if args.design == "exp3":
        from sycophancy.generation import build_exp3_jobs
        pairs = None if args.pairs_per_cell == 0 else args.pairs_per_cell
        probe = build_exp3_jobs(sampled, cfg.conditions, bank, model_bank,
                                cfg.models[0], cfg.generic_dimensions, pairs)
    elif args.design in ("exp2", "exp2solo"):
        from sycophancy.generation import build_exp2_jobs
        from sycophancy.model_statuses import ModelStatusBank
        pairs = None if args.pairs_per_cell == 0 else args.pairs_per_cell
        probe = build_exp2_jobs(sampled, cfg.conditions, bank, ModelStatusBank(),
                                cfg.models[0], cfg.generic_dimensions, pairs,
                                suffix=cfg.user_suffix)
    else:
        probe = build_jobs(sampled, cfg.conditions, bank, cfg.models[0], personas,
                           suffix=cfg.user_suffix)
    per_model = len(probe) * args.samples_per_cell
    total = per_model * len(cfg.models)
    by_cond = Counter(j["condition"] for j in probe)

    print(f"Grid: {per_model} generations per model x {len(cfg.models)} models "
          f"= {total} total")
    print(f"  generic dimensions: {cfg.generic_dimensions}")
    for cond in cfg.conditions:
        n = by_cond.get(cond, 0)
        print(f"    {cond:<12} {n:>5}  ({n / max(1, summary['rows']):.0f} per prompt)")

    write_json(paths.sampled_prompts, sampled)

    # Each variant may be launched as its own process (one per GPU pair), so
    # merge into the manifest rather than overwriting it -- otherwise the last
    # writer wins and the run's own record of its settings is wrong.
    manifest = {}
    if paths.run_config.exists():
        import json as _json
        manifest = _json.loads(paths.run_config.read_text())
    incoming = {
        "n_samples": cfg.n_samples, "seed": cfg.seed, "models": cfg.models,
        "model_ids": {k: v.hf_id for k, v in variants.items()},
        "tensor_parallel_size": args.tensor_parallel_size,
        "design": args.design,
        "conditions": cfg.conditions, "generic_dimensions": cfg.generic_dimensions,
        # Named here even before the rows are copied, so a manifest never
        # describes a nine-condition design while the grid ran eight.
        "inherit_control_from": args.inherit_control,
        "personas_per_cell": cfg.personas_per_cell,
        "prompt_types": cfg.prompt_types, "include_augmented": cfg.include_augmented,
        "anchor_prompt_type": cfg.anchor_prompt_type,
        "user_suffix": cfg.user_suffix,
        "dataset": (SALAD_DATASET_ID if args.design == "exp3"
                    else f"{cfg.dataset_id}:{cfg.dataset_split}"),
        "image_max_side": args.image_max_side if args.design == "exp3" else None,
        "pressure_suffix": (SYCOPHANCY_SUFFIX if args.design != "exp3"
                            or args.pressure_suffix else ""),
        "sampling": {k: vars(v) for k, v in sampling.items()},
        "dataset_summary": summary,
        "grid": {"prompts": summary["rows"], "conditions": len(cfg.conditions),
                 "personas_per_cell": personas, "samples_per_cell": args.samples_per_cell,
                 "generations_this_invocation": per_model * len(cfg.models)},
    }
    # models is a list, the others are dicts -- merge each in its own shape.
    incoming["models"] = sorted(set(manifest.get("models") or []) |
                                set(incoming["models"]))
    for key in ("model_ids", "sampling"):
        merged = dict(manifest.get(key) or {})
        merged.update(incoming[key])
        incoming[key] = merged
    manifest.update(incoming)
    write_json(paths.run_config, manifest)

    engine_kwargs = {"gpu_memory_utilization": args.gpu_memory_utilization}
    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len

    out_path = run_experiment(cfg, sampled, paths, resume=not args.no_resume,
                              design=args.design,
                              pairs_per_cell=(None if args.pairs_per_cell == 0
                                              else args.pairs_per_cell),
                              image_dir=args.image_dir,
                              image_max_side=args.image_max_side,
                              suffix=(SYCOPHANCY_SUFFIX if args.pressure_suffix
                                      else None),
                              **engine_kwargs)
    print(f"\nDone -> {out_path}")
    print(f"Run directory: {paths.root}")
    print(f"Next: scripts/judge/score.py --run {paths.root}")


if __name__ == "__main__":
    main()
