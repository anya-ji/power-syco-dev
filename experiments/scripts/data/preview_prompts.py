#!/usr/bin/env python3
"""Show the exact prompts a run would send, without loading any model."""

import argparse

import _bootstrap  # noqa: F401

from sycophancy.config import (
    CONDITIONS, DEFAULT_GENERIC_DIMENSIONS, DEFAULT_PROMPT_TYPES, MODEL_VARIANTS,
    ExperimentConfig,
)
from sycophancy.dataset import load_sample
from sycophancy.generation import build_user_message
from sycophancy.statuses import StatusBank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--n-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generic-dimensions", nargs="+",
                    default=DEFAULT_GENERIC_DIMENSIONS)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--prompt-types", nargs="+", default=DEFAULT_PROMPT_TYPES)
    ap.add_argument("--personas-per-cell", type=int, default=1,
                    help="0 = every persona in the bank")
    ap.add_argument("--show-template", metavar="MODEL_KEY",
                    help="also render the final templated string for this variant "
                         "(loads the tokenizer, not the weights)")
    args = ap.parse_args()

    personas = None if args.personas_per_cell == 0 else args.personas_per_cell
    cfg = ExperimentConfig(n_samples=args.n_samples, seed=args.seed,
                           prompt_types=args.prompt_types)
    sampled = load_sample(cfg)
    bank = StatusBank(generic_dimensions=args.generic_dimensions)

    tokenizer = None
    variant = None
    if args.show_template:
        from transformers import AutoTokenizer
        variant = MODEL_VARIANTS[args.show_template]
        tokenizer = AutoTokenizer.from_pretrained(variant.hf_id, trust_remote_code=True)

    for item in sampled:
        print("=" * 78)
        print(f"[{item['category']} / {item['prompt_type']}] "
              f"{build_user_message(item['prompt'])}")
        print(f"  safety_fact: {item['safety_fact']}")
        for cond in args.conditions:
            for sp in bank.build_all(cond, item["category"], personas):
                label = sp.status_label or "(no role)"
                print(f"\n  {cond:<12} [{sp.persona_idx}] {label}  ({sp.source})")
                print(f"    system: {sp.system_prompt_or_none}")
                if tokenizer is not None and variant.is_chat:
                    msgs = ([{"role": "system", "content": sp.system_prompt}]
                            if sp.system_prompt else [])
                    msgs.append({"role": "user",
                                 "content": build_user_message(item["prompt"])})
                    rendered = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=variant.enable_thinking)
                    print(f"    templated: {rendered!r}")
        print()


if __name__ == "__main__":
    main()
