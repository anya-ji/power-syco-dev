"""vLLM generation across the (prompt x condition x persona) grid.

Prompts are rendered to strings up front with the HF tokenizer and handed to
``LLM.generate`` as text, rather than going through ``LLM.chat``. That keeps one
code path for all three variants: thinking and non-thinking differ only by the
chat template's ``enable_thinking`` flag, and the base checkpoint -- which has
no chat template at all -- falls back to a raw completion format.

Results stream to JSONL and reruns resume, so a preempted job does not redo
finished cells.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import (
    SYCOPHANCY_SUFFIX,
    THINK_CLOSE_TAG,
    ExperimentConfig,
    ModelVariant,
    SamplingConfig,
    resolve_variants,
    sampling_for,
)
from .statuses import StatusBank

# Base models have no instruction format; this mirrors the pilot's framing.
BASE_TEMPLATE = "{system}User: {user}\nAssistant:"


def build_user_message(prompt: str) -> str:
    """Append the sycophancy pressure phrase to a SAGE-Eval prompt."""
    return prompt + SYCOPHANCY_SUFFIX


def split_thinking(text: str) -> tuple[str, str]:
    """Split a thinking-mode completion into (reasoning, answer).

    Returns ``("", text)`` when no ``</think>`` is present -- which for a
    thinking model usually means the token budget ran out mid-reasoning, leaving
    no answer to judge.
    """
    if THINK_CLOSE_TAG not in text:
        return "", text
    reasoning, _, answer = text.partition(THINK_CLOSE_TAG)
    return reasoning.removeprefix("<think>").strip(), answer.strip()


class VLLMRunner:
    """One loaded vLLM engine for a single model variant."""

    def __init__(self, variant: ModelVariant, sampling: SamplingConfig, **engine_kwargs):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.variant = variant
        self.sampling = sampling
        self.tokenizer = AutoTokenizer.from_pretrained(variant.hf_id, trust_remote_code=True)

        print(f"Loading {variant.display} ({variant.hf_id}) "
              f"on {variant.tensor_parallel_size} GPU(s)")
        self.llm = LLM(
            model=variant.hf_id,
            tensor_parallel_size=variant.tensor_parallel_size,
            trust_remote_code=True,
            **engine_kwargs,
        )

    def render(self, system_prompt: str | None, user_message: str) -> str:
        """Render one prompt to the exact string the model will see."""
        if not self.variant.is_chat:
            system = f"[System: {system_prompt}]\n" if system_prompt else ""
            return BASE_TEMPLATE.format(system=system, user=user_message)

        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_message})
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.variant.enable_thinking,
        )

    def sampling_params(self):
        from vllm import SamplingParams

        return SamplingParams(**self.sampling.to_vllm_kwargs())

    def generate(self, texts: list[str]) -> list[list[dict]]:
        """Run the batch. vLLM schedules internally, so pass everything at once.

        Returns one list of ``n`` samples per input prompt.
        """
        outputs = self.llm.generate(texts, self.sampling_params())
        results = []
        for out in outputs:
            samples = []
            for sample_idx, completion in enumerate(out.outputs):
                raw = completion.text
                reasoning, answer = (
                    split_thinking(raw) if self.variant.enable_thinking
                    else ("", raw.strip())
                )
                samples.append({
                    "sample_idx": sample_idx,
                    "raw_output": raw,
                    "reasoning": reasoning,
                    "response": answer,
                    "finish_reason": completion.finish_reason,
                    "n_output_tokens": len(completion.token_ids),
                    # A thinking run that hit the cap before </think> has no answer.
                    "truncated": completion.finish_reason == "length",
                })
            results.append(samples)
        return results


def build_jobs(
    sampled: list[dict],
    conditions: list[str],
    bank: StatusBank,
    model_key: str,
    personas_per_cell: int | None = 1,
) -> list[dict]:
    """Expand the sample into one job per (prompt, condition, persona) cell."""
    jobs = []
    for prompt_idx, item in enumerate(sampled):
        user_msg = build_user_message(item["prompt"])
        for condition in conditions:
            for sp in bank.build_all(condition, item["category"], personas_per_cell):
                jobs.append({
                    "model": model_key,
                    "prompt_idx": prompt_idx,
                    "safety_fact": item["safety_fact"],
                    "category": item["category"],
                    "prompt_type": item.get("prompt_type"),
                    "augmentation_type": item.get("augmentation_type"),
                    "prompt": item["prompt"],
                    "user_msg": user_msg,
                    "condition": condition,
                    "persona_idx": sp.persona_idx,
                    "status_label": sp.status_label,
                    "status_source": sp.source,
                    "status_dimension": sp.dimension,
                    "system_prompt": sp.system_prompt_or_none,
                })
    return jobs


def job_key(record: dict) -> tuple:
    """Identity of one generation, used to skip work already on disk."""
    return (
        record["model"],
        record["prompt_idx"],
        record["condition"],
        record.get("status_dimension", "none"),
        record.get("persona_idx", 0),
        record.get("sample_idx", 0),
    )


def load_done(path: Path) -> set[tuple]:
    """Read the keys of rows already written to a JSONL file."""
    if not Path(path).exists():
        return set()
    done = set()
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(job_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn final line from a killed job
    return done


def run_variant(
    variant: ModelVariant,
    jobs: list[dict],
    out_path: Path,
    sampling: SamplingConfig,
    resume: bool = True,
    **engine_kwargs,
) -> int:
    """Generate every pending job for one model variant."""
    out_path = Path(out_path)
    # A cell is done only when all n of its samples are on disk; a partially
    # written cell is regenerated whole so the samples stay from one batch.
    done = load_done(out_path) if resume else set()
    pending = [
        j for j in jobs
        if not all(
            (*job_key(j)[:5], i) in done for i in range(sampling.n)
        )
    ]
    if not pending:
        print(f"[{variant.key}] already complete, skipping model load")
        return 0
    if done:
        print(f"[{variant.key}] resuming: {len(pending)} of {len(jobs)} cells remain")

    runner = VLLMRunner(variant, sampling, **engine_kwargs)
    texts = [
        runner.render(
            None if job["system_prompt"] == "(none)" else job["system_prompt"],
            job["user_msg"],
        )
        for job in pending
    ]
    results = runner.generate(texts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    n_trunc = 0
    with out_path.open("a") as f:
        for job, samples in zip(pending, results):
            for sample in samples:
                f.write(json.dumps({**job, "model_id": variant.hf_id,
                                    "mode": variant.mode, **sample}) + "\n")
                written += 1
                n_trunc += sample["truncated"]

    if n_trunc:
        print(f"[{variant.key}] WARNING: {n_trunc}/{written} hit the "
              f"{sampling.max_tokens} token cap"
              + (" (thinking truncated, no answer to judge)"
                 if variant.enable_thinking else ""))
    return written


def run_experiment(
    cfg: ExperimentConfig, sampled: list[dict], resume: bool = True, **engine_kwargs
) -> Path:
    """Run every configured variant in turn, one engine at a time."""
    variants = cfg.variants or resolve_variants(cfg.models)

    bank = StatusBank(generic_dimensions=cfg.generic_dimensions)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.results_dir / "generations.jsonl"

    for model_key in cfg.models:
        variant = variants[model_key]
        sampling = cfg.sampling.get(model_key) or sampling_for(variant)
        jobs = build_jobs(sampled, cfg.conditions, bank, model_key, cfg.personas_per_cell)
        n = run_variant(variant, jobs, out_path, sampling, resume, **engine_kwargs)
        if n:
            print(f"[{model_key}] wrote {n} rows -> {out_path}")
    return out_path
