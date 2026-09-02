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

from .artifacts import RunPaths, dump_prompts, dump_samples
from .config import (
    BASE_COMPLETION_TEMPLATE,
    NO_SUFFIX,
    SYCOPHANCY_SUFFIX,
    THINK_CLOSE_TAG,
    ExperimentConfig,
    ModelVariant,
    SamplingConfig,
    resolve_variants,
    sampling_for,
)
from .statuses import StatusBank

# Identifying fields written above each dumped prompt.
PROMPT_FIELDS = ("model", "prompt_idx", "category", "condition",
                 "status_dimension", "status_label", "persona_idx")


def build_user_message(prompt: str, suffix: str = SYCOPHANCY_SUFFIX) -> str:
    """The user turn: the dataset prompt, plus any pressure phrase.

    exp1/exp2 append "I think it's fine." to a SAGE prompt. exp3 passes
    ``suffix=""`` and sends the SaLAD query verbatim, so its numbers stay
    comparable to the benchmark's published ones.
    """
    return prompt + suffix


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

    def render(self, system_prompt: str | None, user_message: str,
               with_image: bool = False) -> str:
        """Render one prompt to the exact string the model will see.

        ``with_image`` puts an image part ahead of the text in the user turn, so
        the template emits the vision placeholder tokens that vLLM later
        expands against ``multi_modal_data``.
        """
        if not self.variant.is_chat:
            system = f"[System: {system_prompt}]\n" if system_prompt else ""
            return BASE_COMPLETION_TEMPLATE.format(system=system, user=user_message)

        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        content = ([{"type": "image"}, {"type": "text", "text": user_message}]
                   if with_image else user_message)
        msgs.append({"role": "user", "content": content})
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.variant.enable_thinking,
        )

    def sampling_params(self):
        from vllm import SamplingParams

        return SamplingParams(**self.sampling.to_vllm_kwargs())

    def generate(self, texts: list[str], images: list | None = None) -> list[list[dict]]:
        """Run the batch. vLLM schedules internally, so pass everything at once.

        ``images`` is one already-decoded image per prompt (exp3), or None for
        the text-only designs. The same image object is reused across every
        condition that shows it, so 100 stimuli stay 100 decoded images rather
        than one per cell, and vLLM's multimodal cache sees a repeated input.

        Returns one list of ``n`` samples per input prompt.
        """
        inputs: list = texts
        if images is not None:
            inputs = [
                {"prompt": t, "multi_modal_data": {"image": im}} if im is not None
                else {"prompt": t}
                for t, im in zip(texts, images)
            ]
        outputs = self.llm.generate(inputs, self.sampling_params())
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
    suffix: str = SYCOPHANCY_SUFFIX,
) -> list[dict]:
    """Expand the sample into one job per (prompt, condition, persona) cell."""
    jobs = []
    for prompt_idx, item in enumerate(sampled):
        user_msg = build_user_message(item["prompt"], suffix)
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


def build_exp2_jobs(
    sampled: list[dict],
    conditions: list[str],
    user_bank,
    model_bank,
    model_key: str,
    generic_dimensions: list[str],
    pairs_per_cell: int | None = None,
    suffix: str = SYCOPHANCY_SUFFIX,
) -> list[dict]:
    """Expand the sample into one job per exp2 cell.

    Exp2 gives both sides a status, so a cell is identified by
    (condition, dimension, pair) rather than exp1's (condition, persona).
    ``status_dimension`` keeps exp1's field name so the analysis code works
    unchanged across both designs.
    """
    from .model_statuses import build_cells

    jobs = []
    for prompt_idx, item in enumerate(sampled):
        user_msg = build_user_message(item["prompt"], suffix)
        for condition in conditions:
            for cell in build_cells(condition, item["category"], user_bank,
                                    model_bank, generic_dimensions, pairs_per_cell):
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
                    "block": cell.block,
                    "user_level": cell.user_level,
                    "model_level": cell.model_level,
                    "status_dimension": cell.dimension,
                    "persona_idx": cell.pair_idx,
                    "user_label": cell.user_label,
                    "model_label": cell.model_label,
                    "status_source": "exp2",
                    # None for control -- no system message is sent at all.
                    "system_prompt": cell.system_prompt or "(none)",
                })
    return jobs


EXP3_ITEM_FIELDS = ("salad_id", "image", "safety_type", "pair_id", "pair_idx")


def build_exp3_jobs(
    sampled: list[dict],
    conditions: list[str],
    user_bank,
    model_bank,
    model_key: str,
    generic_dimensions: list[str],
    pairs_per_cell: int | None = None,
    suffix: str = NO_SUFFIX,
) -> list[dict]:
    """Exp2's grid over SaLAD items, carrying the image and its gold label.

    The condition structure is exp2's unchanged -- that is the point of exp3,
    which swaps only the stimuli and the model. Each job additionally names the
    image the user showed and whether the item is a genuine hazard or a safe
    request, because the judge picks its rubric from that label.

    ``suffix`` defaults to none, sending the SaLAD query exactly as published.
    Passing exp1/exp2's pressure phrase instead gives a paired run that differs
    from the default one in that phrase alone.
    """
    jobs = build_exp2_jobs(sampled, conditions, user_bank, model_bank,
                           model_key, generic_dimensions, pairs_per_cell,
                           suffix=suffix)
    for job in jobs:
        item = sampled[job["prompt_idx"]]
        job.update({f: item.get(f) for f in EXP3_ITEM_FIELDS})
    return jobs


def load_images(sampled: list[dict], image_dir: Path, max_side: int = 1024) -> list:
    """One decoded, downscaled image per sampled item, in prompt_idx order.

    Downscaling is a real experimental choice, not just a speed knob: Qwen3-VL
    tokenises at native resolution, so a 12-megapixel photo would spend
    thousands of tokens per prompt. A 1024 px long edge keeps the hazards in
    these scenes -- an overloaded power strip, missing goggles -- clearly
    visible while bounding the vision cost per cell.
    """
    from PIL import Image

    image_dir = Path(image_dir)
    out = []
    for item in sampled:
        with Image.open(image_dir / item["image"]) as im:
            im = im.convert("RGB")
            if max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)))
            out.append(im)
    return out


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
    paths: RunPaths | None = None,
    images: list | None = None,
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
    with_image = images is not None
    texts = [
        runner.render(
            None if job["system_prompt"] == "(none)" else job["system_prompt"],
            job["user_msg"],
            with_image,
        )
        for job in pending
    ]
    # One image per *stimulus*, indexed by prompt_idx and shared across the 101
    # cells that show it.
    pending_images = ([images[job["prompt_idx"]] for job in pending]
                      if with_image else None)
    # Save the exact strings sent to the model before generating, so the
    # prompts survive even if the run dies mid-generation.
    if paths is not None:
        dump_prompts(
            paths.model_prompts / f"{variant.key}.txt", pending, texts,
            PROMPT_FIELDS, f"RENDERED MODEL PROMPTS — {variant.display} ({variant.hf_id})",
        )
        dump_samples(paths.samples, pending, texts, PROMPT_FIELDS, variant.key)
        print(f"[{variant.key}] prompts -> {paths.model_prompts / (variant.key + '.txt')}")

    results = runner.generate(texts, pending_images)

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
    cfg: ExperimentConfig,
    sampled: list[dict],
    paths: RunPaths,
    resume: bool = True,
    design: str = "exp1",
    pairs_per_cell: int | None = None,
    image_dir: Path | None = None,
    image_max_side: int = 1024,
    suffix: str | None = None,
    **engine_kwargs,
) -> Path:
    """Run every configured variant in turn, one engine at a time."""
    variants = cfg.variants or resolve_variants(cfg.models)
    images = None
    if design == "exp3":
        from .config import SALAD_IMAGE_DIR
        from .salad import salad_banks

        bank, model_bank = salad_banks(cfg.generic_dimensions)
        image_dir = image_dir or SALAD_IMAGE_DIR
        # Decoded once and reused by every variant and every cell.
        images = load_images(sampled, image_dir, image_max_side)
        print(f"Loaded {len(images)} images from {image_dir} "
              f"(long edge <= {image_max_side}px)")
    else:
        from .model_statuses import ModelStatusBank
        bank, model_bank = StatusBank(generic_dimensions=cfg.generic_dimensions), None

    for model_key in cfg.models:
        variant = variants[model_key]
        sampling = cfg.sampling.get(model_key) or sampling_for(variant)
        out_path = paths.generations_shard(model_key)
        if design == "exp3":
            jobs = build_exp3_jobs(sampled, cfg.conditions, bank, model_bank,
                                   model_key, cfg.generic_dimensions,
                                   pairs_per_cell,
                                   NO_SUFFIX if suffix is None else suffix)
        elif design in ("exp2", "exp2solo"):
            # exp2solo differs only in which conditions cfg carries; the cell
            # builder already knows how to dress one side.
            jobs = build_exp2_jobs(sampled, cfg.conditions, bank, ModelStatusBank(),
                                   model_key, cfg.generic_dimensions, pairs_per_cell,
                                   suffix=cfg.user_suffix)
        else:
            jobs = build_jobs(sampled, cfg.conditions, bank, model_key,
                              cfg.personas_per_cell, suffix=cfg.user_suffix)
        n = run_variant(variant, jobs, out_path, sampling, resume, paths,
                        images, **engine_kwargs)
        if n:
            print(f"[{model_key}] wrote {n} rows -> {out_path}")
    return paths.raw
