"""Run directory layout and human-readable prompt dumps.

Everything a run produces lands under one timestamped directory so a run is
self-contained and reproducible:

    <results>/<run_name>/
      run_config.json          settings + dataset summary
      sampled_prompts.json     the exact SAGE rows used
      prompts/templates/*.txt  the templates, verbatim
      prompts/model/*.txt      every rendered model prompt
      prompts/judge/*.txt      every rendered judge prompt
      prompts/samples/*.txt    a few single prompts for eyeballing
      raw/generations.jsonl    raw model output + parsed answer
      raw/judged.jsonl         raw judge output + parsed verdict
      tables/*.csv             aggregates
      figures/*.png            plots for browsing
      report/*.pdf             vector plots + the compiled report
      dashboard.html
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SEP = "=" * 100


@dataclass(frozen=True)
class RunPaths:
    """Every path a run writes to."""

    root: Path

    @property
    def prompts(self) -> Path: return self.root / "prompts"
    @property
    def templates(self) -> Path: return self.prompts / "templates"
    @property
    def model_prompts(self) -> Path: return self.prompts / "model"
    @property
    def judge_prompts(self) -> Path: return self.prompts / "judge"
    @property
    def samples(self) -> Path: return self.prompts / "samples"
    @property
    def raw(self) -> Path: return self.root / "raw"
    @property
    def tables(self) -> Path: return self.root / "tables"
    @property
    def figures(self) -> Path: return self.root / "figures"
    @property
    def generations(self) -> Path: return self.raw / "generations.jsonl"

    def generations_shard(self, model_key: str) -> Path:
        """Per-model output file.

        Each model writes its own shard so variants can run concurrently on
        separate GPUs without interleaving lines in a shared file.
        """
        return self.raw / f"generations__{model_key}.jsonl"

    def all_generations(self) -> list[Path]:
        """Every generation file present, shards first, then any merged file."""
        shards = sorted(self.raw.glob("generations__*.jsonl"))
        return shards or ([self.generations] if self.generations.exists() else [])
    @property
    def judged(self) -> Path: return self.raw / "judged.jsonl"
    @property
    def run_config(self) -> Path: return self.root / "run_config.json"
    @property
    def sampled_prompts(self) -> Path: return self.root / "sampled_prompts.json"
    @property
    def dashboard(self) -> Path: return self.root / "dashboard.html"

    def mkdirs(self) -> "RunPaths":
        for d in (self.templates, self.model_prompts, self.judge_prompts,
                  self.samples, self.raw, self.tables, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        return self


def new_run(results_dir: Path, name: str | None = None) -> RunPaths:
    """Resolve a run directory, defaulting to a UTC timestamp."""
    name = name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return RunPaths(Path(results_dir) / name).mkdirs()


def latest_run(results_dir: Path) -> RunPaths:
    """The most recently created run directory."""
    runs = sorted(
        (p for p in Path(results_dir).iterdir() if p.is_dir() and (p / "raw").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"no runs found under {results_dir}")
    return RunPaths(runs[-1])


def write_templates(paths: RunPaths, design: str = "exp1",
                    user_suffix: str | None = None) -> None:
    """Dump the prompt templates verbatim, before any substitution.

    ``user_suffix`` is what the run actually appended, which is not always the
    module default -- recording the default instead would misdescribe a run
    launched with the phrase turned off.
    """
    from .config import BASE_COMPLETION_TEMPLATE, SYCOPHANCY_SUFFIX, SYSTEM_PROMPT_TEMPLATE
    from .judge import JUDGE_PROMPT

    suffix = SYCOPHANCY_SUFFIX if user_suffix is None else user_suffix
    files = {
        "system_prompt_template.txt": SYSTEM_PROMPT_TEMPLATE,
        "sycophancy_suffix.txt": suffix or "(none -- the SAGE prompt is sent verbatim)",
        "base_completion_format.txt": BASE_COMPLETION_TEMPLATE,
        "judge_rubric.txt": JUDGE_PROMPT,
    }
    if design == "exp3":
        # exp3 sends the query verbatim and is judged by SaLAD's own two
        # rubrics, not SAGE's one.
        files["sycophancy_suffix.txt"] = (
            "(none -- exp3 sends the SaLAD query exactly as published)"
        )
        from .judge_salad import (
            IMAGE_SAFE_PROMPT, IMAGE_UNSAFE_PROMPT, PAPER_SAFE_PROMPT,
            PAPER_UNSAFE_PROMPT, RUBRIC_SOURCE,
        )

        files.update({
            "judge_rubric_salad_unsafe.txt": PAPER_UNSAFE_PROMPT,
            "judge_rubric_salad_safe.txt": PAPER_SAFE_PROMPT,
            "judge_rubric_salad_unsafe_with_image.txt": IMAGE_UNSAFE_PROMPT,
            "judge_rubric_salad_safe_with_image.txt": IMAGE_SAFE_PROMPT,
            "judge_rubric_salad_source.txt": RUBRIC_SOURCE,
        })
    for name, text in files.items():
        (paths.templates / name).write_text(text.rstrip("\n") + "\n")
    print(f"  templates -> {paths.templates} ({len(files)} files)")


def header(record: dict, fields: tuple[str, ...]) -> str:
    parts = [f"{f}={record.get(f)!r}" for f in fields if f in record]
    return " | ".join(parts)


def dump_prompts(
    path: Path,
    records: list[dict],
    texts: list[str],
    fields: tuple[str, ...],
    title: str,
) -> None:
    """Append rendered prompts to one readable txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        if path.stat().st_size == 0:
            f.write(f"{title}\n{SEP}\n\n")
        for rec, text in zip(records, texts):
            f.write(f"{SEP}\n{header(rec, fields)}\n{SEP}\n{text}\n\n")


def dump_samples(
    directory: Path, records: list[dict], texts: list[str],
    fields: tuple[str, ...], prefix: str, n: int = 6,
) -> None:
    """Write a handful of individual prompt files for quick inspection."""
    directory.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    written = 0
    for rec, text in zip(records, texts):
        key = (rec.get("condition"), rec.get("status_dimension"))
        if key in seen:
            continue
        seen.add(key)
        name = f"{prefix}__{rec.get('condition','na')}__{rec.get('status_dimension','na')}.txt"
        (directory / name.replace("/", "_")).write_text(
            f"{header(rec, fields)}\n{SEP}\n{text}\n"
        )
        written += 1
        if written >= n:
            break


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
