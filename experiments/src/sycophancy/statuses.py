"""Resolve conditions to system prompts by drawing personas from the banks.

Both banks hold 5 high and 5 low personas per group (plus 2 unused ``control``
entries): 4 generic dimensions x 12 = 48, and 7 domains x 12 = 84.

The pilot drew exactly one persona per (condition, category) cell, seeded on
that pair. That is reproducible but confounds the status effect with the
identity of the one persona drawn -- "pediatrician" is not interchangeable with
"child protective services caseworker". ``personas_per_cell=None`` instead
enumerates the whole bank so the effect can be averaged over personas.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .config import (
    ALL_GENERIC_DIMENSIONS,
    CATEGORY_TO_DOMAIN,
    DEFAULT_GENERIC_DIMENSIONS,
    DOMAIN_STATUSES,
    GENERIC_STATUSES,
    SYSTEM_PROMPT_TEMPLATE,
)


@dataclass(frozen=True)
class StatusPrompt:
    """A resolved system prompt plus the persona metadata behind it."""

    condition: str
    system_prompt: str | None
    status_label: str | None
    persona_idx: int
    source: str      # "domain", "generic", or "none"
    dimension: str   # domain name, generic dimension name, or "none"

    @property
    def system_prompt_or_none(self) -> str:
        return self.system_prompt or "(none)"


class StatusBank:
    """Loads the two status JSON files and resolves conditions to prompts."""

    def __init__(
        self,
        generic_path: Path = GENERIC_STATUSES,
        domain_path: Path = DOMAIN_STATUSES,
        generic_dimensions: list[str] | None = None,
        category_to_domain: dict[str, str] | None = None,
    ) -> None:
        generic = json.loads(Path(generic_path).read_text())
        domain = json.loads(Path(domain_path).read_text())

        self.generic_by_dim = {d["dimension"]: d for d in generic}
        self.domain_by_name = {d["domain"]: d for d in domain}
        # Which bank a dataset category draws from. exp3 swaps the stimulus set
        # for SaLAD, whose categories are its own; the bank is carried on the
        # instance so every consumer -- including exp2's cell builder -- follows
        # the same mapping without a second config lookup.
        self.category_to_domain = dict(category_to_domain or CATEGORY_TO_DOMAIN)

        dims = list(generic_dimensions or DEFAULT_GENERIC_DIMENSIONS)
        if dims == ["all"]:
            dims = list(ALL_GENERIC_DIMENSIONS)
        unknown = [d for d in dims if d not in self.generic_by_dim]
        if unknown:
            raise KeyError(
                f"unknown generic dimension(s) {unknown}; "
                f"available: {sorted(self.generic_by_dim)}"
            )
        self.generic_dimensions = dims

    @property
    def dimensions(self) -> list[str]:
        return sorted(self.generic_by_dim)

    @property
    def domains(self) -> list[str]:
        return sorted(self.domain_by_name)

    def _groups(self, condition: str, category: str) -> list[tuple[str, str, list[dict]]]:
        """Candidate personas for a cell, as (source, dimension, entries) groups.

        Domain conditions yield one group (the category's own bank). Irrelevant
        conditions yield one group per configured generic dimension, so each
        dimension can be averaged separately before they are combined.
        """
        level = "high" if condition.endswith("_high") else "low"
        if condition.startswith("domain_"):
            domain_key = self.category_to_domain.get(category)
            bank = self.domain_by_name.get(domain_key) if domain_key else None
            if bank is None:
                raise KeyError(
                    f"no domain bank for category {category!r}; "
                    f"add it to CATEGORY_TO_DOMAIN and domain_statuses.json"
                )
            return [("domain", domain_key, bank[level])]
        if condition.startswith("irrel_"):
            return [
                ("generic", dim, self.generic_by_dim[dim][level])
                for dim in self.generic_dimensions
            ]
        raise ValueError(f"unknown condition: {condition!r}")

    def build_all(
        self, condition: str, category: str, personas_per_cell: int | None = 1
    ) -> list[StatusPrompt]:
        """Every persona to run for this cell.

        ``personas_per_cell=None`` returns the whole bank (average over
        personas); an integer returns that many, drawn with a seed fixed by
        (condition, category) so reruns and other models reuse the same picks.
        """
        if condition == "control":
            return [StatusPrompt("control", None, None, 0, "none", "none")]

        out: list[StatusPrompt] = []
        for source, dimension, entries in self._groups(condition, category):
            indices = list(range(len(entries)))
            if personas_per_cell is not None:
                k = min(personas_per_cell, len(entries))
                # Seeded per (condition, category, dimension) so a subset draw is
                # stable across models and reruns.
                indices = random.Random(
                    f"{condition}|{category}|{dimension}"
                ).sample(indices, k)
            out.extend(
                StatusPrompt(
                    condition=condition,
                    system_prompt=SYSTEM_PROMPT_TEMPLATE.format(
                        snippet=entries[i]["system_prompt_snippet"]
                    ),
                    status_label=entries[i].get("label"),
                    persona_idx=i,
                    source=source,
                    dimension=dimension,
                )
                for i in indices
            )
        return out

    def build(self, condition: str, category: str) -> StatusPrompt:
        """Single-persona convenience wrapper (the pilot's behaviour)."""
        return self.build_all(condition, category, personas_per_cell=1)[0]

    def stats(self) -> dict:
        """Persona counts per bank -- reported by the status stats script."""
        def counts(group: dict) -> dict:
            return {k: len(v) for k, v in group.items() if isinstance(v, list)}

        return {
            "active_generic_dimensions": list(self.generic_dimensions),
            "generic_dimensions": {
                d["dimension"]: counts(d) for d in self.generic_by_dim.values()
            },
            "domains": {d["domain"]: counts(d) for d in self.domain_by_name.values()},
            "generic_total": sum(
                sum(counts(d).values()) for d in self.generic_by_dim.values()
            ),
            "domain_total": sum(
                sum(counts(d).values()) for d in self.domain_by_name.values()
            ),
        }
