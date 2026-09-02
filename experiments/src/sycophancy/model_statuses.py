"""Exp2: status is given to BOTH sides, blocked by domain relevance.

Exp1 varied only the user. Exp2 gives the assistant a status of its own and
crosses the two *within* a relevance block:

    control                      no role for either side
    domain  : user {high,low} x model {high,low}     (4 cells)
    irrel   : user {high,low} x model {high,low}     (4 cells)

Nine cells. Relevance is a property of the block, not of each side separately:
inside ``domain`` both roles come from the item's own expertise bank, and inside
``irrel`` both come from the same generic status dimension. Crossing relevance
per side as well would give sixteen mixed cells whose interpretation
("domain-expert assistant talking to a Supreme Court justice") is murky, and it
is not what the design asks.

Model roles mirror the user banks re-voiced to second person, so both sides draw
on one vocabulary and a high/high cell pairs comparable standings.

:data:`EXP2_SOLO_CONDITIONS` is the same grid with the sides *un*crossed --
each cell dresses exactly one side and leaves the other silent:

    control                      no role for either side
    domain  : user {high,low}, model {high,low}      (4 cells)
    irrel   : user {high,low}, model {high,low}      (4 cells)

Nine cells again, and the same 101 per prompt, but they answer a different
question. In the crossed grid every cell states something about both parties,
so a "user effect" is measured against a model that is always claiming a
standing of its own; the solo grid measures each side against a genuinely
silent partner, which is the comparison the crossed design cannot make.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .config import CATEGORY_TO_DOMAIN, MODEL_STATUSES

BLOCKS = ["domain", "irrel"]
LEVELS = ["high", "low"]
CONTROL = "control"

#: the nine condition ids, in report order
EXP2_CONDITIONS = [CONTROL] + [
    f"{b}_u{u}_m{m}" for b in BLOCKS for u in LEVELS for m in LEVELS
]

#: exp2-solo: one side carries a role and the other carries nothing, so the grid
#: keeps the same shape (control + 2 blocks x 4 cells) but each cell isolates a
#: single side instead of crossing the two.
SIDES = ["u", "m"]
EXP2_SOLO_CONDITIONS = [CONTROL] + [
    f"{b}_{s}{lv}" for b in BLOCKS for s in SIDES for lv in LEVELS
]

_BLOCK_LABEL = {"domain": "Domain", "irrel": "Irrel."}

EXP2_LABEL = {CONTROL: "Control\n(no roles)"}
EXP2_LABEL.update({
    f"{b}_u{u}_m{m}": f"{_BLOCK_LABEL[b]}\nuser {u} / model {m}"
    for b in BLOCKS for u in LEVELS for m in LEVELS
})
EXP2_LABEL.update({
    f"{b}_{s}{lv}": f"{_BLOCK_LABEL[b]}\n{'user' if s == 'u' else 'model'} {lv} only"
    for b in BLOCKS for s in SIDES for lv in LEVELS
})
EXP2_LABEL_FLAT = {k: v.replace("\n", " ") for k, v in EXP2_LABEL.items()}

#: control sends no system message at all -- not a bland assistant persona,
#: which would itself be a role and would not be a true zero point.
CONTROL_SYSTEM_PROMPT = None


def parse_condition(cond: str) -> tuple[str, str | None, str | None] | None:
    """``(block, user_level, model_level)``; ``None`` for control.

    Two id shapes share this parser. The crossed conditions name both sides
    (``domain_uhigh_mlow`` -> ``("domain", "high", "low")``); the solo
    conditions name one, and the absent side comes back as ``None``
    (``domain_mhigh`` -> ``("domain", None, "high")``). A ``None`` level means
    that side gets no clause at all, not a neutral one.
    """
    if cond == CONTROL:
        return None
    parts = cond.split("_")
    if len(parts) == 3:
        block, u, m = parts
        return block, u[1:], m[1:]
    block, spec = parts
    side, level = spec[0], spec[1:]
    if side not in SIDES or level not in LEVELS:
        raise ValueError(f"unparseable condition {cond!r}")
    return block, (level if side == "u" else None), (level if side == "m" else None)


@dataclass(frozen=True)
class Cell:
    """One (user role, model role) pairing for a prompt."""

    condition: str
    block: str
    user_level: str | None
    model_level: str | None
    dimension: str            # domain name, generic dimension, or "none"
    user_label: str | None
    model_label: str | None
    user_clause: str | None
    model_clause: str | None
    pair_idx: int

    @property
    def system_prompt(self) -> str | None:
        """The system message, or None for control (no system message sent)."""
        parts = [p for p in (self.model_clause, self.user_clause) if p]
        return " ".join(parts) if parts else CONTROL_SYSTEM_PROMPT


class ModelStatusBank:
    """The assistant's own standing, mirrored from the user banks."""

    def __init__(self, path: Path = MODEL_STATUSES):
        raw = json.loads(Path(path).read_text())
        self.domain = {g["domain"]: g for g in raw["domain"]}
        self.generic = {g["dimension"]: g for g in raw["generic"]}

    def entries(self, block: str, level: str, key: str) -> list[dict]:
        bank = self.domain if block == "domain" else self.generic
        if key not in bank:
            raise KeyError(f"no model bank for {block}/{key}")
        return bank[key][level]

    def stats(self) -> dict:
        return {
            "domain": {k: {lv: len(v[lv]) for lv in LEVELS}
                       for k, v in self.domain.items()},
            "generic": {k: {lv: len(v[lv]) for lv in LEVELS}
                        for k, v in self.generic.items()},
        }


def build_cells(
    condition: str,
    category: str,
    user_bank,
    model_bank: ModelStatusBank,
    generic_dimensions: list[str],
    pairs_per_cell: int | None = None,
) -> list[Cell]:
    """Every (user, model) pairing to run for one condition on one prompt.

    Roles are paired rather than fully crossed: pairing keeps the cell linear in
    bank size (5 pairs) where a full cross would be quadratic (25), multiplying
    the grid without answering a different question.

    Pairing is a seeded random permutation, not index order. Because the model
    bank mirrors the user bank, index order would pair every persona with itself
    on the same-level cells ("You are a pediatrician. The user is a
    pediatrician."), making those cells structurally unlike the cross-level
    ones. On same-level cells the permutation is a derangement, so the two sides
    always hold different personas.
    """
    parsed = parse_condition(condition)
    if parsed is None:
        return [Cell(CONTROL, "none", None, None, "none",
                     None, None, None, None, 0)]

    block, u_level, m_level = parsed
    if block == "domain":
        # The user bank knows which domain a category maps to -- exp1/exp2 use
        # SAGE's map, exp3 SaLAD's -- so take it from there rather than the
        # module-level default.
        keys = [getattr(user_bank, "category_to_domain", CATEGORY_TO_DOMAIN)
                .get(category)]
        if keys[0] is None:
            raise KeyError(f"no domain bank for category {category!r}")
    else:
        keys = list(generic_dimensions)

    cells = []
    for key in keys:
        if u_level is None:
            u_entries = None
        elif block == "domain":
            u_entries = user_bank.domain_by_name[key][u_level]
        else:
            u_entries = user_bank.generic_by_dim[key][u_level]
        m_entries = (model_bank.entries(block, m_level, key)
                     if m_level is not None else None)

        rng = random.Random(f"{condition}|{category}|{key}")
        # Solo conditions have nothing to pair: the persona on the one side
        # present is the whole cell, and the other side stays empty. Keeping
        # them on the same index/sampling path means a solo cell draws the same
        # five personas a crossed cell would.
        sizes = [len(e) for e in (u_entries, m_entries) if e is not None]
        n = min(sizes)
        perm = (_pairing(n, rng, same_level=(u_level == m_level))
                if u_entries is not None and m_entries is not None
                else list(range(n)))

        idx = list(range(n))
        if pairs_per_cell is not None:
            idx = rng.sample(idx, min(pairs_per_cell, n))

        for i in idx:
            ue = u_entries[i] if u_entries is not None else None
            me = m_entries[perm[i]] if m_entries is not None else None
            cells.append(Cell(
                condition=condition, block=block,
                user_level=u_level, model_level=m_level, dimension=key,
                user_label=ue.get("label") if ue else None,
                model_label=me.get("label") if me else None,
                user_clause=ue["system_prompt_snippet"] if ue else None,
                model_clause=me["system_prompt_snippet"] if me else None,
                pair_idx=i,
            ))
    return cells


def _pairing(n: int, rng: random.Random, same_level: bool) -> list[int]:
    """Permutation mapping user index -> model index.

    With ``same_level`` the permutation is a derangement (no fixed points), so
    a mirrored persona is never paired with itself.
    """
    if n < 2:
        return list(range(n))
    for _ in range(100):
        perm = list(range(n))
        rng.shuffle(perm)
        if not same_level or all(i != perm[i] for i in range(n)):
            return perm
    # Fall back to a rotation, which is a derangement by construction.
    return [(i + 1) % n for i in range(n)]
