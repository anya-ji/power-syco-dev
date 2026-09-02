"""Exp2-solo inference: one dressed side per cell, silent partner.

Same estimator as :mod:`analysis_exp2` -- a cluster bootstrap whose resampling
unit is the safety fact -- reusing that module's engine outright. Only the
contrasts differ, because the design does.

The crossed grid has a 2x2 inside each block, so its natural quantities are two
main effects and an interaction. The solo grid has no 2x2: inside a block the
four cells are four *separate* arms (user high, user low, model high, model
low), each against a partner that says nothing. Two families of contrast come
out of that:

``level``
    ``high - low`` within one side. The same question exp2's main effects ask,
    but now with nothing on the other side to interact with, so it is the
    effect of status per se rather than status-given-the-partner-also-claims-one.

``presence``
    ``mean(high, low) - control``. What merely *having* a role on that side does,
    averaged over which level it is. This is the quantity the crossed design
    cannot estimate: there, every non-control cell dresses both sides at once,
    so "has a role" and "which role" are never separated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis_exp2 import N_BOOT, _grid, _resample, _summarise, stars  # noqa: F401
from .model_statuses import BLOCKS, CONTROL, EXP2_SOLO_CONDITIONS, LEVELS, SIDES

SIDE_NAME = {"u": "user", "m": "model"}
EFFECTS = ["user (high-low)", "model (high-low)"]
PRESENCE = ["user role (vs control)", "model role (vs control)"]

#: the eight dressed cells, ordered so each block's four are contiguous as
#: u-high, u-low, m-high, m-low -- the order the slicing below assumes
SOLO_CONDITIONS = [f"{b}_{s}{lv}" for b in BLOCKS for s in SIDES for lv in LEVELS]


def _arms(r: np.ndarray) -> dict:
    """Level and presence contrasts from four cells ordered uh, ul, mh, ml.

    ``r`` carries the control in its last column, so presence has something to
    subtract; the level contrasts ignore it.
    """
    uh, ul, mh, ml, ctrl = (r[..., i] for i in range(5))
    return {
        "user (high-low)": uh - ul,
        "model (high-low)": mh - ml,
        "user role (vs control)": (uh + ul) / 2 - ctrl,
        "model role (vs control)": (mh + ml) / 2 - ctrl,
    }


def _solo(df, group_keys, models, n_boot, seed, blocks=None):
    """Shared body: solo contrasts within each (group, block).

    The control is one cell for the whole run, not one per block, so it is
    fetched once and appended to each block's four columns.
    """
    blocks = blocks or BLOCKS
    conds = [c for c in SOLO_CONDITIONS
             if c.split("_")[0] in blocks] + [CONTROL]
    keys = ["model", *group_keys] if models is not None else list(group_keys)
    d = df[df["model"].isin(models)] if models is not None else df
    _, groups, S, N = _grid(d, keys, conds)
    obs, boot = _resample(S, N, n_boot, seed)
    ci = len(conds) - 1                       # control sits last

    out = []
    for gi, g in enumerate(groups):
        for bi, block in enumerate(blocks):
            cols = list(range(bi * 4, bi * 4 + 4)) + [ci]
            o = _arms(obs[gi, cols][None, :])
            b = _arms(boot[:, gi, cols])
            for name in EFFECTS + PRESENCE:
                out.append({**dict(zip(keys, g)), "block": block, "effect": name,
                            **_summarise(float(o[name][0]), b[name])})
    return pd.DataFrame(out)


def solo_effects(df: pd.DataFrame, models: list[str] | None = None,
                 extra_group: str | None = None, n_boot: int = N_BOOT,
                 seed: int = 0) -> pd.DataFrame:
    """Level and presence contrasts per (model, block).

    ``extra_group`` splits the estimates further -- pass ``"prompt_type"`` to get
    the same four contrasts within each prompt template.
    """
    models = models or sorted(df["model"].unique())
    return _solo(df, [extra_group] if extra_group else [], models, n_boot, seed)


def condition_means(df: pd.DataFrame, models: list[str] | None = None,
                    n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Pass rate per (model, condition) with a cluster-bootstrap interval."""
    models = models or sorted(df["model"].unique())
    conds = [c for c in EXP2_SOLO_CONDITIONS if c in set(df["condition"])]
    _, groups, S, N = _grid(df[df["model"].isin(models)], ["model"], conds)
    obs, boot = _resample(S, N, n_boot, seed)
    out = []
    for gi, (model,) in enumerate(groups):
        for ci, cond in enumerate(conds):
            r = _summarise(float(obs[gi, ci]), boot[:, gi, ci])
            r.pop("p")      # a p against zero is meaningless for a mean
            out.append({"model": model, "condition": cond, "rate": r.pop("estimate"),
                        **r, "n": int(N[:, gi, ci].sum())})
    return pd.DataFrame(out)


def arm_means(df: pd.DataFrame, models: list[str] | None = None,
              n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Condition means labelled by (block, side, level), carrying their intervals.

    The solo grid's analogue of :func:`analysis_exp2.cell_means`: there the
    non-control conditions land on a 2x2, here they land on a
    block x side x level list.
    """
    cm = condition_means(df, models, n_boot, seed)
    cm = cm[cm["condition"] != CONTROL].copy()
    parts = cm["condition"].str.extract(
        r"^(?P<block>\w+?)_(?P<side>[um])(?P<level>high|low)$")
    cm["block"] = parts["block"]
    cm["side"] = parts["side"].map(SIDE_NAME)
    cm["level"] = parts["level"]
    return cm


def vs_control(df: pd.DataFrame, models: list[str] | None = None,
               n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Each condition minus the no-role control, per model.

    Unadjusted and descriptive, as in exp2 -- but note these carry more weight
    here than there: with one side silent, a condition-vs-control difference is
    itself the effect of that one role, not a mixture of two.
    """
    models = models or sorted(df["model"].unique())
    conds = [c for c in EXP2_SOLO_CONDITIONS if c in set(df["condition"])]
    _, groups, S, N = _grid(df[df["model"].isin(models)], ["model"], conds)
    obs, boot = _resample(S, N, n_boot, seed)
    ctrl = conds.index(CONTROL)
    out = []
    for gi, (model,) in enumerate(groups):
        for ci, cond in enumerate(conds):
            if cond == CONTROL:
                continue
            r = _summarise(float(obs[gi, ci] - obs[gi, ctrl]),
                           boot[:, gi, ci] - boot[:, gi, ctrl])
            out.append({"model": model, "condition": cond,
                        "delta_vs_control": r.pop("estimate"), **r})
    return pd.DataFrame(out)


def dimension_effects(df: pd.DataFrame, models: list[str] | None = None,
                      n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Solo contrasts within each generic status channel (irrel block).

    Control carries no dimension, so it cannot be grouped by one. It is
    broadcast across the dimensions instead -- the same no-role baseline for
    each -- which is what makes the presence contrasts comparable across
    channels.
    """
    models = models or sorted(df["model"].unique())
    sub = df[(df["block"] == "irrel") | (df["condition"] == CONTROL)].copy()
    dims = sorted(sub.loc[sub["block"] == "irrel", "status_dimension"].dropna().unique())
    ctrl = sub[sub["condition"] == CONTROL]
    parts = [sub[sub["condition"] != CONTROL]]
    for dim in dims:
        parts.append(ctrl.assign(status_dimension=dim))
    out = _solo(pd.concat(parts, ignore_index=True), ["status_dimension"],
                models, n_boot, seed, blocks=["irrel"])
    return out.rename(columns={"status_dimension": "dimension"})


def category_effects(df: pd.DataFrame, model: str, n_boot: int = N_BOOT,
                     seed: int = 0) -> pd.DataFrame:
    """Model-status level effect per safety category, per block, for one model."""
    out = _solo(df[df["model"] == model], ["category"], None, n_boot, seed)
    return out[out["effect"] == "model (high-low)"].drop(columns="effect")


def allpass_ci(df: pd.DataFrame, models: list[str] | None = None,
               n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Bootstrap interval for the all-persona pass rate, per condition."""
    from .analysis import collapse_samples

    models = models or sorted(df["model"].unique())
    per_fact = (collapse_samples(df)
                .groupby(["model", "condition", "safety_fact"])["passes"]
                .min().reset_index())
    rng = np.random.default_rng(seed)
    out = []
    for model in models:
        for cond in [c for c in EXP2_SOLO_CONDITIONS if c in set(df["condition"])]:
            v = per_fact[(per_fact["model"] == model)
                         & (per_fact["condition"] == cond)]["passes"].to_numpy()
            if v.size == 0:
                continue
            draws = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            out.append({"model": model, "condition": cond,
                        "se": float(draws.std(ddof=1)), "ci_lo": float(lo),
                        "ci_hi": float(hi), "n_facts": int(v.size)})
    return pd.DataFrame(out)


def side_asymmetry(df: pd.DataFrame, models: list[str] | None = None,
                   n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Does the model's own status move it more than the user's does?

    ``|model level effect| - |user level effect|`` is not a clean contrast, so
    this reports the signed difference of the two level effects and of the two
    presence effects instead. Both are computed inside one bootstrap draw, so
    the correlation between the two sides is carried rather than assumed away --
    which a difference of two separately-published intervals would not do.
    """
    models = models or sorted(df["model"].unique())
    conds = SOLO_CONDITIONS + [CONTROL]
    _, groups, S, N = _grid(df[df["model"].isin(models)], ["model"], conds)
    obs, boot = _resample(S, N, n_boot, seed)
    ci = len(conds) - 1
    out = []
    for gi, (model,) in enumerate(groups):
        for bi, block in enumerate(BLOCKS):
            cols = list(range(bi * 4, bi * 4 + 4)) + [ci]
            o, b = _arms(obs[gi, cols][None, :]), _arms(boot[:, gi, cols])
            out.append({"model": model, "block": block,
                        "contrast": "model level - user level",
                        **_summarise(
                            float(o["model (high-low)"][0] - o["user (high-low)"][0]),
                            b["model (high-low)"] - b["user (high-low)"])})
            out.append({"model": model, "block": block,
                        "contrast": "model presence - user presence",
                        **_summarise(
                            float(o["model role (vs control)"][0]
                                  - o["user role (vs control)"][0]),
                            b["model role (vs control)"]
                            - b["user role (vs control)"])})
    return pd.DataFrame(out)
