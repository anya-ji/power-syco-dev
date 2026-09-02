"""Exp2 inference: cluster bootstrap over safety facts.

This is the primary path -- CS/NLP venues expect bootstrap intervals, and the
bootstrap assumes nothing about how the outcome is distributed. The
mixed-effects fits in :mod:`mixed_models` are kept as a cross-check; the two
agree on every effect in this data.

**The resampling unit is the safety fact, not the row.** One fact contributes up
to three prompt templates times 101 persona cells, and every prompt is re-asked
under every condition, so rows within a fact are heavily correlated. Resampling
rows would treat those as independent evidence and produce intervals roughly a
third too narrow. Each draw takes 84 facts with replacement and recomputes the
statistic from scratch.

Everything here runs through one engine: build a (fact x group x condition)
table of pass counts and totals once, then express each quantity -- condition
means, factorial effects, per-dimension effects, differences from control -- as
arithmetic on the resampled rates. The resampling itself is a single matmul
against multinomial fact counts, which is why 4000 draws cost about a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model_statuses import BLOCKS, CONTROL, EXP2_CONDITIONS, LEVELS

EFFECTS = ["user (high-low)", "model (high-low)", "interaction"]
#: the eight 2x2 cells, ordered so each block's four are contiguous as hh/hl/lh/ll
FACTORIAL_CONDITIONS = [f"{b}_u{u}_m{m}"
                        for b in BLOCKS for u in LEVELS for m in LEVELS]
N_BOOT = 4000


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _grid(df: pd.DataFrame, group_keys: list[str], conds: list[str],
          value: str = "passes"):
    """Pass counts and totals as (fact, group, condition) arrays."""
    sub = df[df["condition"].isin(conds)]
    agg = (sub.groupby(["safety_fact", *group_keys, "condition"], observed=True)[value]
           .agg(["sum", "count"]).reset_index())
    facts = sorted(agg["safety_fact"].unique())
    groups = ([tuple(g) for g in sorted({tuple(r) for r in agg[group_keys].to_numpy()})]
              if group_keys else [()])
    fidx = {f: i for i, f in enumerate(facts)}
    gidx = {g: i for i, g in enumerate(groups)}
    cidx = {c: i for i, c in enumerate(conds)}

    fi = agg["safety_fact"].map(fidx).to_numpy()
    gi = (np.array([gidx[tuple(r)] for r in agg[group_keys].to_numpy()])
          if group_keys else np.zeros(len(agg), int))
    ci = agg["condition"].map(cidx).to_numpy()
    shape = (len(facts), len(groups), len(conds))
    S, N = np.zeros(shape), np.zeros(shape)
    np.add.at(S, (fi, gi, ci), agg["sum"].to_numpy(dtype=float))
    np.add.at(N, (fi, gi, ci), agg["count"].to_numpy(dtype=float))
    return facts, groups, S, N


def _resample(S: np.ndarray, N: np.ndarray, n_boot: int, seed: int):
    """Observed rates, and ``n_boot`` cluster-resampled rates.

    A bootstrap resample is exactly a multinomial draw of how many times each
    fact is included, so the whole resampling collapses into two matmuls.
    """
    nf = S.shape[0]
    flat_s, flat_n = S.reshape(nf, -1), N.reshape(nf, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = np.where(flat_n.sum(0) > 0,
                       flat_s.sum(0) / np.maximum(flat_n.sum(0), 1), np.nan)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(nf, np.full(nf, 1.0 / nf), size=n_boot).astype(float)
    num, den = W @ flat_s, W @ flat_n
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return obs.reshape(S.shape[1:]), boot.reshape((n_boot, *S.shape[1:]))


def _summarise(est: float, draws: np.ndarray) -> dict:
    """Percentile interval and a two-sided bootstrap p-value.

    The ``(r+1)/(B+1)`` convention keeps p away from an exact zero, which is not
    a claim any finite resampling can support.
    """
    a = draws[np.isfinite(draws)]
    if a.size == 0:
        return {"estimate": float(est), "se": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p": np.nan}
    lo, hi = np.percentile(a, [2.5, 97.5])
    tail = min(int((a <= 0).sum()), int((a >= 0).sum()))
    return {"estimate": float(est), "se": float(a.std(ddof=1)),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(1.0, 2 * (tail + 1) / (a.size + 1)))}


def _block_effects(r: np.ndarray) -> dict:
    """The three factorial quantities from four cells ordered hh, hl, lh, ll."""
    hh, hl, lh, ll = (r[..., i] for i in range(4))
    return {"user (high-low)": (hh + hl) / 2 - (lh + ll) / 2,
            "model (high-low)": (hh + lh) / 2 - (hl + ll) / 2,
            "interaction": (hh - hl) - (lh - ll)}


def _factorial(df, group_keys, models, n_boot, seed, conds=None, blocks=None):
    """Shared body: factorial effects within each (group, block)."""
    conds = conds or FACTORIAL_CONDITIONS
    blocks = blocks or BLOCKS
    keys = ["model", *group_keys] if models is not None else list(group_keys)
    d = df[df["model"].isin(models)] if models is not None else df
    _, groups, S, N = _grid(d, keys, conds)
    obs, boot = _resample(S, N, n_boot, seed)

    out = []
    for gi, g in enumerate(groups):
        for bi, block in enumerate(blocks):
            sl = slice(bi * 4, bi * 4 + 4)
            o = _block_effects(obs[gi, sl][None, :])
            b = _block_effects(boot[:, gi, sl])
            for name in EFFECTS:
                out.append({**dict(zip(keys, g)), "block": block, "effect": name,
                            **_summarise(float(o[name][0]), b[name])})
    return pd.DataFrame(out)


def factorial_effects(df: pd.DataFrame, models: list[str] | None = None,
                      extra_group: str | None = None, n_boot: int = N_BOOT,
                      seed: int = 0) -> pd.DataFrame:
    """Main effects and interaction per (model, block).

    ``extra_group`` splits the estimates further -- pass ``"prompt_type"`` to get
    the same three effects within each prompt template.
    """
    models = models or sorted(df["model"].unique())
    return _factorial(df, [extra_group] if extra_group else [],
                      models, n_boot, seed)


GRADIENT = "power gradient"
SYMMETRY = "matched - mismatched"
ASYMMETRY_CONTRASTS = [GRADIENT, SYMMETRY]


def asymmetry_contrasts(df: pd.DataFrame, models: list[str] | None = None,
                        extra_group: str | None = None,
                        n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Who outranks whom, holding the two status levels fixed.

    The main effects answer "does the user's status matter" and "does the
    model's status matter" separately. Neither answers the question the design
    was built for, which is what happens when the two sides are *unequal*. Two
    contrasts do, both computed within a block so relevance is held constant:

    ``power gradient``
        ``u_high/m_low - u_low/m_high``. Both cells contain one high-status and
        one low-status party; only the direction of the gap differs. Negative
        means the model warns less when the person asking outranks it.

    ``matched - mismatched``
        ``(u_high/m_high + u_low/m_low)/2 - (u_high/m_low + u_low/m_high)/2``.
        Whether a status gap matters at all, regardless of direction. This is
        the interaction rescaled, and it is what separates "the model responds
        to rank order" from "the model responds to mismatch"; a gradient without
        a symmetry effect means the former.
    """
    models = models or sorted(df["model"].unique())
    keys = ["model"] + ([extra_group] if extra_group else [])
    _, groups, S, N = _grid(df[df["model"].isin(models)], keys,
                            FACTORIAL_CONDITIONS)
    obs, boot = _resample(S, N, n_boot, seed)
    out = []
    for gi, g in enumerate(groups):
        meta = dict(zip(keys, g))
        for bi, block in enumerate(BLOCKS):
            sl = slice(bi * 4, bi * 4 + 4)
            hh, hl, lh, ll = (obs[gi, sl][i] for i in range(4))
            b_hh, b_hl, b_lh, b_ll = (boot[:, gi, sl][:, i] for i in range(4))
            out.append({**meta, "block": block, "contrast": GRADIENT,
                        **_summarise(hl - lh, b_hl - b_lh)})
            out.append({**meta, "block": block, "contrast": SYMMETRY,
                        **_summarise((hh + ll) / 2 - (hl + lh) / 2,
                                     (b_hh + b_ll) / 2 - (b_hl + b_lh) / 2)})
    return pd.DataFrame(out)


def to_warning_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Re-express pass/fail as one behaviour: did the model warn?

    The two halves are scored by opposite rubrics, so ``passes`` means different
    things in each and the halves cannot be compared. Warning is a single act
    that is right on one half and wrong on the other:

        unsafe item   warned  = passed      (a hit)
        safe item     warned  = not passed  (a false alarm)

    On this scale a manipulation that moves both halves the same way has shifted
    the model's threshold for warning; one that moves them apart has changed how
    well it tells the two kinds of request apart.
    """
    out = df.copy()
    out["passes"] = np.where(out["safety_type"] == "unsafe",
                             out["passes"], 1 - out["passes"])
    return out


def warning_rates(df: pd.DataFrame, models: list[str] | None = None,
                  n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Warning rate per (model, condition, half), with a bootstrap interval.

    Rows are the hit rate on unsafe items and the false-alarm rate on safe ones,
    so one figure can show what every status manipulation does to the model's
    willingness to warn.
    """
    models = models or sorted(df["model"].unique())
    w = to_warning_scale(df[df["model"].isin(models)])
    conds = [c for c in EXP2_CONDITIONS if c in set(w["condition"])]
    _, groups, S, N = _grid(w, ["model", "safety_type"], conds)
    obs, boot = _resample(S, N, n_boot, seed)
    out = []
    for gi, (model, half) in enumerate(groups):
        for ci, cond in enumerate(conds):
            out.append({"model": model, "safety_type": half, "condition": cond,
                        "measure": "hit rate" if half == "unsafe"
                                   else "false-alarm rate",
                        **_summarise(float(obs[gi, ci]), boot[:, gi, ci])})
    return pd.DataFrame(out)


def warning_vs_control(df: pd.DataFrame, models: list[str] | None = None,
                       n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Each condition's warning rate minus the no-role control's, per half.

    The same comparison :func:`vs_control` makes on the pass scale, but computed
    separately for hits and false alarms so a figure can mark which status
    manipulations move each of them off the control.
    """
    models = models or sorted(df["model"].unique())
    w = to_warning_scale(df[df["model"].isin(models)])
    conds = [c for c in EXP2_CONDITIONS if c in set(w["condition"])]
    _, groups, S, N = _grid(w, ["model", "safety_type"], conds)
    obs, boot = _resample(S, N, n_boot, seed)
    ctrl = conds.index(CONTROL)
    out = []
    for gi, (model, half) in enumerate(groups):
        for ci, cond in enumerate(conds):
            if cond == CONTROL:
                continue
            r = _summarise(float(obs[gi, ci] - obs[gi, ctrl]),
                           boot[:, gi, ci] - boot[:, gi, ctrl])
            out.append({"model": model, "safety_type": half, "condition": cond,
                        "measure": "hit rate" if half == "unsafe"
                                   else "false-alarm rate",
                        "delta_vs_control": r.pop("estimate"), **r})
    return pd.DataFrame(out)


def warning_effects(df: pd.DataFrame, models: list[str] | None = None,
                    n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Factorial status effects on the warning rate, within each half.

    Same estimator as :func:`factorial_effects`, on the warning scale, so a
    threshold shift reads as two same-signed effects instead of one positive and
    one negative.
    """
    models = models or sorted(df["model"].unique())
    return _factorial(to_warning_scale(df), ["prompt_type"], models, n_boot, seed)


def condition_means(df: pd.DataFrame, models: list[str] | None = None,
                    n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Pass rate per (model, condition) with a cluster-bootstrap interval."""
    models = models or sorted(df["model"].unique())
    conds = [c for c in EXP2_CONDITIONS if c in set(df["condition"])]
    _, groups, S, N = _grid(df[df["model"].isin(models)], ["model"], conds)
    obs, boot = _resample(S, N, n_boot, seed)
    out = []
    for gi, (model,) in enumerate(groups):
        for ci, cond in enumerate(conds):
            r = _summarise(float(obs[gi, ci]), boot[:, gi, ci])
            # a p-value against zero is meaningless for a mean; only the
            # interval is wanted here
            r.pop("p")
            out.append({"model": model, "condition": cond, "rate": r.pop("estimate"),
                        **r, "n": int(N[:, gi, ci].sum())})
    return pd.DataFrame(out)


def cell_means(df: pd.DataFrame, models: list[str] | None = None,
               n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Condition means reshaped onto the 2x2 grid, carrying the same intervals."""
    cm = condition_means(df, models, n_boot, seed)
    cm = cm[cm["condition"] != CONTROL].copy()
    parts = cm["condition"].str.extract(
        r"^(?P<block>\w+?)_u(?P<u>high|low)_m(?P<m>high|low)$")
    cm["block"] = parts["block"]
    cm["user_level"], cm["model_level"] = parts["u"], parts["m"]
    return cm


def vs_control(df: pd.DataFrame, models: list[str] | None = None,
               n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Each condition minus the no-role control, per model.

    Reported unadjusted, like everything else here. These are descriptive: the
    paper's claims rest on the factorial effects, whose three contrasts per block
    are the design rather than a search over many candidate comparisons.
    """
    models = models or sorted(df["model"].unique())
    conds = [c for c in EXP2_CONDITIONS if c in set(df["condition"])]
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
    """User and model effects within each generic status channel (irrel block)."""
    models = models or sorted(df["model"].unique())
    conds = [c for c in FACTORIAL_CONDITIONS if c.startswith("irrel_")]
    out = _factorial(df[df["block"] == "irrel"], ["status_dimension"], models,
                     n_boot, seed, conds=conds, blocks=["irrel"])
    return out.rename(columns={"status_dimension": "dimension"})


def category_effects(df: pd.DataFrame, model: str, n_boot: int = N_BOOT,
                     seed: int = 0) -> pd.DataFrame:
    """Model-status effect per safety category, per block, for one model."""
    out = _factorial(df[df["model"] == model], ["category"], None, n_boot, seed)
    return out[out["effect"] == "model (high-low)"].drop(columns="effect")


def allpass_ci(df: pd.DataFrame, models: list[str] | None = None,
               n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Bootstrap interval for the all-persona pass rate.

    All-pass is already one Bernoulli draw per fact, so this resamples that
    per-fact vector directly rather than going through the rate engine.
    """
    from .analysis import collapse_samples

    models = models or sorted(df["model"].unique())
    per_fact = (collapse_samples(df)
                .groupby(["model", "condition", "safety_fact"])["passes"]
                .min().reset_index())
    rng = np.random.default_rng(seed)
    out = []
    for model in models:
        for cond in [c for c in EXP2_CONDITIONS if c in set(df["condition"])]:
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

