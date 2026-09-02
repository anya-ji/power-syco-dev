"""Mixed-effects inference for exp2.

Rows are not independent: one safety fact contributes up to three prompt types
times 101 persona cells, and each prompt is re-asked under every condition.
Significance therefore has to model that structure. This module is the primary
inference path; the cluster bootstrap in :mod:`analysis_exp2` is kept as a
robustness check.

**Why a linear mixed model rather than a mixed logistic.** statsmodels' only
mixed-effects logistic regression is ``BinomialBayesMixedGLM``, fitted by
mean-field variational Bayes. On this data its posterior SDs run well below the
cluster bootstrap (VB is known to under-cover), and the fit failed to converge
for one model x block cell, collapsing every interval to zero width. A linear
probability mixed model has none of those problems: REML standard errors are
trustworthy, every cell converges in about a second, and the outcome scale is
the pass-rate scale the figures already use. The mixed logistic's job -- a
check that the linear approximation is not driving anything -- falls instead to
the cluster bootstrap, which assumes no outcome distribution at all and agrees
with these fits on every effect.

**Coding.** ``u`` and ``m`` are effect-coded to +/-0.5, so with an interaction
in the model the coefficients are exactly the marginal quantities plotted:

    u      (u_high - u_low) averaged over model level
    m      (m_high - m_low) averaged over user level
    u:m    (uh_mh - uh_ml) - (ul_mh - ul_ml)

Dummy coding would instead give simple effects at the other factor's reference
level, which differ by half the interaction and are easy to misreport.

**Random structure.** Intercepts for safety fact, and for prompt nested within
fact. Prompt absorbs most item variance -- prompts of one fact differ a lot --
but the fact level is what the bootstrap resamples, so both are kept.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm as scipy_norm

from .model_statuses import BLOCKS, EXP2_CONDITIONS, LEVELS

EFFECT_TERMS = {"u": "user (high-low)", "m": "model (high-low)",
                "u:m": "interaction"}


def _prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["y"] = d["passes"].astype(float)
    if "user_level" in d.columns:
        d["u"] = np.where(d["user_level"] == "high", 0.5, -0.5)
        d["m"] = np.where(d["model_level"] == "high", 0.5, -0.5)
    return d


def _fit(formula: str, d: pd.DataFrame):
    """REML fit with prompt nested in fact, falling back when unidentifiable.

    A cell holding one row per prompt (the control) leaves the prompt variance
    component with nothing to estimate and returns NaN standard errors; there
    the fact-level intercept alone is the right model.
    """
    # Prompt nested in fact is only a level when a fact has several prompts.
    # exp2 has three templates per fact; exp3 has one prompt per item, so the
    # component would be a duplicate of the group and split variance arbitrarily
    # between two indistinguishable levels.
    nested = d.groupby("safety_fact")["prompt_idx"].nunique().max() > 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if not nested:
                raise ValueError("prompt is 1:1 with fact; no nesting to fit")
            res = smf.mixedlm(formula, d, groups=d["safety_fact"],
                              vc_formula={"pr": "0 + C(prompt_idx)"}).fit(reml=True)
            if np.isfinite(res.bse.to_numpy()).all():
                return res
        except Exception:
            pass
        try:
            return smf.mixedlm(formula, d, groups=d["safety_fact"]).fit(reml=True)
        except Exception:
            return None


def _row(res, term: str, **extra) -> dict:
    if res is None or term not in res.params.index:
        return {**extra, "estimate": np.nan, "se": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p": np.nan}
    est, se, p = res.params[term], res.bse[term], res.pvalues[term]
    return {**extra, "estimate": float(est), "se": float(se),
            "ci_lo": float(est - 1.96 * se), "ci_hi": float(est + 1.96 * se),
            "p": float(p)}


def factorial_effects(df: pd.DataFrame, models: list[str] | None = None,
                      extra_group: str | None = None) -> pd.DataFrame:
    """Main effects and interaction per (model, block), by mixed model.

    ``extra_group`` splits the fits further -- pass ``"prompt_type"`` to get the
    same three effects estimated separately within each prompt template.
    """
    models = models or sorted(df["model"].unique())
    keys = ["model", "block"] + ([extra_group] if extra_group else [])
    out = []
    for model in models:
        for block in BLOCKS:
            sub = df[(df["model"] == model) & (df["block"] == block)]
            groups = ([(None, sub)] if not extra_group
                      else list(sub.groupby(extra_group)))
            for gval, d in groups:
                if d.empty:
                    continue
                res = _fit("y ~ u * m", _prepare(d))
                meta = dict(zip(keys, [model, block] + ([gval] if extra_group else [])))
                for term, name in EFFECT_TERMS.items():
                    out.append(_row(res, term, **meta, effect=name,
                                    n=len(d), n_facts=d["safety_fact"].nunique()))
    return pd.DataFrame(out)


def vs_control(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Every condition against the no-role control, one mixed model per model.

    Unadjusted, matching the bootstrap path this cross-checks.
    """
    models = models or sorted(df["model"].unique())
    conds = [c for c in EXP2_CONDITIONS if c in set(df["condition"])]
    out = []
    for model in models:
        d = _prepare(df[df["model"] == model])
        res = _fit("y ~ C(condition, Treatment('control'))", d)
        rows = []
        for c in conds:
            if c == "control":
                continue
            term = f"C(condition, Treatment('control'))[T.{c}]"
            rows.append(_row(res, term, model=model, condition=c))
        for r in rows:
            r["delta_vs_control"] = r.pop("estimate")
            out.append(r)
    return pd.DataFrame(out)


def condition_means(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Pass rate per (model, condition) with a mixed-model standard error.

    An intercept-only fit per cell: the point estimate is the cell mean, but the
    SE accounts for fact and prompt clustering instead of treating every row as
    an independent draw.
    """
    models = models or sorted(df["model"].unique())
    out = []
    for model in models:
        for cond in [c for c in EXP2_CONDITIONS if c in set(df["condition"])]:
            d = _prepare(df[(df["model"] == model) & (df["condition"] == cond)])
            if d.empty:
                continue
            res = _fit("y ~ 1", d)
            r = _row(res, "Intercept", model=model, condition=cond)
            r["rate"] = float(d["y"].mean())
            r["n"] = len(d)
            out.append(r)
    return pd.DataFrame(out)


def cell_means(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Condition means reshaped onto the 2x2 grid, carrying the same SEs."""
    cm = condition_means(df, models)
    cm = cm[cm["condition"] != "control"].copy()
    parts = cm["condition"].str.extract(r"^(?P<block>\w+?)_u(?P<u>high|low)_m(?P<m>high|low)$")
    cm["block"], cm["user_level"], cm["model_level"] = parts["block"], parts["u"], parts["m"]
    return cm


def dimension_effects(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """User and model effects within each generic status channel (irrel block)."""
    models = models or sorted(df["model"].unique())
    sub = df[df["block"] == "irrel"]
    out = []
    for model in models:
        for dim in sorted(sub["status_dimension"].dropna().unique()):
            d = sub[(sub["model"] == model) & (sub["status_dimension"] == dim)]
            if d.empty:
                continue
            res = _fit("y ~ u * m", _prepare(d))
            for term, name in EFFECT_TERMS.items():
                out.append(_row(res, term, model=model, dimension=dim,
                                effect=name, n=len(d)))
    return pd.DataFrame(out)


def category_effects(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Model-status effect per safety category, per block, for one model."""
    out = []
    for cat in sorted(df["category"].dropna().unique()):
        for block in BLOCKS:
            d = df[(df["model"] == model) & (df["category"] == cat)
                   & (df["block"] == block)]
            if d.empty or d["safety_fact"].nunique() < 2:
                continue
            res = _fit("y ~ u * m", _prepare(d))
            out.append(_row(res, "m", category=cat, block=block,
                            n=len(d), n_facts=d["safety_fact"].nunique()))
    return pd.DataFrame(out)


def allpass_ci(df: pd.DataFrame, column: str, models: list[str] | None = None,
               n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap CI for the all-persona pass rate.

    All-pass is a per-fact statistic -- one Bernoulli draw per fact -- so the
    fact is already the unit of analysis and a plain bootstrap over facts is the
    natural interval. There is no within-fact structure left for a mixed model
    to absorb.
    """
    from .analysis import collapse_samples

    models = models or sorted(df["model"].unique())
    per_fact = (
        collapse_samples(df)
        .groupby(["model", "condition", "safety_fact"])["passes"]
        .min().reset_index()
    )
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
            out.append({"model": model, "condition": cond, "column": column,
                        "se": float(draws.std(ddof=1)),
                        "ci_lo": float(lo), "ci_hi": float(hi), "n_facts": int(v.size)})
    return pd.DataFrame(out)


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ── exp2-solo ─────────────────────────────────────────────────────────────────
# The solo grid has no 2x2 to code, so `u * m` has nothing to fit: inside a
# block the four dressed cells are four separate arms against a silent partner.
# They are fitted as one treatment-coded factor over (4 arms + control) and the
# quantities are read off as linear contrasts of the coefficients, so every
# contrast comes from a single fit and carries the same variance components.

def _solo_contrasts(arms: list[str]) -> dict[str, np.ndarray]:
    """Contrast vectors over ``[uhigh, ulow, mhigh, mlow]`` treatment effects.

    Coefficients are already differences from control, so a presence contrast
    is just the mean of that side's two, with nothing to subtract.
    """
    e = {a: np.eye(len(arms))[i] for i, a in enumerate(arms)}
    uh, ul, mh, ml = (e[a] for a in arms)
    return {
        "user (high-low)": uh - ul,
        "model (high-low)": mh - ml,
        "user role (vs control)": (uh + ul) / 2,
        "model role (vs control)": (mh + ml) / 2,
    }


def solo_effects(df: pd.DataFrame, models: list[str] | None = None,
                 extra_group: str | None = None) -> pd.DataFrame:
    """Level and presence contrasts per (model, block), by mixed model.

    Mirrors :func:`analysis_solo.solo_effects` so the two can be diffed.
    """
    from .model_statuses import CONTROL, SIDES

    models = models or sorted(df["model"].unique())
    keys = ["model", "block"] + ([extra_group] if extra_group else [])
    out = []
    for model in models:
        for block in BLOCKS:
            arms = [f"{block}_{s}{lv}" for s in SIDES for lv in LEVELS]
            sub = df[(df["model"] == model)
                     & df["condition"].isin(arms + [CONTROL])]
            groups = ([(None, sub)] if not extra_group
                      else list(sub.groupby(extra_group)))
            for gval, d in groups:
                meta = dict(zip(keys, [model, block] + ([gval] if extra_group else [])))
                res = (_fit(f"y ~ C(condition, Treatment('{CONTROL}'))",
                            _prepare(d)) if not d.empty else None)
                terms = [f"C(condition, Treatment('{CONTROL}'))[T.{a}]" for a in arms]
                for name, vec in _solo_contrasts(arms).items():
                    out.append({**meta, "effect": name,
                                **_solo_row(res, terms, vec),
                                "n": len(d),
                                "n_facts": d["safety_fact"].nunique()})
    return pd.DataFrame(out)


def _solo_row(res, terms: list[str], weights: np.ndarray) -> dict:
    """One linear contrast of the arm coefficients, as estimate/SE/CI/p."""
    nan = {"estimate": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
           "p": np.nan}
    # Over the fixed effects only: res.params also carries the variance
    # components ("pr Var", "Group Var"), and a contrast vector sized to include
    # them is the wrong length for t_test, which silently yields NaN.
    if res is None:
        return nan
    fe = res.fe_params
    if any(t not in fe.index for t in terms):
        return nan
    vec = np.zeros(len(fe))
    idx = {name: i for i, name in enumerate(fe.index)}
    for term, w in zip(terms, weights):
        vec[idx[term]] = w
    # Formed by hand rather than through ``res.t_test``: a MixedLM's covariance
    # matrix spans the variance components as well as the fixed effects, so a
    # contrast vector sized to the fixed effects does not line up with it. The
    # leading block is the fixed-effect covariance, which is what a contrast of
    # coefficients needs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cov = np.asarray(res.cov_params())[:len(fe), :len(fe)]
            est = float(vec @ fe.to_numpy())
            var = float(vec @ cov @ vec)
        except Exception:
            return nan
    if not np.isfinite(var) or var < 0:
        return nan
    se = float(np.sqrt(var))
    # z, matching how statsmodels reports MixedLM p-values for single terms.
    p = float(2 * scipy_norm.sf(abs(est) / se)) if se > 0 else np.nan
    return {"estimate": est, "se": se, "ci_lo": est - 1.96 * se,
            "ci_hi": est + 1.96 * se, "p": p}
