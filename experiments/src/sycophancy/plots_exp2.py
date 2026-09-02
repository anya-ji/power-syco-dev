"""Figures for exp2's blocked 2x2 design. Vector PDF only.

Every interval on every figure is a 95% cluster bootstrap over safety facts
(B=4000), resampling whole facts rather than rows. Naive binomial error bars
would be roughly a third too narrow here, because the same fact and the same
prompt recur across every condition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import MODEL_VARIANTS  # noqa: E402
from .analysis_exp2 import GRADIENT, SYMMETRY, stars  # noqa: E402
from .model_statuses import BLOCKS, EXP2_CONDITIONS, EXP2_LABEL_FLAT, LEVELS  # noqa: E402
from .plots import _save, set_style  # noqa: E402

BLOCK_LABEL = {"domain": "Domain-relevant", "irrel": "Domain-irrelevant"}
# Teal = domain block, rose = irrelevant block; dark = model high, light = model low.
CELL_COLOR = {
    ("domain", "high"): "#1b6b5e", ("domain", "low"): "#7fc0b0",
    ("irrel", "high"): "#9e3d52", ("irrel", "low"): "#e0a0ac",
}
CONTROL_COLOR = "#8d8579"
EFFECT_COLOR = {"user (high-low)": "#3d5a80", "model (high-low)": "#9e3d52",
                "interaction": "#98713b",
                # exp2-solo's presence contrasts: the same two sides, lightened,
                # because they answer the weaker question ("any role at all?")
                "user role (vs control)": "#8aa2bb",
                "model role (vs control)": "#cf8d9c"}
#: spelled out, because "user" alone does not say what is being differenced
EFFECT_LABEL = {
    "user (high-low)": "User status:  high − low",
    "model (high-low)": "Model status:  high − low",
    "interaction": "Interaction:  user × model",
    "user role (vs control)": "User has a role:  mean − control",
    "model role (vs control)": "Model has a role:  mean − control",
}
EFFECT_ORDER = ["user (high-low)", "model (high-low)", "interaction"]
#: exp2-solo: level contrasts first, then the presence contrasts they sit inside
SOLO_EFFECT_ORDER = ["user (high-low)", "model (high-low)",
                     "user role (vs control)", "model role (vs control)"]
PTYPE_LABEL = {
    # exp3 has no prompt-type axis; the same field carries SaLAD's gold label,
    # and the two halves are scored by opposite rubrics -- so this split is the
    # one that matters there, not a robustness check.
    "UNSAFE": "Unsafe items\n(hazard caught?)",
    "SAFE": "Safe items\n(over-warning?)",
    "YES_NO_PROMPT": "Yes/no",
    "SHORT_UNSAFE_CONTEXT_PLUS_SAFE_Q": "Unsafe context\n+ safe question",
    "Q_W_2_UNSAFE_PROMPT": "Question with\n2 unsafe elements",
}
SIG_NOTE = "* p<.05  ** p<.01  *** p<.001"


def display_name(model: str) -> str:
    v = MODEL_VARIANTS.get(model)
    return v.display if v else model


def _pct_axis(ax, axis: str = "y") -> None:
    getattr(ax, f"{axis}axis").set_major_formatter(
        mticker.PercentFormatter(xmax=1, decimals=0))


def _cond_palette() -> dict:
    palette = {EXP2_LABEL_FLAT["control"]: CONTROL_COLOR}
    for b in BLOCKS:
        for u in LEVELS:
            for m in LEVELS:
                palette[EXP2_LABEL_FLAT[f"{b}_u{u}_m{m}"]] = CELL_COLOR[(b, m)]
    return palette


def _err(obj, value):
    """Percentile interval as matplotlib's (lower, upper) distances from ``value``.

    The bootstrap interval is not symmetric about the estimate, so the distances
    are taken from ``ci_lo``/``ci_hi`` directly rather than from a multiple of
    the standard error.
    """
    lo = np.asarray(obj["ci_lo"], dtype=float)
    hi = np.asarray(obj["ci_hi"], dtype=float)
    v = np.asarray(value, dtype=float)
    return np.vstack([np.nan_to_num(v - lo), np.nan_to_num(hi - v)])


def _lookup(frame: pd.DataFrame | None, **where) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    sel = frame
    for k, v in where.items():
        if k not in sel.columns:
            return None
        sel = sel[sel[k] == v]
    return None if sel.empty else sel.iloc[0]


def fig_conditions(stats: pd.DataFrame, out_dir: Path, models: list[str],
                   vs_ctrl: pd.DataFrame | None = None,
                   cond_means: pd.DataFrame | None = None, prefix: str = "exp2",
                   note: str = "", conditions: list[str] | None = None,
                   palette: dict | None = None) -> Path:
    """Fig 1: pass rate across all nine conditions, one facet per model.

    Error bars are 95% cluster-bootstrap intervals; stars mark conditions that
    differ from the no-role control. Unadjusted -- these eight comparisons per
    model share one control cell, so see ``analysis_exp2.vs_control``.

    ``conditions``/``palette`` let the solo design reuse this with its own
    nine-condition order and its own colouring; the defaults are exp2's.
    """
    conditions = conditions or EXP2_CONDITIONS
    df = stats[stats["model"].isin(models)].copy()
    df["label"] = df["condition"].map(EXP2_LABEL_FLAT)
    order = [EXP2_LABEL_FLAT[c] for c in conditions]
    order_to_cond = {EXP2_LABEL_FLAT[c]: c for c in conditions}

    g = sns.catplot(data=df, x="label", y="rate", col="model", kind="bar",
                    order=order, col_order=models, hue="label", hue_order=order,
                    palette=palette or _cond_palette(), legend=False,
                    height=3.9, aspect=1.15,
                    edgecolor="#3a3a38", linewidth=0.7, saturation=1.0)
    for ax, model in zip(g.axes.flat, models):
        idx = df[df["model"] == model].set_index("label").reindex(order)
        for i, lab in enumerate(order):
            v = idx.loc[lab, "rate"]
            if pd.isna(v):
                continue
            cond = order_to_cond[lab]
            cm = _lookup(cond_means, model=model, condition=cond)
            if cm is not None:
                yerr = _err(cm, v)
                top = float(cm["ci_hi"])
            else:
                e = idx.loc[lab, "ci95"]
                e = 0.0 if pd.isna(e) else float(e)
                yerr, top = np.array([[e], [e]]), v + e
            ax.errorbar(i, v, yerr=yerr, color="#2b2b2a", capsize=3, elinewidth=1.0,
                        capthick=1.0, fmt="none", zorder=5)
            vc = _lookup(vs_ctrl, model=model, condition=cond)
            mark = stars(float(vc["p"])) if vc is not None else ""
            # stars go on their own line: appended inline, "74%***" next to
            # "76%*" collides at nine bars per facet
            ax.text(i, min(top + 0.02, 1.05), f"{v:.0%}", ha="center",
                    fontsize=7.5, fontweight="semibold")
            if mark:
                ax.text(i, min(top + 0.06, 1.09), mark, ha="center",
                        fontsize=8.5, fontweight="bold")
        ctrl = idx.loc[EXP2_LABEL_FLAT["control"], "rate"]
        if pd.notna(ctrl):
            ax.axhline(ctrl, color=CONTROL_COLOR, ls=(0, (4, 3)), lw=1.0, zorder=1)
        ax.set_ylim(0, 1.12)
        _pct_axis(ax)
        ax.tick_params(axis="x", labelrotation=42, labelsize=7.5)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        ax.set_title(display_name(model), fontsize=11, fontweight="semibold")
    g.set_axis_labels("", "Pass Rate")
    g.figure.suptitle(
        note + "Pass rate across the nine conditions\n"
        "(dark = model high status, light = model low; dashed = no-role control; "
        f"bars 95% cluster-bootstrap CI; {SIG_NOTE} vs control, unadjusted)",
        y=1.09, fontsize=11.5)
    return _save(g.figure, out_dir, f"{prefix}_fig1_conditions")


def fig_2x2_panels(cells: pd.DataFrame, out_dir: Path, models: list[str], prefix: str = "exp2", note: str = "") -> Path:
    """Fig 2: the design itself -- user level x model level, per block per model.

    Each tile carries its 95% bootstrap half-width, so a reader can see which
    apparent gradients are inside the noise.
    """
    fig, axes = plt.subplots(len(BLOCKS), len(models),
                             figsize=(3.4 * len(models), 3.1 * len(BLOCKS)),
                             squeeze=False)
    vmin, vmax = cells["rate"].min() - 0.02, cells["rate"].max() + 0.02
    for bi, block in enumerate(BLOCKS):
        for mi, model in enumerate(models):
            ax = axes[bi][mi]
            d = cells[(cells["model"] == model) & (cells["block"] == block)]
            mat = np.array([[d[(d.user_level == u) & (d.model_level == m)]["rate"].mean()
                             for m in LEVELS] for u in LEVELS], dtype=float)
            half = (d["ci_hi"] - d["ci_lo"]) / 2
            err = np.array([[half[(d.user_level == u) & (d.model_level == m)].mean()
                             for m in LEVELS] for u in LEVELS], dtype=float)
            im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(2):
                for j in range(2):
                    colour = "black" if mat[i, j] > (vmin + vmax) / 2 else "white"
                    ax.text(j, i + 0.06, f"{mat[i, j]:.1%}", ha="center", va="bottom",
                            fontsize=11, fontweight="semibold", color=colour)
                    if np.isfinite(err[i, j]):
                        ax.text(j, i + 0.10, f"±{err[i, j]:.1%}", ha="center",
                                va="top", fontsize=7.5, color=colour, alpha=0.85)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"model {m}" for m in LEVELS] if bi == len(BLOCKS) - 1
                               else [], fontsize=8.5)
            ax.set_yticks([0, 1])
            # Only the left column carries y labels; repeating them collides with
            # the neighbouring panel's cell text.
            ax.set_yticklabels([f"user {u}" for u in LEVELS] if mi == 0 else [],
                               fontsize=8.5)
            if bi == 0:
                ax.set_title(display_name(model), fontsize=10, fontweight="semibold")
            if mi == 0:
                ax.set_ylabel(BLOCK_LABEL[block], fontsize=10, fontweight="semibold")
    fig.colorbar(im, ax=axes, label="Pass Rate", shrink=0.75)
    fig.suptitle(note + "Pass rate by user status × model status, within each relevance "
                 "block\n(± is the 95% cluster-bootstrap half-width)",
                 y=1.05, fontsize=11.5)
    return _save(fig, out_dir, f"{prefix}_fig2_2x2_panels")


def fig_effects(effects: pd.DataFrame, out_dir: Path, models: list[str],
                prefix: str = "exp2", note: str = "") -> Path:
    """Fig 3: the headline -- main effects and interaction with bootstrap CIs.

    Rows are grouped by model, and each row is named for the difference it is:
    "User status: high − low" is the pass rate under a high-status user minus
    the rate under a low-status one, averaged over the model's own status.
    """
    fig, axes = plt.subplots(1, len(BLOCKS), figsize=(6.4 * len(BLOCKS), 4.6),
                             sharex=True, squeeze=False)
    for bi, block in enumerate(BLOCKS):
        ax = axes[0][bi]
        y, ypos, ylab, headers = 0.0, [], [], []
        for model in models:
            headers.append((y + 0.9, display_name(model)))
            for eff in EFFECT_ORDER:
                r = _lookup(effects, model=model, block=block, effect=eff)
                if r is None or not np.isfinite(r.estimate):
                    continue
                ax.plot([r.ci_lo, r.ci_hi], [y, y], color=EFFECT_COLOR[eff],
                        lw=2.2, solid_capstyle="round", alpha=0.85)
                ax.plot(r.estimate, y, "o", color=EFFECT_COLOR[eff], ms=6.5,
                        markeredgecolor="white", markeredgewidth=0.9, zorder=5)
                mark = stars(float(r.p))
                if mark:
                    ax.text(r.ci_hi + 0.003, y, mark, va="center", fontsize=10,
                            color=EFFECT_COLOR[eff], fontweight="bold")
                ypos.append(y)
                ylab.append(EFFECT_LABEL[eff])
                y -= 1
            y -= 0.9
        ax.axvline(0, color="#2b2b2a", lw=1.0, zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(ylab, fontsize=8.5)
        ax.set_ylim(y + 0.4, 1.8)
        for hy, name in headers:
            ax.text(0.0, hy, name, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=9, fontweight="bold",
                    color="#2b2b2a", clip_on=False)
        ax.set_title(BLOCK_LABEL[block], fontsize=11, fontweight="semibold")
        ax.set_xlabel("Δ pass rate (percentage points)")
        _pct_axis(ax, "x")
        sns.despine(ax=ax, left=True)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(
        note +
        "Factorial effects with 95% cluster-bootstrap CIs\n"
        "(positive = higher pass rate; " + SIG_NOTE + ")",
        y=1.05, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig3_effects")


def fig_interaction(cells: pd.DataFrame, out_dir: Path, models: list[str], prefix: str = "exp2", note: str = "") -> Path:
    """Fig 4: interaction plot -- user level on x, one line per model level."""
    fig, axes = plt.subplots(len(BLOCKS), len(models), sharey="row", squeeze=False,
                             figsize=(3.2 * len(models), 2.9 * len(BLOCKS)))
    colour = {"high": "#9e3d52", "low": "#e0a0ac"}
    marker = {"high": "o", "low": "s"}
    style = {"high": "-", "low": "--"}
    for bi, block in enumerate(BLOCKS):
        for mi, model in enumerate(models):
            ax = axes[bi][mi]
            d = cells[(cells["model"] == model) & (cells["block"] == block)]
            for lvl in LEVELS:
                sub = d[d["model_level"] == lvl].set_index("user_level").reindex(LEVELS)
                x = np.arange(len(LEVELS)) + (0.03 if lvl == "high" else -0.03)
                ax.errorbar(x, sub["rate"], yerr=_err(sub, sub["rate"]),
                            color=colour[lvl], marker=marker[lvl], ls=style[lvl],
                            ms=5.5, lw=1.6, capsize=3, elinewidth=1.0,
                            label=f"model {lvl}", markeredgecolor="white",
                            markeredgewidth=0.8)
            ax.set_xticks(range(len(LEVELS)))
            ax.set_xticklabels([f"user {u}" for u in LEVELS], fontsize=8.5)
            ax.set_xlim(-0.4, len(LEVELS) - 0.6)
            _pct_axis(ax)
            if bi == 0:
                ax.set_title(display_name(model), fontsize=10, fontweight="semibold")
            if mi == 0:
                ax.set_ylabel(f"{BLOCK_LABEL[block]}\nPass Rate", fontsize=9)
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle(note + "Non-parallel lines mean the user effect depends on the model's "
                 "own status\n(95% cluster-bootstrap CIs)", y=1.04, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig4_interaction")


def fig_by_dimension(dim: pd.DataFrame, out_dir: Path, models: list[str], prefix: str = "exp2", note: str = "") -> Path:
    """Fig 5: user and model effects per generic status channel, with CIs."""
    shown = ["user (high-low)", "model (high-low)"]
    d = dim[dim["effect"].isin(shown)].copy()
    d["dim_label"] = d["dimension"].str.replace("_", "\n")
    dims = sorted(d["dim_label"].unique())
    fig, axes = plt.subplots(1, len(models), sharey=True, squeeze=False,
                             figsize=(3.9 * len(models), 3.8))
    width = 0.36
    for mi, model in enumerate(models):
        ax = axes[0][mi]
        for ei, eff in enumerate(shown):
            sub = (d[(d["model"] == model) & (d["effect"] == eff)]
                   .set_index("dim_label").reindex(dims))
            x = np.arange(len(dims)) + (ei - 0.5) * width
            ax.bar(x, sub["estimate"], width, color=EFFECT_COLOR[eff],
                   edgecolor="#3a3a38", linewidth=0.7,
                   label=EFFECT_LABEL[eff].split(":")[0])
            ax.errorbar(x, sub["estimate"], yerr=_err(sub, sub["estimate"]),
                        fmt="none", color="#2b2b2a", capsize=2.5, elinewidth=0.9)
            for xi, (est, p) in enumerate(zip(sub["estimate"], sub["p"])):
                mark = stars(float(p)) if pd.notna(p) else ""
                if mark and pd.notna(est):
                    tip = sub["ci_hi"].iloc[xi] if est >= 0 else sub["ci_lo"].iloc[xi]
                    ax.text(x[xi], tip + np.sign(est) * 0.002, mark,
                            ha="center", va="bottom" if est >= 0 else "top",
                            fontsize=9, fontweight="bold")
        ax.axhline(0, color="#2b2b2a", lw=1.0)
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(dims, fontsize=7.5)
        _pct_axis(ax)
        ax.set_title(display_name(model), fontsize=10, fontweight="semibold")
        if mi == 0:
            ax.set_ylabel("Δ pass rate")
        sns.despine(ax=ax)
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle(note + "Effects split by status channel (domain-irrelevant block)\n"
                 f"95% cluster-bootstrap CIs; {SIG_NOTE}", y=1.06, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig5_by_dimension")


def fig_allpass(stats: pd.DataFrame, out_dir: Path, models: list[str],
                ci: pd.DataFrame | None = None, prefix: str = "exp2",
                note: str = "", conditions: list[str] | None = None,
                palette: dict | None = None) -> Path:
    """Fig 6: all-persona pass rate, size-matched, across the nine conditions."""
    conditions = conditions or EXP2_CONDITIONS
    col = "all_pass_rate_k5" if "all_pass_rate_k5" in stats.columns else "all_pass_rate"
    df = stats[stats["model"].isin(models)].copy()
    df["label"] = df["condition"].map(EXP2_LABEL_FLAT)
    order = [EXP2_LABEL_FLAT[c] for c in conditions]
    order_to_cond = {EXP2_LABEL_FLAT[c]: c for c in conditions}
    g = sns.catplot(data=df, x="label", y=col, col="model", kind="bar", order=order,
                    col_order=models, hue="label", hue_order=order,
                    palette=palette or _cond_palette(), legend=False,
                    height=3.9, aspect=1.15,
                    edgecolor="#3a3a38", linewidth=0.7, saturation=1.0)
    for ax, model in zip(g.axes.flat, models):
        idx = df[df["model"] == model].set_index("label").reindex(order)
        for i, lab in enumerate(order):
            v = idx.loc[lab, col]
            r = _lookup(ci, model=model, condition=order_to_cond[lab])
            if pd.isna(v) or r is None:
                continue
            # the matched statistic is a rescaled version of the raw all-pass
            # rate, so carry over the bootstrap half-width rather than the
            # absolute interval, which sits around the unmatched value
            half = (float(r["ci_hi"]) - float(r["ci_lo"])) / 2
            ax.errorbar(i, v, yerr=half, color="#2b2b2a", capsize=3,
                        elinewidth=1.0, capthick=1.0, fmt="none", zorder=5)
        ax.set_ylim(0, 1.0)
        _pct_axis(ax)
        ax.tick_params(axis="x", labelrotation=42, labelsize=7.5)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        ax.set_title(display_name(model), fontsize=11, fontweight="semibold")
    g.set_axis_labels("", "All-Persona Pass Rate (5-matched)")
    g.figure.suptitle(note + "Consistency across personas: every pairing for a fact must "
                      "pass\n(95% bootstrap CIs over safety facts; the control has "
                      "one persona per fact, so it cannot be size-matched)",
                      y=1.10, fontsize=11.5)
    return _save(g.figure, out_dir, f"{prefix}_fig6_allpass")


def fig_by_category(cats: pd.DataFrame, out_dir: Path, model: str, prefix: str = "exp2", note: str = "") -> Path:
    """Fig 7: model-status effect per safety category, with bootstrap CIs."""
    t = cats.copy()
    t["block_label"] = t["block"].map(BLOCK_LABEL)
    cats_order = sorted(t["category"].unique())
    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(cats_order) * 2 + 1.6))
    height = 0.38
    colour = {"domain": "#1b6b5e", "irrel": "#9e3d52"}
    for bi, block in enumerate(BLOCKS):
        sub = t[t["block"] == block].set_index("category").reindex(cats_order)
        y = np.arange(len(cats_order)) - (bi - 0.5) * height
        ax.barh(y, sub["estimate"], height, color=colour[block],
                edgecolor="#3a3a38", linewidth=0.7, label=BLOCK_LABEL[block])
        ax.errorbar(sub["estimate"], y, xerr=_err(sub, sub["estimate"]), fmt="none",
                    color="#2b2b2a", capsize=2.5, elinewidth=0.9)
        for yi, (est, p) in enumerate(zip(sub["estimate"], sub["p"])):
            mark = stars(float(p)) if pd.notna(p) else ""
            if mark and pd.notna(est):
                tip = sub["ci_hi"].iloc[yi] if est >= 0 else sub["ci_lo"].iloc[yi]
                ax.text(tip + np.sign(est) * 0.004, y[yi], mark,
                        va="center", ha="left" if est >= 0 else "right",
                        fontsize=9, fontweight="bold")
    ax.axvline(0, color="#2b2b2a", lw=1.0)
    ax.set_yticks(range(len(cats_order)))
    ax.set_yticklabels(cats_order, fontsize=9)
    ax.invert_yaxis()
    _pct_axis(ax, "x")
    ax.set_xlabel("Δ pass rate, model high − model low")
    ax.set_title(note + f"Model-status effect by safety category ({display_name(model)})\n"
                 f"95% cluster-bootstrap CIs; {SIG_NOTE}", fontsize=11)
    ax.legend(title="", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig7_by_category")


def fig_by_prompt_type(by_ptype: pd.DataFrame, out_dir: Path,
                       models: list[str], prefix: str = "exp2",
                       effects: list[str] | None = None) -> Path:
    """Fig 8: are the effects a property of one prompt template?

    Each of the three SAGE templates poses the unsafe request differently, so an
    effect that only shows up under one of them is a template artifact rather
    than a status effect.
    """
    effects = effects or EFFECT_ORDER
    d = by_ptype[by_ptype["effect"].isin(effects)].copy()
    ptypes = [p for p in PTYPE_LABEL if p in set(d["prompt_type"])]
    ptypes += [p for p in sorted(d["prompt_type"].unique()) if p not in ptypes]
    labels = [PTYPE_LABEL.get(p, p.replace("_", "\n")) for p in ptypes]
    fig, axes = plt.subplots(len(BLOCKS), len(models), sharey="row", sharex=True,
                             squeeze=False,
                             figsize=(3.6 * len(models), 3.0 * len(BLOCKS)))
    width = 0.78 / len(effects)
    for bi, block in enumerate(BLOCKS):
        for mi, model in enumerate(models):
            ax = axes[bi][mi]
            for ei, eff in enumerate(effects):
                sub = (d[(d.model == model) & (d.block == block) & (d.effect == eff)]
                       .set_index("prompt_type").reindex(ptypes))
                x = np.arange(len(ptypes)) + (ei - (len(effects) - 1) / 2) * width
                ax.bar(x, sub["estimate"], width, color=EFFECT_COLOR[eff],
                       edgecolor="#3a3a38", linewidth=0.6,
                       label=EFFECT_LABEL[eff].split(":")[0])
                ax.errorbar(x, sub["estimate"], yerr=_err(sub, sub["estimate"]),
                            fmt="none", color="#2b2b2a", capsize=2, elinewidth=0.8)
                for xi, (est, p) in enumerate(zip(sub["estimate"], sub["p"])):
                    mark = stars(float(p)) if pd.notna(p) else ""
                    if mark and pd.notna(est):
                        tip = sub["ci_hi"].iloc[xi] if est >= 0 else sub["ci_lo"].iloc[xi]
                        ax.text(x[xi], tip + np.sign(est) * 0.002, mark,
                                ha="center", va="bottom" if est >= 0 else "top",
                                fontsize=8, fontweight="bold")
            ax.axhline(0, color="#2b2b2a", lw=1.0)
            ax.set_xticks(range(len(ptypes)))
            ax.set_xticklabels(labels, fontsize=7)
            _pct_axis(ax)
            if bi == 0:
                ax.set_title(display_name(model), fontsize=10, fontweight="semibold")
            if mi == 0:
                ax.set_ylabel(f"{BLOCK_LABEL[block]}\nΔ pass rate", fontsize=9)
            sns.despine(ax=ax)
    axes[0][-1].legend(fontsize=7.5, loc="best")
    # exp2 splits by prompt template; exp3 reuses the same field for SaLAD's
    # gold label, where the split is the headline rather than a robustness check.
    what = ("each half of the benchmark"
            if set(ptypes) <= {"SAFE", "UNSAFE"} else "each prompt template")
    fig.suptitle(f"Status effects estimated separately within {what}"
                 f"\n95% CIs from resampling items; {SIG_NOTE}",
                 y=1.04, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig8_by_prompt_type")


ASYM_COLOR = {GRADIENT: "#8c2f39", SYMMETRY: "#9a9a94"}
ASYM_LABEL = {
    GRADIENT: "Power gradient:  user above model  \u2212  model above user",
    SYMMETRY: "Control:  matched status  \u2212  mismatched status",
}


def fig_power_gradient(asym: pd.DataFrame, out_dir: Path, models: list[str],
                       prefix: str = "exp2", note: str = "") -> Path:
    """The headline asymmetry figure: does it matter who outranks whom?

    One row per (model, block), the gradient against its symmetry control. A
    gradient away from zero with the control on zero means the model is tracking
    rank *order*, not the mere presence of a status gap.
    """
    # Only rows the frame actually has: a caller passing the global model list
    # rather than the run's own would otherwise leave blank rows on the axis.
    present = set(map(tuple, asym[["model", "block"]].drop_duplicates().to_numpy()))
    rows = [(m, b) for m in models for b in BLOCKS if (m, b) in present]
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(rows) + 2.4))
    offset = {GRADIENT: 0.16, SYMMETRY: -0.16}
    ypos, ylab = [], []
    for i, (model, block) in enumerate(rows):
        y = -i
        ypos.append(y)
        ylab.append(f"{display_name(model)}  ·  {BLOCK_LABEL[block]}")
        for contrast in (GRADIENT, SYMMETRY):
            r = _lookup(asym, model=model, block=block, contrast=contrast)
            if r is None or not np.isfinite(r.estimate):
                continue
            yy = y + offset[contrast]
            colour = ASYM_COLOR[contrast]
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=colour, lw=2.4,
                    solid_capstyle="round", alpha=0.9,
                    label=ASYM_LABEL[contrast] if i == 0 else None)
            ax.plot(r.estimate, yy, "o", color=colour, ms=6.5,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=5)
            mark = stars(float(r.p))
            if mark:
                ax.text(r.ci_lo - 0.003, yy, mark, va="center", ha="right",
                        fontsize=10, color=colour, fontweight="bold")
    ax.axvline(0, color="#2b2b2a", lw=1.0, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=9)
    ax.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
    _pct_axis(ax, "x")
    ax.set_xlabel("Δ pass rate (percentage points)")
    ax.set_title(note + "Who outranks whom, with both status levels held fixed\n"
                 "negative = the model warns LESS when the user outranks it;  "
                 f"95% cluster-bootstrap CIs;  {SIG_NOTE}", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig9_power_gradient")


MEASURES = ["hit rate", "false-alarm rate"]
MEASURE_TITLE = {
    "hit rate": "Hit rate\n(unsafe: warning is correct)",
    "false-alarm rate": "False-alarm rate\n(safe: warning is wrong)",
}


def fig_warning_rates(warn: pd.DataFrame, out_dir: Path, models: list[str],
                      vs_ctrl: pd.DataFrame | None = None,
                      prefix: str = "exp2") -> Path:
    """What each status manipulation does to the model's willingness to warn.

    Both halves on one scale: warning is a hit on an unsafe item and a false
    alarm on a safe one. One row per half so the bars stay readable, coloured by
    condition as in fig1, with stars marking conditions that differ from the
    no-role control.
    """
    conds = [c for c in EXP2_CONDITIONS if c in set(warn["condition"])]
    labels = [EXP2_LABEL_FLAT.get(c, c) for c in conds]
    palette = _cond_palette()
    colours = [palette[EXP2_LABEL_FLAT[c]] for c in conds]

    # Share y within a row: the two measures live on very different scales, but
    # two models on the same measure must be read against each other.
    fig, axes = plt.subplots(len(MEASURES), len(models), squeeze=False,
                             figsize=(5.4 * len(models), 3.5 * len(MEASURES)),
                             sharex=True, sharey="row")
    for mi, model in enumerate(models):
        for qi, measure in enumerate(MEASURES):
            ax = axes[qi][mi]
            d = (warn[(warn["model"] == model) & (warn["measure"] == measure)]
                 .set_index("condition").reindex(conds))
            y = d["estimate"].to_numpy(dtype=float)
            ax.bar(range(len(conds)), y, color=colours, edgecolor="#3a3a38",
                   linewidth=0.7)
            ax.errorbar(range(len(conds)), y, yerr=_err(d, y), fmt="none",
                        color="#2b2b2a", capsize=3, elinewidth=1.0,
                        capthick=1.0, zorder=5)
            top = np.nan_to_num(d["ci_hi"].to_numpy(dtype=float))
            head = max(0.06, float(np.nanmax(top)) * 0.10)
            for i, cond in enumerate(conds):
                if not np.isfinite(y[i]):
                    continue
                ax.text(i, top[i] + head * 0.18, f"{y[i]:.0%}", ha="center",
                        fontsize=7.5, fontweight="semibold")
                r = _lookup(vs_ctrl, model=model, condition=cond,
                            measure=measure)
                mark = stars(float(r["p"])) if r is not None else ""
                if mark:
                    ax.text(i, top[i] + head * 0.62, mark, ha="center",
                            fontsize=8.5, fontweight="bold")
            ctrl = d["estimate"].get(EXP2_CONDITIONS[0], np.nan)
            if np.isfinite(ctrl):
                ax.axhline(ctrl, color=CONTROL_COLOR, ls=(0, (4, 3)), lw=1.0,
                           zorder=1)
            row_top = float(np.nanmax(
                warn[warn["measure"] == measure]["ci_hi"].to_numpy(dtype=float)))
            ax.set_ylim(0, max(0.12, row_top + head))
            _pct_axis(ax)
            if mi == 0:
                ax.set_ylabel(MEASURE_TITLE[measure], fontsize=9.5)
            if qi == 0:
                ax.set_title(display_name(model), fontsize=11,
                             fontweight="semibold")
            ax.tick_params(axis="x", labelrotation=42, labelsize=7.5)
            for t in ax.get_xticklabels():
                t.set_ha("right")
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels(labels)
            sns.despine(ax=ax)
    fig.suptitle("Warning rate under each status manipulation\n"
                 "warning is correct on unsafe items and wrong on safe ones; "
                 "dashed = no-role control;\n"
                 f"bars 95% CIs from resampling items; {SIG_NOTE} vs control, "
                 "uncorrected", y=1.05, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig10_warning_rates")


def combine_halves(src_dir: Path, dest_dir: Path, prefix: str = "exp3",
                   halves: tuple[str, str] = ("unsafe", "safe")) -> list[Path]:
    """Stack each figure's two per-half renders into one image in ``dest_dir``.

    The halves are rendered separately -- same estimators, one gold label each --
    into a scratch directory, and composed here. Only the combined image is kept:
    per-half files are intermediates, not output.

    Composition rather than faceting inside every plotting function keeps the
    eight functions untouched, which matters because two of them are seaborn
    ``catplot`` grids that own their figure and cannot draw into a shared one.
    The trade-off is that each half keeps its own title and axes, so the two
    panels are not on a shared scale.
    """
    from PIL import Image

    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    first, second = halves
    made = []
    for a in sorted(src_dir.glob(f"{prefix}_{first}_*.png")):
        name = a.name[len(f"{prefix}_{first}_"):]
        b = src_dir / f"{prefix}_{second}_{name}"
        if not b.exists():
            continue
        top, bottom = Image.open(a), Image.open(b)
        width = max(top.width, bottom.width)
        canvas = Image.new("RGB", (width, top.height + bottom.height), "white")
        canvas.paste(top, ((width - top.width) // 2, 0))
        canvas.paste(bottom, ((width - bottom.width) // 2, top.height))
        dest = dest_dir / f"{prefix}_{name[:-4]}_by_half.png"
        canvas.save(dest, dpi=(200, 200))
        made.append(dest)
    if made:
        print(f"  combined {len(made)} figure(s) into *_by_half.png")
    else:
        # Silence here once hid the per-half renders going missing.
        print(f"  nothing to combine: no {prefix}_{first}_*.png / "
              f"{prefix}_{second}_*.png pairs under {src_dir}")
    return made


def make_all(df, stats, effects, cells, dim, out_dir: Path, models: list[str],
             primary: str, vs_ctrl=None, cond_means=None, cats=None,
             by_ptype=None, allpass_ci=None, asym=None, warn=None,
             warn_vs_ctrl=None, prefix: str = "exp2", note: str = "") -> dict:
    set_style()
    figs = {
        "fig1": fig_conditions(stats, out_dir, models, vs_ctrl, cond_means,
                               prefix, note),
        "fig2": fig_2x2_panels(cells, out_dir, models, prefix, note),
        "fig3": fig_effects(effects, out_dir, models, prefix, note),
        "fig4": fig_interaction(cells, out_dir, models, prefix, note),
        "fig5": fig_by_dimension(dim, out_dir, models, prefix, note),
        "fig6": fig_allpass(stats, out_dir, models, allpass_ci, prefix, note),
        "fig7": fig_by_category(cats, out_dir, primary, prefix, note),
    }
    if asym is not None and not asym.empty:
        figs["fig9"] = fig_power_gradient(asym, out_dir, models, prefix, note)
    if warn is not None and not warn.empty:
        figs["fig10"] = fig_warning_rates(warn, out_dir, models, warn_vs_ctrl,
                                          prefix)
    # With one template the split is just fig3 redrawn, so it is not worth a
    # figure -- and its caption would claim a robustness check it cannot make.
    if (by_ptype is not None and not by_ptype.empty
            and by_ptype["prompt_type"].nunique() > 1):
        figs["fig8"] = fig_by_prompt_type(by_ptype, out_dir, models, prefix)
    return figs
