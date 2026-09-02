"""Figures for exp2-solo: one dressed side per cell. Vector PDF only.

Shares exp2's estimator and most of its rendering -- every interval here is the
same 95% cluster bootstrap over safety facts (B=4000). What changes is the set
of contrasts. There is no 2x2 inside a block, so there is no interaction to
draw and no interaction plot; in their place are the two *presence* contrasts,
which the crossed design cannot estimate at all.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .analysis_exp2 import stars  # noqa: E402
from .analysis_solo import EFFECTS, PRESENCE, SIDE_NAME  # noqa: E402
from .model_statuses import (  # noqa: E402
    BLOCKS, CONTROL, EXP2_LABEL_FLAT, EXP2_SOLO_CONDITIONS, LEVELS, SIDES,
)
from .plots import SOLO_COLOR, _save, set_style  # noqa: E402
from .plots_exp2 import (  # noqa: E402
    BLOCK_LABEL, EFFECT_COLOR, EFFECT_LABEL, SIG_NOTE, SOLO_EFFECT_ORDER,
    _err, _lookup, _pct_axis, display_name, fig_allpass, fig_by_category,
    fig_by_dimension, fig_by_prompt_type, fig_conditions,
)

CONTROL_COLOR = "#8d8579"
SIDE_LABEL = {"user": "User dressed,\nassistant silent",
              "model": "Assistant dressed,\nuser silent"}
ASYM_COLOR = {"model level - user level": "#8c2f39",
              "model presence - user presence": "#9a9a94"}
ASYM_LABEL = {
    "model level - user level": "Level:  model (high−low) − user (high−low)",
    "model presence - user presence": "Presence:  model role − user role",
}


def _palette() -> dict:
    """Label -> colour, keyed the way :func:`fig_conditions` wants it."""
    palette = {EXP2_LABEL_FLAT[CONTROL]: CONTROL_COLOR}
    for b in BLOCKS:
        for s in SIDES:
            for lv in LEVELS:
                palette[EXP2_LABEL_FLAT[f"{b}_{s}{lv}"]] = SOLO_COLOR[f"{s}{lv}"]
    return palette


def fig_arms(arms: pd.DataFrame, out_dir: Path, models: list[str],
             cond_means: pd.DataFrame | None = None, prefix: str = "exp2solo",
             note: str = "") -> Path:
    """Fig 2: the four arms of each block, against the control line.

    One panel per (block, model). Within a panel the two sides sit side by side
    at their two levels, and the dashed line is the no-role control -- which is
    the comparison the whole design exists to make, so it is drawn rather than
    left to the reader.
    """
    fig, axes = plt.subplots(len(BLOCKS), len(models), sharey="row", squeeze=False,
                             figsize=(3.5 * len(models), 3.0 * len(BLOCKS)))
    x = np.arange(len(SIDES) * len(LEVELS))
    for bi, block in enumerate(BLOCKS):
        for mi, model in enumerate(models):
            ax = axes[bi][mi]
            cells, colors = [], []
            for s in SIDES:
                for lv in LEVELS:
                    r = _lookup(arms, model=model, block=block,
                                side=SIDE_NAME[s], level=lv)
                    cells.append(r)
                    colors.append(SOLO_COLOR[f"{s}{lv}"])
            vals = [np.nan if r is None else float(r["rate"]) for r in cells]
            lo = [np.nan if r is None else float(r["ci_lo"]) for r in cells]
            hi = [np.nan if r is None else float(r["ci_hi"]) for r in cells]
            ax.bar(x, vals, 0.68, color=colors, edgecolor="#3a3a38", linewidth=0.7)
            ax.errorbar(x, vals, fmt="none", color="#2b2b2a", capsize=2.5,
                        elinewidth=0.9,
                        yerr=_err({"ci_lo": lo, "ci_hi": hi}, vals))
            ctrl = _lookup(cond_means, model=model, condition=CONTROL)
            if ctrl is not None:
                ax.axhline(float(ctrl["rate"]), color=CONTROL_COLOR, ls="--",
                           lw=1.2, zorder=4)
                ax.axhspan(float(ctrl["ci_lo"]), float(ctrl["ci_hi"]),
                           color=CONTROL_COLOR, alpha=0.13, lw=0, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{SIDE_NAME[s]}\n{lv}" for s in SIDES
                                for lv in LEVELS], fontsize=7.5)
            _pct_axis(ax)
            if bi == 0:
                ax.set_title(display_name(model), fontsize=10, fontweight="semibold")
            if mi == 0:
                ax.set_ylabel(f"{BLOCK_LABEL[block]}\nPass rate", fontsize=9)
            sns.despine(ax=ax)
    fig.suptitle(note + "Each arm dresses one side only; the dashed line and band "
                 "are the no-role control\n(95% cluster-bootstrap CIs)",
                 y=1.05, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig2_arms")


def fig_effects(effects: pd.DataFrame, out_dir: Path, models: list[str],
                prefix: str = "exp2solo", note: str = "") -> Path:
    """Fig 3: the headline -- level and presence contrasts with bootstrap CIs.

    Four rows per model rather than exp2's three. The top two are the same
    high−low differences exp2 reports, now measured against a silent partner;
    the bottom two are what having any role at all does on that side.
    """
    fig, axes = plt.subplots(1, len(BLOCKS), figsize=(6.8 * len(BLOCKS), 5.2),
                             sharex=True, squeeze=False)
    for bi, block in enumerate(BLOCKS):
        ax = axes[0][bi]
        y, ypos, ylab, headers = 0.0, [], [], []
        for model in models:
            headers.append((y + 0.9, display_name(model)))
            for eff in SOLO_EFFECT_ORDER:
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
        "Solo-role effects with 95% cluster-bootstrap CIs\n"
        "(positive = higher pass rate; " + SIG_NOTE + ")",
        y=1.05, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig3_effects")


def fig_side_asymmetry(asym: pd.DataFrame, out_dir: Path, models: list[str],
                       prefix: str = "exp2solo", note: str = "") -> Path:
    """Fig 4: does the assistant's own status move it more than the user's?

    Both contrasts are differences of two effects taken inside one bootstrap
    draw, so the interval accounts for how the two sides covary. Positive means
    the assistant's own standing is the stronger lever.
    """
    contrasts = list(ASYM_LABEL)
    fig, axes = plt.subplots(1, len(BLOCKS), sharey=True, squeeze=False,
                             figsize=(4.4 * len(BLOCKS), 3.6))
    width = 0.36
    for bi, block in enumerate(BLOCKS):
        ax = axes[0][bi]
        for ci, contrast in enumerate(contrasts):
            sub = (asym[(asym["block"] == block) & (asym["contrast"] == contrast)]
                   .set_index("model").reindex(models))
            x = np.arange(len(models)) + (ci - 0.5) * width
            ax.bar(x, sub["estimate"], width, color=ASYM_COLOR[contrast],
                   edgecolor="#3a3a38", linewidth=0.7, label=ASYM_LABEL[contrast])
            ax.errorbar(x, sub["estimate"], yerr=_err(sub, sub["estimate"]),
                        fmt="none", color="#2b2b2a", capsize=2.5, elinewidth=0.9)
            for xi, (est, p) in enumerate(zip(sub["estimate"], sub["p"])):
                mark = stars(float(p)) if pd.notna(p) else ""
                if mark and pd.notna(est):
                    tip = sub["ci_hi"].iloc[xi] if est >= 0 else sub["ci_lo"].iloc[xi]
                    ax.text(x[xi], tip + np.sign(est) * 0.002, mark, ha="center",
                            va="bottom" if est >= 0 else "top", fontsize=9,
                            fontweight="bold")
        ax.axhline(0, color="#2b2b2a", lw=1.0)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([display_name(m) for m in models], fontsize=8)
        _pct_axis(ax)
        ax.set_title(BLOCK_LABEL[block], fontsize=10, fontweight="semibold")
        if bi == 0:
            ax.set_ylabel("Δ pass rate\n(model side − user side)", fontsize=9)
        sns.despine(ax=ax)
    axes[0][-1].legend(fontsize=7.5, loc="best")
    fig.suptitle(note + "Which side is the stronger lever? Positive = the "
                 "assistant's own status\n"
                 f"95% cluster-bootstrap CIs; {SIG_NOTE}", y=1.06, fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out_dir, f"{prefix}_fig4_side_asymmetry")


def make_all(df, stats, effects, arms, dim, out_dir: Path, models: list[str],
             primary: str, vs_ctrl=None, cond_means=None, cats=None,
             by_ptype=None, allpass_ci=None, asym=None,
             prefix: str = "exp2solo", note: str = "") -> dict:
    """Numbering mirrors exp2's, so fig3 is the headline in both."""
    set_style()
    figs = {
        "fig1": fig_conditions(stats, out_dir, models, vs_ctrl, cond_means,
                               prefix, note, conditions=EXP2_SOLO_CONDITIONS,
                               palette=_palette()),
        "fig2": fig_arms(arms, out_dir, models, cond_means, prefix, note),
        "fig3": fig_effects(effects, out_dir, models, prefix, note),
        "fig5": fig_by_dimension(dim, out_dir, models, prefix, note),
        "fig6": fig_allpass(stats, out_dir, models, allpass_ci, prefix, note,
                            conditions=EXP2_SOLO_CONDITIONS, palette=_palette()),
        "fig7": fig_by_category(cats, out_dir, primary, prefix, note),
    }
    if asym is not None and not asym.empty:
        figs["fig4"] = fig_side_asymmetry(asym, out_dir, models, prefix, note)
    # With one template the split is fig3 redrawn, and its caption would claim a
    # robustness check it cannot make.
    if (by_ptype is not None and not by_ptype.empty
            and by_ptype["prompt_type"].nunique() > 1):
        figs["fig8"] = fig_by_prompt_type(by_ptype, out_dir, models, prefix,
                                          effects=EFFECTS)
    return figs
