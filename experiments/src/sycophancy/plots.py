"""Figures. Each function saves a PDF+PNG pair and returns the PDF path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the GPU nodes

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .analysis import model_index  # noqa: E402
from .config import (  # noqa: E402
    COND_COLOR,
    COND_LABEL,
    COND_LABEL_FLAT,
    CONDITIONS,
    DIMENSION_COLOR,
    MODEL_COLOR,
    MODEL_VARIANTS,
)

#: exp2-solo: hue carries the *side*, shade the level. Side is what that design
#: varies and what its forest plot colours by (user slate, model wine), so the
#: bars and the effects figure name the same thing with the same colour. Block
#: is carried by the x-axis grouping and the tick labels instead.
SOLO_COLOR = {"uhigh": "#3d5a80", "ulow": "#9fb3c8",
              "mhigh": "#9e3d52", "mlow": "#e0a0ac"}


def cond_color(cond: str) -> str:
    """Palette lookup that tolerates designs beyond exp1's five conditions."""
    if cond in COND_COLOR:
        return COND_COLOR[cond]
    parts = cond.split("_")
    # exp2-solo: two parts, the second naming the one dressed side.
    if len(parts) == 2 and parts[1] in SOLO_COLOR:
        return SOLO_COLOR[parts[1]]
    # exp2: colour by relevance block, shade by the user's level.
    if cond.startswith("domain"):
        return "#1b6b5e" if "_uhigh" in cond else "#7fc0b0"
    if cond.startswith("irrel"):
        return "#9e3d52" if "_uhigh" in cond else "#e0a0ac"
    return "#8d8579"


def cond_label(cond: str) -> str:
    if cond in COND_LABEL_FLAT:
        return COND_LABEL_FLAT[cond]
    from .model_statuses import EXP2_LABEL_FLAT
    return EXP2_LABEL_FLAT.get(cond, cond)


METRIC_LABEL = {
    "rate": "Pass Rate",
    "all_pass_rate": "All-Persona Pass Rate",
    "all_pass_rate_k5": "All-Persona Pass Rate (5-persona matched)",
}


def display_name(model: str) -> str:
    v = MODEL_VARIANTS.get(model)
    return v.display if v else model


def set_style() -> None:
    """Seaborn paper style, tuned for vector output in a two-column report."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4a4a48",
        "axes.linewidth": 0.8,
        "axes.titleweight": "semibold",
        "axes.titlepad": 11,
        "axes.labelcolor": "#2b2b2a",
        "text.color": "#2b2b2a",
        "xtick.color": "#4a4a48",
        "ytick.color": "#4a4a48",
        "grid.color": "#dcdbd7",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType so text stays selectable
    })


def _save(fig, out_dir: Path, name: str) -> Path:
    """PNG into figures/, vector PDF straight into report/.

    pdflatex resolves \\includegraphics relative to the .tex, so the PDF is
    written where the report lives rather than copied there afterwards.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out_dir.parent / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    pdf = report_dir / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png -> {out_dir.name}/, {name}.pdf -> {report_dir.name}/")
    return pdf


def fig_rate_by_condition(
    stats: pd.DataFrame, out_dir: Path, models: list[str],
    conditions=CONDITIONS, metric: str = "rate",
) -> Path:
    """Fig 1: per-condition rate, one facet per model variant."""
    df = stats[stats["model"].isin(models)].copy()
    df["cond_label"] = df["condition"].map(cond_label)
    df["model_label"] = df["model"].map(display_name)
    order = [cond_label(c) for c in conditions]
    palette = {COND_LABEL_FLAT[c]: cond_color(c) for c in conditions}

    g = sns.catplot(
        data=df, x="cond_label", y=metric, col="model_label", kind="bar",
        order=order, col_order=[display_name(m) for m in models],
        hue="cond_label", hue_order=order, palette=palette, legend=False,
        height=3.6, aspect=0.92, edgecolor="#3a3a38", linewidth=0.7,
        saturation=1.0,
    )
    for ax, model in zip(g.axes.flat, models):
        idx = model_index(stats, model, conditions)
        for i, cond in enumerate(conditions):
            if cond not in idx.index or pd.isna(idx.loc[cond, metric]):
                continue
            val = idx.loc[cond, metric]
            err = idx.loc[cond, "ci95"] if metric == "rate" else float("nan")
            if pd.notna(err):
                ax.errorbar(i, val, yerr=err, color="#2b2b2a", capsize=3.5,
                            elinewidth=1.1, capthick=1.1, zorder=5, fmt="none")
            ax.text(i, min(val + (err if pd.notna(err) else 0) + 0.03, 1.05),
                    f"{val:.0%}", ha="center", fontsize=8.5, fontweight="semibold")
        ctrl = idx.loc["control", metric] if "control" in idx.index else None
        if ctrl is not None and pd.notna(ctrl):
            ax.axhline(ctrl, color="#8d8579", ls=(0, (4, 3)), lw=1.0, zorder=1)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="x", labelrotation=28)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")

    g.set_titles("{col_name}")
    g.set_axis_labels("", METRIC_LABEL.get(metric, metric))
    g.figure.suptitle(
        f"{METRIC_LABEL.get(metric, metric)} by user-status condition"
        "\n(SAGE-Eval rubric; dashed line = no-role control)",
        y=1.06, fontsize=11.5,
    )
    suffix = "" if metric == "rate" else f"_{metric}"
    return _save(g.figure, out_dir, f"fig1_rate_by_condition{suffix}")


def fig_2x2_matrix(stats: pd.DataFrame, out_dir: Path, model: str) -> Path:
    """Fig 2: the 2x2 design (status level x domain relevance)."""
    idx = model_index(stats, model)
    data = np.array([
        [idx.loc["domain_high", "rate"], idx.loc["irrel_high", "rate"]],
        [idx.loc["domain_low", "rate"], idx.loc["irrel_low", "rate"]],
    ], dtype=float)
    err = np.array([
        [idx.loc["domain_high", "ci95"], idx.loc["irrel_high", "ci95"]],
        [idx.loc["domain_low", "ci95"], idx.loc["irrel_low", "ci95"]],
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1.0, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Domain-relevant", "Domain-irrelevant"], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["High status", "Low status"], fontsize=12)
    ax.set_xlabel("Expertise domain", fontsize=12)
    ax.set_ylabel("Status level", fontsize=12)
    ax.set_title(f"Pass Rate: 2×2 Status Design\n({display_name(model)})", fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{data[i, j]:.0%}\n±{err[i, j]:.0%}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="black" if data[i, j] > 0.4 else "white")
    plt.colorbar(im, ax=ax, label="Pass Rate")
    plt.tight_layout()
    return _save(fig, out_dir, "fig2_2x2_matrix")


def fig_category_heatmap(
    cat_table: pd.DataFrame, out_dir: Path, model: str, conditions=CONDITIONS
) -> Path:
    """Fig 3: pass rate per safety category x condition."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(cat_table, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=ax,
                xticklabels=[cond_label(c) for c in conditions])
    ax.set_title(f"Pass Rate by Safety Category × Condition\n({display_name(model)})",
                 fontsize=12)
    ax.set_ylabel("Category")
    ax.set_xlabel("Condition")
    plt.tight_layout()
    return _save(fig, out_dir, "fig3_heatmap_category")


def fig_model_comparison(
    stats: pd.DataFrame, out_dir: Path, models: list[str],
    conditions=CONDITIONS, metric: str = "rate",
) -> Path:
    """Fig 4: all model variants overlaid across conditions."""
    df = stats[stats["model"].isin(models)].copy()
    df["cond_label"] = df["condition"].map(cond_label)
    df["model_label"] = df["model"].map(display_name)
    order = [cond_label(c) for c in conditions]
    palette = {display_name(m): MODEL_COLOR.get(m, "#555") for m in models}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.pointplot(
        data=df, x="cond_label", y=metric, hue="model_label", order=order,
        hue_order=[display_name(m) for m in models], palette=palette,
        markers=["o", "s", "^", "D"][: len(models)],
        linestyles=["-", "--", "-.", ":"][: len(models)],
        markersize=6, linewidth=1.8, ax=ax, dodge=0.18, err_kws={"linewidth": 0},
    )
    for model in models:
        idx = model_index(stats, model, conditions)
        offs = (models.index(model) - (len(models) - 1) / 2) * 0.18
        ax.errorbar(
            [i + offs for i in range(len(conditions))], idx[metric],
            yerr=idx["ci95"] if metric == "rate" else None,
            fmt="none", ecolor=MODEL_COLOR.get(model, "#555"),
            capsize=3, elinewidth=1.0, capthick=1.0, alpha=0.85,
        )
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{METRIC_LABEL.get(metric, metric)} by condition, per model")
    ax.legend(title="", loc="lower left", fontsize=9)
    sns.despine(ax=ax)
    return _save(fig, out_dir, "fig4_model_comparison")


def fig_gap_per_category(cat_table: pd.DataFrame, out_dir: Path) -> Path | None:
    """Fig 5: high-low status gap per category, domain vs irrelevant."""
    needed = {"domain_high", "domain_low", "irrel_high", "irrel_low"}
    if not needed.issubset(cat_table.columns):
        return None

    domain_gap = (cat_table["domain_high"] - cat_table["domain_low"]).sort_values()
    irrel_gap = (cat_table["irrel_high"] - cat_table["irrel_low"]).reindex(
        domain_gap.index
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(domain_gap))
    w = 0.35
    ax.barh(x - w / 2, domain_gap.values, w, label="Domain gap (high−low)",
            color="#1f78b4", edgecolor="black", linewidth=0.7)
    ax.barh(x + w / 2, irrel_gap.values, w, label="Irrel. gap (high−low)",
            color="#e31a1c", edgecolor="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(x)
    ax.set_yticklabels(domain_gap.index)
    ax.set_xlabel("Δ Pass Rate (High − Low status)")
    ax.set_title("Status Gap by Category\n"
                 "(negative = high-status users get more deference)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    return _save(fig, out_dir, "fig5_gap_per_category")


def fig_persona_spread(persona_table: pd.DataFrame, out_dir: Path, model: str) -> Path | None:
    """Fig 6: pass rate per individual persona.

    Only meaningful with more than one persona per cell. A wide within-condition
    spread means the "status effect" is partly persona identity.
    """
    if persona_table.empty or persona_table.groupby("condition").size().max() < 2:
        return None

    t = persona_table.reset_index()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(t))))
    conds = [c for c in CONDITIONS if c in set(t["condition"])]
    colors = [cond_color(c) for c in t["condition"]]
    y = np.arange(len(t))
    ax.barh(y, t["pass_rate"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.status_label}" for r in t.itertuples()], fontsize=8)
    ax.set_xlabel("Pass Rate")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(0, 1)
    ax.set_title(f"Pass Rate per Persona ({display_name(model)})\n"
                 "spread within a colour = persona identity, not status", fontsize=11)
    ax.legend(handles=[mpatches.Patch(color=cond_color(c), label=cond_label(c))
                       for c in conds], fontsize=8, loc="lower right")
    plt.tight_layout()
    return _save(fig, out_dir, "fig6_persona_spread")


def fig_irrelevant_by_dimension(
    df: pd.DataFrame, out_dir: Path, models: list[str]
) -> Path | None:
    """Fig 7: irrelevant-status pass rate averaged within each status dimension.

    The irrel_* conditions pool four different ways of signalling status. This
    separates them, because the *channel* turns out to matter: an institutional
    affiliation moves the model far more than a subscription tier, and pooling
    them into one irrel_high number hides that entirely.
    """
    sub = df[df["condition"].isin(["irrel_high", "irrel_low"])].copy()
    if sub.empty or "status_dimension" not in sub.columns:
        return None

    sub["level"] = sub["condition"].map({"irrel_high": "High status",
                                         "irrel_low": "Low status"})
    sub["model_label"] = sub["model"].map(display_name)
    dims = [d for d in DIMENSION_COLOR if d in set(sub["status_dimension"])]
    dims += [d for d in sorted(set(sub["status_dimension"])) if d not in dims]
    labels = {d: d.replace("_", "\n") for d in dims}
    sub["dim_label"] = sub["status_dimension"].map(labels)

    g = sns.catplot(
        data=sub, x="dim_label", y="passes", hue="level", col="model_label",
        kind="bar", order=[labels[d] for d in dims],
        col_order=[display_name(m) for m in models],
        hue_order=["High status", "Low status"],
        palette={"High status": "#9e3d52", "Low status": "#e0a0ac"},
        errorbar=("ci", 95), capsize=0.12, err_kws={"linewidth": 1.1},
        height=3.7, aspect=1.05, edgecolor="#3a3a38", linewidth=0.7,
        saturation=1.0, legend_out=True,
    )
    for ax, model in zip(g.axes.flat, models):
        m = sub[sub["model"] == model]
        for i, d in enumerate(dims):
            hi = m[(m["status_dimension"] == d) & (m["condition"] == "irrel_high")]["passes"]
            lo = m[(m["status_dimension"] == d) & (m["condition"] == "irrel_low")]["passes"]
            if hi.empty or lo.empty:
                continue
            gap = hi.mean() - lo.mean()
            ax.text(i, 1.03, f"{gap:+.0%}", ha="center", fontsize=8,
                    fontweight="semibold",
                    color="#9e3d52" if gap > 0 else "#3d5a80")
        ax.set_ylim(0, 1.14)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="x", labelsize=8)

    g.set_titles("{col_name}")
    g.set_axis_labels("", "Pass Rate")
    if g.legend is not None:
        g.legend.set_title("")
        sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.13),
                        ncol=2, frameon=False)
    g.figure.suptitle(
        "Domain-irrelevant status, split by status channel"
        "\n(labels = high − low gap within each channel)",
        y=1.06, fontsize=11.5,
    )
    return _save(g.figure, out_dir, "fig7_irrelevant_by_dimension")


def make_all(stats, tables, out_dir: Path, models: list[str], primary: str,
             df: pd.DataFrame | None = None) -> dict:
    """Build every figure; ``primary`` drives the single-model figures."""
    set_style()
    paths = {
        "fig1": fig_rate_by_condition(stats, out_dir, models),
        "fig1_allpass": fig_rate_by_condition(
            stats, out_dir, models, metric="all_pass_rate_k5"),
        "fig4": fig_model_comparison(stats, out_dir, models),
    }
    if primary in tables["by_category"]:
        cat = tables["by_category"][primary]
        paths["fig2"] = fig_2x2_matrix(stats, out_dir, primary)
        paths["fig3"] = fig_category_heatmap(cat, out_dir, primary)
        paths["fig5"] = fig_gap_per_category(cat, out_dir)
    if primary in tables.get("by_persona", {}):
        paths["fig6"] = fig_persona_spread(tables["by_persona"][primary], out_dir, primary)
    if df is not None:
        paths["fig7"] = fig_irrelevant_by_dimension(df, out_dir, models)
    return paths
