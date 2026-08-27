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
    MODEL_VARIANTS,
)

METRIC_LABEL = {"rate": "Pass Rate", "sage_score": "SAGE Safety Score"}


def display_name(model: str) -> str:
    v = MODEL_VARIANTS.get(model)
    return v.display if v else model


def set_style() -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    for path in (pdf, out_dir / f"{name}.png"):
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")
    return pdf


def fig_rate_by_condition(
    stats: pd.DataFrame, out_dir: Path, models: list[str],
    conditions=CONDITIONS, metric: str = "rate",
) -> Path:
    """Fig 1: per-condition pass rate, one panel per model variant."""
    fig, axes = plt.subplots(
        1, len(models), figsize=(5.2 * len(models), 5.5), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        idx = model_index(stats, model, conditions)
        xs = range(len(conditions))
        err = idx["ci95"] if metric == "rate" and "ci95" in idx.columns else None
        ax.bar(
            xs, idx[metric], yerr=err, capsize=5,
            color=[COND_COLOR[c] for c in conditions],
            edgecolor="black", linewidth=0.8,
            error_kw={"elinewidth": 1.8, "capthick": 1.8}, zorder=3,
        )
        if "control" in idx.index and pd.notna(idx.loc["control", metric]):
            ax.axhline(idx.loc["control", metric], color="#555", linestyle="--",
                       linewidth=1.2, alpha=0.7, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([COND_LABEL[c] for c in conditions], fontsize=9)
        ax.set_title(display_name(model), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.3)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        for i, (_, row) in enumerate(idx.iterrows()):
            if pd.isna(row[metric]):
                continue
            bump = row["ci95"] if err is not None and pd.notna(row["ci95"]) else 0
            # Clamp so a bar at ceiling does not push its label into the title.
            y = min(row[metric] + bump + 0.03, 1.20)
            ax.text(i, y, f"{row[metric]:.0%}",
                    ha="center", fontsize=9, fontweight="bold")

    axes[0].set_ylabel(METRIC_LABEL.get(metric, metric))
    fig.suptitle(
        f"{METRIC_LABEL.get(metric, metric)} by User Status Condition\n"
        "(SAGE-Eval rubric, sycophancy pressure: “I think it's fine.”)",
        fontsize=12,
    )
    fig.legend(
        handles=[mpatches.Patch(color=COND_COLOR[c], label=COND_LABEL_FLAT[c])
                 for c in conditions],
        loc="lower center", ncol=len(conditions), fontsize=9,
        frameon=True, bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    suffix = "" if metric == "rate" else f"_{metric}"
    return _save(fig, out_dir, f"fig1_rate_by_condition{suffix}")


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
                xticklabels=[COND_LABEL_FLAT[c] for c in conditions])
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
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = [("o", "-", 2.5), ("s", "--", 2.0), ("^", "-.", 2.0), ("D", ":", 2.0)]
    for model, (marker, ls, lw) in zip(models, styles):
        idx = model_index(stats, model, conditions)
        ax.errorbar(
            range(len(conditions)), idx[metric],
            yerr=idx["ci95"] if metric == "rate" else None,
            marker=marker, linestyle=ls, linewidth=lw,
            markersize=9, capsize=5, label=display_name(model),
        )
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COND_LABEL_FLAT[c] for c in conditions], fontsize=10)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(f"{METRIC_LABEL.get(metric, metric)} by Condition, per Model",
                 fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()
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
    colors = [COND_COLOR[c] for c in t["condition"]]
    y = np.arange(len(t))
    ax.barh(y, t["pass_rate"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.status_label}" for r in t.itertuples()], fontsize=8)
    ax.set_xlabel("Pass Rate")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(0, 1)
    ax.set_title(f"Pass Rate per Persona ({display_name(model)})\n"
                 "spread within a colour = persona identity, not status", fontsize=11)
    ax.legend(handles=[mpatches.Patch(color=COND_COLOR[c], label=COND_LABEL_FLAT[c])
                       for c in conds], fontsize=8, loc="lower right")
    plt.tight_layout()
    return _save(fig, out_dir, "fig6_persona_spread")


def make_all(stats, tables, out_dir: Path, models: list[str], primary: str) -> dict:
    """Build every figure; ``primary`` drives the single-model figures."""
    set_style()
    paths = {
        "fig1": fig_rate_by_condition(stats, out_dir, models),
        "fig1_sage": fig_rate_by_condition(stats, out_dir, models, metric="sage_score"),
        "fig4": fig_model_comparison(stats, out_dir, models),
    }
    if primary in tables["by_category"]:
        cat = tables["by_category"][primary]
        paths["fig2"] = fig_2x2_matrix(stats, out_dir, primary)
        paths["fig3"] = fig_category_heatmap(cat, out_dir, primary)
        paths["fig5"] = fig_gap_per_category(cat, out_dir)
    if primary in tables.get("by_persona", {}):
        paths["fig6"] = fig_persona_spread(tables["by_persona"][primary], out_dir, primary)
    return paths
