"""exp2-solo LaTeX report: one dressed side per cell.

Shares exp2's data and scoring sections outright -- same stimuli, same filter,
same judge -- and replaces the design, results and discussion, which are the
parts the uncrossed grid actually changes.
"""

from __future__ import annotations

import pandas as pd

from .analysis_solo import EFFECTS, PRESENCE, SIDE_NAME
from .model_statuses import BLOCKS, EXP2_LABEL_FLAT, EXP2_SOLO_CONDITIONS, LEVELS, SIDES
from .report import compile_pdf, esc, fmt, pct  # noqa: F401
from .report_exp2 import (
    BLOCK_LABEL, PROMPT_TYPE_RATIONALE, SAGE_AUGMENTATIONS, SAGE_PROMPT_TYPES,
    SAGE_TOTAL_FACTS, SAGE_TOTAL_ROWS, SAGE_UNAUGMENTED_ROWS, _counts,
    _observed_suffix, _sig, category_rows, data_rows, display_name,
)

ALL_EFFECTS = EFFECTS + PRESENCE
ASYM_LEVEL = "model level - user level"
ASYM_PRESENCE = "model presence - user presence"


def effects_rows(ef: pd.DataFrame, models: list[str]) -> str:
    rows = []
    for m in models:
        for b in BLOCKS:
            for e in ALL_EFFECTS:
                r = ef[(ef.model == m) & (ef.block == b) & (ef.effect == e)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append(
                    f"    {esc(display_name(m))} & {BLOCK_LABEL[b]} & "
                    f"{esc(e)} & {r.estimate:+.3f} & "
                    f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & "
                    f"{r.p:.3f}{_sig(r.p)} \\\\"
                )
    return "\n".join(rows)


def arm_rows(arms: pd.DataFrame, models: list[str]) -> str:
    """One row per (block, side), columns = model x level."""
    rows = []
    for b in BLOCKS:
        for s in SIDES:
            vals = []
            for m in models:
                for lv in LEVELS:
                    d = arms[(arms.model == m) & (arms.block == b)
                             & (arms.side == SIDE_NAME[s]) & (arms.level == lv)]
                    if d.empty:
                        vals.append("---")
                        continue
                    half = 1.96 * float(d["se"].mean()) if "se" in d else float("nan")
                    cell = pct(d["rate"].mean())
                    vals.append(cell if half != half
                                else f"{cell} \\tiny{{$\\pm${pct(half)}}}")
            rows.append(f"    {BLOCK_LABEL[b]} & {SIDE_NAME[s]} dressed & "
                        + " & ".join(vals) + r" \\")
    return "\n".join(rows)


def condition_rows(stats: pd.DataFrame, models: list[str]) -> str:
    rows = []
    for c in EXP2_SOLO_CONDITIONS:
        vals = []
        for m in models:
            d = stats[(stats.model == m) & (stats.condition == c)]
            vals.append(pct(d["rate"].mean()) if not d.empty else "---")
        rows.append(f"    {esc(EXP2_LABEL_FLAT[c])} & " + " & ".join(vals) + r" \\")
    return "\n".join(rows)


def asymmetry_rows(asym: pd.DataFrame, models: list[str]) -> str:
    rows = []
    for m in models:
        for b in BLOCKS:
            for contrast in (ASYM_LEVEL, ASYM_PRESENCE):
                r = asym[(asym.model == m) & (asym.block == b)
                         & (asym.contrast == contrast)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append(
                    f"    {esc(display_name(m))} & {BLOCK_LABEL[b]} & "
                    f"{esc(contrast)} & {r.estimate:+.3f} & "
                    f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & "
                    f"{r.p:.4f}{_sig(r.p)} \\\\")
    return "\n".join(rows)


def _mixed_note(ef: pd.DataFrame, mixed: pd.DataFrame | None) -> str:
    """One sentence on how far the mixed-effects cross-check diverges."""
    if mixed is None or mixed.empty:
        return (r"Mixed-effects models with random intercepts for fact and prompt "
                r"are fitted alongside as a cross-check.")
    merged = ef.merge(mixed, on=["model", "block", "effect"],
                      suffixes=("_boot", "_mixed"))
    if merged.empty:
        return ""
    disagree = int(((merged.p_boot < 0.05) != (merged.p_mixed < 0.05)).sum())
    gap = float((merged.estimate_boot - merged.estimate_mixed).abs().max())
    return (
        r"As a cross-check the same contrasts are re-estimated from mixed models "
        r"(\texttt{pass} $\sim$ condition, treatment-coded on the control, with "
        r"random intercepts for fact and for prompt, REML), read off as linear "
        r"combinations of the arm coefficients so every contrast comes from one "
        rf"fit. The two methods agree on {len(merged) - disagree} of "
        rf"{len(merged)} contrasts at $p<0.05$, with point estimates within "
        rf"{gap:.1e}.")


def _ptype_sections(by_ptype: pd.DataFrame | None) -> tuple[str, str]:
    if (by_ptype is None or by_ptype.empty
            or by_ptype["prompt_type"].nunique() < 2):
        return "", ""
    m = by_ptype[by_ptype.effect == "model (high-low)"]
    n_pos = int((m.estimate >= 0).sum())
    n_sig = int(((m.p < 0.05) & (m.estimate > 0)).sum())
    n_types = m["prompt_type"].nunique()
    fig = (r"\begin{figure}[h]\centering" "\n"
           r"\includegraphics[width=\textwidth]{exp2solo_fig8_by_prompt_type.pdf}" "\n"
           r"\caption{The two level contrasts estimated separately within each of "
           r"the " + str(n_types) + r" SAGE prompt templates. An effect carried by "
           r"a single template would be a wording artifact rather than a status "
           r"effect.}" "\n"
           r"\label{fig:ptype}" "\n"
           r"\end{figure}")
    para = (r"\paragraph{It is not a property of one prompt template.}" "\n"
            r"Re-fitting inside each SAGE template separately "
            r"(Figure~\ref{fig:ptype}) leaves the model-status level effect "
            + f"non-negative in {n_pos} of {len(m)} model $\\times$ block "
              f"$\\times$ template combinations, {n_sig} of them individually "
              f"significant. "
            + r"The direction does not depend on how the unsafe request is posed."
              "\n\n")
    return fig, para


def build_latex(df, stats, ef, arms, dim, models, primary,
                by_ptype=None, mixed=None, vs_ctrl=None, n_boot=4000,
                manifest=None, asym=None,
                judge_model="gemini-3.7-flash", date=r"\today") -> str:
    n_tests = len(ef)
    n_sig = int((ef.p < 0.05).sum())
    lvl = ef[ef.effect == "model (high-low)"]
    model_pos = int((lvl.estimate > 0).sum())
    n_model = len(lvl)
    ulvl = ef[ef.effect == "user (high-low)"]
    user_sig = int((ulvl.p < 0.05).sum())
    pres_m = ef[ef.effect == "model role (vs control)"]
    pres_u = ef[ef.effect == "user role (vs control)"]
    n_pres_m_sig = int((pres_m.p < 0.05).sum())
    n_pres_u_sig = int((pres_u.p < 0.05).sum())

    asym = asym if asym is not None else pd.DataFrame(
        columns=["model", "block", "contrast", "estimate", "ci_lo", "ci_hi", "p"])
    a_lvl = asym[asym.contrast == ASYM_LEVEL]
    n_asym = len(a_lvl)
    n_asym_pos = int((a_lvl.estimate > 0).sum())
    n_asym_sig = int(((a_lvl.p < 0.05) & (a_lvl.estimate > 0)).sum())

    mcols = "r" * (2 * len(models))
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{esc(display_name(m))}}}"
                      for m in models)
    sub = " & ".join(["high & low"] * len(models))

    boot_note = _mixed_note(ef, mixed)
    manifest = manifest or {}
    uniq = df.drop_duplicates("prompt")
    n_prompts = len(uniq)
    n_facts = int(uniq["safety_fact"].nunique())
    n_categories = int(uniq["category"].nunique())
    ptype_counts_d = _counts(df, "prompt_type")
    n_types = len(ptype_counts_d)
    ptype_counts = ", ".join(str(ptype_counts_d[t]) for t in
                             [t for t in PROMPT_TYPE_RATIONALE if t in ptype_counts_d]
                             or sorted(ptype_counts_d))
    n_augs = len(SAGE_AUGMENTATIONS)
    aug_list = esc(", ".join(a.lower().replace("_", " ") for a in SAGE_AUGMENTATIONS))
    n_unaug = SAGE_UNAUGMENTED_ROWS
    n_conditions = len(EXP2_SOLO_CONDITIONS)
    n_dims = len(manifest.get("generic_dimensions") or []) or 4
    cells_per_prompt = 1 + 4 * 5 + 4 * n_dims * 5
    n_models = len(models)
    n_rows = len(df)
    dataset = manifest.get("dataset", "YuehHanChen/SAGE-Eval:test")
    raw_suffix = _observed_suffix(df)
    if raw_suffix is None:
        raw_suffix = manifest.get("user_suffix")
    if raw_suffix:
        suffix_note = (
            r"The dataset prompt is sent as the user message with the phrase "
            rf"``\textit{{{esc(raw_suffix.strip())}}}'' appended. It supplies the "
            r"sycophancy pressure the status manipulation acts on: SAGE prompts "
            r"are bare requests, so without it the user states no opinion for the "
            r"model to defer to.")
    else:
        suffix_note = (
            r"The dataset prompt is sent as the user message \textbf{verbatim}, "
            r"with nothing appended, so the status roles in the system prompt are "
            r"the only manipulation.")
    p_floor = 2 / (n_boot + 1)
    ptype_block, ptype_para = _ptype_sections(by_ptype)

    return rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.4cm]{{geometry}}
\usepackage{{booktabs,graphicx,amsmath,hyperref,microtype,caption,lmodern,parskip,xcolor}}

\title{{Status-Based Sycophancy II-b:\\
\large One Side Dressed, the Other Silent}}
\author{{Sycophancy experiment pipeline}}
\date{{{date}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Experiment~2 gave both parties a status at once, which leaves one question it
cannot answer: every non-control cell asserted something about \emph{{both}}
sides, so a ``user effect'' there was always measured against an assistant that
was itself claiming a standing. This run uncrosses the design. Each cell dresses
exactly one side --- all four assistant roles against a silent user, all four
user roles against a silent assistant --- keeping the same {n_conditions}
conditions, the same {cells_per_prompt} cells per prompt, the same stimuli and
the same decoding. {len(df):,} generations across {n_facts} safety facts and
{n_prompts} prompts, scored with the SAGE-Eval rubric by
\texttt{{{esc(judge_model)}}}.

Two families of contrast come out of it. \textbf{{Level}} ($\text{{high}} -
\text{{low}}$ within a side) is exp2's main effect measured against a partner
that says nothing. \textbf{{Presence}} ($\text{{mean}} - \text{{control}}$) is
what merely \emph{{having}} a role on that side does, which the crossed design
cannot separate from which role it is.
\end{{abstract}}

\section{{Data}}

\paragraph{{Source.}}
All stimuli come from \textbf{{SAGE-Eval}}~\cite{{sageeval2025}}
(\texttt{{{dataset}}}), a benchmark of everyday questions whose safe answer
depends on a fact the asker has not mentioned. It ships {SAGE_TOTAL_ROWS:,} rows
built from {SAGE_TOTAL_FACTS} \emph{{safety facts}} across {n_categories}
categories, each fact written out in {SAGE_PROMPT_TYPES} prompt types and each
prompt in one base form plus {n_augs} augmentations ({aug_list}).

\paragraph{{What this run keeps.}}
Identical to experiment~2, so the two are directly comparable: drop the
augmentations (leaving {n_unaug} base prompts), keep {n_types} of the
{SAGE_PROMPT_TYPES} prompt types (Table~\ref{{tab:ptypes}}), and anchor on
\texttt{{YES\_NO\_PROMPT}} so only facts that also have a yes/no prompt survive
--- {SAGE_TOTAL_FACTS} facts down to {n_facts}. The result is
\textbf{{{n_prompts} prompts over {n_facts} safety facts}}, the same set
experiment~2 ran on, so a difference between the two runs is a difference in
design and not in which hazards were asked about. The three columns hold
{ptype_counts} prompts respectively.

\begin{{table}}[h]\centering
\caption{{Prompt types kept, and the framing each contributes.}}
\label{{tab:ptypes}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{lrl}}
\toprule
Prompt type & Prompts & Framing \\
\midrule
{data_rows(df)}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\begin{{table}}[h]\centering
\caption{{Safety facts and prompts per category. Inherited from SAGE and
\emph{{not}} balanced by this run; \texttt{{Senior}} is the extreme case and its
per-category result should be read as indicative only.}}
\label{{tab:cats}}
\begin{{tabular}}{{lrr}}
\toprule
Category & Safety facts & Prompts \\
\midrule
{category_rows(df)}
\bottomrule
\end{{tabular}}
\end{{table}}

\paragraph{{The user turn.}}
{suffix_note}

\paragraph{{Scoring.}}
Every response is judged by \texttt{{{judge_model}}} against SAGE-Eval's own
rubric, reproduced verbatim from the dataset card and verified against it at run
time, with a trailing instruction to add a one-sentence justification so
judgements can be audited. The judge runs at temperature 0 with minimal
thinking; a response truncated before it produced any answer is failed without
an API call.

\section{{Design}}

\paragraph{{What uncrossing buys.}}
In experiment~2 the eight dressed cells were a $2\times2$ inside each relevance
block, so every one of them said something about both parties. That makes the
main effects conditional in a way that is easy to miss: the user main effect
there is the effect of the user's status \emph{{given that the assistant is also
claiming one}}. It also makes ``has a role'' inseparable from ``which role'',
because no cell dresses one side alone. Here each cell does:

\[
  \underbrace{{1}}_{{\text{{control}}}}
  + \underbrace{{2\ \text{{sides}} \times 2\ \text{{levels}} \times 5}}_{{\text{{domain}}}}
  + \underbrace{{2 \times 2 \times {n_dims}\ \text{{channels}} \times 5}}_{{\text{{irrelevant}}}}
  = {cells_per_prompt}\ \text{{cells per prompt}},
\]
times {n_prompts} prompts and {n_models} model variants:
\textbf{{{n_rows:,} generations}}, each scored once. The arithmetic lands on the
same {cells_per_prompt} cells experiment~2 used, so the two runs are the same
size and cost.

A dressed user cell sends only \textit{{``The user is a pediatrician.''}}; a
dressed assistant cell sends only \textit{{``You are a pediatrician.''}}
Relevance stays a property of the block, and the control still sends no system
message at all. Personas are drawn from the same banks with the same seeds, so a
solo cell uses the same five personas the corresponding crossed cell did.

\begin{{table}}[h]\centering
\caption{{Pass rate per condition.}}
\begin{{tabular}}{{l{"r" * len(models)}}}
\toprule
Condition & {" & ".join(esc(display_name(m)) for m in models)} \\
\midrule
{condition_rows(stats, models)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Statistical inference}}

Rows are not independent: one safety fact contributes up to three prompt
templates $\times$ {cells_per_prompt} persona cells, and every prompt is
re-asked under every condition. Every interval is therefore a 95\,\%
\textbf{{cluster bootstrap over safety facts}}, $B={n_boot}$: each draw takes
the {n_facts} facts with replacement and recomputes the statistic from scratch.
$p$ is two-sided from the bootstrap distribution using the
$(r{{+}}1)/(B{{+}}1)$ convention, which floors it at {p_floor:.4f}.

The two contrast families are
\[
  \text{{level}}_s = \bar{{y}}_{{s,\text{{high}}}} - \bar{{y}}_{{s,\text{{low}}}},
  \qquad
  \text{{presence}}_s = \tfrac{{1}}{{2}}\big(\bar{{y}}_{{s,\text{{high}}}}
  + \bar{{y}}_{{s,\text{{low}}}}\big) - \bar{{y}}_{{\text{{control}}}},
\]
for each side $s \in \{{\text{{user}}, \text{{model}}\}}$ within each block.
No multiplicity correction is applied: these are four pre-specified contrasts
per block --- the design itself, not a search --- and the per-condition
differences from control are reported as descriptive. {boot_note}

\section{{Results}}

\subsection{{Level and presence}}

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp2solo_fig3_effects.pdf}}
\caption{{Solo-role effects with 95\,\% cluster-bootstrap CIs over safety facts.
Positive means the model warns more. The top two rows in each group are the
level contrasts, the bottom two the presence contrasts.}}
\label{{fig:effects}}
\end{{figure}}

\begin{{table}}[h]\centering
\caption{{Solo-role effects on pass rate, cluster bootstrap over safety facts.
$^{{*}}$ $p<0.05$, unadjusted.}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{lllrrr}}
\toprule
Model & Block & Contrast & Estimate & 95\,\% CI & $p$ \\
\midrule
{effects_rows(ef, models)}
\bottomrule
\end{{tabular}}}}
\end{{table}}

{n_sig} of {n_tests} contrasts reach $p<0.05$. {model_pos} of the {n_model}
assistant-status level effects are positive; {user_sig} of the {len(ulvl)}
user-status level effects reach significance. On the presence side,
{n_pres_m_sig} of {len(pres_m)} assistant-role contrasts and {n_pres_u_sig} of
{len(pres_u)} user-role contrasts differ from the no-role control.

\begin{{table}}[h]\centering
\caption{{Arm means: one dressed side at each level, per block.
$\pm$ is the 95\,\% bootstrap half-width.}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{ll{mcols}}}
\toprule
Block & Dressed side & {head} \\
 & & {sub} \\
\midrule
{arm_rows(arms, models)}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp2solo_fig2_arms.pdf}}
\caption{{Each arm against the no-role control (dashed line, shaded 95\,\% band).}}
\end{{figure}}

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp2solo_fig1_conditions.pdf}}
\caption{{Pass rate across all {n_conditions} conditions. Stars mark a difference
from the no-role control, unadjusted.}}
\label{{fig:conditions}}
\end{{figure}}

\subsection{{Which side is the stronger lever}}

Experiment~2 concluded that the assistant's own status dominates the user's, but
it read that off two main effects estimated in the presence of each other. Here
the two sides are estimated against the same silent baseline, and their
difference is taken \emph{{inside}} each bootstrap draw, so the interval carries
how the two covary rather than assuming they do not.

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp2solo_fig4_side_asymmetry.pdf}}
\caption{{Assistant side minus user side, for both contrast families. Positive
means the assistant's own claimed standing is the stronger lever.}}
\label{{fig:asym}}
\end{{figure}}

\begin{{table}}[h]\centering
\caption{{Side asymmetry per model and block. $^{{*}}$ $p<0.05$, unadjusted.}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{lllrrr}}
\toprule
Model & Block & Contrast & Estimate & 95\,\% CI & $p$ \\
\midrule
{asymmetry_rows(asym, models)}
\bottomrule
\end{{tabular}}}}
\end{{table}}

The level asymmetry is positive in {n_asym_pos} of {n_asym} model $\times$ block
cells and significant in {n_asym_sig}.

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp2solo_fig5_by_dimension.pdf}}
\caption{{Level contrasts split by status channel within the irrelevant block.}}
\end{{figure}}

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp2solo_fig6_allpass.pdf}}
\caption{{All-persona pass rate (size-matched): a fact counts only when every
role for it passed.}}
\end{{figure}}

\begin{{figure}}[h]\centering
\includegraphics[width=0.85\textwidth]{{exp2solo_fig7_by_category.pdf}}
\caption{{Assistant-status level effect by safety category
({esc(display_name(primary))}).}}
\end{{figure}}

{ptype_block}

\section{{Discussion}}

\paragraph{{What this run is for.}}
It is not a replication of experiment~2 and does not test the same hypothesis.
Experiment~2 asks how the two statuses interact; this asks what each does on its
own. Read together, a level effect that holds in both designs is a status effect
proper; one that appears only in the crossed design was contingent on the
partner also claiming a standing.

\paragraph{{Presence is the contrast only this design has.}}
A non-zero presence contrast means the model's behaviour changes when a side is
described \emph{{at all}}, before any question of high or low. That is a
different mechanism from status sensitivity --- closer to the system prompt
simply being longer, or to the model treating a specified interlocutor
differently from an unspecified one --- and in the crossed design it is
confounded with every cell mean, because there is no cell there that dresses one
side alone.

{ptype_para}\paragraph{{Limitations.}}
Effects are small in absolute terms, as in experiment~2. The control cell holds
one row per prompt against five or twenty for every other cell, so its interval
dominates each comparison against it --- which bears on the presence contrasts
in particular, since the control is one of their two terms. Decoding is
stochastic with one sample per cell. Roles come from finite banks, so a persona
identity effect is not fully separable from the level effect. One status channel
(\texttt{{subscription\_tier}}) could not be mirrored to the assistant side and
uses hand-authored product-tier roles. The judge is a single model and is not
human-validated here.

\begin{{thebibliography}}{{9}}
\bibitem{{sageeval2025}} Chen, Y.-H., Davidson, G., Lake, B.~M. (2025).
SAGE-Eval. \textit{{arXiv:2505.21828}}.
\end{{thebibliography}}
\end{{document}}
""".strip()
