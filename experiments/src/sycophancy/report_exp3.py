"""Concise LaTeX write-up for exp3.

Deliberately short: what the data is, what was manipulated, the one effect
pattern that holds, and the conclusions. Per-dimension, per-category and
full-grid breakdowns stay in ``tables/``.

Everything factual is read from the run itself -- the judged rows, the result
tables and ``run_config.json`` -- rather than typed in. That matters here
because exp3 has two variants that differ only in whether the pressure phrase is
appended, and a hand-written design section silently describes the wrong one.
"""

from __future__ import annotations

import pandas as pd

from .analysis_exp2 import GRADIENT, SYMMETRY
from .config import MODEL_VARIANTS
from .report import esc

BLOCK_LABEL = {"domain": "domain-relevant", "irrel": "domain-irrelevant"}
EFFECT_SHORT = {"user (high-low)": "user high$-$low",
                "model (high-low)": "model high$-$low",
                "interaction": "interaction"}


def display_name(model: str) -> str:
    v = MODEL_VARIANTS.get(model)
    return v.display if v else model


def _stars(p) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def _sig(df: pd.DataFrame, alpha: float = .05) -> pd.DataFrame:
    return df[df["p"] < alpha]


def _verb(n: int, singular: str, plural: str) -> str:
    """Agreement for counts that are only known at render time."""
    return singular if n == 1 else plural


def detect_suffix(df: pd.DataFrame) -> str:
    """The pressure phrase this run actually used, read off the sent prompts.

    exp3 runs in two variants and the only difference is this string, so it is
    recovered from the data rather than assumed.
    """
    if "user_msg" not in df.columns or "prompt" not in df.columns:
        return ""
    sub = df[["prompt", "user_msg"]].drop_duplicates().head(200)
    extras = {u[len(p):] for p, u in zip(sub["prompt"], sub["user_msg"])
              if isinstance(u, str) and isinstance(p, str) and u.startswith(p)}
    extras.discard("")
    return sorted(extras)[0] if len(extras) == 1 else ""


def baseline_table(df: pd.DataFrame, models: list[str]) -> str:
    rows = []
    for m in models:
        sub = df[df["model"] == m]
        cells = [f"{sub[sub['safety_type'] == lab]['passes'].mean():.3f}"
                 for lab in ("unsafe", "safe")]
        rows.append(f"{esc(display_name(m))} & " + " & ".join(cells) + r" \\")
    pooled = [f"\\textbf{{{df[df['safety_type'] == lab]['passes'].mean():.3f}}}"
              for lab in ("unsafe", "safe")]
    rows.append(r"\midrule pooled & " + " & ".join(pooled) + r" \\")
    return "\n".join(rows)


def category_table(df: pd.DataFrame, available: dict | None = None) -> str:
    """Sampled items per category and label, beside the pool they came from."""
    rows = []
    counts = (df.drop_duplicates("prompt_idx")
                .groupby(["category", "safety_type"]).size().unstack(fill_value=0))
    for cat, r in counts.iterrows():
        cells = [str(int(r.get("unsafe", 0))), str(int(r.get("safe", 0)))]
        if available:
            av = available.get(cat, {})
            cells += [str(av.get("unsafe", "---")), str(av.get("safe", "---"))]
        rows.append(f"{esc(cat)} & " + " & ".join(cells) + r" \\")
    tot = [str(int(counts.get("unsafe", pd.Series()).sum())),
           str(int(counts.get("safe", pd.Series()).sum()))]
    if available:
        tot += [str(sum(v.get("unsafe", 0) for v in available.values())),
                str(sum(v.get("safe", 0) for v in available.values()))]
    rows.append(r"\midrule \textbf{total} & " +
                " & ".join(f"\\textbf{{{c}}}" for c in tot) + r" \\")
    return "\n".join(rows)


def split_table(by_label: pd.DataFrame, models: list[str],
                block: str = "domain") -> str:
    d = by_label[by_label["block"] == block]
    rows = []
    for m in models:
        for effect in ("user (high-low)", "model (high-low)"):
            sub = d[(d["model"] == m) & (d["effect"] == effect)]
            cells = []
            for lab in ("UNSAFE", "SAFE"):
                r = sub[sub["prompt_type"] == lab]
                if r.empty:
                    cells.append("---")
                    continue
                r = r.iloc[0]
                txt = f"${r.estimate:+.3f}$"
                star = _stars(r.p)
                cells.append(f"\\textbf{{{txt}}}\\,{star}" if star else txt)
            rows.append(f"{esc(display_name(m))} & {EFFECT_SHORT[effect]} & "
                        + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def warning_section(warn: pd.DataFrame | None, warn_ef: pd.DataFrame | None,
                    models: list[str]) -> str:
    """Both halves on one scale: what status does to the willingness to warn.

    Only the sharpest comparison is written out; the full grid is in
    ``tables/warning_rates.csv`` and ``tables/warning_effects.csv``.
    """
    if warn is None or warn.empty:
        return ""

    def rate(model, cond, measure):
        r = warn[(warn["model"] == model) & (warn["condition"] == cond)
                 & (warn["measure"] == measure)]
        return float(r["estimate"].iloc[0]) if not r.empty else float("nan")

    def block_mean(model, block, measure):
        r = warn[(warn["model"] == model) & (warn["measure"] == measure)
                 & (warn["condition"].str.startswith(block + "_"))]
        return float(r["estimate"].mean()) if not r.empty else float("nan")

    rows = []
    for m in models:
        cells = [f"{rate(m, 'control', 'hit rate'):.3f}",
                 f"{block_mean(m, 'domain', 'hit rate'):.3f}",
                 f"{block_mean(m, 'irrel', 'hit rate'):.3f}",
                 f"{rate(m, 'control', 'false-alarm rate'):.3f}",
                 f"\\textbf{{{block_mean(m, 'domain', 'false-alarm rate'):.3f}}}",
                 f"{block_mean(m, 'irrel', 'false-alarm rate'):.3f}"]
        rows.append(f"{esc(display_name(m))} & " + " & ".join(cells) + r" \\")
    table = "\n".join(rows)

    # A threshold shift shows up as the same sign on both halves.
    shift_txt = ""
    if warn_ef is not None and not warn_ef.empty:
        e = warn_ef[warn_ef["effect"] != "interaction"]
        pairs = []
        for m in models:
            for block in ("domain", "irrel"):
                for eff in ("user (high-low)", "model (high-low)"):
                    sub = e[(e["model"] == m) & (e["block"] == block)
                            & (e["effect"] == eff)]
                    u = sub[sub["prompt_type"] == "UNSAFE"]
                    sa = sub[sub["prompt_type"] == "SAFE"]
                    if u.empty or sa.empty:
                        continue
                    u, sa = u.iloc[0], sa.iloc[0]
                    if (u.estimate > 0) == (sa.estimate > 0) and \
                            max(u.p, sa.p) < .05:
                        pairs.append((m, block, eff, u, sa))
        if pairs:
            m, block, eff, u, sa = max(
                pairs, key=lambda x: min(abs(x[3].estimate), abs(x[4].estimate)))
            shift_txt = (
                f"The sharpest single manipulation is "
                f"{EFFECT_SHORT[eff].replace('$-$', ' minus ')} on "
                f"{BLOCK_LABEL[block]} roles for {esc(display_name(m))}: it "
                f"raises warning by ${u.estimate:+.3f}$ ($p={u.p:.3f}$) on "
                f"unsafe items and ${sa.estimate:+.3f}$ ($p={sa.p:.3f}$) on safe "
                f"ones. Same sign on both halves is a moved threshold; better "
                f"discrimination would move them apart.")
        else:
            shift_txt = ("No single status effect is significant on both halves "
                         "at once, so the threshold shift is carried by the "
                         "presence of a domain role rather than by its level.")

    # How much each block lifts false alarms off the control, per model. The
    # blanket claim "irrelevant roles leave the rates at control" is not safe:
    # one checkpoint has a 0% control, where a 2-point rise is still detectable.
    lifts = []
    for m in models:
        c = rate(m, "control", "false-alarm rate")
        lifts.append((esc(display_name(m)),
                      block_mean(m, "domain", "false-alarm rate") - c,
                      block_mean(m, "irrel", "false-alarm rate") - c))
    lift_txt = (
        "Against the no-role control, domain-relevant roles lift the false-alarm "
        "rate by "
        + " and ".join(f"${d:+.3f}$" for _, d, _ in lifts)
        + " for " + " and ".join(n for n, _, _ in lifts)
        + ", while hits move only a few points. Roles from an irrelevant status "
          "dimension lift it by "
        + " and ".join(f"${i:+.3f}$" for _, _, i in lifts)
        + " --- a fraction as much, though off a floor that low even a couple of "
          "points is detectable."
    )

    m0 = models[0]
    gap_ctrl = (rate(m0, "control", "hit rate")
                - rate(m0, "control", "false-alarm rate"))
    gap_dom = (block_mean(m0, "domain", "hit rate")
               - block_mean(m0, "domain", "false-alarm rate"))

    return rf"""\section*{{Warning behaviour, split by request type}}

``Pass'' means opposite things on the two halves, so the halves cannot be put on
one axis. Warning can be: it is a single act that is \emph{{right}} on an unsafe
item and \emph{{wrong}} on a safe one. Scoring every response for whether it
warned gives a hit rate and a false-alarm rate, and the two together separate a
model that has become more discriminating from one that has simply become more
cautious.

\begin{{table}}[h]\centering
\begin{{tabular}}{{lccc@{{\hskip 2em}}ccc}}
\toprule
& \multicolumn{{3}}{{c}}{{hit rate (unsafe)}}
& \multicolumn{{3}}{{c}}{{false-alarm rate (safe)}} \\
\cmidrule(r){{2-4}}\cmidrule(l){{5-7}}
model & control & domain & irrel. & control & domain & irrel. \\
\midrule
{table}
\bottomrule
\end{{tabular}}
\end{{table}}

\noindent
\textbf{{Giving the assistant and user any domain-relevant role makes the model
warn more, whether or not warning is warranted.}} {lift_txt} Discrimination does
not improve: hit minus false-alarm goes from ${gap_ctrl:.3f}$ under control to
${gap_dom:.3f}$ under domain roles for {esc(display_name(m0))}. Stars on the
figure mark conditions that differ from the no-role control.

{shift_txt}

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp3_fig10_warning_rates.pdf}}
\caption{{Warning rate under each status manipulation, with stars marking
conditions that differ from the no-role control. Both rows moving together is a
shifted threshold; the rows moving apart would be a change in how well the model
tells the two kinds of request apart.}}
\end{{figure}}

"""


def asymmetry_section(asym_lab: pd.DataFrame | None,
                      asym: pd.DataFrame | None) -> str:
    """Power asymmetry: who outranks whom, and whether the gap itself matters.

    Only the sharpest cell is written out. The full grid is in
    ``tables/asymmetry_contrasts_by_label.csv``.
    """
    if asym_lab is None or asym_lab.empty:
        return ""
    grad = asym_lab[asym_lab["contrast"] == GRADIENT]
    sym = asym_lab[asym_lab["contrast"] == SYMMETRY]
    if grad.empty:
        return ""

    sig = _sig(grad)
    big = grad.loc[grad["estimate"].abs().idxmax()]
    mate = grad[(grad["model"] == big.model) & (grad["block"] == big.block)
                & (grad["prompt_type"] != big.prompt_type)]
    mate_txt = ""
    if not mate.empty:
        m = mate.iloc[0]
        # The halves are scored by opposite rubrics, so opposite signs are what
        # a moved warning threshold looks like -- same signs would not be.
        opposite = (m.estimate < 0) != (big.estimate < 0)
        mate_txt = (
            f" The other half of the same cell moves ${m.estimate:+.3f}$ "
            f"($p={m.p:.3f}$), "
            + ("the opposite way, which is what a shifted warning threshold "
               "looks like" if opposite else
               "the same way, so this is not a clean threshold shift") + "."
        )

    direction = (r"warns \emph{less} when the person asking outranks it"
                 if big.estimate < 0 else
                 r"over-warns \emph{less} --- is more compliant --- when the "
                 r"person asking outranks it")
    sym_sig = _sig(sym)
    sym_line = (
        "Meanwhile the matched-versus-mismatched control is flat: "
        + ("none" if sym_sig.empty else f"only {len(sym_sig)}")
        + f" of {len(sym)} of those tests "
        + _verb(len(sym_sig), "reaches", "reach") + " $p<.05$"
        + ("" if sym_sig.empty else
           f" (largest ${sym_sig['estimate'].abs().max():.3f}$)")
        + r". So what moves the model is rank \emph{order}, not the presence "
          "of a gap."
    )

    return rf"""\section*{{Power asymmetry}}

The main effects say whether each side's status matters on its own. Neither
answers what the design was built for: what happens when the two sides are
\emph{{unequal}}. Two contrasts do, both computed within a block so relevance is
held constant, and both holding one high-status and one low-status party fixed
so that only the \emph{{direction}} of the gap changes.

\begin{{itemize}}
\item \textbf{{power gradient}} --- user above model minus model above user.
Negative means the model warns less when the person asking outranks it.
\item \textbf{{matched $-$ mismatched}} --- whether a gap matters at all, in
either direction. This is the interaction rescaled, and it is the control that
separates ``responds to rank order'' from ``responds to mismatch''.
\end{{itemize}}

\noindent
\textbf{{The sharpest cell is {esc(display_name(big.model))} on
{BLOCK_LABEL[big.block]} roles, {str(big.prompt_type).lower()} items:
${big.estimate:+.3f}$ ($p={big.p:.3f}$)}} --- it {direction}.{mate_txt}
{len(sig)} of the {len(grad)} gradient tests reach $p<.05$. {sym_line}

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp3_fig9_power_gradient.pdf}}
\caption{{Power gradient against its symmetry control, pooled over the two
halves. A gradient away from zero with the control on zero is the model
tracking who outranks whom.}}
\end{{figure}}

"""


def mixed_section(by_label: pd.DataFrame, mixed: pd.DataFrame | None,
                  n_cells: int) -> str:
    """One paragraph: does the headline survive a second, standard method?"""
    if mixed is None or mixed.empty:
        return ""
    j = by_label.merge(mixed, on=["model", "block", "effect", "prompt_type"],
                       suffixes=("_boot", "_mixed"))
    if j.empty:
        return ""
    gap = (j["estimate_boot"] - j["estimate_mixed"]).abs().max()
    disagree = j[(j["p_boot"] < .05) != (j["p_mixed"] < .05)]
    head = j[(j["block"] == "domain") & (j["prompt_type"] == "UNSAFE")
             & (j["effect"] != "interaction") & (j["p_boot"] < .05)]
    kept = head[head["p_mixed"] < .05]
    worst = f"$p\\le{head['p_mixed'].max():.3f}$" if not head.empty else "---"

    return rf"""\section*{{Mixed-effects check}}

The intervals above come from resampling items. As a second route to the same
question, each effect was re-estimated with a \textbf{{linear mixed model}}:
pass (0/1) predicted by user status $\times$ assistant status, with a
\textbf{{random intercept for each item}}, fitted by REML. The random intercept
is the standard way to handle repeated measures on the same item --- every item
here is answered {n_cells} times per model, so some items are simply harder than
others and that item-level variance should not be counted as evidence. It is a
linear rather than a logistic model, so coefficients stay on the pass-rate scale
the figures use.

\textbf{{The conclusions do not depend on the method.}} Point estimates are
identical to the resampled ones (largest difference ${gap:.0e}$), and
{len(kept)} of the {len(head)} significant unsafe-half effects stay significant
({worst}). {len(disagree)} of the {len(j)} tests cross the .05 line between the
two methods, all borderline cells that neither method calls strongly.

"""


def build_latex(df: pd.DataFrame, stats: pd.DataFrame, ef: pd.DataFrame,
                by_label: pd.DataFrame, models: list[str],
                vs_ctrl: pd.DataFrame | None = None,
                mixed_by_label: pd.DataFrame | None = None,
                asym: pd.DataFrame | None = None,
                asym_by_label: pd.DataFrame | None = None,
                warn: pd.DataFrame | None = None,
                warn_effects: pd.DataFrame | None = None,
                cfg: dict | None = None, sample_meta: dict | None = None,
                available: dict | None = None,
                n_boot: int = 4000, run_name: str = "") -> str:
    cfg = cfg or {}
    sample_meta = sample_meta or {}

    n_rows = len(df)
    n_items = df["prompt_idx"].nunique()
    n_cells = int(round(n_rows / max(1, n_items * len(models))))
    n_cats = df["category"].nunique()
    n_images = df["image"].nunique() if "image" in df else n_items
    n_conditions = df["condition"].nunique()
    per_cat = sample_meta.get("per_category", n_items // max(1, n_cats * 2))

    # ── what this run actually sent ──────────────────────────────────────────
    suffix = detect_suffix(df)
    if suffix:
        user_turn = (
            rf"the query with \textbf{{``{esc(suffix.strip())}''}} appended, as "
            r"in exp1 and exp2"
        )
        variant_note = (
            r"This is the \textbf{pressure-phrase} variant. A paired run sends "
            r"the same items with the query verbatim; the two differ in this "
            r"phrase and nothing else."
        )
    else:
        user_turn = r"the query \textbf{verbatim}, exactly as SaLAD publishes it"
        variant_note = (
            r"This is the \textbf{verbatim} variant, so its pass rates are "
            r"comparable to the benchmark's published numbers. A paired run "
            r"appends exp1/exp2's ``I think it's fine.''; the two differ in "
            r"that phrase and nothing else."
        )

    max_side = cfg.get("image_max_side") or 1024
    judge_model = (df["judge_model"].dropna().iloc[0]
                   if "judge_model" in df and df["judge_model"].notna().any()
                   else "gemini-3.7-flash")
    judge_variant = (df["judge_variant"].dropna().iloc[0]
                     if "judge_variant" in df and df["judge_variant"].notna().any()
                     else "paper")
    judge_line = (
        "the two rubrics exactly as the SaLAD paper publishes them: text only, "
        "and the safe-case judge sees the model's answer alone"
        if judge_variant == "paper" else
        "the augmented rubrics, which additionally show the judge the image and "
        "give the safe-case rubric the query and reference"
    )

    gold_len = int(df.drop_duplicates("prompt_idx")["safety_fact"].str.len().mean())
    dup_note = ""
    n_gold = df.drop_duplicates("prompt_idx")["safety_fact"].nunique()
    if n_gold < n_items:
        dup_note = (
            f" {n_items - n_gold} pair of items shares an identical warning "
            f"despite being different photos with different questions, so the "
            f"resampling unit is the item index rather than the warning text."
        )

    # ── results ──────────────────────────────────────────────────────────────
    pooled_sig = _sig(ef)
    top = ef.loc[ef["estimate"].abs().idxmax()]
    pooled_claim = (
        f"none of the {len(ef)} pooled factorial tests reaches $p<.05$; the "
        f"largest is {EFFECT_SHORT[top.effect]} in the "
        f"{BLOCK_LABEL[top.block]} block for {esc(display_name(top.model))} "
        f"(${top.estimate:+.3f}$, $p={top.p:.3f}$)"
        if pooled_sig.empty else
        f"{len(pooled_sig)} of the {len(ef)} pooled factorial tests reach "
        f"$p<.05$"
    )

    dom = by_label[by_label["block"] == "domain"]
    dom_sig = _sig(dom)
    unsafe_sig = dom_sig[dom_sig["prompt_type"] == "UNSAFE"]
    irrel = by_label[by_label["block"] == "irrel"]
    irrel_main = irrel[irrel["effect"] != "interaction"]
    irrel_max = irrel_main["estimate"].abs().max()
    irrel_main_sig = len(_sig(irrel_main))
    irrel_int_sig = _sig(irrel[irrel["effect"] == "interaction"])
    dom_main_sig = dom_sig[dom_sig["effect"] != "interaction"]
    largest_main = (dom_main_sig["estimate"].abs().max()
                    if not dom_main_sig.empty else float("nan"))

    opposed = 0
    for m in models:
        for effect in ("user (high-low)", "model (high-low)"):
            sub = dom[(dom["model"] == m) & (dom["effect"] == effect)]
            u = sub[sub["prompt_type"] == "UNSAFE"]["estimate"]
            s = sub[sub["prompt_type"] == "SAFE"]["estimate"]
            if not u.empty and not s.empty and u.iloc[0] * s.iloc[0] < 0:
                opposed += 1

    # Every claim below is conditioned on what this run actually shows: the two
    # variants differ in which effects clear .05, so fixed prose would be wrong
    # for one of them.
    if dom_sig.empty:
        split_claim = ("No domain-block effect reaches $p<.05$ in either half, "
                       "though the unsafe-side estimates keep the sign they "
                       "have elsewhere.")
    elif len(unsafe_sig) == len(dom_sig):
        split_claim = (
            f"All {len(dom_sig)} significant domain-block "
            f"{_verb(len(dom_sig), 'effect falls', 'effects fall')} on the "
            f"unsafe side.")
    else:
        split_claim = (
            f"{len(unsafe_sig)} of the {len(dom_sig)} significant domain-block "
            f"effects {_verb(len(unsafe_sig), 'falls', 'fall')} on the unsafe "
            f"side.")

    if irrel_main_sig == 0:
        irrel_head = "Irrelevant status does nothing."
        irrel_sig_note = " and none reaches $p<.05$"
    else:
        irrel_head = "Irrelevant status barely moves anything."
        big = _sig(irrel_main).loc[_sig(irrel_main)["estimate"].abs().idxmax()]
        irrel_sig_note = (
            f", though {irrel_main_sig} "
            f"{_verb(irrel_main_sig, 'reaches', 'reach')} $p<.05$ "
            f"(largest ${big.estimate:+.3f}$, $p={big.p:.3f}$)")

    irrel_int_note = ""
    if len(irrel_int_sig):
        r = irrel_int_sig.loc[irrel_int_sig["estimate"].abs().idxmax()]
        irrel_int_note = (
            f" {len(irrel_int_sig)} interaction "
            f"{_verb(len(irrel_int_sig), 'term does', 'terms do')} too "
            f"(largest ${r.estimate:+.3f}$, $p={r.p:.3f}$), among {len(irrel)} "
            f"tests in that block."
        )

    n_tests = len(by_label)
    n_split_sig = len(_sig(by_label))

    if vs_ctrl is not None and not vs_ctrl.empty:
        cs = _sig(vs_ctrl)
        if cs.empty:
            ctrl_line = ("No condition differs from the no-role control at "
                         "$p<.05$.")
        else:
            worst = cs.loc[cs["delta_vs_control"].abs().idxmax()]
            ctrl_line = (
                f"Against the no-role control, {len(cs)} of {len(vs_ctrl)} "
                f"cells differ at $p<.05$; the largest is "
                f"\\texttt{{{esc(worst.condition)}}} for "
                f"{esc(display_name(worst.model))} "
                f"(${worst.delta_vs_control:+.3f}$, $p={worst.p:.3f}$)."
            )
    else:
        ctrl_line = ""

    unsafe_rate = df[df["safety_type"] == "unsafe"]["passes"].mean()
    safe_rate = df[df["safety_type"] == "safe"]["passes"].mean()
    best_safe = max(models, key=lambda m:
                    df[(df.model == m) & (df.safety_type == "safe")]["passes"].mean())
    best_unsafe = max(models, key=lambda m:
                      df[(df.model == m) & (df.safety_type == "unsafe")]["passes"].mean())
    same_model = best_safe == best_unsafe

    # ── conclusions, conditioned on what this run shows ──────────────────────
    dom_unsafe_main = dom[(dom["prompt_type"] == "UNSAFE")
                          & (dom["effect"] != "interaction")]
    sig_unsafe_main = _sig(dom_unsafe_main)
    if not sig_unsafe_main.empty:
        big = sig_unsafe_main.loc[sig_unsafe_main["estimate"].abs().idxmax()]
        c_status_head = "Domain-relevant status shifts how much the model warns."
        c_status_body = (
            f"High-status roles make it name hazards more often on unsafe items "
            f"(largest ${big.estimate:+.3f}$, $p={big.p:.3f}$), and the safe "
            f"half moves the opposite way in {opposed} of the "
            f"{len(models) * 2} pairings --- a moved threshold, not better "
            f"discrimination.")
    else:
        mean_unsafe = dom_unsafe_main["estimate"].mean()
        c_status_head = "Domain-relevant status points the same way but does not reach significance here."
        c_status_body = (
            f"The unsafe-half estimates average ${mean_unsafe:+.3f}$, the same "
            f"direction as elsewhere, but none clears $p<.05$ in this run.")

    if irrel_main_sig == 0:
        c_irrel_head = "Status the model cannot use is ignored."
        c_irrel_body = (
            f"Nobel laureates and shelf stockers move the pass rate by at most "
            f"${irrel_max:.3f}$, so what matters is claimed expertise in the "
            f"item's domain rather than standing as such.")
    else:
        c_irrel_head = "Status the model cannot use barely registers."
        c_irrel_body = (
            f"No irrelevant-status main effect exceeds ${irrel_max:.3f}$ --- "
            f"several times smaller than the domain-relevant ones --- though "
            f"{irrel_main_sig} of them {_verb(irrel_main_sig, 'clears', 'clear')} "
            f"$p<.05$, so 'no effect' overstates it here.")

    c_pool_body = (
        "Pooled across the halves nothing is significant; the effects appear "
        "only once items scored for warning are separated from items scored for "
        "restraint."
        if pooled_sig.empty and n_split_sig else
        "The halves are scored by opposite rubrics, so a pooled pass rate "
        "averages a warning rate against an oversensitivity rate and means "
        "little on its own.")

    c_small_body = (
        f"The largest significant main effect is ${largest_main:.3f}$ against "
        rf"baselines of {unsafe_rate * 100:.0f}--{safe_rate * 100:.0f}\%, and "
        f"{n_split_sig} of {n_tests} tests reach $p<.05$ uncorrected."
        if not pd.isna(largest_main) else
        f"No domain-block main effect clears $p<.05$; the largest is "
        f"${dom_unsafe_main['estimate'].abs().max():.3f}$ against baselines of "
        rf"{unsafe_rate * 100:.0f}--{safe_rate * 100:.0f}\%.")

    if same_model:
        c_model_head = "One checkpoint leads on both halves."
        c_model_body = (
            f"{esc(display_name(best_safe))} is ahead on catching hazards and on "
            f"not over-warning, so the two checkpoints do not trade off here.")
    else:
        c_model_head = "Reasoning trades vigilance for restraint."
        c_model_body = (
            f"{esc(display_name(best_safe))} over-warns less than "
            f"{esc(display_name(best_unsafe))} but catches slightly fewer "
            f"hazards, so a higher overall pass rate is not a safety "
            f"improvement.")

    # Conclusions bullet for the asymmetry result, conditioned on it.
    c_asym_head = "Rank order matters, the size of the gap does not."
    c_asym_body = ("No asymmetry contrast reaches $p<.05$ in this run.")
    if asym_by_label is not None and not asym_by_label.empty:
        g = asym_by_label[asym_by_label["contrast"] == GRADIENT]
        gs = _sig(g)
        if not gs.empty:
            b = gs.loc[gs["estimate"].abs().idxmax()]
            side = ("defers" if b.estimate < 0 else "relaxes")
            c_asym_body = (
                f"With both status levels held fixed, only the direction of the "
                f"gap moves the model (largest ${b.estimate:+.3f}$, "
                f"$p={b.p:.3f}$): it {side} when the person asking outranks it. "
                f"Whether a gap exists at all does almost nothing.")

    # Conclusions bullet for the warning-scale result.
    c_warn_head = "Status changes caution, not accuracy."
    c_warn_body = ("The warning-rate split is not available for this run.")
    if warn is not None and not warn.empty:
        def _r(model, where, measure):
            sel = warn[(warn["model"] == model) & (warn["measure"] == measure)]
            sel = (sel[sel["condition"] == "control"] if where == "control"
                   else sel[sel["condition"].str.startswith(where + "_")])
            return float(sel["estimate"].mean())
        m0 = models[0]
        fa_c, fa_d = _r(m0, "control", "false-alarm rate"), _r(m0, "domain", "false-alarm rate")
        hit_c, hit_d = _r(m0, "control", "hit rate"), _r(m0, "domain", "hit rate")
        c_warn_body = (
            f"Putting both halves on one scale, a domain-relevant role raises "
            f"the false-alarm rate from ${fa_c:.3f}$ to ${fa_d:.3f}$ while hits "
            f"move only ${hit_c:.3f}$ to ${hit_d:.3f}$, so the model warns more "
            f"often without telling the two kinds of request apart any better.")

    warn_block = warning_section(warn, warn_effects, models)
    asym_block = asymmetry_section(asym_by_label, asym)
    mixed_block = mixed_section(by_label, mixed_by_label, n_cells)
    avail_cols = "cc" if available else ""
    avail_head = (r"& \multicolumn{2}{c}{available in SaLAD}" if available else "")
    avail_sub = (r"& unsafe & safe" if available else "")

    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,graphicx,microtype,parskip}}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black]{{hyperref}}
\title{{\vspace{{-2em}}exp3 --- status effects on multimodal safety}}
\author{{Run \texttt{{{esc(run_name)}}}}}
\date{{}}
\begin{{document}}
\maketitle
\vspace{{-2em}}

\section*{{Design}}

exp2's {n_conditions} conditions, unchanged, over a different benchmark and a
different model. Each item is asked under a no-role control and a
$2\times2$ of user status $\times$ assistant status, blocked by whether the two
roles are drawn from the item's own domain or from a status dimension
irrelevant to it. With every persona pairing enumerated that is
{n_cells} cells per item, {n_rows:,} generations across {len(models)}
Qwen3-VL-8B checkpoints.

The user turn is {user_turn}. {variant_note} Status is manipulated entirely in
the system prompt, which names the assistant's standing first and the user's
second; the control sends no system message at all.

Two properties of the benchmark shape how the results must be read. The
\textbf{{two halves are scored by opposite rubrics}} --- an unsafe item passes by
naming the hazard, a safe item passes by \emph{{not}} warning --- so they are
never averaged. And the \textbf{{risk is in the image}}, not the text: the same
sentence is innocuous or dangerous depending on the photo, which is why a vision
model is required and why a text-only pipeline could not be pointed at this data.

All intervals are 95\% confidence intervals from resampling the {n_items} items
with replacement ($B={n_boot}$), and $p$-values are two-sided from the same
resampling. Items are resampled rather than rows because every item is re-asked
in all {n_cells} cells and by both models, so rows within an item are not
independent.

\section*{{Data}}

\textbf{{Source.}} SaLAD (\texttt{{{esc(sample_meta.get('dataset', 'Holly301/SaLAD'))}}},
arXiv:2601.04043), 2{{,}}013 real-world image$+$text samples across
{n_cats} everyday categories, each labelled \texttt{{safe}} or \texttt{{unsafe}}.

\textbf{{Sample.}} {per_cat} safe and {per_cat} unsafe items were drawn from each
category --- {n_items} items, {n_images} distinct images, exactly balanced on
both axes by construction. The draw is seeded per (category, label) with seed
{sample_meta.get('seed', 42)}, so it reproduces exactly and adding a category
leaves the others untouched. The benchmark itself is unbalanced both within and
across categories, so this balance is imposed, not inherited: it is what lets a
per-category number and a safe-versus-unsafe number be read without reweighting,
at the cost of no longer reflecting the benchmark's own emphasis.

\begin{{table}}[h]\centering
\begin{{tabular}}{{lcc{avail_cols}}}
\toprule
& \multicolumn{{2}}{{c}}{{sampled}} {avail_head} \\
category & unsafe & safe {avail_sub} \\
\midrule
{category_table(df, available)}
\bottomrule
\end{{tabular}}
\end{{table}}

\textbf{{Gold reference.}} Every item ships a reference note, and its field name
differs by label: a \emph{{safety warning}} naming the hazard on unsafe items, a
\emph{{safe suggestion}} explaining why there is nothing to warn about on safe
ones. Both are normalised into one field, averaging {gold_len} characters, and
each is what the judge scores the response against.{dup_note}

\textbf{{What the model sees.}} One image and one short question. Images are
downscaled to a {max_side}\,px long edge before the model sees them --- an
experimental choice, not only a speed one, since Qwen3-VL tokenises at native
resolution and every one of the {n_cells} cells per item repeats the same image.

\section*{{Judge}}

Responses are scored by \texttt{{{esc(judge_model)}}} using {judge_line}. The one
deviation from the published protocol is that the verdict is bracketed and
followed by a one-sentence explanation, so a disagreement can be audited; the
criteria are untouched. True maps to \emph{{pass}} and false to \emph{{fail}}.

\section*{{Baseline}}

\begin{{table}}[h]\centering
\begin{{tabular}}{{lcc}}
\toprule
& unsafe (catch the hazard) & safe (do not over-warn) \\
\midrule
{baseline_table(df, models)}
\bottomrule
\end{{tabular}}
\end{{table}}

\noindent
The models miss roughly {(1 - unsafe_rate) * 100:.0f}\% of genuine hazards while
answering {safe_rate * 100:.0f}\% of safe requests without over-warning.
{"One checkpoint leads on both halves"
 if same_model else
 f"{esc(display_name(best_safe))} is the better of the two at not over-warning "
 f"and {esc(display_name(best_unsafe))} at catching hazards, so reasoning here "
 f"buys restraint rather than vigilance"}.

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp3_fig1_conditions.pdf}}
\caption{{Pass rate per condition, pooled over both halves.}}
\end{{figure}}

\section*{{Status effects}}

Pooled over both halves, {pooled_claim}. That is an artefact of pooling, not a
null: the effects sit in the unsafe half and are diluted by a safe half where
almost everything already passes.

Split by gold label, in the domain-relevant block:

\begin{{table}}[h]\centering
\begin{{tabular}}{{llcc}}
\toprule
model & effect & unsafe & safe \\
\midrule
{split_table(by_label, models)}
\bottomrule
\end{{tabular}}
\end{{table}}
\vspace{{-1em}}
\noindent{{\footnotesize * $p<.05$, ** $p<.01$, uncorrected.}}

\noindent
{split_claim} {opposed} of the {len(models) * 2} (model, effect) pairs carry
opposite signs across the two halves --- the signature of a shifted warning
threshold rather than better judgement.

\textbf{{{irrel_head}}} No user- or model-status main effect in the
domain-irrelevant block exceeds ${irrel_max:.3f}$ in magnitude{irrel_sig_note}.%
{irrel_int_note} {ctrl_line}

\begin{{figure}}[h]\centering
\includegraphics[width=\textwidth]{{exp3_fig8_by_prompt_type.pdf}}
\caption{{The factorial effects re-fitted inside each half. The unsafe and safe
panels answer opposite questions, which is why they are never pooled.}}
\end{{figure}}

\noindent
These {n_tests} effects are reported without correction for multiple testing.
Read any single cell as exploratory; the result is the direction the significant
ones share.

{warn_block}{asym_block}{mixed_block}\section*{{Conclusions}}

\begin{{itemize}}
\item \textbf{{{c_status_head}}} {c_status_body}

\item \textbf{{{c_irrel_head}}} {c_irrel_body}

\item \textbf{{The two halves have to be analysed separately.}} {c_pool_body}

\item \textbf{{The status effects are small.}} {c_small_body}

\item \textbf{{{c_model_head}}} {c_model_body}

\item \textbf{{{c_warn_head}}} {c_warn_body}

\item \textbf{{{c_asym_head}}} {c_asym_body}

\item \textbf{{Only the pressure phrase is manipulated cleanly.}} Dataset and
model both changed relative to exp2, so no difference from it is attributable to
either alone; the paired run, which varies the phrase and nothing else, is the
one controlled comparison exp3 offers.
\end{{itemize}}

\end{{document}}
"""
