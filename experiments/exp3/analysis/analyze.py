#!/usr/bin/env python3
"""exp3 analysis: exp2's estimators over SaLAD's two halves.

The design is exp2's, so the inference is too -- 95% cluster bootstrap with the
**stimulus item** as the resampling unit, because one item recurs across all 101
persona cells and both models, and naive binomial bars would be far too narrow.

Two things differ from exp2 and both matter for reading the output:

* **The two halves are scored by opposite rubrics.** An unsafe item passes by
  catching the hazard; a safe item passes by *not* over-warning. Pooling them
  averages a warning rate against an oversensitivity rate, which is why the
  safe/unsafe split (``effects_by_prompt_type.csv``, fig8) is the headline here
  rather than a robustness check. A status effect that raises warnings should
  push the two halves in *opposite* directions.
* **There is no pressure phrase.** exp1/exp2 append "I think it's fine."; exp3
  sends the SaLAD query verbatim, so these are status effects on the unmodified
  benchmark, not status x explicit pressure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _common  # noqa: E402,F401

from sycophancy import analysis_exp2 as B  # noqa: E402
from sycophancy import mixed_models as mm  # noqa: E402
from sycophancy import plots_exp2  # noqa: E402
from sycophancy.analysis import load_judged, write_tables  # noqa: E402
from sycophancy.artifacts import RunPaths, latest_run  # noqa: E402
from sycophancy.config import VL_MODELS, results_dir  # noqa: E402
from sycophancy.model_statuses import EXP2_CONDITIONS  # noqa: E402

PRIMARY = "qwen3vl-8b-think"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--primary-model", default=PRIMARY)
    ap.add_argument("--n-boot", type=int, default=B.N_BOOT)
    ap.add_argument("--no-mixed", action="store_true",
                    help="skip the mixed-effects cross-check")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    paths = RunPaths(args.run) if args.run else latest_run(results_dir("exp3"))
    print(f"Run directory: {paths.root}")
    df = load_judged(paths.judged)

    # The bootstrap clusters on ``safety_fact``. In exp1/exp2 that is a real
    # safety fact shared by several prompts; in exp3 it holds the item's gold
    # reference note, and two distinct items can carry the same one -- SaLAD
    # ids 765 and 768 are different photos with different questions but an
    # identical warning. Left alone they would merge into one cluster, so the
    # note is prefixed with the item index to restore one cluster per stimulus.
    n_items = df["prompt_idx"].nunique()
    if df["safety_fact"].nunique() < n_items:
        merged = n_items - df["safety_fact"].nunique()
        print(f"NOTE: {merged} item(s) share a gold note with another; "
              f"prefixing with the item index so each stimulus is its own "
              f"bootstrap cluster")
        df["safety_fact"] = ("item" + df["prompt_idx"].astype(str).str.zfill(3)
                             + " · " + df["safety_fact"])

    print(f"Loaded {len(df):,} judged rows — {n_items} items, "
          f"{df['model'].nunique()} models")
    if "safety_type" in df:
        by_label = df.groupby("safety_type")["passes"].agg(["mean", "size"])
        print("\nPass rate by gold label (pooled over every condition):")
        for label, row in by_label.iterrows():
            print(f"  {label:<7} {row['mean']:.3f}  ({int(row['size']):,} rows)")
        print("  NB: opposite rubrics — unsafe = hazard caught, "
              "safe = not over-warned. Do not average them.")

    tables = write_tables(df, paths.tables, EXP2_CONDITIONS)
    stats = tables["stats"]

    models = [m for m in VL_MODELS if m in set(df["model"])]
    models += [m for m in df["model"].unique() if m not in models]
    primary = args.primary_model if args.primary_model in models else models[0]

    nb = args.n_boot
    print(f"\nCluster bootstrap over {df['safety_fact'].nunique()} SaLAD items "
          f"(B={nb})…")
    ef = B.factorial_effects(df, models, n_boot=nb)
    ef.to_csv(paths.tables / "factorial_effects.csv", index=False)
    vc = B.vs_control(df, models, n_boot=nb)
    vc.to_csv(paths.tables / "vs_control.csv", index=False)
    cond_means = B.condition_means(df, models, n_boot=nb)
    cond_means.to_csv(paths.tables / "condition_means.csv", index=False)
    cells = B.cell_means(df, models, n_boot=nb)
    cells.to_csv(paths.tables / "cell_means.csv", index=False)
    dim = B.dimension_effects(df, models, n_boot=nb)
    dim.to_csv(paths.tables / "dimension_effects.csv", index=False)
    cats = B.category_effects(df, primary, n_boot=nb)
    cats.to_csv(paths.tables / "category_effects.csv", index=False)
    # prompt_type carries the gold label in exp3, so this is the safe/unsafe split
    by_label_ef = B.factorial_effects(df, models, extra_group="prompt_type",
                                      n_boot=nb)
    by_label_ef.to_csv(paths.tables / "effects_by_label.csv", index=False)
    by_label_ef.to_csv(paths.tables / "effects_by_prompt_type.csv", index=False)
    ap_ci = B.allpass_ci(df, models, n_boot=nb)
    ap_ci.to_csv(paths.tables / "allpass_ci.csv", index=False)
    # Power asymmetry: which side of a status gap the model responds to, and
    # whether the gap matters at all regardless of direction.
    asym = B.asymmetry_contrasts(df, models, n_boot=nb)
    asym.to_csv(paths.tables / "asymmetry_contrasts.csv", index=False)
    asym_lab = B.asymmetry_contrasts(df, models, extra_group="prompt_type",
                                     n_boot=nb)
    asym_lab.to_csv(paths.tables / "asymmetry_contrasts_by_label.csv",
                    index=False)
    # Both halves on one behavioural scale: did the model warn?
    warn = B.warning_rates(df, models, n_boot=nb)
    warn.to_csv(paths.tables / "warning_rates.csv", index=False)
    warn_ef = B.warning_effects(df, models, n_boot=nb)
    warn_ef.to_csv(paths.tables / "warning_effects.csv", index=False)
    warn_vc = B.warning_vs_control(df, models, n_boot=nb)
    warn_vc.to_csv(paths.tables / "warning_vs_control.csv", index=False)

    print(f"\n{'model':<22}{'block':<8}{'effect':<19}{'est':>8}{'95% CI':>19}"
          f"{'p':>9}")
    for _, r in ef.iterrows():
        print(f"{r.model:<22}{r.block:<8}{r.effect:<19}{r.estimate:>+8.3f}"
              f"  [{r.ci_lo:>+.3f},{r.ci_hi:>+.3f}]{r.p:>9.4f}  {B.stars(r.p)}")
    print("\n  95% CIs and two-sided p from resampling the items with "
          "replacement")

    print("\nSame effects split by gold label — the halves should move oppositely "
          "if status shifts how much the model warns:")
    print(f"{'model':<22}{'label':<8}{'block':<8}{'effect':<19}{'est':>8}{'p':>9}")
    for _, r in by_label_ef.iterrows():
        print(f"{r.model:<22}{str(r.prompt_type):<8}{r.block:<8}{r.effect:<19}"
              f"{r.estimate:>+8.3f}{r.p:>9.4f}  {B.stars(r.p)}")

    print("\nPower asymmetry, holding both status levels fixed:"
          "\n  power gradient    = user above model - model above user "
          "(negative: warns less when the asker outranks it)"
          "\n  matched-mismatched = does a gap matter at all, either "
          "direction (the interaction, rescaled)")
    print(f"{'model':<22}{'label':<8}{'block':<8}{'contrast':<20}{'est':>8}"
          f"{'95% CI':>19}{'p':>9}")
    for _, r in asym_lab.iterrows():
        print(f"{r.model:<22}{str(r.prompt_type):<8}{r.block:<8}"
              f"{r.contrast:<20}{r.estimate:>+8.3f}  "
              f"[{r.ci_lo:>+.3f},{r.ci_hi:>+.3f}]{r.p:>9.4f}  {B.stars(r.p)}")

    print("\nWarning rate per condition — warning is a hit on unsafe items "
          "and a false alarm on safe ones:")
    print(f"{'model':<22}{'condition':<20}{'hit':>7}{'false alarm':>13}"
          f"{'hit-FA':>9}")
    for m in models:
        for cond in [c for c in EXP2_CONDITIONS if c in set(df["condition"])]:
            sel = warn[(warn.model == m) & (warn.condition == cond)]
            hit = sel[sel.measure == "hit rate"]["estimate"]
            fa = sel[sel.measure == "false-alarm rate"]["estimate"]
            if hit.empty or fa.empty:
                continue
            h, f = float(hit.iloc[0]), float(fa.iloc[0])
            print(f"{m:<22}{cond:<20}{h:>7.3f}{f:>13.3f}{h - f:>9.3f}")

    print("\nStatus effects on the WARNING rate (positive = warns more; both "
          "halves same sign = threshold shift):")
    print(f"{'model':<22}{'label':<8}{'block':<8}{'effect':<19}{'est':>8}{'p':>9}")
    for _, r in warn_ef[warn_ef.effect != "interaction"].iterrows():
        print(f"{r.model:<22}{str(r.prompt_type):<8}{r.block:<8}{r.effect:<19}"
              f"{r.estimate:>+8.3f}{r.p:>9.4f}  {B.stars(r.p)}")

    print("\nEach condition vs the no-role control (8 tests per model, "
          "uncorrected):")
    for m in models:
        sig = vc[(vc.model == m) & (vc.p < 0.05)]
        if sig.empty:
            print(f"  {m:<22} none differ from control")
        for _, r in sig.iterrows():
            print(f"  {m:<22} {r.condition:<20} {r.delta_vs_control:+.3f} "
                  f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  p={r.p:.4f}")


    if not args.no_mixed:
        print("\nMixed-effects check "
              "(pass ~ u*m + random intercept per item, REML)…")
        mef = mm.factorial_effects(df, models)
        mef.to_csv(paths.tables / "factorial_effects_mixed.csv", index=False)
        # The safe/unsafe split is what the report leads with, so the check has
        # to cover it too, not only the pooled fit.
        mef_lab = mm.factorial_effects(df, models, extra_group="prompt_type")
        mef_lab.to_csv(paths.tables / "effects_by_label_mixed.csv", index=False)
        lab = by_label_ef.merge(mef_lab,
                                on=["model", "block", "effect", "prompt_type"],
                                suffixes=("_boot", "_mixed"))
        n_dis = int(((lab.p_boot < 0.05) != (lab.p_mixed < 0.05)).sum())
        print(f"  split by label: {n_dis}/{len(lab)} disagree at .05; "
              f"max |estimate| gap "
              f"{(lab.estimate_boot - lab.estimate_mixed).abs().max():.1e}")
        merged = ef.merge(mef, on=["model", "block", "effect"],
                          suffixes=("_boot", "_mixed"))
        disagree = merged[(merged.p_boot < 0.05) != (merged.p_mixed < 0.05)]
        print(f"  effects where the two methods disagree at .05: "
              f"{len(disagree)}/{len(merged)}")
        for _, r in disagree.iterrows():
            print(f"    {r.model:<22}{r.block:<8}{r.effect:<19}"
                  f"boot p={r.p_boot:.4f}  mixed p={r.p_mixed:.4f}")
        print("  max |estimate| gap: "
              f"{(merged.estimate_boot - merged.estimate_mixed).abs().max():.2e}")

    if not args.no_figures:
        print(f"\nFigures -> {paths.figures}")
        plots_exp2.make_all(df, stats, ef, cells, dim, paths.figures, models,
                            primary, vs_ctrl=vc, cond_means=cond_means,
                            cats=cats, by_ptype=by_label_ef, allpass_ci=ap_ci,
                            asym=asym, warn=warn, warn_vs_ctrl=warn_vc, prefix="exp3")
    # ── every figure again, once per half, combined into one image ──────────
    # The two halves are scored by opposite rubrics, so a pooled figure averages
    # a warning rate against an oversensitivity rate. Re-running the same
    # estimators on each half is exact -- an item belongs to exactly one half,
    # so filtering the rows is all the split requires.
    #
    # The per-half renders are intermediates: they go to a scratch directory so
    # that figures/ and report/ only ever hold the combined image, and the
    # scratch tree is removed afterwards.
    if not args.no_figures and "safety_type" in df:
        import shutil

        scratch = paths.root / ".halves"
        try:
            for half in ("unsafe", "safe"):
                d = df[df["safety_type"] == half]
                if d.empty:
                    continue
                sub_tables = paths.tables / half
                sub_tables.mkdir(parents=True, exist_ok=True)
                h_tables = write_tables(d, sub_tables, EXP2_CONDITIONS)
                h_ef = B.factorial_effects(d, models, n_boot=nb)
                h_ef.to_csv(sub_tables / "factorial_effects.csv", index=False)
                h_cells = B.cell_means(d, models, n_boot=nb)
                h_cells.to_csv(sub_tables / "cell_means.csv", index=False)
                h_cond = B.condition_means(d, models, n_boot=nb)
                h_cond.to_csv(sub_tables / "condition_means.csv", index=False)
                h_vc = B.vs_control(d, models, n_boot=nb)
                h_vc.to_csv(sub_tables / "vs_control.csv", index=False)
                h_dim = B.dimension_effects(d, models, n_boot=nb)
                h_dim.to_csv(sub_tables / "dimension_effects.csv", index=False)
                h_cats = B.category_effects(d, primary, n_boot=nb)
                h_cats.to_csv(sub_tables / "category_effects.csv", index=False)
                h_ap = B.allpass_ci(d, models, n_boot=nb)
                h_ap.to_csv(sub_tables / "allpass_ci.csv", index=False)
                h_asym = B.asymmetry_contrasts(d, models, n_boot=nb)
                h_asym.to_csv(sub_tables / "asymmetry_contrasts.csv", index=False)

                scored = ("passing = the hazard was named" if half == "unsafe"
                          else "passing = no unnecessary warning")
                note = f"{half.upper()} items only — {scored}\n"
                print(f"\nRendering {half} items ({len(d):,} rows) to combine…")
                plots_exp2.make_all(d, h_tables["stats"], h_ef, h_cells, h_dim,
                                    scratch / "figures", models, primary,
                                    vs_ctrl=h_vc, cond_means=h_cond,
                                    cats=h_cats, allpass_ci=h_ap, asym=h_asym,
                                    prefix=f"exp3_{half}", note=note)
            plots_exp2.combine_halves(scratch / "figures", paths.figures,
                                      prefix="exp3")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nTables -> {paths.tables}")


if __name__ == "__main__":
    main()
