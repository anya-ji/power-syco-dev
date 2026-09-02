#!/usr/bin/env python3
"""exp2-solo analysis: cluster-bootstrap level and presence contrasts.

Same estimator and the same resampling unit as ``analyze.py`` -- the safety
fact -- against the uncrossed grid, where each cell dresses one side only. The
mixed-effects fits run as a cross-check (``--no-mixed`` to skip).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _common  # noqa: E402,F401

from sycophancy import analysis_solo as B  # noqa: E402
from sycophancy import mixed_models as mm  # noqa: E402
from sycophancy import plots_solo  # noqa: E402
from sycophancy.analysis import load_judged, write_tables  # noqa: E402
from sycophancy.artifacts import RunPaths, latest_run  # noqa: E402
from sycophancy.config import DEFAULT_MODELS, results_dir  # noqa: E402
from sycophancy.model_statuses import EXP2_SOLO_CONDITIONS  # noqa: E402

DEFAULT_RUN = "exp2solo_245x101"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None,
                    help=f"run directory (default: {DEFAULT_RUN} under "
                         f"exp2/results, else the most recent)")
    ap.add_argument("--primary-model", default="qwen3-8b-think")
    ap.add_argument("--n-boot", type=int, default=B.N_BOOT)
    ap.add_argument("--no-mixed", action="store_true",
                    help="skip the mixed-effects cross-check")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if args.run is None:
        default = results_dir("exp2") / DEFAULT_RUN
        paths = RunPaths(default) if default.exists() else latest_run(results_dir("exp2"))
    else:
        paths = RunPaths(args.run)
    print(f"Run directory: {paths.root}")
    df = load_judged(paths.judged)
    print(f"Loaded {len(df):,} judged rows — {df['safety_fact'].nunique()} facts, "
          f"{df['prompt'].nunique()} prompts, {df['model'].nunique()} models")

    unknown = sorted(set(df["condition"]) - set(EXP2_SOLO_CONDITIONS))
    if unknown:
        raise SystemExit(
            f"{paths.root.name} carries conditions this analysis does not "
            f"know: {unknown}. It is a crossed exp2 run — use analyze.py."
        )

    tables = write_tables(df, paths.tables, EXP2_SOLO_CONDITIONS)
    stats = tables["stats"]

    models = [m for m in DEFAULT_MODELS if m in set(df["model"])]
    models += [m for m in df["model"].unique() if m not in models]

    nb = args.n_boot
    print(f"\nCluster bootstrap over {df['safety_fact'].nunique()} safety facts "
          f"(B={nb})…")
    ef = B.solo_effects(df, models, n_boot=nb)
    ef.to_csv(paths.tables / "solo_effects.csv", index=False)
    vc = B.vs_control(df, models, n_boot=nb)
    vc.to_csv(paths.tables / "vs_control.csv", index=False)
    cond_means = B.condition_means(df, models, n_boot=nb)
    cond_means.to_csv(paths.tables / "condition_means.csv", index=False)
    arms = B.arm_means(df, models, n_boot=nb)
    arms.to_csv(paths.tables / "arm_means.csv", index=False)
    dim = B.dimension_effects(df, models, n_boot=nb)
    dim.to_csv(paths.tables / "dimension_effects.csv", index=False)
    cats = B.category_effects(df, args.primary_model, n_boot=nb)
    cats.to_csv(paths.tables / "category_effects.csv", index=False)
    by_ptype = B.solo_effects(df, models, extra_group="prompt_type", n_boot=nb)
    by_ptype.to_csv(paths.tables / "effects_by_prompt_type.csv", index=False)
    ap_ci = B.allpass_ci(df, models, n_boot=nb)
    ap_ci.to_csv(paths.tables / "allpass_ci.csv", index=False)
    asym = B.side_asymmetry(df, models, n_boot=nb)
    asym.to_csv(paths.tables / "side_asymmetry.csv", index=False)

    print(f"\n{'model':<18}{'block':<8}{'contrast':<26}{'est':>8}{'95% CI':>19}"
          f"{'p':>9}")
    for _, r in ef.iterrows():
        print(f"{r.model:<18}{r.block:<8}{r.effect:<26}{r.estimate:>+8.3f}"
              f"  [{r.ci_lo:>+.3f},{r.ci_hi:>+.3f}]{r.p:>9.4f}  {B.stars(r.p)}")
    print(f"\n  p is two-sided from the bootstrap distribution; its floor is "
          f"2/(B+1) = {2 / (nb + 1):.4f}")

    print(f"\n{'model':<18}{'block':<8}{'contrast':<32}{'est':>8}{'95% CI':>19}"
          f"{'p':>9}")
    for _, r in asym.iterrows():
        print(f"{r.model:<18}{r.block:<8}{r.contrast:<32}{r.estimate:>+8.3f}"
              f"  [{r.ci_lo:>+.3f},{r.ci_hi:>+.3f}]{r.p:>9.4f}  {B.stars(r.p)}")

    print("\nEach condition vs the no-role control (unadjusted):")
    for m in models:
        sig = vc[(vc.model == m) & (vc.p < 0.05)]
        if sig.empty:
            print(f"  {m:<18} none differ from control")
        for _, r in sig.iterrows():
            print(f"  {m:<18} {r.condition:<16} {r.delta_vs_control:+.3f} "
                  f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  p={r.p:.4f}")

    if not args.no_mixed:
        print("\nMixed-effects cross-check "
              "(pass ~ condition + (1|fact) + (1|prompt), REML)…")
        mef = mm.solo_effects(df, models)
        mef.to_csv(paths.tables / "solo_effects_mixed.csv", index=False)
        merged = ef.merge(mef, on=["model", "block", "effect"],
                          suffixes=("_boot", "_mixed"))
        disagree = merged[(merged.p_boot < 0.05) != (merged.p_mixed < 0.05)]
        print(f"  contrasts where the two methods disagree at .05: "
              f"{len(disagree)}/{len(merged)}")
        for _, r in disagree.iterrows():
            print(f"    {r.model:<18}{r.block:<8}{r.effect:<26}"
                  f"boot p={r.p_boot:.4f}  mixed p={r.p_mixed:.4f}")
        print("  max |estimate| gap: "
              f"{(merged.estimate_boot - merged.estimate_mixed).abs().max():.2e}")

    if not args.no_figures:
        print(f"\nFigures -> {paths.figures}")
        plots_solo.make_all(df, stats, ef, arms, dim, paths.figures, models,
                            args.primary_model, vs_ctrl=vc,
                            cond_means=cond_means, cats=cats, by_ptype=by_ptype,
                            allpass_ci=ap_ci, asym=asym)
    print(f"\nTables -> {paths.tables}")


if __name__ == "__main__":
    main()
