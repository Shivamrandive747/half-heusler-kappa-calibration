"""Figure -- the machine-learning half of the method, and what it delivers from structure alone.

Three panels, all read from artefacts (nothing typed):
  (a) STEP 1, THE MODEL AGAINST PUBLISHED CALCULATIONS. Each compound's model prediction against its
      published full-BTE value at 300 K, with the compound's whole X-site chemistry cluster removed from
      training (validate_bte_generator.py -> bte_generator_validation.csv). The headline numbers in the
      panel are the registry's (ml_generator_validation), which are per compound over all temperatures.
  (b) WHICH REGRESSOR. Four regressors on identical rows and folds (compare_models_production.py ->
      model_comparison_production.json): median error per compound, R^2 per row. The production model
      is CatBoost; the bar is labelled, not argued.
  (c) FROM STRUCTURE ALONE. TmPbAu and HfTeOs beside the published calculations of their own families'
      members (grey) and the model's held-out predictions for those members (open). The new compound's
      model value (open blue) is moved to the experimental scale by the shared transfer function
      (filled blue). The two steps' errors are stated in the caption side by side, never combined.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import figlib as F

EX = "data/exports/kappa_v2"


def ticks(axis, values):
    """Plain-number ticks on a log axis (figlib.log_ticks formats only one axis)."""
    import matplotlib.ticker as mticker
    axis.set_major_locator(mticker.FixedLocator(values))
    axis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in values]))
    axis.set_minor_locator(mticker.NullLocator())


def main() -> int:
    import matplotlib.pyplot as plt
    N = json.load(open(f"{EX}/paper_numbers.json"))
    mv = N["ml_generator_validation"]
    so = N["structure_only"]["main_text"]
    c, p = N["calibration"]["c"], N["calibration"]["p"]
    V = pd.read_csv(f"{EX}/bte_generator_validation.csv")
    V300 = V[V["T"] == 300].groupby("compound").agg(pub=("k_published", "median"), gen=("k_generated", "median"),
                                                     fam=("fam", "first"))
    M = json.load(open(f"{EX}/model_comparison_production.json"))

    F.use_style()
    fig = plt.figure(figsize=(F.DOUBLE, F.DOUBLE * 0.38))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 1.0], wspace=0.48)
    axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))

    # (a) parity
    lo, hi = 0.3, 120.0
    F.band_2x(axa, lo, hi, label=None)
    axa.scatter(V300.pub, V300.gen, s=11, c=F.OURS, lw=0.3, edgecolors="white", zorder=3)
    F.parity_axes(axa, lo, hi, "published calculation $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)",
                  "model, cluster held out (W m$^{-1}$ K$^{-1}$)")
    for ax_ in (axa.xaxis, axa.yaxis):
        ticks(ax_, (0.5, 1, 2, 5, 10, 20, 50, 100))
    stats = (f"{mv['median_ape']:.1f}% median error\n{mv['within_2x_pct']:.0f}% within 2$\\times$\n"
             f"{mv['n_compounds']} compounds, {mv['n_clusters']} clusters\n({len(V300)} shown, at 300 K)")
    axa.text(0.96, 0.04, stats, transform=axa.transAxes, va="bottom", ha="right", fontsize=6.2,
             bbox=dict(fc="white", ec="#CCCCCC", lw=0.4, pad=2.2, alpha=0.94), zorder=6)
    F.panel_label(axa, "a")

    # (b) regressors
    names = {"catboost": "CatBoost", "lightgbm": "LightGBM", "gpr": "Gaussian\nprocess", "krr": "kernel\nridge"}
    # the metrics ML papers report (RMSE, MAE on log10 kappa, R^2), ranked by RMSE; the median error
    # per compound stays in the text and the supplement
    keys = sorted(names, key=lambda k: M[k]["rmse_log"])
    y = np.arange(len(keys))
    hgt = 0.36
    for i, k in enumerate(keys):
        base = F.OURS if k == "catboost" else F.REFUSED
        axb.barh(i - hgt / 2, M[k]["rmse_log"], height=hgt, color=base, zorder=3,
                 label="RMSE" if i == 0 else None)
        axb.barh(i + hgt / 2, M[k]["mae_log"], height=hgt, color=base, alpha=0.55, zorder=3,
                 label="MAE" if i == 0 else None)
        axb.text(max(M[k]["rmse_log"], M[k]["mae_log"]) + 0.012, i, f"$R^2$ {M[k]['r2_log']:.2f}",
                 va="center", fontsize=6.2, color="#333333")
    axb.set_yticks(y)
    axb.set_yticklabels([names[k] for k in keys], fontsize=6.6)
    axb.invert_yaxis()
    axb.set_xlim(0, max(M[k]["rmse_log"] for k in keys) * 1.55)
    axb.set_xlabel("error in log$_{10}\\,\\kappa_L$ (dark RMSE, light MAE)")
    axb.set_title(f"identical rows ({M['catboost']['n_rows']}), identical folds", fontsize=7, pad=3)
    F.panel_label(axb, "b", loc="upper right")

    # (c) structure alone
    for i, (cpd, fam) in enumerate((("TmPbAu", "Pb-Au"), ("HfTeOs", "Te-Os"))):
        mem = V300[V300.fam == fam]
        xs = i + np.linspace(-0.20, -0.06, max(len(mem), 2))[:len(mem)]
        axc.scatter(xs, mem.pub, s=13, c="#8A8A8A", zorder=3,
                    label="family members: published calculation" if i == 0 else None)
        axc.scatter(xs, mem.gen, s=15, facecolors="none", edgecolors="#8A8A8A", lw=0.8, zorder=3,
                    label="family members: model, held out" if i == 0 else None)
        kt, ke = so[cpd]["kappa_theory_300"], so[cpd]["kappa_exp_shared_300"]
        axc.scatter([i + 0.12], [kt], s=30, facecolors="white", edgecolors=F.OURS, lw=1.4, zorder=5,
                    label="new compound: model (theory scale)" if i == 0 else None)
        axc.scatter([i + 0.12], [ke], s=30, c=F.OURS, zorder=5,
                    label="new compound: after transfer function" if i == 0 else None)
        axc.annotate("", xy=(i + 0.12, ke * 1.10), xytext=(i + 0.12, kt / 1.10),
                     arrowprops=dict(arrowstyle="->", color=F.OURS, lw=0.9))
        axc.text(i + 0.19, kt, f"{kt:.2f}", fontsize=6.3, color=F.OURS, va="center")
        axc.text(i + 0.19, ke, f"{ke:.2f}", fontsize=6.3, color=F.OURS, va="center", fontweight="bold")
    axc.set_yscale("log")
    axc.set_ylim(0.8, 90)
    ticks(axc.yaxis, (1, 2, 3, 5, 10, 20, 50))
    axc.set_xticks([0, 1])
    axc.set_xticklabels(["TmPbAu\n(Pb–Au)", "HfTeOs\n(Te–Os)"], fontsize=6.8)
    axc.set_xlim(-0.45, 1.6)
    axc.set_ylabel("$\\kappa_L$ at 300 K (W m$^{-1}$ K$^{-1}$)")
    axc.set_title(f"shared transfer function, $\\times${c:.2f} at 300 K", fontsize=7, pad=3)
    axc.legend(loc="upper right", fontsize=5.4, frameon=True, borderaxespad=0.3, handletextpad=0.3)
    F.panel_label(axc, "c", loc="lower left")

    F.save(fig, "fig10_ml")
    print(f"  (a) {len(V300)} compounds at 300 K; registry {mv['median_ape']}% / {mv['within_2x_pct']}%")
    print("  (b) " + ", ".join(f"{k} {M[k]['median_ape']:.1f}% R2 {M[k]['r2_log']:.2f}" for k in keys))
    print(f"  (c) c={c}, p={p}; " + ", ".join(f"{k} {so[k]['kappa_theory_300']} -> {so[k]['kappa_exp_shared_300']}"
                                             for k in so))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
