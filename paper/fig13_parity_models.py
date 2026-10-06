"""Supplementary figure -- out-of-fold parity for each of the four regressors.

Exactly the predictions compare_models_production.py scores (model_comparison_oof.csv): identical rows,
identical element-system folds, production weights. log10 kappa on both axes; the factor-of-two band
is shaded. R^2 and RMSE (per row, log10) in each panel are read from model_comparison_production.json.
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
NAMES = {"catboost": "CatBoost", "lightgbm": "LightGBM", "gpr": "Gaussian process", "krr": "kernel ridge"}


def main() -> int:
    import matplotlib.pyplot as plt
    P = pd.read_csv(f"{EX}/model_comparison_oof.csv")
    M = json.load(open(f"{EX}/model_comparison_production.json"))
    F.use_style()
    fig, axs = plt.subplots(1, 4, figsize=(F.DOUBLE, F.DOUBLE * 0.29), sharex=True, sharey=True)
    lo, hi = -1.0, 2.5
    for ax, (k, nm), l in zip(axs, NAMES.items(), "abcd"):
        ax.fill_between([lo, hi], [lo - np.log10(2), hi - np.log10(2)], [lo + np.log10(2), hi + np.log10(2)],
                        color=F.BAND, lw=0, zorder=1)
        ax.plot([lo, hi], [lo, hi], color="#777777", lw=0.8, ls="--", zorder=2)
        ax.scatter(P.log10_kappa_true, P[f"log10_kappa_{k}"], s=6, lw=0,
                   c=F.OURS if k == "catboost" else "#6E6E6E", alpha=0.7, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(nm, fontsize=7.5, pad=3)
        ax.text(0.96, 0.04, f"$R^2$ {M[k]['r2_log']:.2f}\nRMSE {M[k]['rmse_log']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
                bbox=dict(fc="white", ec="#CCCCCC", lw=0.4, pad=1.8, alpha=0.94))
        ax.set_xlabel("log$_{10}\\,\\kappa_L$ (label)")
        F.panel_label(ax, l)
    axs[0].set_ylabel("log$_{10}\\,\\kappa_L$ (out of fold)")
    F.save(fig, "fig13_parity_models")
    print(f"  {len(P)} rows, {P.compound.nunique()} compounds; " +
          ", ".join(f"{k} R2 {M[k]['r2_log']:.3f}" for k in NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
