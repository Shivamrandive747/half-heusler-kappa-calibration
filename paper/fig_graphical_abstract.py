"""Graphical abstract (Computational Materials Science: one wide image, >= 1328 x 531 px).

The pipeline left to right -- crystal structure -> machine-learning model -> kappa_BTE -> transfer
function -> experimental kappa_L -- with an inset of the end-to-end blind test against measurement
(in-domain compounds, chemistry cluster held out, reference seed). Numbers are read from the registry.
Colours follow figlib's semantic roles: measurement in vermillion (F.MEASURED -- the measured axis and
a rug of the measured values), our calibrated result in the accent blue (F.OURS).
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


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import compute_seed_averaged as S

    N = json.load(open(f"{EX}/paper_numbers.json"))
    h = N["in_domain_table"]["seed_medians"]["calibrated"]
    mv = N["ml_generator_validation"]
    c, p = N["calibration"]["c"], N["calibration"]["p"]
    raw = pd.read_csv(f"{EX}/target_blind_test.csv")
    raw = raw[raw.klass == "half"] if "klass" in raw.columns else raw
    M = S.per_cmp(S.nested(S.in_domain(raw)))

    F.use_style()
    fig = plt.figure(figsize=(6.6, 2.64), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x0, x1, title, body, col):
        ax.add_patch(FancyBboxPatch((x0, 0.30), x1 - x0, 0.44, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc="white", ec=col, lw=1.4))
        ax.text((x0 + x1) / 2, 0.67, title, ha="center", va="center", fontsize=7.6, fontweight="bold", color=col)
        ax.text((x0 + x1) / 2, 0.47, body, ha="center", va="center", fontsize=6.4, color="#222222", linespacing=1.3)

    def arrow(x0, x1, label=""):
        ax.annotate("", xy=(x1, 0.52), xytext=(x0, 0.52),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2))
        if label:
            ax.text((x0 + x1) / 2, 0.58, label, ha="center", va="bottom", fontsize=5.6, color="#555555")

    blue, grey, dark = F.OURS, "#6E8FA8", "#1A1A1A"
    box(0.015, 0.155, "Crystal structure", "half Heusler XYZ\ncubic $C1_b$, VEC 18", dark)
    arrow(0.16, 0.20)
    box(0.205, 0.37, "Machine learning", f"CatBoost, {N['shap_ranking']['n_descriptors']} descriptors\nvs DFT: {mv['median_ape']:.1f}% median\n(cluster held out)", grey)
    arrow(0.375, 0.415, "$\\kappa_L^{\\mathrm{BTE}}$")
    box(0.42, 0.60, "Transfer function", f"$\\kappa_L^{{\\mathrm{{exp}}}}=\\kappa_L^{{\\mathrm{{BTE}}}}$\n"
        f"$\\times\\min[c\\,(T/300)^{{p}},1]$\n$c={c:.2f},\\ p={p:.2f}$", blue)
    arrow(0.605, 0.645, "$\\kappa_L^{\\mathrm{exp}}$")

    axi = fig.add_axes([0.70, 0.15, 0.27, 0.70])
    lo, hi = 0.4, 30
    F.band_2x(axi, lo, hi, label=None)
    axi.scatter(M.ref, M.k, s=9, c=blue, lw=0.3, edgecolors="white", zorder=3)
    axi.plot([lo, hi], [lo, hi], color="#777777", lw=0.7, ls="--", zorder=2)
    axi.set_xscale("log")
    axi.set_yscale("log")
    axi.set_xlim(lo, hi)
    axi.set_ylim(lo, hi)
    # measurement wears the measurement colour (figlib role): the axis it is read on, and a rug of
    # the measured values along it
    axi.plot(M.ref, np.full(len(M), lo * 1.07), ls="none", marker="|", ms=5, mew=0.8,
             color=F.MEASURED, zorder=4)
    axi.set_xlabel("measured $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)", fontsize=6.4, labelpad=1,
                   color=F.MEASURED)
    axi.set_ylabel("predicted $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)", fontsize=6.4, labelpad=1, color=blue)
    import matplotlib.ticker as mticker
    for a_ in (axi.xaxis, axi.yaxis):
        a_.set_major_locator(mticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
        a_.set_major_formatter(mticker.FixedFormatter(["0.5", "1", "2", "5", "10", "20"]))
        a_.set_minor_locator(mticker.NullLocator())
    axi.tick_params(labelsize=5.6)
    axi.set_title(f"blind test vs experiment ({len(M)} compounds)\n{h['median_ape']['median']:.1f}% median error, "
                  f"{h['within_2x']['median']:.1f}% within 2$\\times$\n(five-seed median; points: one seed)",
                  fontsize=6.0, pad=2)

    out = Path(__file__).resolve().parent / "figures"
    fig.savefig(out / "graphical_abstract.png", dpi=300)
    fig.savefig(out / "graphical_abstract.pdf")
    from PIL import Image
    w, hgt = Image.open(out / "graphical_abstract.png").size
    print(f"  wrote figures/graphical_abstract.png ({w} x {hgt} px) and .pdf; {len(M)} compounds in the inset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
