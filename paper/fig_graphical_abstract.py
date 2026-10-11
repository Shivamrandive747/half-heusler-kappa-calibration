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

    blue = F.OURS
    # 2026-10-11: redrawn in the style of Figure 1 (fig08_method) -- same millimetre canvas, the
    # same stage-family fills (ML reddish purple, transfer function accent blue), the same 8 pt
    # titles / 7 pt-plus body, and no text below 7 pt at the printed size (168 mm wide).
    import matplotlib.ticker as mticker
    from matplotlib.patches import FancyArrowPatch
    from fig08_method import FAM

    F.use_style()
    import matplotlib as mpl
    mpl.rcParams["figure.constrained_layout.use"] = False
    W, H = 168.0, 67.0
    fig = plt.figure(figsize=(W * F.MM, H * F.MM), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()

    def fit(t, x0, x1, pad=2.0):
        w = t.get_window_extent(renderer=renderer).width / fig.dpi / F.MM
        assert w <= (x1 - x0) - pad, f"text {t.get_text()!r} is {w:.1f} mm, box {x1 - x0:.1f} mm"

    Y0, Y1 = 15.0, 51.0
    ym = (Y0 + Y1) / 2
    STRUCT = dict(fc="#ECEEF0", ec="#3A3F44", tc="#2B2B2B", bc="#2E2E2E", ls="-")

    def box(x0, x1, title, lines, s, lw=1.2):
        ax.add_patch(FancyBboxPatch((x0, Y0), x1 - x0, Y1 - Y0,
                                    boxstyle="round,pad=0,rounding_size=2.0",
                                    fc=s["fc"], ec=s["ec"], lw=lw, ls=s["ls"], zorder=3))
        xc = (x0 + x1) / 2
        fit(ax.text(xc, Y1 - 3.0, title, ha="center", va="top", fontsize=9, fontweight="bold",
                    color=s["tc"], zorder=4), x0, x1, pad=1.6)
        pitch = 4.4
        yc = (Y1 - 8.5 + Y0) / 2
        for i, ln in enumerate(lines):
            fit(ax.text(xc, yc + ((len(lines) - 1) / 2 - i) * pitch, ln, ha="center",
                        va="center", fontsize=7.5, color=s["bc"], zorder=4), x0, x1)

    def arrow(x0, x1, label="", col="#5F6368"):
        ax.add_patch(FancyArrowPatch((x0, ym), (x1, ym),
                                     arrowstyle="-|>,head_length=2.8,head_width=1.7",
                                     mutation_scale=1, lw=1.2, color=col, shrinkA=0, shrinkB=0,
                                     zorder=2))
        if label:
            ax.text((x0 + x1) / 2, ym + 2.2, label, ha="center", va="bottom", fontsize=8,
                    color=col)

    XS, XM, XT = (1.0, 28.5), (34.0, 66.5), (74.0, 105.5)
    box(*XS, "Crystal structure", ["half Heusler XYZ", "cubic $C1_b$, VEC 18"], STRUCT)
    arrow(XS[1], XM[0])
    box(*XM, "Machine learning",
        [f"CatBoost, {N['shap_ranking']['n_descriptors']} descriptors",
         f"vs DFT: {mv['median_ape']:.1f}% median", "cluster held out"], FAM["ml"])
    arrow(XM[1], XT[0], "$\\kappa_L^{\\mathrm{BTE}}$", col=F.PURPLE)
    box(*XT, "Transfer function",
        ["$\\kappa_L^{\\mathrm{exp}}=\\kappa_L^{\\mathrm{BTE}}\\times$",
         "$\\min[c\\,(T/300)^{p},1]$", f"$c={c:.2f},\\ p={p:.2f}$"], FAM["tf"], lw=1.7)
    arrow(XT[1], 113.0, "$\\kappa_L^{\\mathrm{exp}}$", col=F.OURS)

    # ---- inset: the end-to-end blind test against measurement --------------------------------
    IX, IY, IS = 125.0, 14.0, 39.0
    axi = fig.add_axes([IX / W, IY / H, IS / W, IS / H])
    lo, hi = 0.4, 30
    F.band_2x(axi, lo, hi, label=None)
    axi.scatter(M.ref, M.k, s=10, c=blue, lw=0.3, edgecolors="white", zorder=3)
    axi.set_xscale("log")
    axi.set_yscale("log")
    axi.set_xlim(lo, hi)
    axi.set_ylim(lo, hi)
    axi.set_aspect("equal", adjustable="box")
    # measurement wears the measurement colour (figlib role): the axis it is read on, and a rug of
    # the measured values along it
    axi.plot(M.ref, np.full(len(M), lo * 1.07), ls="none", marker="|", ms=5, mew=0.8,
             color=F.MEASURED, zorder=4)
    axi.set_xlabel("measured $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)", fontsize=F.FS_LABEL, labelpad=1,
                   color=F.MEASURED)
    axi.set_ylabel("predicted $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)", fontsize=F.FS_LABEL,
                   labelpad=1, color=blue)
    for a_ in (axi.xaxis, axi.yaxis):
        a_.set_major_locator(mticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
        a_.set_major_formatter(mticker.FixedFormatter(["0.5", "1", "2", "5", "10", "20"]))
        a_.set_minor_locator(mticker.NullLocator())
    axi.tick_params(labelsize=F.FS_TICK, pad=1.5)
    xc_i = IX + IS / 2
    ax.text(xc_i, IY + IS + 9.6, f"Blind test vs experiment, n = {len(M)}", ha="center",
            va="bottom", fontsize=8, fontweight="bold", color="#1A1A1A")
    ax.text(xc_i, IY + IS + 5.4, f"{h['median_ape']['median']:.1f}% median error, "
            f"{h['within_2x']['median']:.1f}% within 2$\\times$", ha="center", va="bottom",
            fontsize=7.5, color=blue)
    ax.text(xc_i, IY + IS + 1.6, "five-seed median; points: one seed", ha="center", va="bottom",
            fontsize=F.FS_MIN, color="#555555")

    out = Path(__file__).resolve().parent / "figures"
    fig.savefig(out / "graphical_abstract.png", dpi=300)
    fig.savefig(out / "graphical_abstract.pdf")
    from PIL import Image
    w, hgt = Image.open(out / "graphical_abstract.png").size
    print(f"  wrote figures/graphical_abstract.png ({w} x {hgt} px) and .pdf; {len(M)} compounds in the inset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
