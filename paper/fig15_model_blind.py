"""Figure -- from crystal structure to measurement, and how close that is to the field's own spread.

One row, one argument (redesigned 2026-11-18), every number read from an artefact (nothing typed):
  (a) STEP 1, MODEL vs PUBLISHED CALCULATION. Each compound's model prediction against its published
      full-BTE value at 300 K, with its whole X-site chemistry cluster removed from training
      (bte_generator_validation.csv). The statistics are the registry's ml_generator_validation, per
      compound over all temperatures.
  (b) STEP 2, THE ROUTE AGAINST MEASUREMENT. The blind-test points of fig04_blind.blind_points (seed-0
      file, calibration refitted without the compound's own chemistry cluster). Each in-domain
      compound is drawn TWICE and joined: the model's uncorrected, calculation-scale value (open
      grey) and the value after the transfer function (blue), so the reader sees the correction pull
      the cloud onto the diagonal in one panel. The statistics printed are the five-seed medians.
  (c) COMPARED TO WHAT. The error-budget strip (fig16_error_budget.draw): how far two laboratories,
      or two calculations, disagree on the same compound, then every route and the two nulls on the
      same multiplicative axis.

What left this figure: the SHAP ranking (now only in Section S4 of the supplement, where the full
attribution already was) and the four-route dot-and-whisker panel (the same four medians are in the
accuracy table and in panel (c)). The old layout was a 2 x 2 block 140 mm tall plus a separate
column figure; this is one row 74 mm tall.

Runs AFTER make_paper_numbers.py (it reads the registry).
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import figlib as F

EX = "data/exports/kappa_v2"


def ticks(axis, values):
    import matplotlib.ticker as mticker
    axis.set_major_locator(mticker.FixedLocator(values))
    axis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in values]))
    axis.set_minor_locator(mticker.NullLocator())


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from fig04_blind import blind_points
    import fig16_error_budget as EB

    N = json.load(open(f"{EX}/paper_numbers.json"))
    for k in ("ml_generator_validation", "in_domain_table", "accuracy_summary", "error_budget"):
        assert "_unavailable" not in N[k], f"registry {k} unavailable -- run make_paper_numbers.py"
    mv = N["ml_generator_validation"]
    it = N["in_domain_table"]
    sm = it["seed_medians"]["calibrated"]

    V = pd.read_csv(f"{EX}/bte_generator_validation.csv")
    V300 = V[V["T"] == 300].groupby("compound").agg(pub=("k_published", "median"),
                                                     gen=("k_generated", "median"))
    R = blind_points()
    ins, out = R["ins"], R["out"]
    # the blind-test table's n and the drawn in-domain set must be the same compounds' count
    assert len(ins) == int(it["n"]), f"drawn in-domain n {len(ins)} != registry {it['n']}"

    # ---- explicit millimetre layout ------------------------------------------------------------
    F.use_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    W, SQ = F.DOUBLE / F.MM, 44.0
    Y0, HS = 10.0, 58.0                       # bottom offset; height of the strip (c)
    H = Y0 + HS + 6.0
    fig = plt.figure(figsize=(W * F.MM, H * F.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

    axa = ax_mm(13.0, Y0, SQ, SQ)
    axb = ax_mm(70.0, Y0, SQ, SQ)
    XS = 150.0
    axs = ax_mm(XS, Y0, W - XS - 10.5, HS)
    unit = "(W m$^{-1}$ K$^{-1}$)"
    HEAD = 33.0                               # points above a square panel at which its heading sits

    def heading(ax, head, sub):
        ax.annotate(head, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, HEAD + 10.5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=F.FS_LABEL,
                    fontweight="bold")
        ax.annotate(sub, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, HEAD + 9.0),
                    textcoords="offset points", ha="center", va="top", fontsize=F.FS_MIN,
                    color="#333333", linespacing=1.2)

    # ---- (a) model vs published calculation ----------------------------------------------------
    lo, hi = 0.3, 160.0
    v = np.concatenate([V300.pub.values, V300.gen.values])
    assert v.min() >= lo and v.max() <= hi, f"(a) data {v.min():.2f}-{v.max():.2f} outside {lo}-{hi}"
    F.band_2x(axa, lo, hi, color=F.BAND, label=None)
    axa.scatter(V300.pub, V300.gen, s=7, c=F.OURS, lw=0.3, edgecolors="white", zorder=3)
    F.parity_axes(axa, lo, hi, f"calculated $\\kappa_L^{{\\mathrm{{BTE}}}}$ {unit}",
                  f"model $\\kappa_L^{{\\mathrm{{BTE}}}}$ {unit}")
    for ax_ in (axa.xaxis, axa.yaxis):
        ticks(ax_, (1, 3, 10, 30, 100))
    axa.tick_params(labelsize=F.FS_TICK, pad=1.5)
    heading(axa, "Model against calculation",
            f"{mv['median_ape']:.1f}% median error, {mv['within_2x_pct']:.0f}% within 2$\\times$\n"
            f"{mv['n_compounds']} compounds, chemistry held out")
    F.panel_label(axa, "a")

    # ---- (b) the route against measurement, before and after the correction --------------------
    F.warn_clip(np.concatenate([ins.ref, ins.cal, ins.raw, out.ref, out.cal]), "fig15 (b)")
    clo, chi = F.KAPPA_LIM
    GREY = "#9AA0A6"
    F.band_2x(axb, clo, chi, color=F.BAND, label=None)
    for r in ins.itertuples():
        axb.plot([r.ref, r.ref], [r.raw, r.cal], color=GREY, lw=0.5, zorder=2, solid_capstyle="butt")
    axb.scatter(ins.ref, ins.raw, s=7, facecolors="white", edgecolors=GREY, lw=0.6, zorder=3)
    axb.scatter(out.ref, out.cal, s=11, facecolors="none", edgecolors="#8C8C8C", lw=0.7,
                marker="s", zorder=3)
    axb.scatter(ins.ref, ins.cal, s=13, c=F.OURS, marker="o", lw=0.35, edgecolors="white", zorder=4)
    F.parity_axes(axb, clo, chi, f"measured $\\kappa_L$ {unit}", f"predicted $\\kappa_L$ {unit}")
    F.kappa_ticks(axb)
    axb.tick_params(labelsize=F.FS_TICK, pad=1.5)
    n_off = sum(F.edge_markers(axb, g.ref.values, g.cal.values, axis=a, color=col)
                for g, col in ((out, "#8C8C8C"), (ins, F.OURS)) for a in ("y", "x"))
    if n_off:
        print(f"  (b) {n_off} point(s) drawn on the axis edge")
    hdl = [Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=GREY, mew=0.6,
                  label="model, uncorrected"),
           Line2D([], [], ls="none", marker="o", ms=3.8, mfc=F.OURS, mec="white", mew=0.35,
                  label=f"corrected, in domain ({len(ins)})"),
           Line2D([], [], ls="none", marker="s", ms=3.2, mfc="none", mec="#8C8C8C", mew=0.7,
                  label=f"corrected, outside domain ({len(out)})")]
    axb.legend(handles=hdl, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False,
               fontsize=F.FS_MIN, handletextpad=0.2, labelspacing=0.15, borderaxespad=0.15,
               borderpad=0.0)
    axb.annotate("Model route against measurement", xy=(0.5, 1.0), xycoords="axes fraction",
                 xytext=(0, HEAD + 10.5), textcoords="offset points", ha="center", va="bottom",
                 fontsize=F.FS_LABEL, fontweight="bold")
    axb.annotate(f"{sm['median_ape']['median']:.1f}% median error, "
                 f"{sm['within_2x']['median']:.1f}% within 2$\\times$",
                 xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, HEAD + 9.0),
                 textcoords="offset points", ha="center", va="top", fontsize=F.FS_MIN,
                 color="#333333")
    F.panel_label(axb, "b")

    # ---- (c) compared to what: the field's own spread ------------------------------------------
    summary = EB.draw(axs, EB.rows(), heads={"Same compound, two sources": "Spread between sources"})
    fig.text((XS - 32.0) / W, (Y0 + HS + 1.6) / H, "(c)  Against the spread of the field",
             ha="left", va="bottom", fontsize=F.FS_LABEL, fontweight="bold")

    F.save(fig, "fig15_model_blind")
    print(f"  figure size: {W:.0f} x {H:.1f} mm")
    print(f"  (a) {len(V300)} drawn; registry {mv['median_ape']}% / {mv['within_2x_pct']}% "
          f"/ {mv['n_compounds']} compounds / {mv['n_clusters']} clusters")
    print(f"  (b) seed-0 within 2x {ins.ratio.between(.5, 2).mean() * 100:.1f}% "
          f"(registry median of seeds {sm['within_2x']['median']}); "
          f"uncorrected above corrected for {(ins.raw > ins.cal).sum()} of {len(ins)}")
    print("  (c) " + "; ".join(f"{k} {v['median_factor']}" for k, v in summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
