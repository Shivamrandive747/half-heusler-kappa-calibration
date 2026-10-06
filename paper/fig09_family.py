"""Figure 4 -- per-family calibration: accuracy by family, by compound, and the three best curves.

The paper's predictions apply the transfer function to a PUBLISHED calculation with the constant
of the compound's own bonding family (family_calibration.py, Section 3.6). This figure shows that
route's accuracy, scored the way Section 3.6 scores it, and what a good held-out fit looks like.

  (a) PER FAMILY. Validated families (held-out median error below 25 %): every compound held out and
      predicted from its family-mates alone, paired with the shared constant on the same compounds.
      Families calibrated on their own reference paper -- multi-member families whose held-out error
      is at or above 25 % (held-out figures in the supplementary table) and single-member anchored
      families -- are drawn hatched as in-sample fits, labelled "in sample, not a test". A family
      that misses the bar even fitted on itself (Sb-Ir) is left to the supplement.
  (b) PER COMPOUND. Shared against family error for all held-out compounds; points below the
      diagonal are where the family constant does better.
  (c)-(e) The three best held-out family fits as kappa_L(T) curves, selected and drawn by
      fig03_curves.select_best / draw_curve (the same code that writes curves_best3.json), on one
      shared y-scale and labelled as the three best of the eligible set with the median over all.

Every number is read from data/exports/kappa_v2/family_deployed_pooled.csv, deployed_route.json and
family_calibration.json (written by compute_deployed_route.py and family_calibration.py), or
recomputed by fig03_curves.select_best and asserted equal to the scoreboard.
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
import fig03_curves as C

EX = "data/exports/kappa_v2"
BAR = 25.0


def family_tables():
    """Held-out family table (validated) and the in-sample table (calibrated + anchored)."""
    P = pd.read_csv(f"{EX}/family_deployed_pooled.csv")
    held = P[P.arm == "family"]
    fitted = P[P.arm == "family (in-sample)"]
    fam = (held.groupby("family").agg(fam_err=("ape", "median"), glob_err=("ape_global", "median"),
                                      n=("ape", "size"))
               .sort_values("fam_err"))
    one = fitted.groupby("family").agg(fam_err=("ape", "median"), n=("ape", "size"))
    # cross-check against the family table, so the figure cannot drift from Section 3.6
    fc = json.load(open(f"{EX}/family_calibration.json"))["families"]
    for f, r in fam.iterrows():
        assert abs(r.fam_err - fc[f]["loo_family_ape"]) < 0.06, (f, r.fam_err, fc[f]["loo_family_ape"])
        assert abs(r.glob_err - fc[f]["loo_global_ape"]) < 0.06, (f, r.glob_err, fc[f]["loo_global_ape"])
    # (a) shows two groups (author's ruling 2026-10-06): families that PASS the held-out test, as
    # paired held-out bars; and families calibrated on their own reference paper, drawn as in-sample
    # fits -- the multi-member families at or above the bar, then the single-member anchored ones.
    own = P.groupby("family").ape_anchored.median()
    cal = pd.DataFrame({"fam_err": own[[f for f in fam.index if fam.loc[f, "fam_err"] >= BAR]],
                        "n": fam.n[fam.fam_err >= BAR]}).sort_values("fam_err")
    one = one[one.fam_err < BAR]
    one = pd.concat([cal, one.assign(fam_err=own[one.index])]) if len(cal) else one
    fam = fam[fam.fam_err < BAR]
    return P, held, fam, one


def main() -> int:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    P, held, fam, one = family_tables()
    dep = json.load(open(f"{EX}/deployed_route.json"))
    fa = dep["family_arm_compounds"]
    assert fa["n"] == len(held), (fa["n"], len(held))
    pred = pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv")
    issues_in = set(pred[pred.status == "ISSUED"].family.dropna())
    sel = C.select_best()

    # ---- explicit millimetre layout: 190 mm wide, every gap chosen ----------------------------
    F.use_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["hatch.linewidth"] = 0.6
    W = 190.0
    HB, YB = 33.0, 8.5                 # bottom row: height, bottom offset (room for the x label)
    YS = YB + HB + 1.5                 # legend strip, between the rows
    HT, YT = 54.0, YB + HB + 17.5      # top row: height, bottom offset
    H = YT + HT + 1.5
    fig = plt.figure(figsize=(W * F.MM, H * F.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

    axa = ax_mm(21.5, YT, 97.0, HT)
    axb = ax_mm(134.5, YT, HT, HT)
    x0, gap = 11.5, 2.5
    wc = (W - x0 - 1.0 - 2 * gap) / 3
    axc = ax_mm(x0, YB, wc, HB)
    axes_c = [axc] + [ax_mm(x0 + k * (wc + gap), YB, wc, HB, sharey=axc) for k in (1, 2)]

    # ---- (a) per family ------------------------------------------------------------------------
    rows = list(fam.index) + list(one.index)
    y = np.arange(len(rows), dtype=float)
    y[len(fam):] += 0.9                                   # a gap before the in-sample group
    h = 0.40
    xmax = float(max(fam.glob_err.max(), fam.fam_err.max(), one.fam_err.max() if len(one) else 0))
    # the 25 % bar, drawn per group so it never runs through a group heading
    for a_, b_ in ((0, len(fam) - 1), (len(fam), len(rows) - 1)):
        if b_ >= a_:
            axa.vlines(BAR, y[a_] - 0.55, y[b_] + 0.6, color="#888888", lw=0.7, ls=":", zorder=2)
    fs_val = 6.0
    for i, f in enumerate(fam.index):
        r = fam.loc[f]
        axa.barh(y[i] - h / 2, r.glob_err, height=h, color=F.REFUSED, zorder=3)
        axa.barh(y[i] + h / 2, r.fam_err, height=h, color=F.OURS, zorder=3)
        for v, dy, col, wt in ((r.glob_err, -h / 2, "#6A6A6A", "normal"),
                               (r.fam_err, h / 2, F.OURS, "bold")):
            # a white knock-out only where the number would sit on the dotted 25 % line
            knock = v < BAR < v + xmax * 0.09
            axa.text(v + xmax * 0.012, y[i] + dy, f"{v:.1f}", va="center", fontsize=fs_val,
                     color=col, fontweight=wt, zorder=4,
                     bbox=dict(boxstyle="square,pad=0.0", fc="white", ec="none") if knock else None)
    for j, f in enumerate(one.index):
        k = len(fam) + j
        r = one.loc[f]
        axa.barh(y[k], r.fam_err, height=h * 1.25, color="white", edgecolor=F.OURS,
                 hatch="////", lw=0.6, zorder=3)
        axa.text(r.fam_err + xmax * 0.012, y[k], f"{r.fam_err:.1f}", va="center",
                 fontsize=fs_val, color=F.OURS)
    nn = {**fam.n.to_dict(), **one.n.to_dict()}
    axa.set_yticks(y)
    axa.set_yticklabels([f"{f}{'*' if f in issues_in else ''} ({int(nn[f])})" for f in rows],
                        fontsize=6.6)
    axa.tick_params(axis="y", which="both", length=0, pad=2)
    axa.grid(axis="y", visible=False)
    axa.set_xlim(0, xmax * 1.16)
    axa.set_ylim(y[-1] + 0.65, y[0] - 1.40)               # inverted, room for the group headings
    axa.text(BAR, y[-1], " 25%", fontsize=6.0, color="#666666", ha="left", va="center")
    axa.set_xlabel("median error per family (%)")
    head = dict(fontsize=6.6, fontweight="bold", ha="left", va="center",
                transform=axa.get_yaxis_transform())
    axa.text(0.012, y[0] - 0.78, f"validated: each compound held out "
             f"({int(fam.n.sum())} compounds)", color=F.OURS, **head)
    if len(one):
        axa.text(0.012, y[len(fam)] - 0.78, "calibrated on own paper: in sample, not a test",
                 color=F.OURS, **head)
    hdl = [Patch(fc=F.REFUSED, label="shared constant, held out"),
           Patch(fc=F.OURS, label="own family constant, held out"),
           Patch(fc="white", ec=F.OURS, hatch="////", lw=0.6,
                 label="own paper (in sample, not a test)"),
           Line2D([], [], ls="none", marker="$*$", color="#222222", ms=5,
                  label="predictions issued in family")]
    axa.legend(handles=hdl, loc="lower right", fontsize=6.2, borderaxespad=0.5,
               handlelength=1.4, handleheight=0.9)
    F.panel_label(axa, "a", loc="upper right")

    # ---- (b) per compound ------------------------------------------------------------------------
    lim = float(max(held.ape_global.max(), held.ape.max())) * 1.08
    axb.fill_between([0, lim], [0, 0], [0, lim], color=F.BAND, zorder=1, lw=0)
    axb.plot([0, lim], [0, lim], ls="--", lw=0.8, color="#555555", zorder=2)
    axb.scatter(held.ape_global, held.ape, s=14, c=F.OURS, edgecolors="white", lw=0.4, zorder=4)
    axb.set_xlim(0, lim)
    axb.set_ylim(0, lim)
    axb.set_aspect("equal", adjustable="box")
    axb.set_xlabel("shared-constant error (%)")
    axb.set_ylabel("family-constant error (%)", labelpad=1.5)
    xy_b = np.column_stack([held.ape_global.values, held.ape.values])
    gc = fa["global_on_same_compounds"]

    def mk_stats(x, yy, ha, va):
        return axb.text(x, yy, f"{fa['n']} compounds, each held out\n"
                               f"median {gc['median_ape']:.1f}% $\\rightarrow$ {fa['median_ape']:.1f}%\n"
                               f"within 2$\\times$ {gc['within2x_pct']:.1f}% $\\rightarrow$ "
                               f"{fa['within2x_pct']:.1f}%\n"
                               f"family better for {fa['n_family_better']} of {fa['n']}",
                        transform=axb.transAxes, ha=ha, va=va, fontsize=6.2, linespacing=1.3,
                        color=F.OURS, zorder=20)
    _, occ = C.place_free(axb, mk_stats, xy_b, corners=[(0.035, 0.84, "left", "top")])

    def mk_below(x, yy, ha, va):
        return axb.text(x, yy, "below line:\nfamily better", transform=axb.transAxes, ha=ha,
                        va=va, fontsize=6.0, color="#666666", zorder=20)
    C.place_free(axb, mk_below, xy_b, occupied=occ,
                 corners=[(0.965, 0.035, "right", "bottom"), (0.965, 0.30, "right", "bottom")])
    F.panel_label(axb, "b", loc="upper left")

    # ---- (c)-(e) the three best held-out curves ------------------------------------------------
    drawn = [C.draw_curve(ax, sel, q) for ax, q in zip(axes_c, sel["pick"].itertuples())]
    C.shared_logy(axes_c, drawn, pad=1.22)
    for k, (ax, q, xy) in enumerate(zip(axes_c, sel["pick"].itertuples(), drawn)):
        ax.margins(x=0.04)
        span = float(xy[:, 0].max() - xy[:, 0].min())
        ax.xaxis.set_major_locator(mticker.MultipleLocator(200 if span > 600 else 100))
        ax.set_xlabel("$T$ (K)", labelpad=1.0)
        ax.tick_params(labelsize=6.6, pad=1.5)
        if k:
            ax.tick_params(labelleft=False)

        def mk_letter(x, yy, ha, va, ax=ax, k=k):
            return ax.text(x, yy, f"({'cde'[k]})", transform=ax.transAxes, fontsize=8.5,
                           fontweight="bold", ha=ha, va=va, zorder=20,
                           bbox=dict(boxstyle="square,pad=0.28", fc="white", ec="#BBBBBB",
                                     lw=0.5, alpha=0.92))
        _, occ = C.place_free(ax, mk_letter, xy)

        def mk_note(x, yy, ha, va, ax=ax, q=q):
            return ax.text(x, yy, f"{q.compound} ({q.family})\nheld-out error {q.ape:.1f}%",
                           transform=ax.transAxes, fontsize=6.6, ha=ha, va=va, color=F.OURS,
                           zorder=20, linespacing=1.2)
        C.place_free(ax, mk_note, xy, occupied=occ,
                     corners=[(0.965, 0.965, "right", "top"), (0.965, 0.035, "right", "bottom"),
                              (0.20, 0.965, "left", "top"), (0.20, 0.035, "left", "bottom")])
    axes_c[0].set_ylabel(F.KAPPA_LABEL, labelpad=1.5)

    # the strip between the rows: the selection caveat, then the curve key
    hc, lc = C.legend_handles()
    fig.legend(handles=hc, labels=lc, loc="lower left", bbox_to_anchor=(x0 / W, YS / H),
               ncol=4, frameon=False, fontsize=6.4, handlelength=1.8, columnspacing=1.2,
               borderaxespad=0.0, borderpad=0.0)
    lab = C.honest_label(sel)
    fig.text(x0 / W, (YS + 4.6) / H, f"(c)–(e): {lab}", ha="left", va="bottom", fontsize=6.6,
             fontweight="bold", color="#222222")

    print(fam.round(1).to_string())
    print(f"in-sample fits: {dict(one.fam_err.round(1))}")
    print(f"figure size: {W:.0f} x {H:.1f} mm")
    F.save(fig, "fig09_family")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
