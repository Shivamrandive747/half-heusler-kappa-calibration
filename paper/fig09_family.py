"""Figure 4 -- per-family calibration: by family, by compound, and held-out fits as curves.

The paper's predictions apply the transfer function to a PUBLISHED calculation with the constant
of the compound's own bonding family (family_calibration.py, Section 3.6). This figure shows that
route's accuracy, scored the way Section 3.6 scores it, and what the held-out fits look like.

  (a) PER FAMILY. Validated families (held-out median error below 25 %): every compound held out and
      predicted from its family-mates alone, paired with the shared constant on the same compounds.
      Families calibrated on their own reference paper -- multi-member families whose held-out error
      is at or above 25 % (held-out figures in the supplementary table) and single-member anchored
      families -- are drawn hatched as in-sample fits, under the heading "Calibrated on own papers (in sample)". A family
      that misses the bar even fitted on itself (Sb-Ir) is left to the supplement.
  (b) PER COMPOUND, AS A BEFORE/AFTER PAIR (2026-11-18, author: the error-against-error scatter did
      not show what it was meant to). Every held-out compound is one vertical line from its error
      with the shared constant (open grey ring) to its error with the family constant (blue dot),
      and the compounds are ordered by that change: improved on the left, tied in the middle,
      worsened on the right. The three group sizes are the counts of the sign test, and the two
      horizontal lines are the two medians the text quotes.
  (c) HELD-OUT FITS AS CURVES, ONE PER FAMILY (2026-11-18, author: more curves, to show the
      correction at work). Of the eligible fits in validated families (fig03_curves.select_best),
      the one with the lowest held-out error in EACH family, labelled as the best of its family
      with the median over all eligible fits beside it. This keeps the author's ruling of
      2026-10-13 (best fits, said to be the best) and widens it from one family to every family
      that has an eligible curve. ALL eligible fits, including the worst, are drawn in the same
      style as fig09_family_curves, for the supplement.
      CURVE_RULE = "every_second" draws instead every second eligible fit from the best to the
      worst (a stratified row whose median equals the full set's); it was drawn and set aside.

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
SHARED = "#7F858B"        # the shared constant in panel (b): grey, open marker, dashed line
T_LIM = (160.0, 1130.0)   # one temperature axis for every curve panel
CURVE_LABELS = ["measured", "published calculation", "family constant, held out",
                "temperature of a published calculation"]
CURVE_RULE = "best_per_family"     # or "every_second"; see the module docstring, panel (c)
N_COL_ALL = 7                      # columns of the supplement grid of all eligible fits


def pick_curves(ok: pd.DataFrame):
    """The fits drawn in the main text, and the heading that says how they were chosen.

    `ok` is every eligible fit, sorted by held-out error (lowest first).
    """
    n, med = len(ok), float(ok.ape.median())
    if CURVE_RULE == "best_per_family":
        shown = ok.drop_duplicates("family")      # sorted by error: a family's first row is its best
        head = (f"Held-out fits as curves: the lowest-error fit of each validated family "
                f"({len(shown)} of the {n} eligible fits; median error over all {n}: {med:.1f}%)")
    elif CURVE_RULE == "every_second":
        shown = ok.iloc[::2]
        assert abs(float(shown.ape.median()) - med) < 1e-9, "shown median must equal the full set's"
        assert shown.iloc[-1].compound == ok.iloc[-1].compound, "the worst fit must be shown"
        head = (f"Held-out fits as curves: every second of the {n} eligible, from the lowest error "
                f"to the highest (median of those shown and of all {n}: {med:.1f}%)")
    else:
        raise ValueError(CURVE_RULE)
    return shown, head


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


def curve_panel(ax, sel, q, first: bool, wide: bool = False):
    """One held-out fit: the curve, one shared temperature axis, its name and error in a free corner."""
    import matplotlib.ticker as mticker

    xy = C.draw_curve(ax, sel, q)
    ax.set_xlim(*T_LIM)
    ax.xaxis.set_major_locator(mticker.FixedLocator((200, 400, 600, 800, 1000) if wide
                                                    else (300, 600, 900)))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(100))
    ax.tick_params(labelsize=F.FS_TICK, pad=1.5)
    ax.tick_params(axis="x", which="minor", length=1.5)
    if not first:
        ax.tick_params(labelleft=False)
    note = (f"{q.compound} ({q.family})\nheld-out error {q.ape:.1f}%" if wide
            else f"{q.compound}\n{q.family}, {q.ape:.1f}%")

    def mk(x, yy, ha, va):
        return ax.text(x, yy, note, transform=ax.transAxes,
                       fontsize=F.FS_MIN, ha=ha, va=va, color=F.OURS, zorder=20, linespacing=1.15)
    C.place_free(ax, mk, xy, pad=0.015,
                 corners=[(0.96, 0.96, "right", "top"), (0.04, 0.04, "left", "bottom"),
                          (0.96, 0.04, "right", "bottom"), (0.04, 0.96, "left", "top")])
    return xy


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    P, held, fam, one = family_tables()
    dep = json.load(open(f"{EX}/deployed_route.json"))
    fa = dep["family_arm_compounds"]
    gc = fa["global_on_same_compounds"]
    assert fa["n"] == len(held), (fa["n"], len(held))
    pred = pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv")
    issues_in = set(pred[pred.status == "ISSUED"].family.dropna())

    # every eligible held-out fit, in order of error; the main text draws the ones pick_curves names
    sel = C.select_best(n_show=None, verbose=False)
    ok = sel["pick"].reset_index(drop=True)
    shown, head_c = pick_curves(ok)

    # ---- explicit millimetre layout: 190 mm wide, every gap chosen ----------------------------
    F.use_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["hatch.linewidth"] = 0.6
    W = 190.0
    x0, gap = 12.0, 2.0                 # curve row: left offset (y label), gap between panels
    YB, HB = 8.5, 28.0                  # curve row: bottom offset (x label), height
    YL = YB + HB + 1.6                  # the curve key, one line above the curve row
    YH = YL + 4.3                       # the heading of the curve row, above the key
    YT = YH + 4.6 + 10.5                # top row: bottom offset (heading + its own x labels)
    HT = 52.0
    H = YT + HT + 1.5
    fig = plt.figure(figsize=(W * F.MM, H * F.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

    axa = ax_mm(23.0, YT, 57.0, HT)
    axb = ax_mm(93.0, YT, W - 93.0 - 1.5, HT)
    n_c = len(shown)
    wc = (W - x0 - 1.0 - (n_c - 1) * gap) / n_c
    axes_c = [ax_mm(x0 + k * (wc + gap), YB, wc, HB) for k in range(n_c)]

    # ---- (a) per family ------------------------------------------------------------------------
    rows = list(fam.index) + list(one.index)
    y = np.arange(len(rows), dtype=float)
    y[len(fam):] += 0.9                                   # a gap before the in-sample group
    h = 0.46
    xmax = float(max(fam.glob_err.max(), fam.fam_err.max(), one.fam_err.max() if len(one) else 0))
    # the 25 % bar, drawn per group so it never runs through a group heading
    for a_, b_ in ((0, len(fam) - 1), (len(fam), len(rows) - 1)):
        if b_ >= a_:
            axa.vlines(BAR, y[a_] - (0.55 if a_ == 0 else 0.42), y[b_] + 0.6, color="#888888",
                       lw=0.7, ls=":", zorder=2)
    fs_val = F.FS_MIN
    for i, f in enumerate(fam.index):
        r = fam.loc[f]
        axa.barh(y[i] - h / 2, r.glob_err, height=h, color=F.REFUSED, zorder=3)
        axa.barh(y[i] + h / 2, r.fam_err, height=h, color=F.OURS, zorder=3)
        # Values sit INSIDE the end of each bar when it is long enough (2026-10-11): at 7 pt two
        # labels outside a pair of touching bars collide whenever the two values are close
        # (Co-Sb: 23.0 against 24.9). Short bars keep the label just outside.
        for v, dy, inside_col, out_col, wt in ((r.glob_err, -h / 2, "#2E2E2E", "#6A6A6A", "normal"),
                                               (r.fam_err, h / 2, "white", F.OURS, "bold")):
            fits_in = v > xmax * 0.13
            axa.text(v - xmax * 0.012 if fits_in else v + xmax * 0.012, y[i] + dy, f"{v:.1f}",
                     va="center", ha="right" if fits_in else "left", fontsize=fs_val,
                     color=inside_col if fits_in else out_col, fontweight=wt, zorder=4)
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
                        fontsize=F.FS_MIN)
    axa.tick_params(axis="y", which="both", length=0, pad=2)
    axa.grid(axis="y", visible=False)
    axa.set_xlim(0, xmax * 1.16)
    axa.set_ylim(y[-1] + 0.65, y[0] - 1.40)               # inverted, room for the group headings
    # the dotted 25 % line is named in the caption; at this width a label on it hits the heading
    axa.set_xlabel("median error per family (%)")
    head = dict(fontsize=F.FS_LABEL, fontweight="bold", ha="left", va="center",
                transform=axa.get_yaxis_transform())
    # The two group headings, worded exactly as the text and caption use them (2026-10-11).
    axa.text(0.012, y[0] - 0.80, "Validated (held out)", color=F.OURS, **head)
    if len(one):
        axa.text(0.012, y[len(fam)] - 0.80, "Calibrated or anchored (in sample)",
                 color=F.OURS, **head)
    # Legend (2026-10-11): what each bar style IS; whether it is held out or in sample is said
    # once, by the group headings, not repeated in every entry.
    hdl = [Patch(fc=F.REFUSED, label="shared constant"),
           Patch(fc=F.OURS, label="family constant"),
           Patch(fc="white", ec=F.OURS, hatch="////", lw=0.6, label="family constant, own papers"),
           Line2D([], [], ls="none", marker="$*$", color="#222222", ms=5.5,
                  label="predictions issued")]
    axa.legend(handles=hdl, loc="lower right", fontsize=F.FS_MIN, borderaxespad=0.5,
               handlelength=1.4, handleheight=0.9, labelspacing=0.35)
    F.panel_label(axa, "a", loc="upper right")

    # ---- (b) per compound: shared constant -> family constant, ordered by the change -------------
    d = held.assign(delta=held.ape - held.ape_global)
    better = d[d.delta < 0].sort_values("delta")            # largest improvement first
    tied = d[d.delta == 0].sort_values("ape")
    worse = d[d.delta > 0].sort_values("delta")
    nb, nt, nw = len(better), len(tied), len(worse)
    # the three groups ARE the sign-test counts the text quotes; the medians are the text's too
    assert nb == fa["n_family_better"] and nb + nt + nw == fa["n"], (nb, nt, nw)
    assert abs(float(d.ape.median()) - fa["median_ape"]) < 0.06
    assert abs(float(d.ape_global.median()) - gc["median_ape"]) < 0.06
    order = pd.concat([better, tied, worse])
    xs = np.arange(1, len(order) + 1, dtype=float)
    for xi, r in zip(xs, order.itertuples()):
        if r.delta != 0:
            axb.plot([xi, xi], [r.ape_global, r.ape], color=F.OURS if r.delta < 0 else SHARED,
                     lw=1.1, solid_capstyle="butt", zorder=3)
    is_tie = (order.delta == 0).values
    axb.plot(xs[~is_tie], order.ape_global.values[~is_tie], ls="none", marker="o", ms=4.4,
             mfc="white", mec=SHARED, mew=1.0, zorder=4)
    # a tie: the blue dot sits inside a wider grey ring, so both constants are still visible
    axb.plot(xs[is_tie], order.ape_global.values[is_tie], ls="none", marker="o", ms=6.2,
             mfc="white", mec=SHARED, mew=1.0, zorder=4)
    axb.plot(xs, order.ape.values, ls="none", marker="o", ms=4.2, mfc=F.OURS, mec="white",
             mew=0.4, zorder=5)
    # the two medians, labelled at the right-hand end of their lines
    x_hi = len(order) + 4.5
    for v, col, ls in ((gc["median_ape"], SHARED, "--"), (fa["median_ape"], F.OURS, "-")):
        axb.plot([0.2, len(order) + 0.75], [v, v], color=col, lw=0.9, ls=ls, zorder=2)
        axb.text(len(order) + 0.95, v, f"{v:.1f}%", ha="left", va="center", fontsize=F.FS_MIN,
                 color=col, fontweight="bold" if col == F.OURS else "normal")
    for xv in (nb + 0.5, nb + nt + 0.5):
        axb.axvline(xv, color="#C9CCCF", lw=0.6, zorder=1)
    axb.set_xlim(0.2, x_hi)
    axb.set_ylim(0, 136)
    axb.grid(axis="x", visible=False)
    axb.set_xticks([xs[:nb].mean(), xs[nb:nb + nt].mean(), xs[nb + nt:].mean()])
    # short labels: "tied" has only four compounds' width, and the key names the two constants
    axb.set_xticklabels([f"family better ({nb})", f"tied ({nt})", f"shared better ({nw})"],
                        fontsize=F.FS_MIN)
    for lab, col in zip(axb.get_xticklabels(), (F.OURS, "#444444", "#555B61")):
        lab.set_color(col)
    axb.tick_params(axis="x", which="both", length=0, pad=2.5)
    axb.set_xlabel(f"{fa['n']} held-out compounds, ordered by the change in error", labelpad=2.0)
    axb.set_ylabel("held-out error (%)", labelpad=1.5)
    hb = [Line2D([], [], ls="--", lw=0.9, color=SHARED, marker="o", ms=4.4, mfc="white",
                 mec=SHARED, mew=1.0, label="shared constant"),
          Line2D([], [], ls="-", lw=0.9, color=F.OURS, marker="o", ms=4.2, mfc=F.OURS,
                 mec="white", mew=0.4, label="family constant")]
    lgb = axb.legend(handles=hb, title="Each line: one compound, held out\nHorizontal lines: medians",
                     title_fontsize=F.FS_MIN, alignment="left", loc="upper center",
                     bbox_to_anchor=(0.50, 0.985), fontsize=F.FS_MIN, handlelength=2.6,
                     labelspacing=0.35, borderaxespad=0.0)
    lgb.get_title().set_linespacing(1.25)
    F.panel_label(axb, "b", loc="upper right")

    # ---- (c) held-out fits as curves, chosen by CURVE_RULE ----------------------------------------
    wide = wc > 35.0
    drawn = [curve_panel(ax, sel, q, k == 0, wide=wide)
             for k, (ax, q) in enumerate(zip(axes_c, shown.itertuples()))]
    C.shared_logy(axes_c, drawn, pad=1.18)
    for ax in axes_c:
        ax.set_xlabel("$T$ (K)", labelpad=1.0)
    axes_c[0].set_ylabel(F.KAPPA_LABEL, labelpad=1.5)
    hc, _ = C.legend_handles()
    fig.legend(handles=hc, labels=CURVE_LABELS, loc="lower left", bbox_to_anchor=(x0 / W, YL / H),
               ncol=4, frameon=False, fontsize=F.FS_MIN, handlelength=1.8, columnspacing=1.3,
               borderaxespad=0.0, borderpad=0.0)
    F.panel_text(axes_c[0], x0 / W, (YH + 0.2) / H, "c", ha="left", va="bottom",
                 transform=fig.transFigure)
    fig.text((x0 + 8.5) / W, (YH + 0.7) / H, head_c,
             ha="left", va="bottom", fontsize=F.FS_MIN, fontweight="bold", color="#222222")

    # ---- supplement figure: ALL eligible held-out fits, in order of error -------------------------
    n_col = N_COL_ALL
    wc = (W - x0 - 1.0 - (n_col - 1) * gap) / n_col
    HB = 27.0
    n_row = int(np.ceil((len(ok) + 1) / n_col))            # one cell is kept for the key
    RG = 7.0                                               # gap between the rows (tick labels)
    H2 = YB + n_row * HB + (n_row - 1) * RG + 6.5
    fig2 = plt.figure(figsize=(W * F.MM, H2 * F.MM))
    axes2 = []
    for k in range(len(ok)):
        rr, cc = divmod(k, n_col)
        yb = YB + (n_row - 1 - rr) * (HB + RG)
        axes2.append(fig2.add_axes([(x0 + cc * (wc + gap)) / W, yb / H2, wc / W, HB / H2]))
    drawn2 = [curve_panel(ax, sel, q, k % n_col == 0)
              for k, (ax, q) in enumerate(zip(axes2, ok.itertuples()))]
    C.shared_logy(axes2, drawn2, pad=1.18)
    for k, ax in enumerate(axes2):
        if k % n_col == 0:
            ax.set_ylabel(F.KAPPA_LABEL, labelpad=1.5)
        if k + n_col >= len(ok):                           # nothing below this panel
            ax.set_xlabel("$T$ (K)", labelpad=1.0)
    # the key sits in the empty last cell
    rr, cc = divmod(len(ok), n_col)
    fig2.legend(handles=hc, labels=["measured", "published\ncalculation",
                                    "family constant,\nheld out",
                                    "temperature of a\npublished calculation"],
                loc="lower left",
                bbox_to_anchor=((x0 + cc * (wc + gap) + 0.5) / W, (YB + 1.0) / H2),
                ncol=1, frameon=False, fontsize=F.FS_MIN, handlelength=1.6, labelspacing=0.55,
                borderaxespad=0.0, borderpad=0.0)
    fig2.text(x0 / W, (H2 - 1.0) / H2,
              f"All {len(ok)} eligible held-out fits in validated families "
              f"(≥{C.MIN_POINTS} temperatures spanning ≥{C.MIN_SPAN:.0f} K), in order of "
              f"held-out error; median {ok.ape.median():.1f}%",
              ha="left", va="top", fontsize=7.5, fontweight="bold", color="#222222")

    print(fam.round(1).to_string())
    print(f"in-sample fits: {dict(one.fam_err.round(1))}")
    print(f"(b) family better {nb}, tied {nt}, shared better {nw}")
    print("(c) shown: " + ", ".join(f"{q.compound} {q.ape:.1f}%" for q in shown.itertuples()))
    print(f"figure sizes: {W:.0f} x {H:.1f} mm and {W:.0f} x {H2:.1f} mm")
    F.save(fig, "fig09_family")
    F.save(fig2, "fig09_family_curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
