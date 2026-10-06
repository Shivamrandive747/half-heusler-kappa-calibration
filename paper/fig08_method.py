"""Figure 8 (printed as Figure 1) -- the method end to end.

WHY THE PAPER NEEDS ONE. Sections 2 and 3 describe a corpus, a tiering rule, a regression model, a
transfer function, a holdout protocol and four screens, and a reader has to assemble them from
prose. Most papers of this kind carry a schematic; ours did not.

WHAT IT IS BUILT TO SAY. One thing above all: measurements and calculations are never pooled. They
run in separate lanes from the corpus to the point where the transfer function maps one onto the
other, and that separation is the reason the paper can claim to predict experiment rather than to
reproduce other people's DFT. Drawing them as two lanes makes the claim structural instead of
something the reader has to take on trust from a sentence in Section 2.

REDESIGN (2026-10-05, author request: "more contrasting and easier to follow").
  * Seven numbered steps run along a header strip, one per column. Columns 1-3 carry BOTH lanes, so
    "step 2" is the same operation applied separately to calculations and to measurements -- the
    shared number says "in parallel", the two lanes say "never pooled".
  * Every box is filled by its STAGE FAMILY, keyed by a legend along the foot. The families keep the
    paper's role colours wherever a role exists: calculations stay grey and dashed (provisional),
    measurements are the darkest ink (here a dark fill, the strongest contrast on the page), the
    transfer function wears the accent blue that means "our contribution", and the outcomes use the
    prediction encoding (issued = solid blue, conditional = open/dashed blue, flagged = the
    recessive refusal grey). Only the two stage families with no role elsewhere -- machine learning
    and validation -- take new Okabe-Ito hues (reddish purple, bluish green); the CVD check below
    runs on the full set.
  * The two routes into the transfer function are drawn as two arrows: the published calculation
    is used directly, and the surrogate supplies one only where none is published. A single arrow
    from the surrogate would have told the reader that every calibrated value passes through ML,
    which is false.
  * The dashed green enclosure marks exactly what a fold of the validation refits -- the model AND
    the constants -- because holding out one but not the other is the single easiest way to get
    this kind of result wrong, and this paper has made that mistake once already.
  * No equation numbers: the author asked for none on the figure.

Every count is read from an artefact; nothing in the drawing is typed.
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

import figlib as F

# ---- stage families: (fill, edge, title colour, body colour, edge linestyle) ----------------------
FAM = {
    "calc": dict(fc="#ECEEF0", ec="#6E757C", tc="#3F454B", bc="#2E2E2E", ls=(0, (3.2, 1.8)),
                 label="Calculations (published DFT-BTE)"),
    "meas": dict(fc=F.MEASURED_LIGHT, ec=F.MEASURED, tc="#A33F00", bc="#2E2E2E", ls="-",
                 label="Measurements"),
    "ml":   dict(fc="#F7E4EF", ec=F.PURPLE, tc="#8A3566", bc="#2E2E2E", ls="-",
                 label="Machine learning"),
    "tf":   dict(fc="#DCEBF6", ec=F.OURS, tc=F.OURS, bc="#1C2B38", ls="-",
                 label="Transfer function (this work)"),
    "val":  dict(fc="#D8F1E8", ec=F.GREEN, tc="#00694E", bc="#2E2E2E", ls="-",
                 label="Validation & screening"),
}
INK = "#2B2B2B"
ARROW = "#5F6368"


def corpus_counts() -> dict:
    """The two corpus boxes, read from the training set rather than typed into the drawing.

    They had been hardcoded, and both had drifted: the calculation box claimed 32 sources where the
    corpus held 27 even before duplicated preprints were collapsed, and the measurement box said
    164 papers after every other file in the paper had been corrected to 163. A schematic is the
    last place a reader checks a number and the first place one goes stale.
    """
    import pandas as pd
    from pymatgen.core import Composition

    import source_identity as SI
    from run_loco_chemistry import sites

    seen: dict = {}

    def half(f):
        try:
            r = Composition(str(f)).reduced_formula
        except Exception:  # noqa: BLE001
            return None
        if r not in seen:
            try:
                seen[r] = (sites(r) or ["?"])[0] == "half"
            except Exception:  # noqa: BLE001
                seen[r] = False
        return r if seen[r] else None

    d = SI.dedupe(pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv",
                              low_memory=False))
    d["red"] = d.formula.map(half)
    d["t"] = pd.to_numeric(d.method_tier, errors="coerce")
    d["k"] = pd.to_numeric(d.kappa_L, errors="coerce")
    d = d.dropna(subset=["red"])
    d = d[d.k > 0]
    t1, t0 = d[d.t == 1], d[d.t == 0]
    return dict(calc_n=t1.red.nunique(), calc_src=t1.source_doi.nunique(),
                meas_n=t0.red.nunique(), meas_src=t0.source_doi.nunique())


def main() -> int:
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    C = corpus_counts()
    print(f"  corpus boxes: {C['calc_n']} calculated / {C['calc_src']} sources, "
          f"{C['meas_n']} measured / {C['meas_src']} papers")
    # the in-domain count is read from the registry; it was typed as 38 and outlived the 39
    n_dom = json.load(open("data/exports/kappa_v2/paper_numbers.json"))["in_domain_table"]["n"]
    # THE CONSTANT IS PER FAMILY wherever a family has a measured member. Read, not typed.
    fc = json.load(open("data/exports/kappa_v2/family_calibration.json"))
    fams = fc["families"]
    cs = sorted(v["c"] for v in fams.values())
    gc, gp = fc["global"]["c"], fc["global"]["p"]
    # The status counts are READ, not typed. They were hardcoded at 5/5/10/6 and silently went stale
    # when the calibration changed which compounds qualify.
    P = pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv")
    n_cond = len(pd.read_csv("data/Target_Materials/CONDITIONAL_PREDICTIONS.csv"))
    ns = P.status.value_counts()
    print(f"  in domain {n_dom}; {len(fams)} families, c_f {cs[0]}-{cs[-1]}; shared c={gc} p={gp}; "
          f"issued {ns.get('ISSUED', 0)} flagged {ns.get('FLAGGED', 0)} conditional {n_cond}")

    print("  stage-family edge colours:")
    F.check_cvd([FAM[k]["ec"] for k in FAM] + [F.REFUSED])

    # ---- canvas in MILLIMETRES, so every size below is a print size ----------------------------
    W, H = 190.0, 96.0
    F.use_style()
    fig = plt.figure(figsize=(W * F.MM, H * F.MM), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    TS, BS = 6.9, 6.1          # title / body type, pt -- at 190 mm this prints at full size

    def box(x0, x1, y0, y1, fam, title, body, *, lw=1.0, tfs=TS, bfs=BS, top_pad=2.3,
            gap=1.3, z=3):
        s = FAM[fam]
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=s["fc"], ec=s["ec"], lw=lw, ls=s["ls"], zorder=z))
        xc = (x0 + x1) / 2
        ax.text(xc, y1 - top_pad, title, ha="center", va="top", fontsize=tfs,
                fontweight="bold", color=s["tc"], zorder=z + 1)
        if body:
            # body centred in the space under the title, so a short body does not leave the
            # lower half of a tall box empty
            ytop = y1 - top_pad - tfs * 0.3528 - gap
            ax.text(xc, (ytop + y0 + 1.0) / 2, body, ha="center", va="center",
                    fontsize=bfs, color=s["bc"], zorder=z + 1, linespacing=1.32)

    def arrow(p0, p1, col=ARROW, lw=0.9, rad=0.0, ls="-", z=2):
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>,head_length=2.6,head_width=1.5",
                                     mutation_scale=1, lw=lw, color=col, ls=ls,
                                     shrinkA=0, shrinkB=0, zorder=z))

    def elbow(pts, col=ARROW, lw=0.9, z=2):
        """Orthogonal connector: straight segments, arrowhead on the last one."""
        xs, ys = zip(*pts)
        ax.plot(xs[:-1], ys[:-1], color=col, lw=lw, solid_capstyle="butt", zorder=z)
        arrow(pts[-2], pts[-1], col=col, lw=lw, z=z)

    # ---- columns (mm) ------------------------------------------------------------------------
    X = [(6.5, 28.5), (31.5, 53.5), (56.5, 79.0), (82.5, 113.5), (117.0, 143.0),
         (146.5, 168.5), (171.5, 190.0)]
    cx = [(a + b) / 2 for a, b in X]
    TOP = (60.0, 80.0)          # calculation lane
    BOT = (13.0, 33.5)          # measurement lane
    MID = (37.5, 76.5)          # transfer function, validation, screens
    ym = (MID[0] + MID[1]) / 2

    # ---- numbered step strip ------------------------------------------------------------------
    steps = ["Collect", "Tier / convert", "Surrogate / domain", "Transfer", "Validate",
             "Screen", "Report"]
    ax.add_patch(FancyArrowPatch((cx[0] - 9, 90.5), (W - 0.5, 90.5),
                                 arrowstyle="-|>,head_length=2.6,head_width=1.6",
                                 mutation_scale=1, lw=0.7, color="#C3C7CB", zorder=1))
    for i, (x, name) in enumerate(zip(cx, steps), start=1):
        ax.add_patch(Circle((x, 90.5), 2.5, fc="#3A3F44", ec="white", lw=0.8, zorder=3))
        ax.text(x, 90.5, str(i), ha="center", va="center", fontsize=7, fontweight="bold",
                color="white", zorder=4)
        ax.text(x, 86.4, name, ha="center", va="center", fontsize=6.4, color="#3A3F44",
                fontweight="bold", zorder=4)

    # ---- the holdout enclosure: what one validation fold refits ---------------------------
    ax.add_patch(FancyBboxPatch((X[2][0] - 1.6, MID[0] - 1.6), X[3][1] - X[2][0] + 3.2,
                                TOP[1] + 1.6 - (MID[0] - 1.6),
                                boxstyle="round,pad=0,rounding_size=2.2",
                                fc="none", ec=F.GREEN, lw=0.9, ls=(0, (4, 2.2)), zorder=1.5))
    ax.text(X[2][0] + 0.6, MID[0] + 0.3, "refitted in every fold\nwith the held-out\n"
            "cluster removed", ha="left", va="bottom", fontsize=5.7, color="#00694E",
            fontstyle="italic", linespacing=1.2, zorder=4)

    # ---- lane tags ------------------------------------------------------------------------
    for (y0, y1), fam, name in ((TOP, "calc", "CALCULATIONS"), (BOT, "meas", "MEASUREMENTS")):
        s = FAM[fam]
        ax.add_patch(FancyBboxPatch((0.6, y0), 3.9, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=s["ec"], ec="none", zorder=3))
        ax.text(2.55, (y0 + y1) / 2, name, rotation=90, ha="center", va="center",
                fontsize=5.6, fontweight="bold", color="white", zorder=4)

    # ---- calculations lane ----------------------------------------------------------------
    box(*X[0], *TOP, "calc", "Published $\\kappa_L$",
        f"DFT-BTE values\n{C['calc_n']} half Heuslers\n{C['calc_src']} sources")
    box(*X[1], *TOP, "calc", "Tier by method",
        "tier 1 = full BTE\non DFT forces;\nonly tier 1 is fitted")
    box(*X[2], *TOP, "ml", "ML surrogate",
        "CatBoost, trained\non tier 1; supplies\n$\\kappa^\\mathrm{BTE}$ only where\nnone is published")
    arrow((X[0][1], 70), (X[1][0], 70))
    arrow((X[1][1], 70), (X[2][0], 70))

    # ---- measurements lane ----------------------------------------------------------------
    box(*X[0], *BOT, "meas", "Measured $\\kappa_\\mathrm{tot}$",
        f"{C['meas_n']} half Heuslers\n{C['meas_src']} papers")
    box(*X[1], *BOT, "meas", "Wiedemann–Franz",
        "$\\kappa_L=\\kappa_\\mathrm{tot}-LT/\\rho$\none reference\nper compound", tfs=6.4)
    box(*X[2], *BOT, "meas", "Domain screens",
        f"VEC 18, cubic,\nnon-polymorphic\n→ {n_dom} in domain")
    arrow((X[0][1], 23.25), (X[1][0], 23.25), col=F.MEASURED)
    arrow((X[1][1], 23.25), (X[2][0], 23.25), col=F.MEASURED)

    # ---- transfer function ----------------------------------------------------------------
    s = FAM["tf"]
    ax.add_patch(FancyBboxPatch((X[3][0], MID[0]), X[3][1] - X[3][0], MID[1] - MID[0],
                                boxstyle="round,pad=0,rounding_size=1.6",
                                fc=s["fc"], ec=s["ec"], lw=1.7, zorder=3))
    xt = cx[3]
    ax.text(xt, MID[1] - 2.3, "Transfer function", ha="center", va="top", fontsize=7.4,
            fontweight="bold", color=F.OURS, zorder=4)
    tf_lines = [
        (MID[1] - 8.4, "shared $(c,\\,p)$", 6.1, "bold", "#1C2B38"),
        (MID[1] - 12.4, "$\\kappa^\\mathrm{BTE}\\,\\min[\\,c\\,(T/300)^{p},\\,1\\,]$", 6.4,
         "normal", "#1C2B38"),
        (MID[1] - 16.4, f"$c={gc:.2f},\\ p={gp:.2f}$", 6.1, "normal", "#1C2B38"),
        (MID[1] - 22.0, f"per family $c_f$ ({len(fams)} families)", 6.1, "bold", "#1C2B38"),
        (MID[1] - 26.0, "$\\kappa^\\mathrm{BTE}\\,c_f\\,(T/300)^{p}$", 6.4, "normal",
         "#1C2B38"),
        (MID[1] - 30.0, f"$c_f={cs[0]:.2f}$–${cs[-1]:.2f}$", 6.1, "normal", "#1C2B38"),
        (MID[1] - 35.0, "both fitted on published\ncalculation vs measurement", 5.6, "normal",
         "#41505C"),
    ]
    for y, t, fs, fw, col in tf_lines:
        ax.text(xt, y, t, ha="center", va="center", fontsize=fs, fontweight=fw, color=col,
                zorder=4, linespacing=1.2, fontstyle="italic" if fs < 6 else "normal")
    ax.plot([X[3][0] + 3, X[3][1] - 3], [MID[1] - 19.2] * 2, color="#9CC3E0", lw=0.5, zorder=4)

    # two routes into the transfer function -- published calculation used directly, surrogate
    # only where none is published
    elbow([(cx[1], TOP[0]), (cx[1], 50.0), (X[3][0], 50.0)], col=FAM["calc"]["ec"])
    ax.text(cx[2], 51.0, "published $\\kappa^\\mathrm{BTE}$,\nused directly",
            ha="center", va="bottom", fontsize=5.8, color=FAM["calc"]["tc"], zorder=4,
            linespacing=1.15)
    arrow((X[2][1], 70.0), (X[3][0], 70.0), col=F.PURPLE, lw=1.0)
    # measurements: fit the constants, and score the held-out compounds
    xf = X[3][0] + 6.0
    elbow([(X[2][1], 28.5), (xf, 28.5), (xf, MID[0])], col=F.MEASURED)
    ax.text(xf + 1.2, 32.0, "fit $c$, $p$, $c_f$", ha="left", va="center", fontsize=5.8,
            color=F.MEASURED, zorder=4)
    elbow([(X[2][1], 18.5), (cx[4], 18.5), (cx[4], MID[0])], col=F.MEASURED)
    ax.text((X[2][1] + cx[4]) / 2, 19.5, "score the held-out measured compounds",
            ha="center", va="bottom", fontsize=5.8, color=F.MEASURED, zorder=4)

    # ---- validation and screens -------------------------------------------------------------
    box(*X[4], *MID, "val", "Validation",
        "blind test: one X-site\ncluster withheld from\nmodel AND constants\n\n"
        f"all {C['meas_n']} scored;\nheadline: {n_dom} in domain\n\n"
        "family $c_f$: leave\none compound out", lw=1.1)
    box(*X[5], *MID, "val", "Issue screens",
        "family $c_f$ validated\nor anchored\n\ninside the family's\nmeasured range\n\n"
        "structure and\nsources agree", lw=1.1)
    arrow((X[3][1], ym), (X[4][0], ym), col=F.OURS, lw=1.2)
    arrow((X[4][1], ym), (X[5][0], ym), col=F.GREEN, lw=1.0)

    # ---- outputs ----------------------------------------------------------------------------
    outs = [("ISSUED", ns.get("ISSUED", 0), "passes\nevery check",
             dict(fc=F.OURS, ec=F.OURS, ls="-", lw=1.2), "white", "#E3EEF7"),
            ("CONDITIONAL", n_cond, "awaits named\nmeasurements",
             dict(fc="white", ec=F.OURS, ls=(0, (3, 1.6)), lw=1.1), F.OURS, "#1C2B38"),
            ("FLAGGED", ns.get("FLAGGED", 0), "fails a\nnamed check",
             dict(fc="#E6E8EA", ec="#9AA0A6", ls="-", lw=0.9), "#4D545B", "#4D545B")]
    oh, og = 15.6, 2.4
    y_top = ym + 1.5 * oh + og
    for k, (name, n, desc, st, tcol, dcol) in enumerate(outs):
        y1 = y_top - k * (oh + og)
        y0 = y1 - oh
        ax.add_patch(FancyBboxPatch((X[6][0], y0), X[6][1] - X[6][0], oh,
                                    boxstyle="round,pad=0,rounding_size=1.4", zorder=3, **st))
        ax.text(cx[6], y1 - 1.4, str(n), ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=tcol, zorder=4)
        ax.text(cx[6], y1 - 7.0, name, ha="center", va="top", fontsize=6.0, fontweight="bold",
                color=tcol, zorder=4)
        ax.text(cx[6], y1 - 9.9, desc, ha="center", va="top", fontsize=5.4, color=dcol,
                zorder=4, linespacing=1.15)
        arrow((X[5][1], ym), (X[6][0], (y0 + y1) / 2), col=ARROW, lw=0.8)

    # ---- legend of stage families -----------------------------------------------------------
    lx, ly = 6.5, 4.5
    items = [(FAM[k], FAM[k]["label"]) for k in FAM]
    for s, lab in items:
        ax.add_patch(FancyBboxPatch((lx, ly - 1.7), 5.2, 3.4,
                                    boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=s["fc"], ec=s["ec"], lw=0.9, ls=s["ls"], zorder=3))
        t = ax.text(lx + 6.6, ly, lab, ha="left", va="center", fontsize=6.0, color=INK)
        fig.canvas.draw() if lx == 6.5 else None
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        wmm = bb.width / fig.dpi / F.MM
        lx += 6.6 + wmm + 4.5
    # outputs: the three status swatches together
    for st in (outs[0][3], outs[1][3], outs[2][3]):
        ax.add_patch(FancyBboxPatch((lx, ly - 1.7), 3.4, 3.4,
                                    boxstyle="round,pad=0,rounding_size=0.7", zorder=3, **st))
        lx += 4.2
    ax.text(lx + 1.0, ly, "Outputs: issued / conditional / flagged", ha="left", va="center",
            fontsize=6.0, color=INK)

    F.save(fig, "fig08_method")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
