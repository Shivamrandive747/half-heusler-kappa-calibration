"""Figure 6 -- everything the method can say, and what each statement is worth.

One panel, FOUR tiers, ordered by how much evidence stands behind them:

  ISSUED       the compounds the method predicts. Bonding family calibrated, structure on record,
               sources in agreement. These are the ones the abstract commits to.
  FLAGGED      those that clear the family test but fail one specific check -- a non-cubic
               polymorph on record, or a value outside the range the family has ever been
               measured in. Section 5 is explicit that these are "flagged rather than issued":
               the number is reported with its reason, and is NOT a prediction the paper makes.
  CONDITIONAL  compounds that pass every structural and chemical screen but sit in a family
               with no usable measured anchor. Ten are Ni-Bi, which needs TWO sound
               measurements: its only measured member, YNiBi, carries a bipolar contribution its
               own authors report, so the family stands at zero usable anchors rather than one.

Tier SIZES are deliberately not written here. They are counted from the artefacts at draw time and
printed to stdout, because every count this docstring once carried (5 / 5 / 10) went stale when the
calibration changed which compounds qualify.
The REFUSED compounds are deliberately NOT drawn. They contain thorium or uranium, have VEC != 18,
or carry calculations that disagree by more than 2x, so for them the calibrated number is meaningless -- plotting it on a kappa axis,
with a conformal interval, would assert that a quantity we decline to give is nonetheless
estimated to lie in a range. They are named in the caption with the screen each fails. Flagged and
refused together are the "declined" of the abstract; only the issued tier carries a filled
prediction marker.

Drawn as one figure because the reader's question is "what do you actually know, and how well?"
-- which is a question about the whole ladder, not about any one rung.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import figlib as F

PRED = "data/Target_Materials/PAPER_PREDICTIONS.csv"
COND = "data/Target_Materials/CONDITIONAL_PREDICTIONS.csv"
COND_COL = "#8FB8CE"     # conditional: blue-leaning, predictable once a measurement exists
FLAG_COL = "#8A9BA8"     # flagged: grey-leaning, declined -- must NOT read as issued blue


def short_reason(note: str) -> str:
    """The one word that says why a compound is not in the clean tier."""
    n = str(note)
    if n.startswith("POLYMORPHIC"):
        return "polymorphic"
    if n.startswith("MARGINAL"):
        return "above/below family span"
    if n.startswith("EXTRAP"):
        return "extrapolation"
    if n.startswith("FAMILY CALIBRATION FAILS"):
        return "family fails held out"
    if "VEC" in n:
        return "VEC $\\neq$ 18"
    if "radioactive" in n or "actinide" in n:
        return "radioactive"
    if "disagree" in n or "factor of" in n:
        return "sources disagree"
    return "flagged"


def main() -> int:
    import matplotlib.pyplot as plt

    P = pd.read_csv(PRED)
    C = pd.read_csv(COND)
    # the axis is room-temperature kappa: a row quoted at another temperature (500 K) is left
    # off, as the conditional ones quoted there already are, and counted in the printout
    _off = P[P.status.isin(["ISSUED", "FLAGGED"]) & P.kappa_pred_300.isna()]
    if len(_off):
        print(f"  not on the 300 K axis (quoted at another temperature): {sorted(_off.compound)}")
    iss = P[(P.status == "ISSUED") & P.kappa_pred_300.notna()].sort_values("kappa_pred_300")
    flg = P[(P.status == "FLAGGED") & P.kappa_pred_300.notna()].sort_values("kappa_pred_300")
    # Four conditional rows are quoted at 500 K, where their only calculation exists, and carry no
    # 300 K value. This panel is a room-temperature comparison, so they are excluded from it rather
    # than plotted on a mixed abscissa; the count is printed below so the omission is visible.
    _n_all = len(C)
    C = C[C.kappa_pred_300.notna()]
    con = C.sort_values(["family", "kappa_pred_300"])
    if len(C) < _n_all:
        print(f"  {_n_all - len(C)} conditional row(s) quoted away from 300 K, not plotted here")
    ref = P[P.status == "REFUSED"].sort_values("kappa_pred_300")
    print(f"  issued {len(iss)}   flagged {len(flg)}   conditional {len(con)}   "
          f"refused {len(ref)}")
    # every row is either drawn or deliberately left off the 300 K axis -- nothing else may vanish
    assert len(iss) + len(flg) + len(ref) + len(_off) == len(P), "a status was dropped from the ladder"

    F.use_style()
    # the row count follows what is actually drawn, after the bulk families are collapsed; at
    # ONEHALF width the figure scaled past the height of the text block in a two-column float
    _bulk = [f for f, g in con.groupby('family') if len(g) >= 5]
    n = len(iss) + len(flg) + len(con[~con.family.isin(_bulk)]) + len(_bulk) + 3
    fig, ax = plt.subplots(figsize=(F.DOUBLE, 0.215 * n + 1.6))

    y, ticks, labs, seps = 0, [], [], []

    def draw(r, col, filled, bold=False, note=""):
        ax.plot([r.lo90, r.hi90], [y, y], lw=1.0, color=col, alpha=.55, zorder=3)
        ax.plot([r.lo50, r.hi50], [y, y], lw=2.9, color=col, alpha=.92, zorder=4)
        ax.plot([r.kappa_pred_300], [y], marker="o", ms=4.4, mew=1.3, ls="none",
                mfc=(col if filled else "white"), mec=col, zorder=5)
        ax.text(r.hi90 * 1.05, y, f"{r.kappa_pred_300:.1f}", va="center", fontsize=6.6,
                color=col, fontweight="bold" if bold else "normal")
        if note:
            ax.text(r.hi90 * 1.05 * 1.42, y, note, va="center", fontsize=5.9,
                    color="#8A8A8A", fontstyle="italic")

    # ---- tier 1: issued outright -------------------------------------------------
    for r in iss.itertuples():
        # an anchored family's prediction rests on ONE measured compound; the panel says so
        _nt = str(getattr(r, "note", "") or "")
        draw(r, F.OURS, True, bold=True, note="anchored (one member)" if _nt.startswith("ANCHORED") else "")
        ticks.append(y); labs.append(f"{r.compound}  ({r.family})"); y -= 1
    seps.append(y + 0.5); y -= 0.55

    # ---- tier 2: flagged -- declined with a named reason, NOT issued -------------
    for r in flg.itertuples():
        draw(r, FLAG_COL, False, note=short_reason(r.note))
        ticks.append(y); labs.append(f"{r.compound}  ({r.family})"); y -= 1
    seps.append(y + 0.5); y -= 0.55

    # ---- tier 3: conditional on a family measurement -----------------------------
    #
    # A FAMILY THAT CONTRIBUTES MANY NEAR-IDENTICAL ROWS IS DRAWN ONCE. Ten Ni-Bi compounds, all
    # carrying the same global constant because the family has no measured member, produced ten
    # rows that differ only in their calculated input and pushed the figure past the height of the
    # text block. They are one statement -- "this family has no usable measurement, and here is
    # the span it would predict" -- and Table S1 carries the ten values individually.
    BULK_AT = 5
    big = [f for f, g in con.groupby("family") if len(g) >= BULK_AT]
    for r in con[~con.family.isin(big)].itertuples():
        cav = str(getattr(r, "caveat", "") or "")
        note = "Sm valence" if cav.startswith("FLAGGED") else ""
        draw(r, COND_COL, False, note=note)
        ticks.append(y); labs.append(f"{r.compound}  ({r.family})"); y -= 1
    for fam in big:
        g = con[con.family == fam]
        lo, hi = float(g.kappa_pred_300.min()), float(g.kappa_pred_300.max())
        ax.plot([float(g.lo90.min()), float(g.hi90.max())], [y, y], lw=1.0, color=COND_COL,
                alpha=.40, zorder=3)
        ax.plot([lo, hi], [y, y], lw=2.9, color=COND_COL, alpha=.92, zorder=4,
                solid_capstyle="butt")
        for v in (lo, hi):
            ax.plot([v], [y], marker="|", ms=7, mew=1.3, color=COND_COL, zorder=5)
        ax.text(float(g.hi90.max()) * 1.05, y, f"{lo:.1f}–{hi:.1f}", va="center",
                fontsize=6.6, color=COND_COL)
        ax.text(float(g.hi90.max()) * 1.05 * 1.9, y, "no measured member in the family",
                va="center", fontsize=5.9, color="#8A8A8A", fontstyle="italic")
        ticks.append(y); labs.append(f"{len(g)} compounds  ({fam})"); y -= 1
        print(f"  {fam}: {len(g)} conditional rows drawn as one bracket, {lo:.1f}-{hi:.1f}")

    # tier 4 -- REFUSED -- is intentionally not drawn. See the module docstring: for a compound
    # refused on physics the calibrated value is not a quantity, and giving it an interval would
    # claim more than we are willing to claim. The caption names them instead.

    for s in seps:
        ax.axhline(s, color="#CFCFCF", lw=.6, ls=":")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=6.6)
    for t, v in zip(ax.get_yticklabels(), ticks):
        if v < seps[1]:
            t.set_color("#5A7F94")           # conditional
        elif v < seps[0]:
            t.set_color("#6E7F8A")           # flagged -- declined, not the issued blue
    ax.set_xscale("log")
    # limits from the data: CrSnPt sits at 0.15 and was being clipped off the left edge
    # limits from what is DRAWN. The refused compounds are no longer plotted, and leaving
    # them in this list stretched the axis to cover a value the figure does not show.
    drawn = pd.concat([iss, flg])
    allv = (list(drawn.lo90.dropna()) + list(drawn.hi90.dropna())
            + list(C.lo90) + list(C.hi90))
    # The shared paper-wide conductivity scale. The fitted range this replaced already sat close to
    # it; pinning it means a 4 W/m/K prediction here lands where 4 W/m/K lands in Figures 2-5.
    F.warn_clip(allv, "fig06 intervals")
    # the anchored Co-Bi prediction's 90% band (5.6-34.9) overran the shared KAPPA_LIM ceiling and
    # its value label with it; the upper limit is taken from the data so nothing is drawn outside
    _hi = max(float(P.hi90.max()), float(C.hi90.max()) if len(C) else 0.0)
    ax.set_xlim(F.KAPPA_LIM[0], max(F.KAPPA_LIM[1], _hi * 3.2))
    F.kappa_ticks(ax, both=False)
    ax.set_ylim(y + 0.6, 0.9)
    ax.set_xlabel("predicted $\\kappa_L$ at 300 K  (W m$^{-1}$ K$^{-1}$)")

    def frac(v):
        return (v - (y + 0.6)) / (0.9 - (y + 0.6))

    for anchor, text, col in (
            # counts on the 300 K axis out of the totals: the 500 K rows are named in the caption
            (0.55, f"ISSUED  ({len(iss)} of {int((P.status == 'ISSUED').sum())} at 300 K)", F.OURS),
            (seps[0] - 0.45, f"FLAGGED  ({len(flg)} of {int((P.status == 'FLAGGED').sum())} at 300 K)"
             f"   declined \u2013 reason given",
             FLAG_COL),
            # "no measured member" was false for all of them -- Ni-Bi has YNiBi, Co-Bi has ZrCoBi,
            # Ni-Pb has ZrNiPb, Sb-Ir has ZrSbIr. What each family lacks is a member whose
            # measurement the calibration can use, which is the caption's wording too.
            (seps[1] - 0.45,
             f"CONDITIONAL  ({_n_all})   family cannot yet license it"
             + (f"; {_n_all - len(con)} quoted away from 300 K, not drawn" if _n_all > len(con) else ""),
             "#5A7F94")):
        ax.text(.012, frac(anchor), text, transform=ax.transAxes, fontsize=7,
                fontweight="bold", color=col, va="center")

    hl = [plt.Line2D([], [], color=F.OURS, lw=2.9),
          plt.Line2D([], [], color=F.OURS, lw=1.0, alpha=.55),
          plt.Line2D([], [], color=F.OURS, marker="o", mfc=F.OURS, ls="none", ms=4.4),
          plt.Line2D([], [], color=FLAG_COL, marker="o", mfc="white", ls="none", ms=4.4)]
    # placed above the plot rather than over the refused rows it was covering
    # plain "%" -- this style renders text with mathtext, not LaTeX, so "\%" prints the backslash
    ax.legend(hl, ["50% interval", "90% interval", "issued", "not issued"],
              loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
              fontsize=6.2, frameon=False, borderaxespad=.2, handletextpad=.5,
              columnspacing=1.2)
    F.save(fig, "fig06_landscape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
