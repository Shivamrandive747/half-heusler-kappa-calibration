"""Figure -- from structure to measurement: the model, what it learned, and the end-to-end blind test.

Four panels, one argument, every number read from an artefact (nothing typed):
  (a) STEP 1, MODEL vs PUBLISHED CALCULATION. Each compound's model prediction against its published
      full-BTE value at 300 K, with its whole X-site chemistry cluster removed from training
      (bte_generator_validation.csv, the source fig10_ml (a) draws). The statistics are the registry's
      ml_generator_validation, per compound over all temperatures.
  (b) WHAT THE MODEL USES. The eight largest mean |SHAP| attributions of log10 kappa_BTE, read from the
      registry's shap_ranking (written by fig11_shap.py; the beeswarm stays in the supplement).
  (c) STEP 1 + STEP 2, BLIND AGAINST MEASUREMENT. The points of fig04_blind (a), drawn by the same
      function (fig04_blind.blind_points: seed-0 file, calibration refitted without the compound's own
      chemistry cluster). The within-2x figure printed on the panel is the registry's median of five
      seeds, not the seed-0 value.
  (d) AGAINST THE ALTERNATIVES. Median error of the calibrated route, the uncorrected model and the two
      held-out nulls, with within-2x on every row and the paired Wilcoxon p -- the in_domain_table of
      the registry, drawn instead of tabulated. Model rows are medians of five seeds, whiskers their
      range; the nulls are seed-free.

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
TOP = 8

# Readable descriptor names. Words only -- the values come from the registry.
NAME = {
    "is_full_heusler": "full-Heusler indicator",
    "r_X": "atomic radius, X site",
    "bond_min": "shortest bond length",
    "bond_mean": "mean bond length",
    "r_sum": "sum of atomic radii",
    "n_atoms_conv": "atoms per conventional cell",
    "X_X": "electronegativity, X site",
    "M_Y": "atomic mass, Y site",
    "r_mean": "mean atomic radius",
    "r_Z": "atomic radius, Z site",
    "volume_per_atom": "volume per atom",
    "inv_T": "inverse temperature",
    "M_X": "atomic mass, X site",
    "M_Z": "atomic mass, Z site",
    "gamma_mass": "mass disorder",
}


def ticks(axis, values):
    import matplotlib.ticker as mticker
    axis.set_major_locator(mticker.FixedLocator(values))
    axis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in values]))
    axis.set_minor_locator(mticker.NullLocator())


def main() -> int:
    import matplotlib.pyplot as plt
    from fig04_blind import blind_points

    N = json.load(open(f"{EX}/paper_numbers.json"))
    mv = N["ml_generator_validation"]
    sh = N["shap_ranking"]
    it = N["in_domain_table"]
    sm = it["seed_medians"]
    for k in ("ml_generator_validation", "shap_ranking", "in_domain_table"):
        assert "_unavailable" not in N[k], f"registry {k} unavailable -- run make_paper_numbers.py"

    V = pd.read_csv(f"{EX}/bte_generator_validation.csv")
    V300 = V[V["T"] == 300].groupby("compound").agg(pub=("k_published", "median"),
                                                     gen=("k_generated", "median"))
    R = blind_points()
    ins, out = R["ins"], R["out"]
    # the blind-test table's n and the drawn in-domain set must be the same compounds' count
    assert len(ins) == int(it["n"]), f"drawn in-domain n {len(ins)} != registry {it['n']}"

    F.use_style()
    fig = plt.figure(figsize=(F.DOUBLE, 118 * F.MM))
    # the left column is sized to the square parity panels, so no white space is left beside them
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.9])
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    # (d) is two aligned bar charts sharing one row per method: median error and within 2x, so the
    # two statistics a table would carry are both read off an axis rather than off a text column
    gd = gs[1, 1].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.04)
    axd = fig.add_subplot(gd[0])
    axw = fig.add_subplot(gd[1], sharey=axd)
    unit = "(W m$^{-1}$ K$^{-1}$)"

    # ---- (a) model vs published calculation ----------------------------------------------------
    lo, hi = 0.3, 160.0
    v = np.concatenate([V300.pub.values, V300.gen.values])
    assert v.min() >= lo and v.max() <= hi, f"(a) data {v.min():.2f}-{v.max():.2f} outside {lo}-{hi}"
    F.band_2x(axa, lo, hi, color=F.BAND, label=None)
    axa.scatter(V300.pub, V300.gen, s=9, c=F.OURS, lw=0.3, edgecolors="white", zorder=3)
    F.parity_axes(axa, lo, hi, f"published calculation $\\kappa_L^{{\\mathrm{{BTE}}}}$ {unit}",
                  f"model, cluster held out {unit}")
    for ax_ in (axa.xaxis, axa.yaxis):
        ticks(ax_, (0.5, 1, 2, 5, 10, 20, 50, 100))
    axa.text(0.965, 0.035,
             f"{mv['median_ape']:.1f}% median error\n{mv['within_2x_pct']:.0f}% within 2$\\times$\n"
             f"{mv['n_compounds']} compounds, {mv['n_clusters']} clusters\n"
             f"{len(V300)} drawn (those with a 300 K value)",
             transform=axa.transAxes, ha="right", va="bottom", fontsize=6.3, linespacing=1.25,
             zorder=6)
    F.panel_label(axa, "a")

    # ---- (b) mean |SHAP| -------------------------------------------------------------------------
    top = sh["top"][:TOP]
    y = np.arange(len(top))
    vals = [r["mean_abs_shap"] for r in top]
    axb.barh(y, vals, color=F.OURS, height=0.62, zorder=3)
    for i, r in enumerate(top):
        arrow = "$\\downarrow$" if r["corr_value_shap"] < 0 else "$\\uparrow$"
        axb.text(r["mean_abs_shap"] + max(vals) * 0.02, i, f"{r['mean_abs_shap']:.3f} {arrow}",
                 va="center", ha="left", fontsize=6.3, color="#333333")
    axb.set_yticks(y)
    axb.set_yticklabels([NAME.get(r["descriptor"], r["label"]) for r in top], fontsize=6.8)
    axb.set_ylim(len(top) - 0.5, -0.6)
    axb.set_xlim(0, max(vals) * 1.32)
    axb.tick_params(axis="y", which="both", length=0)
    axb.grid(axis="y", visible=False)
    axb.set_xlabel("mean |SHAP| on log$_{10}\\,\\kappa_L^{\\mathrm{BTE}}$")
    axb.text(0.985, 0.04,
             f"CatBoost, {sh['n_rows']} rows, {sh['n_descriptors']} descriptors\n"
             "$\\downarrow$ larger value lowers $\\kappa_L$",
             transform=axb.transAxes, ha="right", va="bottom", fontsize=6.3, color="#444444",
             linespacing=1.3)
    F.panel_label(axb, "b", loc="upper right")

    # ---- (c) blind test against measurement ----------------------------------------------------
    F.warn_clip(np.concatenate([ins.ref, ins.cal, out.ref, out.cal]), "fig15 (c)")
    clo, chi = F.KAPPA_LIM
    F.band_2x(axc, clo, chi, color=F.BAND, label=None)
    axc.scatter(out.ref, out.cal, s=17, facecolors="none", edgecolors="#8C8C8C", lw=0.8,
                marker="s", zorder=3, label=f"outside domain ({len(out)})")
    axc.scatter(ins.ref, ins.cal, s=17, c=F.OURS, marker="o", lw=0.35, edgecolors="white",
                zorder=4, label=f"in domain ({len(ins)})")
    F.parity_axes(axc, clo, chi, f"measured $\\kappa_L$ {unit}", f"predicted $\\kappa_L$ {unit}")
    F.kappa_ticks(axc)
    n_off = sum(F.edge_markers(axc, g.ref.values, g.cal.values, axis=a, color=col)
                for g, col in ((out, "#8C8C8C"), (ins, F.OURS)) for a in ("y", "x"))
    axc.legend(loc="lower right", fontsize=6.3, borderaxespad=0.3, handletextpad=0.2)
    # the empty upper-left triangle (over-prediction of low-kappa compounds by > 2x) carries the
    # headline; nothing is measured there
    axc.text(0.035, 0.86, f"{sm['calibrated']['within_2x']['median']:.1f}% within 2$\\times$",
             transform=axc.transAxes, ha="left", va="top", fontsize=6.8, color=F.OURS,
             fontweight="bold")
    axc.text(0.035, 0.79, f"in domain,\nmedian of {it['n_seeds']} seeds", transform=axc.transAxes, ha="left",
             va="top", fontsize=6.3, color="#444444", linespacing=1.2)
    if n_off:
        print(f"  (c) {n_off} point(s) drawn on the axis edge")
    F.panel_label(axc, "c")

    # ---- (d) against the alternatives ----------------------------------------------------------
    cal, unc = sm["calibrated"], sm["uncorrected_surrogate"]
    nl_ = it["nulls"]
    # (label, median error, its seed range, within 2x, its seed range, bar style); nulls are seed-free
    rows = [
        ("calibrated\n(this work)", cal["median_ape"]["median"], cal["median_ape"]["range"],
         cal["within_2x"]["median"], cal["within_2x"]["range"], dict(color=F.OURS)),
        ("uncorrected\nmodel", unc["median_ape"]["median"], unc["median_ape"]["range"],
         unc["within_2x"]["median"], unc["within_2x"]["range"],
         dict(color="white", edgecolor=F.RAW_DFT, hatch="//////", lw=0.8)),
        ("power-law\nnull", nl_["power"]["median_ape"], None, nl_["power"]["within_2x_pct"], None,
         dict(color=F.REFUSED)),
        ("constant-$\kappa$\nnull", nl_["constant"]["median_ape"], None,
         nl_["constant"]["within_2x_pct"], None, dict(color=F.REFUSED)),
    ]
    rows.sort(key=lambda r: r[1])                          # sorted by value, never by name
    yy = np.arange(len(rows))
    xmax = max(max(r[1], (r[2] or [0, 0])[1]) for r in rows)

    def whisker(ax_, rng, i):
        ax_.plot(rng, [i, i], color="#222222", lw=0.8, zorder=4, solid_capstyle="butt")
        for e in rng:
            ax_.plot([e, e], [i - 0.13, i + 0.13], color="#222222", lw=0.8, zorder=4)

    for i, (lab, med, rng, w2, w2rng, sty) in enumerate(rows):
        axd.barh(i, med, height=0.6, zorder=3, **sty)
        if rng:
            whisker(axd, rng, i)
        ours = "this work" in lab
        tk = dict(fontsize=6.8, fontweight="bold" if ours else "normal",
                  color=F.OURS if ours else "#333333")
        axd.text(max(med, rng[1] if rng else 0) + xmax * 0.03, i, f"{med:.1f}%", va="center",
                 ha="left", **tk)
        # within 2x: the same row, its own axis. The value sits at the bar's root, clear of the
        # whisker at its tip; on a hatched bar it gets a white ground so the hatch misses the digits
        axw.barh(i, w2, height=0.6, zorder=3, **sty)
        if w2rng:
            whisker(axw, w2rng, i)
        axw.text(4, i, f"{w2:.1f}%", va="center", ha="left", zorder=6,
                 **{**tk, "color": "white" if ours else "#222222"},
                 bbox=(dict(fc="white", ec="none", pad=0.6) if "hatch" in sty else None))
    axd.set_yticks(yy)
    axd.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    for ax_ in (axd, axw):
        ax_.tick_params(axis="y", which="both", length=0)
        ax_.grid(axis="y", visible=False)
    axw.tick_params(axis="y", labelleft=False)
    axd.set_ylim(len(rows) - 0.45, -1.15)
    axd.set_xlim(0, xmax * 1.22)
    axd.set_xticks([t for t in (0, 10, 20, 30, 40, 50) if t <= xmax * 1.22])
    axd.set_xlabel("median error (%)")
    axw.set_xlim(0, 100)
    axw.set_xticks([0, 25, 50, 75, 100])
    axw.set_xlabel("within 2$\\times$ (%)")
    pc, pp = it["paired_p_seed_median"]["constant"], it["paired_p_seed_median"]["power"]
    ptxt = (f"paired Wilcoxon vs each null: p = {pc:g}" if pc == pp
            else f"paired Wilcoxon: p = {pc:g} (constant), {pp:g} (power law)")
    axd.text(0.03, 0.975, f"{it['n']} in-domain compounds, held out\n" + ptxt,
             transform=axd.transAxes, ha="left", va="top", fontsize=6.3, color="#444444",
             linespacing=1.25)
    axw.text(0.04, 0.975, f"model rows: median of\n{it['n_seeds']} seeds; whisker = range",
             transform=axw.transAxes, ha="left", va="top", fontsize=6.3, color="#444444",
             linespacing=1.25)
    F.panel_label(axw, "d", loc="lower right")

    F.save(fig, "fig15_model_blind")
    print(f"  (a) {len(V300)} drawn; registry {mv['median_ape']}% / {mv['within_2x_pct']}% "
          f"/ {mv['n_compounds']} compounds / {mv['n_clusters']} clusters")
    print("  (b) " + ", ".join(f"{r['descriptor']} {r['mean_abs_shap']}" for r in top))
    print(f"  (c) seed-0 within 2x {ins.ratio.between(.5, 2).mean() * 100:.1f}% "
          f"(registry median of seeds {sm['calibrated']['within_2x']['median']})")
    print("  (d) " + "; ".join(f"{r[0].replace(chr(10), ' ')} {r[1]} / {r[3]}" for r in rows)
          + f"; p {pc}/{pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
