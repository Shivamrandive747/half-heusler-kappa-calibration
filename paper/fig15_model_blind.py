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
    import matplotlib.ticker as mticker
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

    # EXPLICIT MILLIMETRE LAYOUT (2026-10-11 polish): the two parity panels (a) and (c) are the
    # evidence and get the width -- 57 mm squares, up from ~47 -- while the SHAP ranking (b), which
    # only needs room for its labels and eight bars, is narrowed. (d) keeps its two aligned axes.
    F.use_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    W, SQ = F.DOUBLE / F.MM, 57.0
    YC, YA = 10.0, 10.0 + SQ + 14.5          # bottom row / top row
    H = YA + SQ + 1.0
    fig = plt.figure(figsize=(W * F.MM, H * F.MM))

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

    XR = W - 0.8                              # right edge of the right-hand panels
    axa = ax_mm(13.0, YA, SQ, SQ)
    axb = ax_mm(108.0, YA, XR - 108.0, SQ)
    axc = ax_mm(13.0, YC, SQ, SQ)
    # (d) is two aligned bar charts sharing one row per method: median error and within 2x, so the
    # two statistics a table would carry are both read off an axis rather than off a text column
    axd = ax_mm(108.0, YC, XR - 108.0 - 17.0, SQ)
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
             transform=axa.transAxes, ha="right", va="bottom", fontsize=F.FS_MIN, linespacing=1.25,
             zorder=6)
    F.panel_label(axa, "a")

    # ---- (b) mean |SHAP|, signed by the direction of the effect --------------------------------
    # The bar's side and colour say which way a LARGER descriptor value moves the prediction (the
    # sign of the value-SHAP correlation); its length is the mean |SHAP|. No arrow glyphs needed.
    top = sh["top"][:TOP]
    y = np.arange(len(top))
    sgn = [(-1.0 if r["corr_value_shap"] < 0 else 1.0) for r in top]
    vals = [r["mean_abs_shap"] for r in top]
    sv = [g * v for g, v in zip(sgn, vals)]
    cols = [F.OURS if g < 0 else F.VERMILLION for g in sgn]
    axb.barh(y, sv, color=cols, height=0.62, zorder=3)
    vmax = max(vals)
    for i, (x, v) in enumerate(zip(sv, vals)):
        axb.text(x + (vmax * 0.03 if x > 0 else -vmax * 0.03), i, f"{v:.3f}", va="center",
                 ha="left" if x > 0 else "right", fontsize=F.FS_MIN, color="#333333")
    axb.axvline(0, color="#555555", lw=0.7, zorder=4)
    axb.set_yticks(y)
    axb.set_yticklabels([NAME.get(r["descriptor"], r["label"]) for r in top], fontsize=F.FS_MIN)
    axb.set_ylim(len(top) - 0.5, -1.35)
    axb.set_xlim(-vmax * 1.38, vmax * 0.62)
    axb.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{abs(v):.2f}"))
    axb.tick_params(axis="y", which="both", length=0)
    axb.grid(axis="y", visible=False)
    axb.set_xlabel("mean |SHAP| on log$_{10}\\,\\kappa_L^{\\mathrm{BTE}}$")
    axb.text(-vmax * 0.05, -1.05, "larger value lowers $\\kappa_L$", ha="right", va="center",
             fontsize=F.FS_MIN, color=F.OURS, fontweight="bold")
    axb.text(vmax * 0.05, -1.05, "raises", ha="left", va="center", fontsize=F.FS_MIN,
             color=F.VERMILLION, fontweight="bold")
    axb.text(0.02, 0.03, f"CatBoost, {sh['n_rows']} rows,\n{sh['n_descriptors']} descriptors",
             transform=axb.transAxes, ha="left", va="bottom", fontsize=F.FS_MIN, color="#444444",
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
    axc.legend(loc="lower right", fontsize=F.FS_MIN, borderaxespad=0.3, handletextpad=0.2)
    # the empty upper-left triangle (over-prediction of low-kappa compounds by > 2x) carries the
    # headline; nothing is measured there
    axc.text(0.035, 0.86, f"{sm['calibrated']['within_2x']['median']:.1f}% within 2$\\times$",
             transform=axc.transAxes, ha="left", va="top", fontsize=7.5, color=F.OURS,
             fontweight="bold")
    axc.text(0.035, 0.79, f"in domain,\nmedian of {it['n_seeds']} seeds", transform=axc.transAxes, ha="left",
             va="top", fontsize=F.FS_MIN, color="#444444", linespacing=1.2)
    if n_off:
        print(f"  (c) {n_off} point(s) drawn on the axis edge")
    F.panel_label(axc, "c")

    # ---- (d) against the alternatives: one dot-and-whisker panel (Table 5's numbers) ---------
    acc = N["accuracy_summary"]["routes"]
    assert "_unavailable" not in N["accuracy_summary"], "accuracy_summary unavailable"
    spec = [("ml_shared_in_domain", "calibrated (this work)", F.OURS, True),
            ("ml_uncorrected_in_domain", "uncorrected model", F.VERMILLION, True),
            ("null_power_law_in_domain", "power-law null", "#7F868C", False),
            ("null_constant_in_domain", "constant-$\\kappa$ null", "#7F868C", False)]
    rows = sorted(spec, key=lambda r: acc[r[0]]["median_ape"])   # by value, never by name
    for i, (key, lab, col, seeded) in enumerate(rows):
        r = acc[key]
        lo_, hi_ = r["ci95_median_ape"]
        axd.plot([lo_, hi_], [i, i], color=col, lw=0.9, zorder=3, solid_capstyle="butt")
        for e in (lo_, hi_):
            axd.plot([e, e], [i - 0.12, i + 0.12], color=col, lw=0.9, zorder=3)
        if seeded:
            s0, s1 = r["seed_range"]["median_ape"]
            axd.plot([s0, s1], [i, i], color=col, lw=4.2, alpha=0.45, zorder=4, solid_capstyle="butt")
        ours = key == "ml_shared_in_domain"
        axd.scatter([r["median_ape"]], [i], s=34 if ours else 24, color=col, edgecolors="white",
                    lw=0.6, zorder=5)
        axd.annotate(f"{r['median_ape']:.1f}%", (r["median_ape"], i), xytext=(0, 5.5),
                     textcoords="offset points", ha="center", va="bottom", fontsize=F.FS_MIN,
                     color=col if ours else "#333333", fontweight="bold" if ours else "normal")
        axd.text(1.03, i, f"{r['within_2x_pct']:.1f}%", transform=axd.get_yaxis_transform(),
                 ha="left", va="center", fontsize=F.FS_MIN, color=col if ours else "#333333",
                 fontweight="bold" if ours else "normal")
    axd.text(1.03, -0.85, "within\n2$\\times$", transform=axd.get_yaxis_transform(), ha="left",
             va="center", fontsize=F.FS_MIN, color="#444444", linespacing=1.1)
    axd.set_yticks(range(len(rows)))
    axd.set_yticklabels([r[1] for r in rows], fontsize=F.FS_MIN)
    axd.tick_params(axis="y", which="both", length=0)
    axd.grid(axis="y", visible=False)
    axd.set_ylim(len(rows) - 0.2, -1.35)
    xs = [v for r in rows for v in acc[r[0]]["ci95_median_ape"]]
    axd.set_xlim(max(0, min(xs) - 6), max(xs) + 6)
    axd.set_xlabel("median error (%), held out")
    pc, pp = it["paired_p_seed_median"]["constant"], it["paired_p_seed_median"]["power"]
    ptxt = (f"paired Wilcoxon vs each null: p = {pc:g}" if pc == pp
            else f"paired Wilcoxon: p = {pc:g} (constant), {pp:g} (power law)")
    axd.text(0.02, 0.975, f"{it['n']} in-domain compounds, held out\n" + ptxt,
             transform=axd.transAxes, ha="left", va="top", fontsize=F.FS_MIN, color="#444444",
             linespacing=1.25)
    axd.text(0.02, 0.025, "thin: 95% bootstrap interval\nthick: seed range",
             transform=axd.transAxes, ha="left", va="bottom", fontsize=F.FS_MIN, color="#444444",
             linespacing=1.25)
    F.panel_label(axd, "d", loc="lower right")

    F.save(fig, "fig15_model_blind")
    print(f"  (a) {len(V300)} drawn; registry {mv['median_ape']}% / {mv['within_2x_pct']}% "
          f"/ {mv['n_compounds']} compounds / {mv['n_clusters']} clusters")
    print("  (b) " + ", ".join(f"{r['descriptor']} {r['mean_abs_shap']}" for r in top))
    print(f"  (c) seed-0 within 2x {ins.ratio.between(.5, 2).mean() * 100:.1f}% "
          f"(registry median of seeds {sm['calibrated']['within_2x']['median']})")
    print("  (d) " + "; ".join(f"{k} {acc[k]['median_ape']} / {acc[k]['within_2x_pct']}" for k, *_ in rows)
          + f"; p {pc}/{pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
