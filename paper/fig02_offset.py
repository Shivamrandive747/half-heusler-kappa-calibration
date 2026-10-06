"""Figure 2 -- the DFT-to-experiment offset, before and after the transfer function.

This is the figure the paper's central sentence rests on: published Boltzmann-transport values sit
systematically above laboratory measurements, and a two-parameter transfer removes most of that
offset. Two panels, same axes, same points -- the only difference is the correction, so a reader
can see the cloud move rather than take the median on trust.

Log-log and square, because the quantity being judged is a RATIO. On linear axes a factor-of-two
error at 15 W/m/K would dominate the eye while the same factor at 2 W/m/K vanished, which is the
opposite of how the result is scored.
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
from pymatgen.core import Composition

import figlib as F

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
BTE_METHODS = "bte|boltz|phono3py|shengbte|almabte|iterative|rta"
NOT_BTE = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def yzfam(f):
    try:
        e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
        return f"{e[1]}-{e[2]}" if len(e) == 3 else None
    except Exception:  # noqa: BLE001
        return None


def pairs() -> pd.DataFrame:
    """The PUBLISHED-ROUTE calculation/measurement pairs (shared_constant.pairs(); FIXPASS S5).

    One reference laboratory per compound (reference_choice), deduplicated corpus, the calculation
    the median over sources at each temperature and carried to the measured temperature by
    family_calibration.bte_at (log-log interpolation, else 1/T capped at 2.5x), contested
    calculations resolved, in domain. These are exactly the pairs the shared constant is fitted on,
    so the offset this figure reports is the offset the transfer function corrects.

    The old pairing (kept nowhere) read the raw CSV with no dedupe, paired every tier-0 row with
    the FIRST calculated row within a flat +-50 K, pooled every laboratory, and took medians over
    compound x 100 K points -- so ZrNiSn's many laboratories and duplicated samples weighed far more
    than a one-paper compound, and the reported "median ratio" was a per-point figure.

    Columns: compound, fam, T, k_exp, k_dft.
    """
    import shared_constant as SCN
    d = SCN.pairs()
    return pd.DataFrame(dict(compound=d.compound.values, fam=d.compound.map(yzfam).values,
                             T=d["T"].values.astype(float), k_exp=d.k_ref.values.astype(float),
                             k_dft=d.k_pred.values.astype(float)))


def main() -> int:
    import extend_blind_test as E  # noqa: F401

    # THE SHARED CONSTANT, fitted on published calculations vs measurements (shared_constant.py;
    # paper/evidence/PREREG_shared_constant_on_calculations.md), no longer on the ML model's
    # blind-test predictions. Panel c draws the deployed curve; the calibrated points of panel b and
    # the summary statistics apply, to each compound, the constant refitted WITHOUT its chemistry
    # cluster, because those points are the pairs the constant is fitted on (prereg rule 4).
    import shared_constant as SCN
    c, p = SCN.shared_cp()

    raw = pairs()          # already in domain: build_published keeps in-domain compounds only
    if not len(raw):
        print("no matched pairs found")
        return 1
    raw["k_cal"] = raw.k_dft * SCN.factor_without_own_cluster(raw)

    # THE HEADLINE OFFSET IS PER COMPOUND: each compound's median calculated/measured ratio over its
    # own temperatures, then the median over compounds -- the unit the constant is fitted in. The
    # per-point figure (compound x 100 K bin) is kept as SECONDARY, for the parity panels.
    raw["r_dft"] = raw.k_dft / raw.k_exp
    raw["r_cal"] = raw.k_cal / raw.k_exp
    per_c = raw.groupby("compound").r_dft.median()
    per_c_cal = raw.groupby("compound").r_cal.median()
    fam_of = raw.groupby("compound").fam.first()
    per_f = per_c.groupby(fam_of).median()      # family = median over its compounds' medians

    # AGGREGATE TO ONE POINT PER COMPOUND PER 100 K BIN for the panels, so a compound with many
    # temperatures does not dominate the visual impression.
    raw["Tbin"] = (raw["T"] / 100).round() * 100
    d = (raw.groupby(["compound", "fam", "Tbin"], as_index=False)
            .agg(k_exp=("k_exp", "median"), k_dft=("k_dft", "median"), k_cal=("k_cal", "median"),
                 n_rows=("k_exp", "size")))
    d = d.rename(columns={"Tbin": "T"})
    print(f"  published-route pairs {len(raw)} -> {len(d)} compound-temperature points "
          f"across {d.compound.nunique()} compounds")
    print(f"  per-compound median ratio  {per_c.median():.2f}   calibrated {per_c_cal.median():.2f}"
          f"   (deployed c={c:.3f}, p={p:.3f}; calibrated values use the constant refitted "
          f"without each compound's chemistry cluster)")
    print(f"  per-point (secondary)      {np.median(d.k_dft / d.k_exp):.2f}   calibrated "
          f"{np.median(d.k_cal / d.k_exp):.2f}")

    # THE OFFSET SUMMARY THE PAPER QUOTES, written as an artefact the registry ingests.
    import json
    within2 = float(100 * per_c.between(0.5, 2.0).mean())
    within2_cal = float(100 * per_c_cal.between(0.5, 2.0).mean())
    above_cap = int((d["T"] > 300.0 * (1.0 / c) ** (1.0 / p)).sum())
    summary = dict(
        _generated_by="paper/fig02_offset.py",
        pairing=("published-route pairs (shared_constant.pairs): single reference laboratory per "
                 "compound, median calculation carried to the measured temperature; statistics "
                 "per compound (median over compounds of each compound's median ratio)"),
        n_compounds=int(per_c.size), n_points=int(len(d)), n_pair_rows=int(len(raw)),
        n_families=int(per_f.size),
        median_ratio_calc_over_meas=round(float(per_c.median()), 2),
        median_ratio_calibrated=round(float(per_c_cal.median()), 2),
        within_2x_uncorrected_pct=round(within2, 0), within_2x_calibrated_pct=round(within2_cal, 0),
        compounds_with_ratio_above_1=int((per_c > 1).sum()),
        compounds_with_ratio_below_1=sorted(per_c[per_c <= 1].index),
        families_with_median_ratio_above_1=int((per_f > 1).sum()),
        families_with_median_ratio_below_1=sorted(per_f[per_f <= 1].index),
        per_point_secondary=dict(
            unit="compound x 100 K bin",
            median_ratio_calc_over_meas=round(float(np.median(d.k_dft / d.k_exp)), 2),
            median_ratio_calibrated=round(float(np.median(d.k_cal / d.k_exp)), 2),
            within_2x_uncorrected_pct=round(float(100 * (d.k_dft / d.k_exp).between(0.5, 2).mean()), 0),
            within_2x_calibrated_pct=round(float(100 * (d.k_cal / d.k_exp).between(0.5, 2).mean()), 0)),
        points_above_cap=above_cap, cap_K=int(round(300.0 * (1.0 / c) ** (1.0 / p))),
        c=round(float(c), 3), p=round(float(p), 3),
        calibrated_protocol=("shared constant fitted on published calculations; each compound's "
                             "calibrated points use the constant refitted without its chemistry "
                             "cluster; c, p are the deployed values"))
    Path("data/exports/kappa_v2/offset_summary.json").write_text(json.dumps(summary, indent=2),
                                                                 encoding="utf-8")
    print(f"  wrote data/exports/kappa_v2/offset_summary.json: {summary['compounds_with_ratio_above_1']}"
          f" of {summary['n_compounds']} compounds above 1, "
          f"{summary['families_with_median_ratio_above_1']} of {summary['n_families']} families")

    # ============================================================================================
    # THE FIGURE. Four panels in ONE ROW, 190 mm wide (compacted 2026-10-06; the per-family
    # before/after panel moved out -- the family figure, fig09_family.py, now carries families):
    #
    #   (a) (b)  parity, uncorrected and calibrated -- identical square log axes, identical ticks,
    #            identical limits in both panels (the paper-wide KAPPA_LIM), colour = temperature
    #   (c)      the transfer function itself over the measured/calculated ratio it models
    #   (d)      the PER-COMPOUND ratio distribution, before and after -- the unit the text quotes,
    #            drawn as two vertical swarms so it fits the row
    #
    # NOTHING IS WRITTEN OVER THE DATA. Every statistic lives in a panel title, and the labels that
    # must sit inside an axes (panel letter, the n note in (a)) are placed by `_free_spot`, which
    # tests candidate positions against the plotted points and takes the first one that covers none.
    #
    # ENCODE TEMPERATURE, NOT FAMILY, IN (a)-(c). The transfer function carries an exponent p, so
    # the offset is temperature-dependent by construction.
    # ============================================================================================
    F.use_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.ticker as mticker

    plt.rcParams["figure.constrained_layout.use"] = False   # explicit millimetre layout below
    MMI = F.MM
    W = F.DOUBLE / MMI                                       # 190 mm
    S = 37.5                                                 # side of the square panels, mm
    WD = 27.0                                                # width of the swarm panel (d)
    Y0, TT = 8.8, 10.8                                       # bottom offset; title block height
    H = Y0 + S + TT
    xa = 10.0
    xb = xa + S + 9.3
    xc = xb + S + 9.6
    xcb = xc + S + 1.3                                       # colour bar
    xd = W - WD - 0.6
    fig = plt.figure(figsize=(W * MMI, H * MMI))

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    axa, axb, axc = ax_mm(xa, Y0, S, S), ax_mm(xb, Y0, S, S), ax_mm(xc, Y0, S, S)
    cax = ax_mm(xcb, Y0 + 0.06 * S, 1.8, 0.88 * S)
    axd = ax_mm(xd, Y0, WD, S)

    allv = np.concatenate([d.k_exp.values, d.k_dft.values, d.k_cal.values])
    # The shared paper-wide conductivity scale, not this figure's own data range -- see figlib.
    F.warn_clip(allv, "fig02 parity panels")
    lo, hi = F.KAPPA_LIM
    cmap = F.temp_cmap()
    norm = Normalize(vmin=float(d["T"].min()), vmax=float(d["T"].max()))
    renderer = fig.canvas.get_renderer()
    FS_SUB = 6.3

    def _pts_axes(ax, x, y):
        """Plotted points in axes-fraction coordinates (log axes handled by transData)."""
        xy = ax.transData.transform(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
        return ax.transAxes.inverted().transform(xy)

    def _free_spot(ax, artist_fn, pts, candidates, pad_mm=1.0):
        """Draw `artist_fn(x, y, ha, va)` at the first candidate whose box covers no point.

        `pts` are axes-fraction point centres; each is padded by roughly a marker radius so a dot
        grazing the box edge counts as covered. Returns the artist; raises if nothing is free, so
        an overlap can never ship silently.
        """
        bw, bh = ax.get_window_extent(renderer).width, ax.get_window_extent(renderer).height
        px = pad_mm * MMI * fig.dpi
        for (x, y, ha, va) in candidates:
            art = artist_fn(x, y, ha, va)
            bb = art.get_window_extent(renderer).transformed(ax.transAxes.inverted())
            padx, pady = px / bw, px / bh
            hit = ((pts[:, 0] > bb.x0 - padx) & (pts[:, 0] < bb.x1 + padx)
                   & (pts[:, 1] > bb.y0 - pady) & (pts[:, 1] < bb.y1 + pady))
            if not hit.any():
                return art
            art.remove()
        raise RuntimeError(f"no free spot for a label in {ax} -- move it outside the axes")

    def _panel_letter(ax, letter, pts, cands=None):
        def mk(x, y, ha, va):
            return ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5,
                           fontweight="bold", va=va, ha=ha, zorder=20,
                           bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#BBBBBB",
                                     lw=0.5, alpha=0.92))
        return _free_spot(ax, mk, pts, cands or [(0.035, 0.965, "left", "top"),
                                                 (0.035, 0.035, "left", "bottom"),
                                                 (0.965, 0.035, "right", "bottom"),
                                                 (0.965, 0.965, "right", "top")])

    def _title(ax, head, sub):
        """Bold head line, then the panel's statistics in regular weight -- outside the data."""
        ax.annotate(sub, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 2.0),
                    textcoords="offset points", ha="center", va="bottom", fontsize=FS_SUB,
                    color="#333333", linespacing=1.15)
        rise = 2.0 + (sub.count("\n") + 1) * FS_SUB * 1.15 * 1.18 + 0.8
        ax.annotate(head, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, rise),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.4,
                    fontweight="bold")

    # ---- (a), (b): parity ----------------------------------------------------------------------
    n_cmp, n_pts = d.compound.nunique(), len(d)
    lab_meas = "measured $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)"
    for ax, col, ttl, who in ((axa, "k_dft", "Uncorrected calculation", "calculated"),
                              (axb, "k_cal", "After transfer function", "calibrated")):
        F.band_2x(ax, lo, hi, label=None)
        sc = ax.scatter(d.k_exp, d[col], c=d["T"], cmap=cmap, norm=norm, s=8,
                        marker="o", lw=0.25, edgecolors="white", zorder=3)
        F.parity_axes(ax, lo, hi, lab_meas, who + " $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)")
        ax.xaxis.labelpad = 1.5
        ax.yaxis.labelpad = 1.0
        F.kappa_ticks(ax)
        ax.tick_params(labelsize=6.4, pad=1.5)
        r = d[col] / d.k_exp
        n_out_hi, n_out_lo = int((r > 2).sum()), int((r < 0.5).sum())
        # SAY WHICH WAY THE RATIO GOES -- the direction is the figure's whole claim.
        _title(ax, ttl, f"{who}/measured: median {np.median(r):.2f}\n"
                        f"{100 * r.between(0.5, 2.0).mean():.0f}% within 2$\\times$; "
                        f"{n_out_hi} above, {n_out_lo} below")
        pts = _pts_axes(ax, d.k_exp, d[col])
        _panel_letter(ax, "ab"[col == "k_cal"], pts)

        # n and the band's meaning, said once in (a); (b) plots the same points
        if col == "k_dft":
            def mk_n(x, y, ha, va, ax=ax):
                return ax.text(x, y, f"{n_pts} points\n{n_cmp} compounds\nshaded: 2$\\times$",
                               transform=ax.transAxes, ha=ha, va=va, fontsize=6.0,
                               color="#555555", linespacing=1.12)
            _free_spot(ax, mk_n, pts, [(0.97, 0.03, "right", "bottom"),
                                       (0.03, 0.03, "left", "bottom")])

    # ---- (c): the transfer function and the ratio it was fitted to ------------------------------
    # measured/calculated is exactly what min[c(T/300)^p, 1] models, so the curve should pass
    # through the middle of the cloud. Points above unity are compounds whose calculation already
    # sits below the measurement; no downward correction reaches them.
    r_obs = (d.k_exp / d.k_dft).values
    Tgrid = np.linspace(150, 1250, 400)
    mult = np.minimum(c * (Tgrid / 300.0) ** p, 1.0)
    Tcap = 300.0 * (1.0 / c) ** (1.0 / p)
    n_above = int((d["T"] >= Tcap).sum())
    n_r_above1 = int((r_obs > 1).sum())
    axc.axhline(1.0, color="#9A9A9A", lw=0.7, ls=(0, (4, 3)), zorder=1)
    axc.scatter(d["T"], r_obs, c=d["T"], cmap=cmap, norm=norm, s=8, marker="o",
                lw=0.25, edgecolors="white", zorder=3)
    axc.plot(Tgrid, mult, color=F.OURS, lw=1.6, zorder=4)
    axc.axvline(Tcap, color=F.OURS, lw=0.8, ls=(0, (2, 2)), zorder=2)
    axc.set_xlabel("$T$ (K)", labelpad=1.5)
    axc.set_ylabel("measured / calculated", labelpad=1.5)
    axc.set_xlim(150, 1250)
    axc.xaxis.set_major_locator(mticker.MultipleLocator(300))
    axc.xaxis.set_minor_locator(mticker.MultipleLocator(100))
    axc.tick_params(labelsize=6.4, pad=1.5)
    ytop = float(np.ceil(max(1.25, 1.06 * np.nanmax(r_obs)) * 4) / 4)
    axc.set_ylim(0, ytop)
    axc.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    axc.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))
    # The curve's label goes in the title block, not on the panel: the cloud fills (c) from corner
    # to corner and no position for it covers no point.
    _title(axc, "Transfer function",
           f"$\\min[c(T/300)^{{p}},1]$, $c={c:.2f}$, $p={p:.2f}$\n"
           f"cap {Tcap:.0f} K (dotted); {n_above} of {n_pts} points beyond")
    pts_c = _pts_axes(axc, d["T"], r_obs)
    # the curve's own pixels count as occupied too, so the letter never sits on the line
    pts_c = np.vstack([pts_c, _pts_axes(axc, Tgrid, mult)])
    _panel_letter(axc, "c", pts_c)
    print(f"  (c): {n_r_above1} of {n_pts} points have measured > calculated (ratio above 1)")

    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("$T$ (K)", fontsize=7, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.2, width=0.5, length=2, pad=1.2)
    cb.ax.yaxis.set_major_locator(mticker.MultipleLocator(200))
    cb.outline.set_linewidth(0.5)

    # ---- (d): per-compound distribution, before and after, as two vertical swarms -------------
    rat = np.concatenate([per_c.values, per_c_cal.values])
    rlo = min(0.5, float(np.nanmin(rat))) / 1.18
    rhi = max(2.0, float(np.nanmax(rat))) * 1.15
    rticks = [v for v in (0.25, 0.33, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7) if rlo <= v <= rhi]
    axd.set_yscale("log")
    axd.set_ylim(rlo, rhi)
    axd.axhspan(0.5, 2.0, color="#EDEDED", lw=0, zorder=0)
    for f_ in (0.5, 2.0):
        axd.axhline(f_, color="#C4C4C4", lw=0.5, ls=(0, (4, 2.5)), zorder=1)
    axd.axhline(1.0, color="#6E6E6E", lw=0.8, ls="--", zorder=1)
    axd.yaxis.set_major_locator(mticker.FixedLocator(rticks))
    axd.yaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in rticks]))
    axd.yaxis.set_minor_locator(mticker.NullLocator())
    axd.grid(axis="x", visible=False)
    axd.set_ylabel("calculated / measured", labelpad=1.0)
    axd.tick_params(labelsize=6.4, pad=1.5)
    XL = (-0.62, 1.62)
    axd.set_xlim(*XL)
    mm_per_x = WD / (XL[1] - XL[0])                         # millimetres per x unit on (d)
    mm_per_dec = S / (np.log10(rhi) - np.log10(rlo))        # millimetres per decade on (d)
    diam_mm = 1.45                                          # marker diameter

    def swarm(vals):
        """Beeswarm offsets in mm: each compound at its ratio, nudged sideways off its neighbours."""
        yv = np.log10(np.asarray(vals, float)) * mm_per_dec
        order = np.argsort(yv)
        placed, off = [], np.zeros(len(yv))
        steps = [0.0] + [s_ * k * diam_mm * 0.92 for k in range(1, 14) for s_ in (1, -1)]
        for i in order:
            for o in steps:
                if all((yv[i] - yj) ** 2 + (o - oj) ** 2 >= diam_mm ** 2 for yj, oj in placed):
                    off[i] = o
                    placed.append((yv[i], o))
                    break
        return off

    rows_d = [("uncorr.", per_c, dict(facecolors="white", edgecolors="#6E747A", lw=0.7)),
              ("after", per_c_cal, dict(facecolors=F.OURS, edgecolors="white", lw=0.3))]
    pts_d = []
    for k, (lab, ser, sty) in enumerate(rows_d):
        off = swarm(ser.values) / mm_per_x
        assert np.abs(off).max() < 0.6, "swarm wider than its column -- widen (d)"
        axd.scatter(k + off, ser.values, s=5.5, zorder=4, **sty)
        med = float(ser.median())
        axd.plot([k - 0.45, k + 0.45], [med, med], color=F.OURS if k else "#6E747A", lw=1.5,
                 zorder=3, ls="-" if k else (0, (3, 1.5)), solid_capstyle="butt")
        pts_d.append(_pts_axes(axd, k + off, ser.values))
    axd.set_xticks([0, 1])
    axd.set_xticklabels([lab for lab, _, _ in rows_d], fontsize=6.6)
    axd.tick_params(axis="x", which="both", length=0, top=False, bottom=False)
    _title(axd, f"Per compound ({int(per_c.size)})",
           f"median {per_c.median():.2f} $\\rightarrow$ {per_c_cal.median():.2f}\n"
           f"{100 * per_c.between(0.5, 2.0).mean():.0f}% $\\rightarrow$ "
           f"{100 * per_c_cal.between(0.5, 2.0).mean():.0f}% within 2$\\times$")
    _panel_letter(axd, "d", np.vstack(pts_d))

    print(f"  figure size: {W:.0f} x {H:.1f} mm")
    F.save(fig, "fig02_offset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
