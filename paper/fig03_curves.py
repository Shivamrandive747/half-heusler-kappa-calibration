"""The three best held-out family fits, as kappa_L(T) curves.

Since 2026-10-06 these curves are panels (c)-(e) of the family-calibration figure
(paper/fig09_family.py), which imports `select_best`, `draw_curve` and the helpers below. This script
still writes data/exports/kappa_v2/curves_best3.json, which the registry (make_paper_numbers.py)
reads, and a standalone fig03_curves.pdf for the supplement.

Every other figure reports a summary statistic. A materials reader does not believe a median; they
believe a curve sitting on the data points they recognise. Each panel shows the measured kappa_L(T)
of the reference laboratory, the published transport calculation (uncorrected, dashed) and the
family-calibrated curve (solid).

AUTHOR'S DECISION (2026-10-05): ONLY THE THREE BEST, AND SAID SO. The figure is an illustration,
not evidence of typical accuracy, and it is labelled as the three best fits on the figure itself,
with the median over every eligible compound beside it. Calling these "representative" would be
false, so nothing here claims it.

HELD OUT, AS THE DEPLOYED ROUTE SCORES IT. Each compound is drawn with its family constant fitted
WITHOUT it (c_family_loo, p_family_loo in family_deployed_pooled.csv, written by
compute_deployed_route.py), on the pairs of its own leave-one-out fold (family_calibration.fold_pairs:
the reference ladder re-run with the compound absent from the family rung, PREREG amendment A).
Compounds on the "family (in-sample)" arm -- single-member families, whose constant saw the compound
-- are not eligible. The held-out error is RECOMPUTED here from those constants and asserted equal to
the scoreboard's, so the figure cannot drift from the number the text quotes.

SELECTION RULE (printed and written to curves_best3.json): among held-out family-arm compounds
whose fold reference curve has >= MIN_POINTS distinct temperatures spanning >= MIN_SPAN K, the three
with the lowest held-out family error. The span rule keeps a two-point "curve" from being chosen
because two points are easy to fit.

The calculation is drawn as the published-route median calculation carried in temperature by
family_calibration.published_calc_at (log-log interpolation between calculated temperatures, else
kappa ~ 1/T, capped at 2.5x) -- the same value build_published pairs with each measurement -- and
the temperatures at which a calculation was actually published are marked, so the reader can see
how much of the dashed curve is the 1/T carry rather than a calculated point.
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

import matplotlib.ticker as mticker

import figlib as F

POOLED = "data/exports/kappa_v2/family_deployed_pooled.csv"
OUT_JSON = "data/exports/kappa_v2/curves_best3.json"
MIN_POINTS = 5
MIN_SPAN = 300.0
N_SHOW = 3
CALC_MARK = dict(ls="none", marker="s", ms=3.6, mfc="white", mec=F.RAW_DFT, mew=1.0, zorder=4)


def select_best(n_show: int = N_SHOW, verbose: bool = True) -> dict:
    """Score every held-out family-arm compound and pick the `n_show` best eligible fits.

    Returns a dict: R (all held-out compounds), ok (eligible, sorted by error), pick (the shown
    ones), frames (compound -> its fold pairs with k_cal), curves (compound -> (T grid, published
    calculation, calibrated curve, {T: calculated value at a published temperature})), rule (text),
    and json (the curves_best3.json payload).
    """
    import contextlib
    import io

    import family_calibration as FC

    P = pd.read_csv(POOLED)
    held = P[P.arm == "family"].copy()          # held out; "family (in-sample)" is not eligible
    if verbose:
        print(f"  {len(P)} compounds on the deployed route; {len(held)} scored held out on a "
              f"family arm")

    rows, frames = [], {}
    for r in held.itertuples():
        with contextlib.redirect_stdout(io.StringIO()):
            f = FC.fold_pairs(r.compound)
        te = f[f.compound == r.compound].sort_values("T")
        if te.empty:
            continue
        T = te["T"].values.astype(float)
        k = FC.apply_family(FC.ADOPTED_FORM, (r.c_family_loo, r.p_family_loo), T, te.k_pred.values)
        ape = float(FC.per_compound_ape(te, k).iloc[0])
        # the figure must show the scoreboard's number, not a re-derivation that could drift from it
        assert abs(ape - r.ape) < 1e-6, f"{r.compound}: recomputed {ape} vs scoreboard {r.ape}"
        frames[r.compound] = te.assign(k_cal=k)
        rows.append(dict(compound=r.compound, family=r.family, ape=float(r.ape),
                         c=float(r.c_family_loo), p=float(r.p_family_loo),
                         n_T=int(te["T"].round().nunique()), T_lo=float(T.min()),
                         T_hi=float(T.max()), span=float(T.max() - T.min()),
                         ref_doi=str(r.ref_doi_heldout)))
    R = pd.DataFrame(rows)
    ok = R[(R.n_T >= MIN_POINTS) & (R.span >= MIN_SPAN)].sort_values("ape").reset_index(drop=True)
    pick = ok.head(n_show)
    rule = (f"held-out family-arm compounds (arm == 'family' in family_deployed_pooled.csv; family "
            f"constant fitted without the compound, on its leave-one-out fold reference) whose "
            f"reference curve has >= {MIN_POINTS} distinct temperatures spanning >= "
            f"{MIN_SPAN:.0f} K; the {n_show} with the lowest held-out family error")
    if verbose:
        print(f"  SELECTION RULE: {rule}")
        print(f"  eligible: {len(ok)} of {len(R)} held-out compounds; eligible median error "
              f"{ok.ape.median():.1f}%")
        for q in pick.itertuples():
            print(f"    {q.compound:<8} {q.family:<6} held-out error {q.ape:5.1f}%   c={q.c:.3f} "
                  f"p={q.p:.2f}   {q.n_T} T, {q.T_lo:.0f}-{q.T_hi:.0f} K   ref {q.ref_doi}")

    # the drawn curves, computed once so both figures draw exactly the same thing
    med_calc = FC._prepare()["med"]
    curves = {}
    for q in pick.itertuples():
        Tg = np.linspace(q.T_lo, q.T_hi, 120)
        kc = np.array([FC.published_calc_at(q.compound, t) or np.nan for t in Tg], dtype=float)
        kf = FC.apply_family(FC.ADOPTED_FORM, (q.c, q.p), Tg, kc)
        cal_T = {float(t): float(v) for t, v in (med_calc.get(q.compound) or {}).items()
                 if q.T_lo - 1 <= float(t) <= q.T_hi + 1}
        curves[q.compound] = (Tg, kc, np.asarray(kf, dtype=float), cal_T)

    payload = dict(
        _generated_by="paper/fig03_curves.py",
        selection_rule=rule,
        min_points=MIN_POINTS, min_span_K=MIN_SPAN,
        n_heldout_family_arm=int(len(R)), n_eligible=int(len(ok)),
        eligible_median_heldout_error_pct=round(float(ok.ape.median()), 1),
        protocol=("per-family SHAPE_NOCAP calibration held out: c_family_loo, p_family_loo from "
                  "compute_deployed_route.py, scored on family_calibration.fold_pairs(compound)"),
        not_representative=("the three BEST fits by construction; the full per-compound "
                            "distribution is family_deployed_pooled.csv"),
        compounds=[dict(compound=q.compound, family=q.family,
                        heldout_error_pct=round(q.ape, 1), c_family_loo=round(q.c, 4),
                        p_family_loo=round(q.p, 3), n_temperatures=q.n_T,
                        T_range_K=[round(q.T_lo), round(q.T_hi)], reference_doi=q.ref_doi)
                   for q in pick.itertuples()])
    return dict(R=R, ok=ok, pick=pick, frames=frames, curves=curves, rule=rule, json=payload)


def honest_label(sel: dict) -> str:
    """The selection caveat, every number computed: 'three best of N eligible; median over all N'."""
    n_ok = len(sel["ok"])
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    k = len(sel["pick"])
    return (f"{words.get(k, str(k))} best of {n_ok} eligible held-out fits "
            f"(≥{MIN_POINTS} temperatures over ≥{MIN_SPAN:.0f} K); "
            f"median over all {n_ok}: {sel['ok'].ape.median():.1f}%")


def draw_curve(ax, sel: dict, q) -> np.ndarray:
    """Draw one compound: measured points, uncorrected calculation, held-out family curve.

    Returns the drawn (T, kappa) pairs, for the shared y-range and for label placement.
    """
    s = sel["frames"][q.compound]
    Tg, kc, kf, cal_T = sel["curves"][q.compound]
    ax.plot(Tg, kc, **{**F.ROLE["raw"], "label": None})
    ax.plot(Tg, kf, **{**F.ROLE["ours"], "label": None})
    ax.plot(s["T"], s.k_ref, **{**F.ROLE["measured"], "ms": 3.6, "label": None})
    if cal_T:
        ax.plot(list(cal_T), list(cal_T.values()), **CALC_MARK)
    xy = np.vstack([np.column_stack([s["T"].values, s.k_ref.values]),
                    np.column_stack([Tg, kc]), np.column_stack([Tg, kf])])
    return xy[np.isfinite(xy).all(axis=1)]


def shared_logy(axes, drawn: list, pad: float = 1.30) -> None:
    """One log y range from the data across every panel, labelled at readable values."""
    allk = np.concatenate([d[:, 1] for d in drawn])
    F.warn_clip(allk, "best-fit curves")
    ylo, yhi = float(allk.min()) / pad, float(allk.max()) * pad
    keep = [v for v in (0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30) if ylo <= v <= yhi]
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(mticker.FixedLocator(keep))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter([f"{v:g}" for v in keep]))
        ax.yaxis.set_minor_locator(mticker.NullLocator())


def place_free(ax, make, xy_data, corners=None, pad: float = 0.025, occupied=None):
    """Draw `make(x, y, ha, va)` in the first corner whose box covers no drawn point.

    `xy_data` are data coordinates; `occupied` an optional (n, 2) array of axes-fraction points
    already taken (by an earlier label). Returns (artist, occupied_with_this_box). Raises if no
    corner is free, so a label over data can never ship silently.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transAxes.inverted().transform(ax.transData.transform(xy_data))
    if occupied is not None and len(occupied):
        pts = np.vstack([pts, occupied])
    corners = corners or [(0.035, 0.965, "left", "top"), (0.965, 0.965, "right", "top"),
                          (0.035, 0.035, "left", "bottom"), (0.965, 0.035, "right", "bottom")]
    for (x, y, ha, va) in corners:
        art = make(x, y, ha, va)
        bb = art.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        hit = ((pts[:, 0] > bb.x0 - pad) & (pts[:, 0] < bb.x1 + pad)
               & (pts[:, 1] > bb.y0 - pad) & (pts[:, 1] < bb.y1 + pad))
        if not hit.any():
            gx, gy = np.meshgrid(np.linspace(bb.x0, bb.x1, 8), np.linspace(bb.y0, bb.y1, 5))
            box = np.column_stack([gx.ravel(), gy.ravel()])
            occ = box if occupied is None else np.vstack([occupied, box])
            return art, occ
        art.remove()
    raise RuntimeError(f"no empty corner in {ax} for a label")


def legend_handles():
    import matplotlib.pyplot as plt
    h = [plt.Line2D([], [], **{a: b for a, b in F.ROLE[r].items() if a not in ("zorder", "label")})
         for r in ("measured", "raw", "ours")]
    h.append(plt.Line2D([], [], **{a: b for a, b in CALC_MARK.items() if a != "zorder"}))
    labels = ["measured (reference laboratory)", "published calculation, uncorrected",
              "family-calibrated, held out", "temperature of a published calculation"]
    return h, labels


def write_json(sel: dict) -> None:
    Path(OUT_JSON).write_text(json.dumps(sel["json"], indent=2), encoding="utf-8")
    print(f"  wrote {OUT_JSON}")


def main() -> int:
    import matplotlib.pyplot as plt

    sel = select_best()
    write_json(sel)

    # ---- standalone figure, for the supplement --------------------------------------------------
    F.use_style()
    fig, axes = plt.subplots(1, N_SHOW, figsize=(F.DOUBLE, F.DOUBLE * 0.30), sharey=True)
    drawn = [draw_curve(ax, sel, q) for ax, q in zip(axes, sel["pick"].itertuples())]
    for ax in axes:
        ax.margins(x=0.05)
        ax.set_xlabel("$T$ (K)")
        ax.tick_params(labelsize=7, pad=1.5)
    shared_logy(axes, drawn)
    axes[0].set_ylabel(F.KAPPA_LABEL)
    for k, (ax, q, xy) in enumerate(zip(axes, sel["pick"].itertuples(), drawn)):
        F.panel_label(ax, "abc"[k], dx=0.0, dy=1.10, outside=True)

        def mk(x, y, ha, va, q=q, ax=ax):
            return ax.text(x, y, f"{q.compound} ({q.family})\nheld-out error {q.ape:.1f}%",
                           transform=ax.transAxes, fontsize=7.0, ha=ha, va=va, color=F.OURS,
                           zorder=20, linespacing=1.25)
        place_free(ax, mk, xy)
    h, labels = legend_handles()
    fig.legend(handles=h, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, fontsize=7.0, handlelength=2.0, columnspacing=1.4)
    lab = honest_label(sel)
    fig.text(0.5, 1.075, lab[0].upper() + lab[1:], ha="center", va="bottom", fontsize=7.4,
             color="#222222")
    F.save(fig, "fig03_curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
