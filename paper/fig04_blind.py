"""Figure 4 -- the blind test: how well, where it fails, and against what alternative.

Two panels: the domain restriction and the null comparison are one argument. The error by bonding
family is shown on the deployed per-family route in Figure fig09_family; the global-constant
per-family errors are still written to blind_by_family.json for the registry, but not drawn.

  (a) PARITY, chemistry held out. In-domain compounds in the paper's accent, out-of-domain ones as
      open grey markers. The domain restriction is therefore ARGUED here rather than asserted --
      a reader sees the grey points sitting far off the diagonal and understands the screen without
      being told. Nothing is hidden by restricting the headline; the excluded compounds are on the
      same axes as the included ones.
  (b) AGAINST THE NULLS. Both nulls are fitted with the chemistry held out exactly as the
      calibration is, because a null given less information than the model is not a fair test and
      a referee will say so.
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
from pymatgen.core import Composition, Element

import figlib as F
from make_paper_predictions import structure_status


def ve(e):
    g = Element(str(e)).group
    return g if g <= 12 else g - 10


def vec(f):
    try:
        return int(sum(ve(e) for e in Composition(f).elements))
    except Exception:  # noqa: BLE001
        return None


def yzfam(f):
    try:
        e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
        return f"{e[1]}-{e[2]}" if len(e) == 3 else None
    except Exception:  # noqa: BLE001
        return None


def nulls(d, cluster):
    """Leave-one-chemistry-out constant and power-law nulls, per compound."""
    out = {}
    for name in ("constant", "power"):
        rec = []
        for cl in sorted(set(cluster.values())):
            te = d[d.compound.map(cluster) == cl]
            tr = d[d.compound.map(cluster) != cl]
            if not len(te) or len(tr) < 4:
                continue
            if name == "constant":
                pred = np.full(len(te), float(np.median(tr.k_ref)))
            else:
                lx = np.log(tr["T"].values / 300.0)
                ly = np.log(tr.k_ref.values)
                q, la = np.polyfit(lx, ly, 1)
                pred = np.exp(la) * (te["T"].values / 300.0) ** q
            rec.append(pd.DataFrame(dict(compound=te.compound.values,
                                         ape=np.abs(pred - te.k_ref.values) / te.k_ref.values * 100,
                                         ratio=pred / te.k_ref.values)))
        r = pd.concat(rec).groupby("compound").median()
        out[name] = r
    return out


def blind_points() -> dict:
    """The per-compound blind-test points, exactly as panel (a) draws them.

    Factored out so that fig15_model_blind.py draws the SAME points by the SAME protocol (seed-0 file
    target_blind_test.csv, in-domain compounds calibrated with their chemistry cluster held out,
    out-of-domain ones with the shared constant refitted without their cluster) rather than a copy
    of this code that could drift from it.
    """
    import extend_blind_test as E
    from run_loco_chemistry import cluster_of

    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    d = B[B.klass == "half"].copy()
    d["VEC"] = d.compound.map(vec)
    st = {c_: structure_status(str(c_)) for c_ in d.compound.unique()}
    d["ind"] = (d.VEC == 18) & d.compound.map(lambda c_: st[c_][0] and not st[c_][1])
    d["fam"] = d.compound.map(yzfam)

    dom = d[d.ind]
    # the shared constant is fitted on published calculations vs measurements (shared_constant.py;
    # PREREG_shared_constant_on_calculations), not on these model predictions
    import shared_constant as SCN

    # IN-DOMAIN COMPOUNDS ARE SCORED WITH THE CALIBRATION HELD OUT, matching the nulls in panel (c)
    # and Table 1. Scoring them with a single fit over the whole domain gave 29.7% where the table
    # says 31.0%, and put an in-sample method against out-of-sample nulls in a panel whose title
    # says the nulls are held out. Out-of-domain compounds are scored the same way: the shared
    # constant refitted without their chemistry cluster (prereg rule 4).
    dom_n = F.nested_calibrate(dom, SCN.shared_cp_without_cluster, E.apply_cp, cluster_of)
    outd = d[~d.ind].copy()
    outd["k_cal"] = outd.k_pred * SCN.factor_without_own_cluster(outd)
    d = pd.concat([dom_n, outd], ignore_index=True)
    d["ape2"] = (d.k_cal - d.k_ref).abs() / d.k_ref * 100
    d["ratio2"] = d.k_cal / d.k_ref

    per = d.groupby("compound").agg(ape=("ape2", "median"), ratio=("ratio2", "median"),
                                    ref=("k_ref", "median"), cal=("k_cal", "median"),
                                    ind=("ind", "first"), fam=("fam", "first"))
    ins, out = per[per.ind], per[~per.ind]
    print(f"  in domain {len(ins)}   median {ins.ape.median():.1f}%   "
          f"within2x {ins.ratio.between(.5, 2).mean()*100:.1f}%")
    print(f"  out       {len(out)}   median {out.ape.median():.1f}%   "
          f"within2x {out.ratio.between(.5, 2).mean()*100:.1f}%")

    clus = {cp_: (cluster_of(cp_) or "?") for cp_ in dom.compound.unique()}
    nl = nulls(dom, clus)
    print(f"  null constant  {nl['constant'].ape.median():.1f}%   "
          f"within2x {nl['constant'].ratio.between(.5,2).mean()*100:.1f}%")
    print(f"  null power law {nl['power'].ape.median():.1f}%   "
          f"within2x {nl['power'].ratio.between(.5,2).mean()*100:.1f}%")
    return dict(per=per, ins=ins, out=out, nl=nl)


def main() -> int:
    import matplotlib.pyplot as plt

    R = blind_points()
    per, ins, out, nl = R["per"], R["ins"], R["out"], R["nl"]

    F.use_style()
    fig = plt.figure(figsize=(F.DOUBLE * 0.66, F.DOUBLE * 0.42))
    # (a) carries the headline result and is square by construction: its HEIGHT is set by its
    # width, so too narrow a first column leaves it floating in white space while (b) runs the full
    # height of the figure.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.70], wspace=0.30)
    axa, axc = (fig.add_subplot(gs[i]) for i in range(2))

    # (a) parity
    # The shared paper-wide conductivity scale, so this panel and Fig. 2's parity panels put the
    # same conductivity in the same place. Both are measured-against-modelled on log-log axes, and
    # a reader compares them directly.
    F.warn_clip(np.concatenate([per.ref.values, per.cal.values]), "fig04 (a)")
    lo, hi = F.KAPPA_LIM
    F.band_2x(axa, lo, hi, color=F.BAND, label=None)
    axa.scatter(out.ref, out.cal, s=20, facecolors="none", edgecolors=F.REFUSED,
                lw=0.9, marker="s", zorder=3, label=f"outside domain ({len(out)})")
    axa.scatter(ins.ref, ins.cal, s=20, c=F.OURS, marker="o", lw=0.4,
                edgecolors="white", zorder=4, label=f"in domain ({len(ins)})")
    F.parity_axes(axa, lo, hi, "measured $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)",
                  "predicted $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)")
    F.kappa_ticks(axa)
    # Anything below the shared floor is drawn ON the edge as an open triangle rather than dropped,
    # so the tighter scale costs no data.
    n_off = sum(F.edge_markers(axa, g.ref.values, g.cal.values, axis=a, color=c)
                for g, c in ((out, F.REFUSED), (ins, F.OURS)) for a in ("y", "x"))
    if n_off:
        axa.text(0.03, 0.03, f"{n_off} below axis", transform=axa.transAxes, fontsize=6,
                 color="#666666", ha="left", va="bottom")
    # the boxed panel letter takes the upper-left corner, so the key moves to the lower right
    axa.legend(loc="lower right", fontsize=6.5, borderaxespad=0.4)
    axa.text(0.97, 0.955, f"{ins.ratio.between(.5,2).mean()*100:.0f}% within 2$\\times$",
             transform=axa.transAxes, ha="right", va="top", fontsize=7, color=F.OURS,
             fontweight="bold")
    F.panel_label(axa, "a", dx=-0.24)

    # (b) family
    # n >= 2 hides any family with a single member in this set -- including Co-Bi, through which
    # the paper issues HfCoBi. A starred family missing from its own panel is worse than a short
    # bar, so single-member families are kept and marked.
    fam = (ins.groupby("fam").agg(err=("ape", "median"), n=("ape", "size"))
              .query("n >= 1").sort_values("err"))
    ob = out.groupby("fam").agg(err=("ape", "median"), n=("ape", "size")).query("n >= 2")
    # Families the paper actually issues predictions in are drawn in the accent; the rest recede.
    # The panel then makes the argument rather than merely reporting: predictions are issued where
    # the family has been validated, and the two worst families are exactly the two the prediction
    # code refuses. A reader can check that claim against the bar lengths without being told.
    # DERIVED, never hardcoded. A hardcoded set went wrong in both directions as the calibration
    # moved: it once starred Sb-Pd when both its candidates were flagged, and would now omit it,
    # since TmSbPd is issued while DySbPd remains flagged for its non-cubic polymorph. Reading the
    # file keeps the starred families in step with Figure 5, Table 2 and Section 5.
    # THE PER-FAMILY ERRORS ARE AN ARTEFACT, NOT STDOUT. Section 3.6 quotes these five numbers as
    # body text; until now they existed only in this script's printout, which the registry rule
    # ("nothing is typed by hand") forbids. Written here, ingested by make_paper_numbers.py.
    import json as _j
    _bf = dict(_generated_by="paper/fig04_blind.py",
               seed=0,
               seed_note="model predictions from the seed-0 blind-test file only",
               protocol=("in-domain blind set, global constant, leave-one-chemistry-out, single "
                         "reference seed; NOT the deployed per-family route of family_calibration"),
               n_in_domain=int(len(ins)),
               families={str(f): dict(median_ape=round(float(r.err), 1), n=int(r.n))
                         for f, r in fam.iterrows()},
               out_of_domain={str(f): dict(median_ape=round(float(r.err), 1), n=int(r.n))
                              for f, r in ob.iterrows()})
    Path("data/exports/kappa_v2/blind_by_family.json").write_text(_j.dumps(_bf, indent=2),
                                                                  encoding="utf-8")

    _pred = pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv")
    ISSUES_IN = set(_pred[_pred.status == "ISSUED"].family.dropna())
    print(f"  families with issued predictions (derived): {sorted(ISSUES_IN)}")

    # (b) nulls
    # short labels -- the long forms collide at this panel width; the full functional form of each
    # null belongs in the caption, not squeezed under a 25 mm axis
    # Panel (c) must agree with Table 1, whose model row is the median over five seeds. Panels (a)
    # and (b) are per-compound and are necessarily one seed; the caption says so. Printing the
    # seed-0 value here while the table printed the seed median put 31% beside 33.7% in the same
    # document -- the class of mismatch this project has now hit three times.
    import json as _json
    try:
        _sa = _json.load(open("data/exports/kappa_v2/seed_averaged_indomain.json"))
        _model_median = float(_sa["median_ape_median"])
        print(f"  panel (c) model bar uses the seed-averaged {_model_median}% "
              f"(this seed alone: {ins.ape.median():.1f}%)")
    except Exception:  # noqa: BLE001
        _model_median = float(ins.ape.median())
    # the title already says these are nulls; repeating the word on the tick made the first
    # lines of adjacent labels touch and print as "constantpower-law".
    labels = ["this\nwork", "constant\n$\\kappa$", "power\nlaw"]
    vals = [_model_median, nl["constant"].ape.median(), nl["power"].ape.median()]
    w2 = [ins.ratio.between(.5, 2).mean() * 100,
          nl["constant"].ratio.between(.5, 2).mean() * 100,
          nl["power"].ratio.between(.5, 2).mean() * 100]
    cols = [F.OURS, F.REFUSED, F.REFUSED]
    x = np.arange(3)
    axc.bar(x, vals, color=cols, width=0.62, zorder=3)
    for i, (v, w) in enumerate(zip(vals, w2)):
        axc.text(i, v + max(vals) * 0.03, f"{v:.0f}%", ha="center", fontsize=7.5,
                 fontweight="bold" if i == 0 else "normal",
                 color=F.OURS if i == 0 else "#777777")
        # inside the bar, on two lines, so it cannot run past the bar edge
        axc.text(i, max(vals) * 0.06, f"{w:.0f}%\nwithin\n2$\\times$", ha="center", va="bottom",
                 fontsize=5.8, linespacing=1.15,
                 color="white" if i == 0 else "#4A4A4A")
    axc.set_xticks(x)
    axc.set_xticklabels(labels, fontsize=6.3)
    axc.set_xlim(-0.72, 2.72)
    axc.set_ylabel("median error (%)")
    # 1.30 left a quarter of the panel empty above the tallest bar; 1.14 clears the value labels
    # and no more.
    axc.set_ylim(0, max(vals) * 1.14)
    axc.set_title("nulls fitted with chemistry held out", fontsize=7.5, pad=3)
    F.panel_label(axc, "b", dx=-0.32)

    F.save(fig, "fig04_blind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
