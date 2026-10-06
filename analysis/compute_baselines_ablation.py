"""Two things a referee will ask for that the paper cannot currently answer.

(1) A PUBLISHED-METHOD BASELINE. The paper compares only against uncorrected DFT and two
    compound-blind nulls. A referee will ask how the method fares against an existing published
    predictor on the same compounds. We already hold one: 245 half Heuslers in the corpus carry a
    semi-empirical Slack / Debye-Callaway estimate, which is a real published method with its own
    literature. The paper currently uses those rows only as something to exclude. Scored on the
    same held-out compounds they become the external comparator the paper lacks.

(2) AN ABLATION OF THE EXPONENT. The paper reports a two-parameter transfer function and shows
    that imposing the Callaway constraint p = 1 - c costs accuracy. It never reports what a SINGLE
    constant does. If kappa_expt = c * kappa_BTE scores about as well, then "two-parameter transfer
    function" is one parameter with decoration, and it is much better to find that out here than
    from a referee.

Both are computed under the paper's own protocol: leave-one-chemistry-out, in domain, per-compound
median error, so they drop straight into Table 1.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import wilcoxon

import extend_blind_test as E
import shared_constant as SCN
from make_paper_predictions import structure_status, vec
from run_loco_chemistry import cluster_of

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
OUT = "data/exports/kappa_v2/baselines_ablation.json"
SEMI = "semi-empirical|slack|debye-callaway"


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def per_compound(d, col):
    return (d.assign(a=(d[col] - d.k_ref).abs() / d.k_ref * 100, r=d[col] / d.k_ref)
             .groupby("compound").agg(a=("a", "median"), r=("r", "median")))


def summarise(pc, label):
    return dict(label=label, n=int(len(pc)),
                median_ape=round(float(pc.a.median()), 1),
                within_2x_pct=round(float(pc.r.between(0.5, 2.0).mean() * 100), 1),
                bias=round(float(pc.r.median()), 2))


def fit_c_only(tr):
    """Single-constant transfer: kappa_expt = min(c, 1) * kappa_BTE.

    ONE GRID WITH THE HEADLINE (FIXPASS S9). This had its own grid, 0.20-1.00 in 161 steps, while
    the seed-averaged Table 1 ablation (compute_seed_averaged.fit_c_only) used extend_blind_test.CS;
    the same "one-parameter" row could then come out differently in two tables. It now delegates."""
    from compute_seed_averaged import fit_c_only as _f
    return _f(tr)


def main() -> int:
    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    d = B[B.klass == "half"].copy()
    d["_v"] = d.compound.map(vec)
    _s = {x: structure_status(str(x)) for x in d.compound.unique()}
    d = d[(d._v == 18) & d.compound.map(lambda x: _s[x][0] and not _s[x][1])].copy()
    d["chem"] = d.compound.map(lambda c: cluster_of(c) or "?")
    print(f"in-domain compounds: {d.compound.nunique()}")

    R = {"_generated_by": "compute_baselines_ablation.py",
         "seed": 0,
         "seed_note": "model predictions from target_blind_test.csv (seed 0); seed medians of the "
                      "one-parameter row are in seed_averaged_indomain.json",
         "protocol": "leave-one-chemistry-out, in domain, per-compound median error; both transfer "
                    "arms fitted on published calculations without the held-out cluster"}

    # ---- the paper's method, and the one-parameter ablation, both nested ---------------------
    rows_two, rows_one = [], []
    for cl in sorted(set(d.chem)):
        te, tr = d[d.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        # both arms fitted on the published calculation pairs without this cluster, not on the
        # other clusters' model predictions (PREREG_shared_constant_on_calculations rule 4); the
        # one-parameter arm keeps this script's own c grid
        c2, p2 = SCN.shared_cp_without_cluster(cl)
        if np.isfinite(c2):
            rows_two.append(te.assign(k=te.k_pred.values * E.apply_cp(te["T"].values, c2, p2)))
        c1 = SCN.shared_c_only_without_cluster(cl)   # default fitter: E.CS grid (headline's)
        rows_one.append(te.assign(k=te.k_pred.values * min(c1, 1.0)))
    two = per_compound(pd.concat(rows_two), "k")
    one = per_compound(pd.concat(rows_one), "k")
    R["two_parameter"] = summarise(two, "kappa_BTE * min(c (T/300)^p, 1)")
    R["one_parameter"] = summarise(one, "kappa_BTE * c   (no temperature term)")
    common = two.index.intersection(one.index)
    _, pw = wilcoxon(two.a[common], one.a[common])
    R["exponent_earns_its_keep"] = {
        "delta_pp": round(float(one.a.median() - two.a.median()), 1),
        "wilcoxon_p": float(f"{pw:.4f}"),
        "two_param_better_pct": round(float((two.a[common] < one.a[common]).mean() * 100), 1)}

    # ---- published-method baseline: Slack / Debye-Callaway on the same compounds -------------
    tr_all = pd.read_csv(TRAIN, low_memory=False)
    tr_all["red"] = tr_all.formula.map(red)
    tr_all["k"] = pd.to_numeric(tr_all.kappa_L, errors="coerce")
    tr_all["T"] = pd.to_numeric(tr_all.temperature_K, errors="coerce")
    m = tr_all.method.astype(str).str.lower()
    semi = tr_all[m.str.contains(SEMI, regex=True) & tr_all.k.notna() & (tr_all.k > 0)]
    hits = []
    for cp_, g in d.groupby("compound"):
        s = semi[semi.red == cp_]
        if not len(s):
            continue
        for T, kref in zip(g["T"].values, g.k_ref.values):
            j = int(np.abs(s["T"].values - T).argmin())
            if abs(s["T"].values[j] - T) <= 50:
                hits.append(dict(compound=cp_, k_ref=kref, k_semi=float(s.k.values[j])))
    if hits:
        H = pd.DataFrame(hits)
        pc = (H.assign(a=(H.k_semi - H.k_ref).abs() / H.k_ref * 100, r=H.k_semi / H.k_ref)
                .groupby("compound").agg(a=("a", "median"), r=("r", "median")))
        R["published_semi_empirical"] = {
            **summarise(pc, "Slack / Debye-Callaway (published method)"),
            "note": ("scored on the in-domain compounds that carry such an estimate; it is not a "
                     "fitted model so no holdout applies, which if anything favours it")}
        ours = two.loc[two.index.intersection(pc.index)]
        theirs = pc.loc[ours.index]
        _, pv = wilcoxon(ours.a, theirs.a)
        R["vs_published"] = {"n_common": int(len(ours)),
                             "ours": round(float(ours.a.median()), 1),
                             "theirs": round(float(theirs.a.median()), 1),
                             "wilcoxon_p": float(f"{pv:.4f}"),
                             "ours_better_pct": round(float((ours.a < theirs.a).mean() * 100), 1)}
    else:
        R["published_semi_empirical"] = {"_unavailable": "no in-domain compound carries one"}

    json.dump(R, open(OUT, "w"), indent=2)

    print(f"\n  {'':<44}{'n':>4}{'median':>9}{'within2x':>10}{'bias':>7}")
    for k in ("two_parameter", "one_parameter", "published_semi_empirical"):
        v = R.get(k, {})
        if "median_ape" in v:
            print(f"  {v['label']:<44}{v['n']:>4}{v['median_ape']:>8.1f}%"
                  f"{v['within_2x_pct']:>9.1f}%{v['bias']:>7.2f}")
    e = R["exponent_earns_its_keep"]
    print(f"\n  dropping the exponent costs {e['delta_pp']:+.1f} pp "
          f"(Wilcoxon p = {e['wilcoxon_p']}, two-parameter better on "
          f"{e['two_param_better_pct']:.0f}% of compounds)")
    if "vs_published" in R:
        v = R["vs_published"]
        print(f"  vs published semi-empirical on {v['n_common']} shared compounds: "
              f"ours {v['ours']}% vs {v['theirs']}%, p = {v['wilcoxon_p']}, "
              f"ours better on {v['ours_better_pct']:.0f}%")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
