"""The family power-law baseline: does the method beat it, and where does it add anything?

A review found, and independent computation confirmed, that a power law fitted to the MEASURED
kappa(T) of a compound's own (Y,Z) bonding family matches this paper's transfer function on median
error. That baseline needs no first-principles calculation, no machine learning and no calibration.
It has to go in the paper.

But "matches on median error" is not the whole comparison, and two things decide whether the method
has any defensible advantage:

  1 COVERAGE. The family null needs at least two measured same-family compounds outside the
    held-out cluster. Where it has them the method must justify its extra machinery; where it does
    not, the method is the only option. The decisive case is the five compounds the paper actually
    issues predictions for.

  2 WITHIN-FAMILY DISCRIMINATION. This is the real question. A family power law returns ONE curve
    per family, so every member of a family receives the same prediction at a given temperature. It
    cannot rank ErSbPt against HoSbPt against TmSbPt -- and choosing between the members of a
    substitution series is precisely what a screening user does. The transfer function, carrying a
    compound-specific calculation, can.

Everything is leave-one-chemistry-out, in domain, per-compound median error.
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
from scipy.stats import spearmanr, wilcoxon

import extend_blind_test as E
import shared_constant as SCN
from make_paper_predictions import structure_status, vec
from run_loco_chemistry import cluster_of

OUT = "data/exports/kappa_v2/family_null.json"
PRED = "data/Target_Materials/PAPER_PREDICTIONS.csv"


def issued_set() -> set:
    """The compounds the paper issues, read from the file that decides them.

    This was a hardcoded literal, {ErSbPt, HoSbPt, TmSbPt, GdNiSb, TbNiSb}, and it went stale: the
    issued set is now {TmSbPt, HfCoBi, GdNiSb, TbNiSb}, so the coverage answer below was being
    computed for two compounds the paper no longer issues and missing one it does. A set of
    compound names typed into a script is the exact failure this project keeps rediscovering, so it
    is read from PAPER_PREDICTIONS.csv instead.
    """
    p = pd.read_csv(PRED)
    return set(p[p.status.astype(str).str.upper() == "ISSUED"].compound.astype(str))


def yz(f):
    e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
    return f"{e[1]}-{e[2]}" if len(e) == 3 else None


def main() -> int:
    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    d = B[B.klass == "half"].copy()
    d["_v"] = d.compound.map(vec)
    st = {x: structure_status(str(x)) for x in d.compound.unique()}
    d = d[(d._v == 18) & d.compound.map(lambda x: st[x][0] and not st[x][1])].copy()
    d["chem"] = d.compound.map(lambda c: cluster_of(c) or "?")
    d["fam"] = d.compound.map(yz)

    model, fam = [], []
    for cl in sorted(set(d.chem)):
        te, tr = d[d.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        # shared constant on published calculations without this cluster (prereg rule 4); the
        # family power-law null below is unchanged
        c, p = SCN.shared_cp_without_cluster(cl)
        if np.isfinite(c):
            model.append(te.assign(k=te.k_pred.values * E.apply_cp(te["T"].values, c, p)))
        for f, g in te.groupby("fam"):
            trf = tr[tr.fam == f]
            if trf.compound.nunique() < 2:
                continue
            q, la = np.polyfit(np.log(trf["T"].values / 300.0), np.log(trf.k_ref.values), 1)
            fam.append(g.assign(k=np.exp(la) * (g["T"].values / 300.0) ** q))

    def pc(frames):
        D = pd.concat(frames)
        return (D.assign(a=(D.k - D.k_ref).abs() / D.k_ref * 100, r=D.k / D.k_ref)
                 .groupby("compound").agg(a=("a", "median"), r=("r", "median"),
                                          ref=("k_ref", "median"), k=("k", "median"),
                                          fam=("fam", "first")))

    M, F = pc(model), pc(fam)
    common = M.index.intersection(F.index)
    _, pv = wilcoxon(M.a[common], F.a[common])
    R = {"_generated_by": "compute_family_null.py",
         "seed": 0,
         "seed_note": "model arm from target_blind_test.csv (seed 0); the family null is seed-free",
         "model": dict(n=int(len(M)), median_ape=round(float(M.a.median()), 1),
                       within_2x=round(float(M.r.between(.5, 2).mean() * 100), 1)),
         "family_null": dict(n=int(len(F)), median_ape=round(float(F.a.median()), 1),
                             within_2x=round(float(F.r.between(.5, 2).mean() * 100), 1)),
         "head_to_head": dict(n=int(len(common)),
                              model=round(float(M.a[common].median()), 1),
                              family=round(float(F.a[common].median()), 1),
                              wilcoxon_p=round(float(pv), 3),
                              model_better_pct=round(float((M.a[common] < F.a[common]).mean() * 100), 1))}

    missed = sorted(set(M.index) - set(F.index))
    R["family_null_cannot_cover"] = missed

    # COVERAGE AT THE PREDICTIONS, which is the question a reader actually has.
    #
    # This used to read `ISSUED & set(F.index)`, intersecting the issued compounds with the
    # VALIDATION set. An issued compound is unmeasured by definition, so it can never appear in the
    # validation set and the intersection was empty every time -- a check that could not fail, and
    # so reported nothing. The real question is whether the cheap baseline could have produced a
    # number for each issued compound at all: that needs two measured compounds in its own bonding
    # family, which is exactly the condition applied to the validation set above.
    iss = issued_set()
    meas_per_fam = d.groupby("fam").compound.nunique()
    cov = {}
    for cmpd in sorted(iss):
        f = yz(cmpd)
        n = int(meas_per_fam.get(f, 0))
        cov[cmpd] = dict(family=f, measured_family_members=n, family_null_could_predict=n >= 2)
    R["issued_family_null_coverage"] = cov
    R["issued_covered_by_family_null"] = sorted(c for c, v in cov.items()
                                                if v["family_null_could_predict"])
    R["_issued_coverage_meaning"] = (
        "For each compound the paper issues: how many MEASURED compounds its bonding family has, "
        "and so whether the calculation-free family power law could have produced a number for it. "
        "Where it could, the transfer function has to justify its extra machinery; where it could "
        "not, it is the only route to a value.")
    print(f"  issued coverage: " + ", ".join(
        f"{c} ({v['family']}, {v['measured_family_members']} measured"
        f"{' -- null covers' if v['family_null_could_predict'] else ' -- null CANNOT'})"
        for c, v in cov.items()))
    print(f"  model      n={len(M)}  median {M.a.median():.1f}%")
    print(f"  family null n={len(F)}  median {F.a.median():.1f}%")
    print(f"  head to head on {len(common)}: model {M.a[common].median():.1f}% vs "
          f"{F.a[common].median():.1f}%, p={pv:.3f}, model better on "
          f"{(M.a[common] < F.a[common]).mean()*100:.0f}%")
    print(f"  family null cannot cover {len(missed)}: {missed}")

    # --- THE DECISIVE TEST: can either rank compounds WITHIN a family? --------------------
    print("\n  within-family discrimination (a family null gives every member the same value):")
    rows = []
    for f, g in M.join(F.k.rename("k_fam"), how="inner").groupby("fam"):
        if len(g) < 3:
            continue
        rm, _ = spearmanr(g.ref, g.k)
        rf, _ = spearmanr(g.ref, g.k_fam)
        spread_f = float(g.k_fam.max() / g.k_fam.min())
        rows.append(dict(fam=f, n=int(len(g)), rho_model=round(float(rm), 3),
                         rho_family=round(float(rf), 3), family_spread=round(spread_f, 2)))
        print(f"    {f:<8}n={len(g)}  model rho={rm:+.3f}   family-null rho={rf:+.3f}   "
              f"family-null spread {spread_f:.2f}x")
    R["within_family"] = rows
    if rows:
        mm = float(np.median([r["rho_model"] for r in rows]))
        ff = float(np.median([r["rho_family"] for r in rows if np.isfinite(r["rho_family"])]))
        R["within_family_median_rho"] = dict(model=round(mm, 3), family_null=round(ff, 3))
        print(f"    median within-family rho: model {mm:+.3f}  family null {ff:+.3f}")

    json.dump(R, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
