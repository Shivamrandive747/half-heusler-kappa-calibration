"""Rebuild the conformal prediction intervals on the population the paper actually claims.

The bands quoted in Section 3.6 and applied to every issued prediction were taken from
`interval_validation.json`, whose "half, support>=3" set contains 46 compounds -- including
compounds outside the declared domain -- and whose residuals come from a calibration fitted on the
pooled half AND full Heusler blind set. Neither matches the predictor the paper reports:

  * the domain restriction is central to every other number in the paper;
  * `extend_blind_test.py` states in its own docstring that half and full Heuslers are never
    pooled, so a band built from a pooled fit contradicts the script that produced it.

The consequence is not cosmetic. A band that is too wide is not conservative in any useful sense:
it is a quantitative claim about coverage that the method does not make, and it makes every issued
prediction look less informative than the validation supports.

Construction, unchanged in principle from what the paper describes: absolute residuals in log space
(the error is multiplicative), and the half-width at nominal level alpha is the ceil((m+1)*alpha)-th
order statistic. Coverage is evaluated leave-one-chemistry-out -- the band applied to a held-out
cluster is built only from the residuals of the other clusters -- so the reported coverage is
out-of-sample rather than the level that was requested.
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

import extend_blind_test as E
import shared_constant as SCN
from make_paper_predictions import structure_status, vec, yzfam
from transfer_forms import apply_family
from run_loco_chemistry import cluster_of

OUT = "data/exports/kappa_v2/conformal_indomain.json"
FAMCAL: dict = {}
LEVELS = (0.50, 0.80, 0.90)


def band(res, alpha):
    """Conformal half-width in log10 units: the ceil((m+1)*alpha)-th smallest absolute residual."""
    r = np.sort(np.abs(np.asarray(res)))
    m = len(r)
    if m == 0:
        return np.nan
    k = int(np.ceil((m + 1) * alpha))
    return float(r[min(k, m) - 1])


def _load_famcal() -> dict:
    """The adopted family calibrations, or {} when the file is absent."""
    import os
    fp = "data/exports/kappa_v2/family_calibration.json"
    if not os.path.exists(fp):
        return {}
    fc = json.load(open(fp))
    return {f: (r["c"], r["p"], r.get("form", "SHAPE")) for f, r in fc["families"].items()
            if r.get("adopted")}


LOO: dict = {}


def _load_loo() -> dict:
    """compound -> (c, p) of its family form fitted WITHOUT it (None: single-member family).

    p is the exponent it was scored with when held out -- compute_deployed_route's p_family_loo,
    i.e. shared_constant.shared_p_excluding([compound]) -- not the deployed shared p."""
    import os
    fp = "data/exports/kappa_v2/family_deployed_pooled.csv"
    if not os.path.exists(fp):
        return {}
    P = pd.read_csv(fp)
    if "c_family_loo" not in P.columns:
        return {}
    out = {}
    for r in P.itertuples():
        if r.arm == "family" and pd.notna(r.c_family_loo):
            p_loo = getattr(r, "p_family_loo", None)
            if p_loo is None or pd.isna(p_loo):
                p_loo = SCN.shared_p_excluding([r.compound])
            out[r.compound] = (float(r.c_family_loo), float(p_loo))
        elif r.arm == "family (in-sample)":
            out[r.compound] = None
    return out


SEED0 = "data/exports/kappa_v2/target_blind_test.csv"
QUOTE_T = (300.0, 500.0)    # the temperatures predictions are quoted at (PREREG amendment B)


def seed_files() -> dict:
    """seed -> blind-test file: seed 0 is target_blind_test.csv, seeds 1-4 paper/evidence."""
    import glob
    files = {0: SEED0}
    for f in sorted(glob.glob("paper/evidence/blind_d2_s*.csv")):
        s_ = int(f[:-4].split("_s")[-1])
        files.setdefault(s_, f)
    return files


def residual_rows(path: str) -> pd.DataFrame:
    """Every in-domain blind-test row with its log10 residual under the calibration it would
    actually receive (nested: shared constant without its cluster, family constant without it)."""
    B = pd.read_csv(path)
    d = B[B.klass == "half"].copy()
    d["_v"] = d.compound.map(vec)
    st = {x: structure_status(str(x)) for x in d.compound.unique()}
    d = d[(d._v == 18) & d.compound.map(lambda x: st[x][0] and not st[x][1])].copy()
    d["chem"] = d.compound.map(lambda c: cluster_of(c) or "?")
    rows = []
    for cl in sorted(set(d.chem)):
        te, tr = d[d.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        # the global arm: the shared constant fitted on published calculations without this
        # cluster (shared_constant.py; PREREG_shared_constant_on_calculations rule 4), no longer a
        # fit on the other clusters' model predictions
        c, p = SCN.shared_cp_without_cluster(cl)
        if not np.isfinite(c):
            continue
        # EACH COMPOUND IS SCORED UNDER THE CALIBRATION IT WOULD ACTUALLY RECEIVE.
        #
        # A band describes the residuals of the method as deployed. Families carry their own
        # constants (family_calibration.py), held FIXED across folds: they are derived from the
        # published-BTE route, not from this blind-test frame.
        #
        # A FAMILY CONSTANT MUST NOT HAVE SEEN THE COMPOUND IT IS SCORED ON. A member of the
        # published route uses the constant its family gives when it is LEFT OUT
        # (compute_deployed_route, c_family_loo -- references re-chosen inside that fold); a
        # single-member family has no such constant, so its member is scored on the global arm; a
        # compound outside the published route takes the family constant as it stands. The family
        # arm is uncapped since 2026-10-01, the global arm keeps the cap.
        T_ = te["T"].values.astype(float)
        k = te.k_pred.values * np.minimum(c * (T_ / 300.0) ** p, 1.0)
        if FAMCAL:
            for i, (cmp_, f) in enumerate(zip(te.compound.values, te.compound.map(yzfam).values)):
                if f not in FAMCAL:
                    continue
                cf, pf, form = FAMCAL[f]
                if cmp_ in LOO:
                    if LOO[cmp_] is None:
                        continue                      # single-member family: global arm
                    cf, pf = LOO[cmp_]
                k[i] = float(apply_family(form, (cf, pf), [T_[i]], [te.k_pred.values[i]])[0])
        rows.append(te.assign(k_cal=k, res=np.log10(k) - np.log10(te.k_ref.values)))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["compound", "chem", "T", "res"])


def residual_at(D: pd.DataFrame, Tq: float) -> pd.DataFrame:
    """ONE residual per compound: the one at the bin nearest the quoted temperature Tq (amendment
    B). A compound whose nearest bins are equidistant on both sides takes the median of the two."""
    out = []
    for (cmp_, chem), g in D.groupby(["compound", "chem"]):
        dist = (g["T"] - Tq).abs()
        sel = g[dist <= dist.min() + 1e-9]
        out.append(dict(compound=cmp_, chem=chem, res=float(sel.res.median()),
                        T_used=float(sel["T"].median()), dT=float(dist.min())))
    return pd.DataFrame(out, columns=["compound", "chem", "res", "T_used", "dT"])


def levels_for(per: pd.DataFrame) -> dict:
    """Band and leave-one-cluster-out coverage per nominal level for one residual set."""
    clusters = sorted(set(per.chem))
    out = {}
    for a in LEVELS:
        inside, tot, widths = 0, 0, []
        for cl in clusters:
            te, tr = per[per.chem == cl], per[per.chem != cl]
            if not len(te) or len(tr) < 4:
                continue
            h = band(tr.res.values, a)
            widths.append(h)
            inside += int((te.res.abs() <= h).sum())
            tot += len(te)
        h_all = band(per.res.values, a)
        out[f"{a:.2f}"] = dict(factor=float(10 ** h_all),
                               coverage=(inside / tot * 100 if tot else float("nan")),
                               n_inside=int(inside), n_scored=int(tot),
                               fold_lo=float(10 ** min(widths)) if widths else float("nan"),
                               fold_hi=float(10 ** max(widths)) if widths else float("nan"))
    return out


def _aggregate(per_seed: dict) -> dict:
    """Median over seeds of each level's band and coverage, with every seed's value beside it."""
    out = {}
    for a in (f"{x:.2f}" for x in LEVELS):
        f_ = {s_: v[a]["factor"] for s_, v in per_seed.items()}
        cv = {s_: v[a]["coverage"] for s_, v in per_seed.items()}
        out[a] = dict(
            factor=round(float(np.median(list(f_.values()))), 2),
            empirical_coverage_pct=round(float(np.median(list(cv.values()))), 1),
            factor_per_seed={str(k): round(float(v), 2) for k, v in sorted(f_.items())},
            factor_seed_range=[round(float(min(f_.values())), 2), round(float(max(f_.values())), 2)],
            coverage_per_seed={str(k): round(float(v), 1) for k, v in sorted(cv.items())},
            n_scored_per_seed={str(k): int(per_seed[k][a]["n_scored"]) for k in sorted(per_seed)},
            factor_range_across_folds_seed0=(
                [round(per_seed[0][a]["fold_lo"], 2), round(per_seed[0][a]["fold_hi"], 2)]
                if 0 in per_seed else None))
    return out


def main() -> int:
    global FAMCAL
    FAMCAL = _load_famcal()
    global LOO
    LOO = _load_loo()
    print(f"family calibrations applied: {sorted(FAMCAL) or 'none'}")
    files = seed_files()
    print(f"seed files: {files}")

    # BANDS PER SEED; the deployed band is the median over seeds (FIXPASS S4: this used seed 0 only).
    # RESIDUAL AT THE QUOTED TEMPERATURE (PREREG amendment B): one residual per compound, at the
    # bin nearest 300 K for the 300 K band and nearest 500 K for the 500 K band -- not the median
    # over all temperatures, which described an error no single quoted value carries.
    by_T = {Tq: {} for Tq in QUOTE_T}
    over_T = {}
    n_comp = {}
    for s_, path in sorted(files.items()):
        D = residual_rows(path)
        n_comp[s_] = int(D.compound.nunique())
        for Tq in QUOTE_T:
            by_T[Tq][s_] = levels_for(residual_at(D, Tq))
        over_T[s_] = levels_for(D.groupby(["compound", "chem"]).res.median().reset_index())
        print(f"  seed {s_}: {n_comp[s_]} compounds   90% band at 300 K "
              f"{by_T[300.0][s_]['0.90']['factor']:.2f}x, at 500 K "
              f"{by_T[500.0][s_]['0.90']['factor']:.2f}x")

    R = {"_generated_by": "compute_conformal_indomain.py",
         "population": ("the declared domain (VEC=18, cubic, non-polymorphic); compounds per seed "
                        + json.dumps({str(k): v for k, v in n_comp.items()})),
         "construction": ("absolute log10 residuals; half-width = ceil((m+1)*alpha) order statistic; "
                          "one residual per compound, at the bin nearest the quoted temperature "
                          "(PREREG amendment B)"),
         "seeds": sorted(files),
         "seed_rule": "every band is the median over the model seeds; per-seed values beside it",
         "levels": _aggregate(by_T[300.0]),
         "levels_quoted_T_K": 300,
         "levels_500K": _aggregate(by_T[500.0]),
         "superseded_median_over_temperatures": _aggregate(over_T)}

    print(f"\n  {'nominal':>8}{'300 K':>10}{'500 K':>10}{'coverage':>10}   per seed (300 K)")
    for a in R["levels"]:
        v, w = R["levels"][a], R["levels_500K"][a]
        print(f"  {float(a) * 100:>7.0f}%{v['factor']:>9.2f}x{w['factor']:>9.2f}x"
              f"{v['empirical_coverage_pct']:>9.1f}%   {v['factor_per_seed']}")

    prev = {"0.50": 1.32, "0.80": 2.06, "0.90": 3.21}
    R["superseded_bands"] = dict(
        values=prev,
        why=("taken from interval_validation.json's 'half, support>=3' set: 46 compounds including "
             "out-of-domain ones, with residuals from a calibration fitted on the pooled half and "
             "full Heusler blind set. Both differ from the predictor this paper reports."))

    json.dump(R, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
