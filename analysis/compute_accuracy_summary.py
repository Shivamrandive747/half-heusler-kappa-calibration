"""One accuracy table for every route the paper scores, with a symmetric metric and bootstrap CIs.

WHAT THIS IS. A RE-SCORING of per-compound predictions that already exist. Nothing is retrained,
no calibration procedure is new, and no data changes. Each route is scored by the exact function
its own producer uses:

    routes 1-3   compute_seed_averaged (in_domain, nested, per_cmp) on each of the five seed files
    route 2      compute_null_baselines' model_seed_median loop (all half Heuslers, `cluster` column)
    routes 4-5   compute_seed_averaged.null (constant / power-law, leave-one-chemistry-out)
    routes 6-7   family_deployed_pooled.csv (compute_deployed_route.py), family-arm compounds
    route 8      paper/evidence/shared_constant_test.json (test_shared_constant.py, lofo C0)

The shared constant applied to a held-out cluster is shared_constant.shared_cp_without_cluster,
and the global arm of route 6 is shared_constant.shared_cp_in_fold -- the same cached, deterministic
calls the existing producers make, so they return the constants already deployed, not new ones.
Route 6's per-compound global ratio is not written to family_deployed_pooled.csv (only `ape_global`
is), so it is recomputed here with compute_deployed_route's own expressions, and every recomputed
per-compound APE is asserted equal to the stored `ape_global` (1e-9) before the ratio is used.

DEFINITIONS (identical to the producers):
    per-compound APE    median over the compound's matched temperatures of |pred - ref| / ref
    within 2x           the per-compound median ratio pred/ref in [0.5, 2]
    |log10 ratio| (NEW) median over compounds of |log10(per-compound median ratio)|: symmetric, so
                        a factor-2 over- and under-prediction cost the same
For model routes every statistic is computed per seed file and the reported figure is the median
over the five seeds (target_blind_test.csv = seed 0, paper/evidence/blind_d2_s{1..4}.csv), exactly
as in_domain_table.seed_medians.

BOOTSTRAP. 5000 resamples of compounds, percentile 95% interval, a fresh
numpy.random.default_rng(20261011) per route so a route's interval does not depend on route order.
For model routes the resampled statistic IS the reported one: one set of compound indices is drawn
and applied to all five seeds together, each seed's median (or within-2x share) is computed on the
resample, and the median over seeds is taken. The interval therefore brackets the five-seed median
that is quoted, rather than a seed-0 figure or a median of per-compound seed medians (a different
statistic from the one in the table).

ASSERTIONS. Every median and within-2x is asserted equal, to the printed precision, to the artefact
the registry copies it from (seed_averaged_indomain.json, null_baselines.json, results_indomain.json,
deployed_route.json, shared_constant_test.json); run standalone it is also asserted against
paper_numbers.json. In rerun_downstream.sh it runs with --chain, which skips only the registry
comparison: the registry is rebuilt after this step, and make_paper_numbers cross-checks this file
against the leaves it has just written.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import glob
import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import compute_seed_averaged as CSA
import extend_blind_test as E
import shared_constant as SCN

EX = "data/exports/kappa_v2"
OUT = f"{EX}/accuracy_summary.json"
SEED0 = f"{EX}/target_blind_test.csv"
N_BOOT = 5000
RNG_SEED = 20261011


# ---------------------------------------------------------------------------------- helpers
def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def seed_files() -> dict[int, str]:
    files = {0: SEED0}
    for f in sorted(glob.glob("paper/evidence/blind_d2_s*.csv")):
        files.setdefault(int(f[:-4].split("_s")[-1]), f)
    return files


def symlog(r) -> float:
    return float(np.median(np.abs(np.log10(np.asarray(r, dtype=float)))))


def pct_ci(x) -> list[float]:
    lo, hi = np.percentile(np.asarray(x, dtype=float), [2.5, 97.5])
    return [round(float(lo), 1), round(float(hi), 1)]


def boot_idx(n: int) -> np.ndarray:
    return np.random.default_rng(RNG_SEED).integers(0, n, size=(N_BOOT, n))


def single_route(ape, ratio) -> dict:
    """Point estimates and bootstrap CIs for a route with one prediction per compound."""
    a, r = np.asarray(ape, float), np.asarray(ratio, float)
    w = ((r >= 0.5) & (r <= 2.0)).astype(float)
    idx = boot_idx(len(a))
    return dict(n=int(len(a)), median_ape=round(float(np.median(a)), 1),
                within_2x_pct=round(float(w.mean() * 100), 1),
                median_abs_log10_ratio=round(symlog(r), 3),
                ci95_median_ape=pct_ci(np.median(a[idx], axis=1)),
                ci95_within_2x_pct=pct_ci(w[idx].mean(axis=1) * 100))


def seeded_route(A: pd.DataFrame, Rt: pd.DataFrame) -> dict:
    """A, Rt: compounds x seeds frames of per-compound APE and per-compound median ratio."""
    assert list(A.columns) == list(Rt.columns) and (A.index == Rt.index).all()
    assert not A.isna().any().any(), "a compound is missing from a seed file"
    a, r = A.values, Rt.values
    w = ((r >= 0.5) & (r <= 2.0)).astype(float)
    med_s = np.median(a, axis=0)
    w2_s = w.mean(axis=0) * 100
    sl_s = np.median(np.abs(np.log10(r)), axis=0)
    idx = boot_idx(len(a))
    bm = np.median(np.median(a[idx], axis=1), axis=1)         # (B, seeds) -> (B,)
    bw = np.median(w[idx].mean(axis=1) * 100, axis=1)
    seeds = [int(s) for s in A.columns]
    return dict(n=int(len(a)), n_seeds=len(seeds),
                median_ape=round(float(np.median(med_s)), 1),
                within_2x_pct=round(float(np.median(w2_s)), 1),
                median_abs_log10_ratio=round(float(np.median(sl_s)), 3),
                seed_range=dict(median_ape=[round(float(med_s.min()), 1), round(float(med_s.max()), 1)],
                                within_2x_pct=[round(float(w2_s.min()), 1), round(float(w2_s.max()), 1)],
                                median_abs_log10_ratio=[round(float(sl_s.min()), 3),
                                                        round(float(sl_s.max()), 3)]),
                per_seed={str(s): dict(median_ape=round(float(m), 1), within_2x_pct=round(float(x), 1),
                                       median_abs_log10_ratio=round(float(y), 3))
                          for s, m, x, y in zip(seeds, med_s, w2_s, sl_s)},
                ci95_median_ape=pct_ci(bm), ci95_within_2x_pct=pct_ci(bw))


def stack(per_seed: dict[int, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({s: v for s, v in sorted(per_seed.items())})


# ---------------------------------------------------------------------------------- routes
def in_domain_frames():
    """Per seed: in-domain frame, calibrated / uncorrected per-compound frames (compute_seed_averaged)."""
    D, M, U = {}, {}, {}
    for s, f in sorted(seed_files().items()):
        raw = pd.read_csv(f)
        raw = raw[raw.klass == "half"] if "klass" in raw.columns else raw
        d = CSA.in_domain(raw)
        D[s] = d
        M[s] = CSA.per_cmp(CSA.nested(d))
        U[s] = CSA.per_cmp(d.assign(k=d.k_pred))
    return D, M, U


def all50_frames():
    """Per seed: per-compound APE / ratio on every half Heusler, compute_null_baselines' loop."""
    A, R = {}, {}
    for s, f in sorted(seed_files().items()):
        Hs = pd.read_csv(f)
        Hs = Hs[Hs.klass == "half"].dropna(subset=["T", "k_ref", "k_pred"])
        ap, ra = {}, {}
        for cl in sorted(Hs.cluster.unique()):
            c_, p_ = SCN.shared_cp_without_cluster(cl)
            for cp_, g in Hs[Hs.cluster == cl].groupby("compound"):
                kk = g.k_pred.values * E.apply_cp(g["T"].values, c_, p_)
                ap[cp_] = float(np.median(np.abs(kk - g.k_ref.values) / g.k_ref.values * 100))
                ra[cp_] = float(np.median(kk / g.k_ref.values))
        A[s], R[s] = pd.Series(ap), pd.Series(ra)
    return stack(A), stack(R)


def family_global_ratio(compounds) -> pd.DataFrame:
    """compute_deployed_route's global arm, per held-out compound: (ape, ratio)."""
    import family_calibration as FC
    out = {}
    for cpd in compounds:
        fold = FC.fold_pairs(cpd)
        te = fold[fold.compound == cpd]
        T = te["T"].values.astype(float)
        c_h, p_h = SCN.shared_cp_in_fold(cpd)
        k = te.k_pred.values * E.apply_cp(T, c_h, p_h)
        out[cpd] = dict(ape=float(FC.per_compound_ape(te, k).iloc[0]),
                        ratio=float(np.median(k / te.k_ref.values)))
    return pd.DataFrame(out).T


# ---------------------------------------------------------------------------------- checks
CHECKS: list[tuple[str, float, float, str]] = []      # (label, computed, expected, source)


def check(label, got, want, source):
    CHECKS.append((label, float(got), float(want), source))


def main() -> int:
    chain = "--chain" in sys.argv
    sa = jload(f"{EX}/seed_averaged_indomain.json")
    nb = jload(f"{EX}/null_baselines.json")
    ri = jload(f"{EX}/results_indomain.json")
    dr = jload(f"{EX}/deployed_route.json")
    sct = jload("paper/evidence/shared_constant_test.json")

    out = {"_generated_by": "compute_accuracy_summary.py",
           "_what": ("re-scoring of existing per-compound predictions; no model or calibration "
                     "procedure is refitted beyond the deterministic cached shared-constant calls "
                     "the original producers make"),
           "definitions": {
               "per_compound_ape": "median over matched temperatures of |pred - ref| / ref, in %",
               "within_2x": "per-compound median ratio pred/ref in [0.5, 2]; % of compounds",
               "median_abs_log10_ratio": ("NEW, symmetric: median over compounds of |log10(per-compound "
                                          "median ratio)|; 0.301 = a factor of 2"),
               "model_routes": ("each statistic computed per seed file (seed 0 = target_blind_test.csv, "
                                "seeds 1-4 = paper/evidence/blind_d2_s{1..4}.csv); reported value = "
                                "median over the five seeds, seed_range = min-max over them")},
           "bootstrap": {
               "resamples": N_BOOT, "rng": f"numpy.random.default_rng({RNG_SEED}), fresh per route",
               "interval": "percentile 2.5-97.5, compounds resampled with replacement",
               "model_routes": ("one draw of compound indices is applied to all five seeds jointly; on "
                                "each resample the per-seed median APE (within-2x share) is computed and "
                                "the median over seeds taken, so the CI is for the five-seed median that "
                                "is reported. Chosen over (a) a seed-0 CI, which would bracket a figure "
                                "the paper does not quote, and (b) bootstrapping per-compound seed "
                                "medians, whose median is a different statistic from the table's.")},
           "routes": {}}
    R = out["routes"]

    # ---- 1 & 3: in domain, ML + shared constant / uncorrected ----------------------------------
    D, M, U = in_domain_frames()
    cmp0 = sorted(M[0].index)
    for s in M:
        assert sorted(M[s].index) == cmp0 and sorted(U[s].index) == cmp0, \
            f"seed {s}: in-domain compound set differs from seed 0"
    MA, MR = stack({s: M[s].a for s in M}).loc[cmp0], stack({s: M[s].r for s in M}).loc[cmp0]
    UA, UR = stack({s: U[s].a for s in U}).loc[cmp0], stack({s: U[s].r for s in U}).loc[cmp0]
    R["ml_shared_in_domain"] = dict(
        holdout=("leave-one-chemistry-cluster-out (X-site cluster); shared constant refitted on the "
                 "published calculation pairs without the held-out cluster "
                 "(shared_constant.shared_cp_without_cluster); in domain = VEC 18 and cubic C1b, not "
                 "polymorphic"),
        **seeded_route(MA, MR))
    R["ml_uncorrected_in_domain"] = dict(
        holdout="model predictions as made (each seed's blind test holds out the chemistry cluster); no calibration",
        **seeded_route(UA, UR))
    t1 = sa["table1_seed_medians"]
    for key, blk in (("ml_shared_in_domain", "calibrated"), ("ml_uncorrected_in_domain", "uncorrected_surrogate")):
        check(f"{key}.median_ape", R[key]["median_ape"], t1[blk]["median_ape"]["median"], "seed_averaged_indomain.json")
        check(f"{key}.within_2x_pct", R[key]["within_2x_pct"], t1[blk]["within_2x"]["median"], "seed_averaged_indomain.json")
        for i in (0, 1):
            check(f"{key}.seed_range.median_ape[{i}]", R[key]["seed_range"]["median_ape"][i],
                  t1[blk]["median_ape"]["range"][i], "seed_averaged_indomain.json")
            check(f"{key}.seed_range.within_2x_pct[{i}]", R[key]["seed_range"]["within_2x_pct"][i],
                  t1[blk]["within_2x"]["range"][i], "seed_averaged_indomain.json")

    # ---- 2: all half Heuslers, ML + shared constant ---------------------------------------------
    AA, AR = all50_frames()
    R["ml_shared_all"] = dict(
        holdout=("leave-one-chemistry-cluster-out on every half Heusler in the blind test, no domain "
                 "screen (the blind test's `cluster` column, as compute_null_baselines)"),
        **seeded_route(AA, AR))
    ms = nb["model_seed_median"]
    check("ml_shared_all.median_ape", R["ml_shared_all"]["median_ape"], round(ms["median_ape"], 1), "null_baselines.json")
    check("ml_shared_all.within_2x_pct", R["ml_shared_all"]["within_2x_pct"], ms["within_2x_pct"], "null_baselines.json")
    check("ml_shared_all.n", R["ml_shared_all"]["n"], ms["n_compounds"], "null_baselines.json")
    for i, f in ((0, min), (1, max)):
        check(f"ml_shared_all.seed_range.median_ape[{i}]", R["ml_shared_all"]["seed_range"]["median_ape"][i],
              round(f(ms["per_seed"].values()), 1), "null_baselines.json")

    # ---- 4 & 5: nulls, in domain (seed-free: fitted on reference rows only) ----------------------
    d0 = D[0]
    NC, NP = CSA.per_cmp(CSA.null(d0, "constant")).loc[cmp0], CSA.per_cmp(CSA.null(d0, "power")).loc[cmp0]
    for s in D:     # the nulls see only k_ref and T, so every seed's in-domain frame gives the same null
        assert np.allclose(CSA.per_cmp(CSA.null(D[s], "constant")).loc[cmp0].a, NC.a)
    for key, P, lab, kind in (("null_constant_in_domain", NC, "a single constant kappa: the median of the training clusters' reference rows", "constant"),
                              ("null_power_law_in_domain", NP, "A (T/300)^q fitted on the training clusters' reference rows", "power")):
        R[key] = dict(holdout=f"leave-one-chemistry-cluster-out; {lab}; contains no compound information; seed-free",
                      **single_route(P.a.values, P.r.values))
        check(f"{key}.median_ape", R[key]["median_ape"], ri["nulls"][kind]["median_ape"], "results_indomain.json")
        check(f"{key}.within_2x_pct", R[key]["within_2x_pct"], ri["nulls"][kind]["within_2x_pct"], "results_indomain.json")

    # ---- 6 & 7: published calculation, family-test protocol --------------------------------------
    pool = pd.read_csv(f"{EX}/family_deployed_pooled.csv")
    fam = pool[pool.arm == "family"].set_index("compound").sort_index()
    assert sorted(fam.index) == sorted(dr["family_arm_compounds"]["compounds"]), \
        "family_deployed_pooled.csv family arm differs from deployed_route.json"
    G = family_global_ratio(list(fam.index)).loc[fam.index]
    bad = fam.index[~np.isclose(G.ape.values.astype(float), fam.ape_global.values, rtol=0.0, atol=1e-9)]
    assert not len(bad), f"recomputed global-arm APE differs from family_deployed_pooled.csv for {list(bad)}"
    R["published_calc_shared_family_test"] = dict(
        holdout=("published full calculation x shared constant refitted without the scored compound, on "
                 "the pairs as they stand in that compound's fold (shared_constant.shared_cp_in_fold); the "
                 "31 compounds scored held out on the family arm (compute_deployed_route)"),
        ratio_source=("recomputed with compute_deployed_route's own global-arm expressions (the CSV stores "
                      "ape_global but no global ratio); per-compound APE asserted equal to ape_global, 1e-9"),
        **single_route(fam.ape_global.values, G.ratio.values.astype(float)))
    R["published_calc_family_constant"] = dict(
        holdout=("published full calculation x family constant fitted leave-one-compound-out within the "
                 "family (references re-chosen inside the fold, PREREG amendment A)"),
        ratio_source="family_deployed_pooled.csv `ratio` (deployed/measured, per-compound median)",
        **single_route(fam.ape.values, fam.ratio.values))
    fa = dr["family_arm_compounds"]
    check("published_calc_shared_family_test.n", R["published_calc_shared_family_test"]["n"], fa["n"], "deployed_route.json")
    check("published_calc_shared_family_test.median_ape", R["published_calc_shared_family_test"]["median_ape"],
          fa["global_on_same_compounds"]["median_ape"], "deployed_route.json")
    check("published_calc_shared_family_test.within_2x_pct", R["published_calc_shared_family_test"]["within_2x_pct"],
          fa["global_on_same_compounds"]["within2x_pct"], "deployed_route.json")
    check("published_calc_family_constant.median_ape", R["published_calc_family_constant"]["median_ape"],
          fa["median_ape"], "deployed_route.json")
    check("published_calc_family_constant.within_2x_pct", R["published_calc_family_constant"]["within_2x_pct"],
          fa["within2x_pct"], "deployed_route.json")

    # ---- 8: shared constant on unseen families (no per-compound file exists) ---------------------
    c0 = sct["lofo"]["C0"]
    R["shared_constant_unseen_family"] = dict(
        holdout=("leave-one-bonding-family-out on the published calculation pairs; shared constant refitted "
                 "without the held-out family (test_shared_constant.py, candidate C0)"),
        n=int(c0["n"]), median_ape=float(c0["median_ape"]), within_2x_pct=float(c0["within_2x_pct"]),
        median_abs_log10_ratio=None, ci95_median_ape=None, ci95_within_2x_pct=None,
        _unavailable=("test_shared_constant.py builds the per-compound rows in memory and writes only the "
                      "summary (paper/evidence/shared_constant_test.json); no per-compound file exists, so "
                      "the symmetric metric and the CIs are not computed here (recomputing them would rerun "
                      "the leave-one-family-out fits)"),
        source="paper/evidence/shared_constant_test.json lofo.C0 (the source of the registry key)")

    # ---- paired statistics, in domain: model vs each null ----------------------------------------
    paired = {"population": f"in domain, n = {len(cmp0)}",
              "definition": ("per seed: per-compound model APE minus null APE (percentage points), its "
                             "median over compounds, and a paired Wilcoxon on per-compound APE "
                             "(compute_seed_averaged); reported = median over the five seeds. Negative "
                             "difference = the model is closer."),
              "bootstrap": ("compounds resampled jointly across seeds, per-seed median difference, median "
                            "over seeds; same rng rule as the routes"),
              "holm": ("Holm step-down over the two nulls, applied to the seed-median Wilcoxon p values; a "
                       "median of p values is a summary, not itself a p value, so the per-seed Holm maximum "
                       "is given beside it")}
    pvals, holm_per_seed = {}, {}
    for key, P in (("constant", NC), ("power", NP)):
        diff = MA.values - P.a.values[:, None]                  # compounds x seeds
        idx = boot_idx(len(diff))
        bd = np.median(np.median(diff[idx], axis=1), axis=1)
        ps = [float(wilcoxon(MA[s].values, P.a.values)[1]) for s in MA.columns]
        pvals[key] = float(np.median(ps))
        holm_per_seed[key] = ps
        paired[f"vs_{key}_null"] = dict(
            median_paired_diff_pp=round(float(np.median(np.median(diff, axis=0))), 1),
            median_paired_diff_pp_seed_range=[round(float(np.median(diff, axis=0).min()), 1),
                                              round(float(np.median(diff, axis=0).max()), 1)],
            ci95_median_paired_diff_pp=pct_ci(bd),
            wilcoxon_p_seed_median=round(pvals[key], 4),
            wilcoxon_p_per_seed={str(s): round(p, 4) for s, p in zip(MA.columns, ps)},
            model_better_pct_seed_median=round(float(np.median((diff < 0).mean(axis=0) * 100)), 1))
        check(f"paired.vs_{key}_null.wilcoxon_p_seed_median(3dp)", round(pvals[key], 3),
              sa[f"p_{key}_median"], "seed_averaged_indomain.json")

    def holm(pd_: dict) -> dict:
        ks = sorted(pd_, key=pd_.get)
        m, run, adj = len(ks), 0.0, {}
        for i, k in enumerate(ks):
            run = max(run, min(1.0, (m - i) * pd_[k]))
            adj[k] = run
        return adj
    hs = holm(pvals)
    per_seed_adj = [holm({k: holm_per_seed[k][j] for k in pvals}) for j in range(len(MA.columns))]
    for key in pvals:
        paired[f"vs_{key}_null"]["holm_p_seed_median"] = round(hs[key], 4)
        paired[f"vs_{key}_null"]["holm_p_max_over_seeds"] = round(max(a[key] for a in per_seed_adj), 4)
    out["paired_in_domain"] = paired

    # ---- assertions ----------------------------------------------------------------------------
    if not chain:
        reg = jload(f"{EX}/paper_numbers.json")
        idt, hd = reg["in_domain_table"], reg["headline"]["unscreened_same_protocol"]
        sm = idt["seed_medians"]
        for lab, got, want in (
                ("REG ml_shared_in_domain.median_ape", R["ml_shared_in_domain"]["median_ape"], sm["calibrated"]["median_ape"]["median"]),
                ("REG ml_shared_in_domain.within_2x_pct", R["ml_shared_in_domain"]["within_2x_pct"], sm["calibrated"]["within_2x"]["median"]),
                ("REG ml_shared_in_domain.n", R["ml_shared_in_domain"]["n"], idt["n"]),
                ("REG ml_shared_all.median_ape", R["ml_shared_all"]["median_ape"], hd["median_ape"]),
                ("REG ml_shared_all.within_2x_pct", R["ml_shared_all"]["within_2x_pct"], hd["within_2x_pct"]),
                ("REG ml_shared_all.n", R["ml_shared_all"]["n"], hd["n"]),
                ("REG ml_uncorrected_in_domain.median_ape", R["ml_uncorrected_in_domain"]["median_ape"], sm["uncorrected_surrogate"]["median_ape"]["median"]),
                ("REG ml_uncorrected_in_domain.within_2x_pct", R["ml_uncorrected_in_domain"]["within_2x_pct"], sm["uncorrected_surrogate"]["within_2x"]["median"]),
                ("REG null_constant_in_domain.median_ape", R["null_constant_in_domain"]["median_ape"], idt["nulls"]["constant"]["median_ape"]),
                ("REG null_constant_in_domain.within_2x_pct", R["null_constant_in_domain"]["within_2x_pct"], idt["nulls"]["constant"]["within_2x_pct"]),
                ("REG null_power_law_in_domain.median_ape", R["null_power_law_in_domain"]["median_ape"], idt["nulls"]["power"]["median_ape"]),
                ("REG null_power_law_in_domain.within_2x_pct", R["null_power_law_in_domain"]["within_2x_pct"], idt["nulls"]["power"]["within_2x_pct"]),
                ("REG published_calc_shared_family_test.median_ape", R["published_calc_shared_family_test"]["median_ape"], reg["deployed_route"]["held_out"]["global_on_same_compounds"]["median_ape"]),
                ("REG published_calc_shared_family_test.within_2x_pct", R["published_calc_shared_family_test"]["within_2x_pct"], reg["deployed_route"]["held_out"]["global_within_2x_pct"]),
                ("REG published_calc_family_constant.median_ape", R["published_calc_family_constant"]["median_ape"], reg["deployed_route"]["held_out"]["median_ape"]),
                ("REG published_calc_family_constant.within_2x_pct", R["published_calc_family_constant"]["within_2x_pct"], reg["deployed_route"]["held_out"]["within_2x_pct"]),
                ("REG published_calc_family_constant.n", R["published_calc_family_constant"]["n"], reg["deployed_route"]["held_out"]["n"]),
                ("REG shared_constant_unseen_family.median_ape", R["shared_constant_unseen_family"]["median_ape"], reg["shared_constant_unseen_family"]["median_ape"]),
                ("REG shared_constant_unseen_family.within_2x_pct", R["shared_constant_unseen_family"]["within_2x_pct"], reg["shared_constant_unseen_family"]["within_2x_pct"]),
                ("REG shared_constant_unseen_family.n", R["shared_constant_unseen_family"]["n"], reg["shared_constant_unseen_family"]["n_compounds"]),
                ("REG paired.constant p (3dp)", round(pvals["constant"], 3), idt["paired_p_seed_median"]["constant"]),
                ("REG paired.power p (3dp)", round(pvals["power"], 3), idt["paired_p_seed_median"]["power"])):
            check(lab, got, want, "paper_numbers.json")
    fails = [c for c in CHECKS if abs(c[1] - c[2]) > 1e-9]
    print(f"\n{len(CHECKS)} consistency checks, {len(fails)} failed"
          + ("" if not chain else " (registry comparison skipped: --chain)"))
    for lab, got, want, src in fails:
        print(f"  MISMATCH {lab}: computed {got} vs {want} in {src}")
    if fails:
        print("STOP: not writing", OUT)
        return 1
    out["consistency_checks"] = dict(n=len(CHECKS), failed=0,
                                     registry_compared=not chain,
                                     sources=sorted({c[3] for c in CHECKS}))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n  {'route':<36}{'n':>4}{'median %':>10}{'CI':>16}{'w2x %':>8}{'CI':>16}{'|log r|':>9}")
    for k, v in R.items():
        ci = lambda x: "-" if x is None else f"[{x[0]:.1f},{x[1]:.1f}]"
        sl = "-" if v["median_abs_log10_ratio"] is None else f"{v['median_abs_log10_ratio']:.3f}"
        print(f"  {k:<36}{v['n']:>4}{v['median_ape']:>10.1f}{ci(v['ci95_median_ape']):>16}"
              f"{v['within_2x_pct']:>8.1f}{ci(v['ci95_within_2x_pct']):>16}{sl:>9}")
    for key in ("constant", "power"):
        p = paired[f"vs_{key}_null"]
        print(f"  model vs {key:<8} median diff {p['median_paired_diff_pp']:+.1f} pp "
              f"CI {p['ci95_median_paired_diff_pp']}  Wilcoxon p {p['wilcoxon_p_seed_median']}  "
              f"Holm {p['holm_p_seed_median']} (max over seeds {p['holm_p_max_over_seeds']})")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
