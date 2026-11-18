"""Null baselines refitted with the route's per-compound objective, beside the row-fitted ones.

Specified in paper/evidence/PREREG_null_baselines_per_compound.md, which was committed before this
script was run. The blind test's two null baselines are fitted to ROWS (compute_seed_averaged.null),
the route's constants PER COMPOUND (extend_blind_test.fit_cp). Here each baseline is refitted by the
route's objective -- the median over training compounds of each compound's median |log10 error| --
with the same folds, compounds, references and scoring, and every statistic the paper reports for
the baselines is recomputed for both fits.

POSITIVE CONTROL. The row-fitted baselines run through this same code and must reproduce the
registry (paper_numbers.json, accuracy_summary.json); the script refuses to write otherwise.

Nothing is retrained: the route's per-seed predictions are read from the saved blind-test files.
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
from scipy.stats import binomtest, wilcoxon

import compute_accuracy_summary as CAS
import compute_seed_averaged as CSA

EX = "data/exports/kappa_v2"
OUT = f"{EX}/nulls_fitted_per_compound.json"
KS = np.exp(np.linspace(np.log(0.5), np.log(50.0), 401))      # W/m/K, constant and prefactor
QS = np.round(np.linspace(-2.0, 0.5, 51), 2)                   # exponent of the power-law null


def _score(tr: pd.DataFrame, qs: np.ndarray) -> np.ndarray:
    """Objective on the (K, q) grid: median over compounds of each compound's median |log10 error|."""
    lr = np.log10(tr.k_ref.values.astype(float))
    lt = np.log10(tr["T"].values.astype(float) / 300.0)
    codes = pd.Categorical(tr.compound).codes
    la = np.log10(KS)[:, None, None]
    qq = qs[None, :, None]
    per = []
    for k in range(codes.max() + 1):
        m = codes == k
        per.append(np.median(np.abs(la + qq * lt[m][None, None, :] - lr[m][None, None, :]), axis=2))
    return np.median(np.stack(per, axis=0), axis=0)


def fit_per_compound(tr: pd.DataFrame, kind: str) -> tuple[float, float]:
    qs = np.array([0.0]) if kind == "constant" else QS
    s = _score(tr, qs)
    i, j = np.unravel_index(int(np.argmin(s)), s.shape)       # first minimum in grid order
    return float(KS[i]), float(qs[j])


def null_per_compound(d: pd.DataFrame, kind: str) -> pd.DataFrame:
    """compute_seed_averaged.null with the fit swapped; folds, rows and scoring unchanged."""
    out = []
    for cl in sorted(set(d.chem)):
        te, tr = d[d.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        A, q = fit_per_compound(tr, kind)
        out.append(te.assign(k=A * (te["T"].values.astype(float) / 300.0) ** q))
    return pd.concat(out)


def holm(p: dict) -> dict:
    ks = sorted(p, key=p.get)
    m, run, adj = len(ks), 0.0, {}
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def evaluate(D: dict, MA: pd.DataFrame, cmp0: list, nullfn) -> dict:
    """Every statistic the paper reports for the baselines, for one way of fitting them."""
    d0 = D[min(D)]
    res, pvals, per_seed_p = {}, {}, {}
    for kind in ("constant", "power"):
        P = CSA.per_cmp(nullfn(d0, kind)).loc[cmp0]
        for s in D:                                            # the baselines carry no model seed
            assert np.allclose(CSA.per_cmp(nullfn(D[s], kind)).loc[cmp0].a, P.a)
        diff = MA.values - P.a.values[:, None]                 # compounds x seeds; negative = route closer
        bd = np.median(np.median(diff[CAS.boot_idx(len(diff))], axis=1), axis=1)
        ps = [float(wilcoxon(MA[s].values, P.a.values)[1]) for s in MA.columns]
        pvals[kind], per_seed_p[kind] = float(np.median(ps)), ps
        chem = d0.groupby("compound").chem.first().loc[cmp0]
        cl_w, cl_s, cl_better = [], [], []
        for s in MA.columns:
            cl = pd.DataFrame({"m": MA[s].values, "n": P.a.values, "chem": chem.values}) \
                .groupby("chem")[["m", "n"]].median()
            dd = cl.n - cl.m
            nz = dd[dd != 0]
            cl_w.append(float(wilcoxon(cl.m, cl.n).pvalue))
            cl_s.append(float(binomtest(int((nz > 0).sum()), len(nz)).pvalue))
            cl_better.append(int((dd > 0).sum()))
        res[kind] = dict(
            **CAS.single_route(P.a.values, P.r.values),
            within_30_pct=round(float((P.a <= 30).mean() * 100), 1),
            bias=round(float(P.r.median()), 2),
            typical_factor=round(float(10 ** np.median(np.abs(np.log10(P.r.values)))), 2),
            route_closer_pct_seed_median=round(float(np.median((diff < 0).mean(axis=0) * 100)), 1),
            median_paired_diff_pp=round(float(np.median(np.median(diff, axis=0))), 1),
            ci95_median_paired_diff_pp=CAS.pct_ci(bd),
            wilcoxon_p_seed_median=round(pvals[kind], 4),
            wilcoxon_p_per_seed={str(s): round(p, 4) for s, p in zip(MA.columns, ps)},
            cluster_level=dict(n_clusters=int(chem.nunique()),
                               wilcoxon_p_per_seed=[round(p, 4) for p in cl_w],
                               wilcoxon_p_max=round(max(cl_w), 4),
                               sign_p_range=[round(min(cl_s), 4), round(max(cl_s), 4)],
                               route_better_clusters_per_seed=cl_better))
    hs = holm(pvals)
    worst = [holm({k: per_seed_p[k][j] for k in pvals}) for j in range(len(MA.columns))]
    for kind in pvals:
        res[kind]["holm_p_seed_median"] = round(hs[kind], 4)
        res[kind]["holm_p_max_over_seeds"] = round(max(w[kind] for w in worst), 4)
    return res


def main() -> int:
    D, M, _ = CAS.in_domain_frames()
    cmp0 = sorted(M[min(M)].index)
    MA = CAS.stack({s: M[s].a for s in M}).loc[cmp0]
    MR = CAS.stack({s: M[s].r for s in M}).loc[cmp0]
    route = CAS.seeded_route(MA, MR)

    rows = evaluate(D, MA, cmp0, CSA.null)
    fair = evaluate(D, MA, cmp0, null_per_compound)

    # ---- positive control: the row fit must reproduce the published figures ----------------------
    # In rerun_downstream.sh this runs with --chain, BEFORE make_paper_numbers.py rebuilds the
    # registry, so there the comparison is with accuracy_summary.json only (written one step
    # earlier); run on its own it is compared with the registry as well.
    chain = "--chain" in sys.argv
    acc = json.load(open(f"{EX}/accuracy_summary.json", encoding="utf-8"))
    checks = []
    for kind in ("constant", "power"):
        pa = acc["paired_in_domain"][f"vs_{kind}_null"]
        ar = acc["routes"][f"null_{'constant' if kind == 'constant' else 'power_law'}_in_domain"]
        checks += [
            (f"{kind} median_ape", rows[kind]["median_ape"], ar["median_ape"]),
            (f"{kind} within_2x", rows[kind]["within_2x_pct"], ar["within_2x_pct"]),
            (f"{kind} route closer %", rows[kind]["route_closer_pct_seed_median"],
             pa["model_better_pct_seed_median"]),
            (f"{kind} paired diff pp", rows[kind]["median_paired_diff_pp"], pa["median_paired_diff_pp"]),
            (f"{kind} p seed median", rows[kind]["wilcoxon_p_seed_median"], pa["wilcoxon_p_seed_median"]),
            (f"{kind} holm seed median", rows[kind]["holm_p_seed_median"], pa["holm_p_seed_median"]),
            (f"{kind} holm worst seed", rows[kind]["holm_p_max_over_seeds"], pa["holm_p_max_over_seeds"])]
    if not chain:
        reg = json.load(open(f"{EX}/paper_numbers.json", encoding="utf-8"))
        idt, cls = reg["in_domain_table"], reg["cluster_level_significance"]
        for kind in ("constant", "power"):
            checks += [
                (f"REG {kind} median_ape", rows[kind]["median_ape"], idt["nulls"][kind]["median_ape"]),
                (f"REG {kind} within_2x", rows[kind]["within_2x_pct"], idt["nulls"][kind]["within_2x_pct"]),
                (f"REG {kind} p seed median (3dp)", round(rows[kind]["wilcoxon_p_seed_median"], 3),
                 idt["paired_p_seed_median"][kind]),
                (f"REG {kind} cluster wilcoxon max", rows[kind]["cluster_level"]["wilcoxon_p_max"],
                 cls["wilcoxon_p_max"][kind])]
    bad = [c for c in checks if abs(float(c[1]) - float(c[2])) > 1e-9]
    print(f"positive control: {len(checks)} checks against the registry, {len(bad)} failed")
    for lab, got, want in bad:
        print(f"  MISMATCH {lab}: computed {got}, registry {want}")
    if bad:
        print("STOP: not writing", OUT)
        return 1

    out = dict(
        _generated_by="compute_nulls_fitted_per_compound.py",
        prereg="paper/evidence/PREREG_null_baselines_per_compound.md",
        population=f"in domain, n = {len(cmp0)} compounds, {int(D[min(D)].chem.nunique())} clusters",
        route=dict(median_ape=route["median_ape"], within_2x_pct=route["within_2x_pct"],
                   typical_factor=round(float(10 ** route["median_abs_log10_ratio"]), 2)),
        objective=("median over training compounds of each compound's median |log10(prediction / "
                   "reference)|, first minimum on the grid"),
        grid=dict(K_points=len(KS), K_range=[0.5, 50.0], q_points=len(QS), q_range=[-2.0, 0.5]),
        fitted_to_rows=rows, fitted_per_compound=fair,
        positive_control=dict(n_checks=len(checks), failed=0, registry_compared=not chain),
        HOW_TO_STATE=("the route is compared with each baseline under both fits; the paired difference "
                      "is route minus baseline, so a negative value means the route is closer"))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)

    print(f"\nroute: {route['median_ape']}% median, {route['within_2x_pct']}% within 2x")
    for name, R in (("fitted to rows", rows), ("fitted per compound", fair)):
        print(f"\n  BASELINES {name.upper()}")
        for kind in ("constant", "power"):
            r = R[kind]
            print(f"    {kind:<9} median {r['median_ape']:5.1f}%  within2x {r['within_2x_pct']:5.1f}%  "
                  f"bias {r['bias']:.2f}  factor {r['typical_factor']:.2f}  | route closer "
                  f"{r['route_closer_pct_seed_median']:.1f}%  diff {r['median_paired_diff_pp']:+.1f} pp "
                  f"{r['ci95_median_paired_diff_pp']}  p {r['wilcoxon_p_seed_median']:.4f}  "
                  f"Holm {r['holm_p_seed_median']:.4f} (worst seed {r['holm_p_max_over_seeds']:.4f})  | "
                  f"clusters: Wilcoxon max {r['cluster_level']['wilcoxon_p_max']:.4f}, sign "
                  f"{r['cluster_level']['sign_p_range']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
