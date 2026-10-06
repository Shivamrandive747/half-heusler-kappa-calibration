"""Can the model reproduce a published BTE calculation it has never seen?

THE ONLY NEW STEP. Every prediction in the paper so far multiplies a PUBLISHED kappa_BTE by the
transfer function. Extending to compounds nobody has calculated means generating kappa_BTE ourselves,
and that generator is the one link in the chain with no validation behind it. This script supplies
that validation directly: for every DXMag candidate that DOES have a published Boltzmann-transport
calculation, hide its entire chemistry cluster from the model, predict it, and compare.

WHY THE WHOLE CLUSTER IS HELD OUT AND NOT JUST THE COMPOUND. `ZrCoSb` sits among forty other Zr-Co-Sb
rows; removing one row leaves the answer in the training set under a different formula. The unit of
independence is the chemistry cluster, which is what `run_loco_chemistry.cluster_of` defines and what
`run_target_blind_test.py` already uses for the same reason.

WHAT IT IS NOT. This does not test the transfer function -- that is validated elsewhere against
measurements. It tests only whether a generated kappa_BTE can stand in for a published one. The
benchmark to beat is `surrogate_vs_published.json`: the deployed surrogate differs from published BTE
by a median 15.4%, so a generator materially worse than that is not fit to replace a calculation.

THE NUMBER THAT MATTERS is not the corpus-wide median but the error on compounds resembling the six
we intend to issue, so the per-element and per-candidate breakdowns are reported alongside.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pymatgen.core import Composition

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import s19_kappa_dataset as ds
from pipeline.s20_kappa_train import make_model
from run_loco_chemistry import cluster_of, sites
from make_paper_predictions import vec, yzfam

GAP = "data/Target_Materials/HEUSLER_KAPPA_TIER3_GAP_ROWS.csv"
TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
NOT_BTE = ("not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx")
OUT = "data/exports/kappa_v2/bte_generator_validation.csv"

# the six we intend to issue, and the nine flagged behind them
TIER1 = ["HfAsIr", "ScAsPt", "HfBiIr", "ZrSiNi", "TiSiNi", "YAsPt"]
TIER2 = ["VCoSi", "YGeAu", "TaSiIr", "HfSiNi", "ScSiAu", "HfSiPt", "YSiAu", "TaSiRh", "HfSiPd"]


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main(seed: int = 0) -> int:
    tr = pd.read_csv(TRAIN, low_memory=False)
    tr["red"] = tr.formula.map(red)
    t = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    meth = tr.method.astype(str).str.lower()

    # the reference: published full-BTE values, in the temperature band the paper predicts over
    ref = tr[(t == 1) & tr.k.notna() & (tr.k > 0) & ~meth.str.contains(NOT_BTE, regex=True)
             & tr["T"].between(200, 1300)].dropna(subset=["red"]).copy()
    ref = ref[ref.red.map(lambda r: (sites(r) or ["?"])[0] == "half")]
    ref = ref[ref.red.map(lambda r: vec(r) == 18)]
    ref["Tbin"] = (ref["T"] / 100).round() * 100
    per = (ref.groupby(["red", "Tbin"])
           .agg(k_ref=("k", "median"), formula=("formula", "first")).reset_index())
    print(f"in-domain half Heuslers with a published BTE calculation: {per.red.nunique()}")
    print(f"compound x temperature conditions to predict: {len(per)}")

    b = ds.build(tier_max=1, tier_min=1, group_by="element_system", extra_csv=GAP, verbose=False)
    forms = b["meta"].formula.values
    clus = np.array([cluster_of(f) for f in forms])
    X, y, w = b["X"], b["y"], b["weights"]
    reds = np.array([red(f) for f in forms])
    print(f"training pool: {len(y)} rows, {len(set(forms))} compounds, "
          f"{len(set(clus))} chemistry clusters\n")

    need: dict = {}
    for r in per.red.unique():
        need.setdefault(cluster_of(per[per.red == r].formula.iloc[0]), []).append(r)

    rows = []
    for i, (cl, members) in enumerate(sorted(need.items()), 1):
        te = clus == cl
        trm = ~te
        if trm.sum() < 50:
            continue
        assert not (clus[trm] == cl).any(), f"cluster {cl} leaked into training"
        mdl = make_model("catboost", {"random_seed": seed})
        mdl.fit(X[trm], y[trm], sample_weight=w[trm])
        for r in members:
            idx = np.where(reds == r)[0]
            if not len(idx):
                continue                      # no descriptor row; cannot be predicted honestly
            src = X.iloc[idx[0]].copy()
            for _, cond in per[per.red == r].iterrows():
                q = src.copy()
                q["temperature_K"] = float(cond.Tbin)
                q["inv_T"] = 1000.0 / float(cond.Tbin)
                q["log_T"] = np.log10(float(cond.Tbin))
                pred = float(10 ** mdl.predict(pd.DataFrame([q])[X.columns])[0])
                rows.append(dict(compound=r, cluster=cl, T=float(cond.Tbin),
                                 k_published=float(cond.k_ref), k_generated=round(pred, 4),
                                 fam=yzfam(r)))
        if i % 8 == 0:
            print(f"  [{i}/{len(need)}] clusters fitted")

    D = pd.DataFrame(rows)
    D["ape"] = 100 * (D.k_generated - D.k_published).abs() / D.k_published
    D["ratio"] = D.k_generated / D.k_published
    D.to_csv(OUT, index=False)
    g = D.groupby("compound").agg(ape=("ape", "median"), ratio=("ratio", "median"),
                                  n=("ape", "size"), fam=("fam", "first"))

    print()
    print("=" * 78)
    print(f"GENERATOR vs PUBLISHED BTE -- {len(g)} compounds, whole chemistry held out")
    print("=" * 78)
    print(f"  median error per compound : {g.ape.median():.1f}%")
    print(f"  within 30%                : {100 * (g.ape <= 30).mean():.0f}%")
    print(f"  within 2x                 : {100 * ((g.ratio.between(0.5, 2.0))).mean():.0f}%")
    print(f"  bias (median ratio)       : {g.ratio.median():.2f}")
    print(f"  BENCHMARK to beat         : 15.4%  (deployed surrogate vs published)")
    print()

    els = {}
    for c in g.index:
        for e in Composition(c).elements:
            els.setdefault(str(e), []).append(float(g.ape[c]))
    print("ERROR BY ELEMENT -- the elements our six candidates are built from")
    print(f"  {'element':<9}{'n compounds':>13}{'median error':>14}")
    want = sorted({str(e) for c in TIER1 + TIER2 for e in Composition(c).elements})
    for e in want:
        v = els.get(e)
        print(f"  {e:<9}{(len(v) if v else 0):>13}{(f'{np.median(v):.1f}%' if v else 'NO DATA'):>14}")
    print()
    sub = [c for c in g.index if any(str(e) in want for e in Composition(c).elements)]
    if sub:
        s = g.loc[sub]
        print(f"RESTRICTED to compounds sharing an element with our candidates: n={len(s)}")
        print(f"  median error {s.ape.median():.1f}%   within 2x {100 * s.ratio.between(0.5, 2.0).mean():.0f}%")
    print()
    print("WORST 12 -- where a generated calculation would be most misleading")
    for c, r in g.sort_values("ape", ascending=False).head(12).iterrows():
        print(f"  {c:<10}{str(r.fam):<9}{r.ape:>7.0f}%   generated/published = {r.ratio:.2f}")
    print()
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0))
