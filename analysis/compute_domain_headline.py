"""Regenerate data/exports/kappa_v2/domain_headline.json, which until now had no producer.

WHY THIS EXISTS. `domain_headline.json` carries the canonical row the manuscript's headline traces
back to -- {scope: "VEC = 18 + cubic, non-polymorphic", n: 38, c: 0.51, p: 0.8, median_ape: 29.7,
within_2x: 86.8} -- and is read by make_paper_numbers.py:100 and shipped by make_release.py. It was
committed as a static artefact with no script behind it, so nothing could regenerate it when the
calibration moved. Seven files on the paper's live read path share that defect; this closes the one
that blocks the family-calibration work, because a family arm changes these numbers and the headline
and the predictions would otherwise be calibrated differently with no way to reconcile them.

WHAT IT COMPUTES. Four nested scopes of the half-Heusler blind test. SINCE 2026-10-05
(paper/evidence/PREREG_shared_constant_on_calculations.md) no scope gets its own in-sample fit on the
model's predictions: every compound is scored under the shared constant fitted on published
calculations without its X-site chemistry cluster (shared_constant.shared_cp_without_cluster), and
the c, p columns hold the deployed shared constant. Until then each scope was fitted IN SAMPLE; the
row remains a description of how much the screens tighten the population, not the headline. The out-of-sample number for the same population is 31.0% and lives in
conformal_indomain.json / compute_results_indomain.py; do not confuse them, and never quote a row of
this file as a validation result.

VERIFY MODE. `python compute_domain_headline.py --check` recomputes and diffs against the committed
file without writing. Run that FIRST: if the recomputation does not reproduce the published numbers,
the discrepancy must be understood before the file is regenerated, because the manuscript currently
quotes the committed values.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import argparse
import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import extend_blind_test as E
import shared_constant as SCN
from make_paper_predictions import structure_status, vec

BLIND = "data/exports/kappa_v2/target_blind_test.csv"
OUT = "data/exports/kappa_v2/domain_headline.json"
NOTE = ("screens are pre-existing in make_paper_predictions.py; applying them to validation "
        "removes an inconsistency, it does not introduce a filter")


def score(d: pd.DataFrame, c: float, p: float) -> tuple[float, float]:
    """Per-compound median APE, then the median over compounds; and the within-2x fraction.

    Scoring per compound and never per row is the rule the whole project follows: a compound with
    81 temperature points would otherwise outvote one with a single point, which is how an earlier
    36.5% headline was withdrawn.
    """
    if c is None:
        k = d.k_pred.values * 1.0
    else:
        # c, p are reported only; each compound is scored under the shared constant fitted
        # without its own chemistry cluster (prereg rule 4)
        k = d.k_pred.values.astype(float).copy()
        for cl, idx in d.groupby(d.cluster.fillna("?")).indices.items():
            cc, pp = SCN.shared_cp_without_cluster(cl)
            k[idx] = k[idx] * E.apply_cp(d["T"].values[idx], cc, pp)
    t = d.assign(_ape=100 * np.abs(k - d.k_ref.values) / d.k_ref.values,
                 _r=k / d.k_ref.values)
    per = t.groupby("compound").agg(ape=("_ape", "median"), r=("_r", "median"))
    return float(per.ape.median()), float(100 * per.r.between(0.5, 2.0).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="recompute and diff against the committed file; write nothing")
    a = ap.parse_args()

    B = pd.read_csv(BLIND)
    d = B[B.klass == "half"].copy()
    d["_v"] = d.compound.map(vec)
    st = {x: structure_status(str(x)) for x in d.compound.unique()}
    d["_cubic"] = d.compound.map(lambda x: st[x][0] and not st[x][1])

    scopes = [
        ("all half Heuslers (no screen)", d),
        ("VEC = 18", d[d._v == 18]),
        ("VEC = 18 + cubic, non-polymorphic", d[(d._v == 18) & d._cubic]),
        ("EXCLUDED by the screens", d[~((d._v == 18) & d._cubic)]),
    ]

    rows = []
    for name, sub in scopes:
        if sub.empty:
            continue
        c, p = SCN.shared_cp()            # deployed shared constant, reported per row
        ape, w2 = score(sub, c, p)
        unc, _ = score(sub, None, None)
        rows.append(dict(scope=name, n=int(sub.compound.nunique()),
                         c=round(float(c), 2), p=round(float(p), 2),
                         median_ape=round(ape, 1), within_2x=round(w2, 1),
                         uncorrected=round(unc, 1)))

    excluded = sorted(d[~((d._v == 18) & d._cubic)].compound.unique())
    R = {"_generated_by": "compute_domain_headline.py",
         "_fit": "shared constant fitted on published calculations vs measurements; each "
                 "compound scored with its chemistry cluster removed from that fit (c, p shown are "
                 "the deployed values). Describes the effect of the screens; the headline is the "
                 "seed-averaged nested figure.",
         "rows": rows, "excluded": excluded, "note": NOTE}

    old = json.load(open(OUT)) if __import__("pathlib").Path(OUT).exists() else None
    if old:
        print(f"{'scope':<38}{'field':<12}{'committed':>11}{'recomputed':>12}   ")
        print("-" * 76)
        diffs = 0
        byname = {r["scope"]: r for r in rows}
        for o in old["rows"]:
            n = byname.get(o["scope"])
            if not n:
                print(f"{o['scope']:<38}{'MISSING from recomputation':>35}")
                diffs += 1
                continue
            for f in ("n", "c", "p", "median_ape", "within_2x", "uncorrected"):
                if abs(float(o[f]) - float(n[f])) > 1e-9:
                    print(f"{o['scope'][:37]:<38}{f:<12}{o[f]:>11}{n[f]:>12}   <-- differs")
                    diffs += 1
        so, sn = set(old.get("excluded", [])), set(excluded)
        if so != sn:
            print(f"excluded list differs: only-committed={sorted(so - sn)} "
                  f"only-recomputed={sorted(sn - so)}")
            diffs += 1
        print("-" * 76)
        print("REPRODUCES THE COMMITTED FILE EXACTLY" if not diffs
              else f"{diffs} field(s) differ -- understand why BEFORE regenerating")
        if a.check:
            return 1 if diffs else 0

    if a.check:
        print(json.dumps(R, indent=2)[:600])
        return 0
    json.dump(R, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
