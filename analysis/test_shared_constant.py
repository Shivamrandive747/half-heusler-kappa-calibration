"""The best single shared constant for a family never measured -- per PREREG_shared_constant.md.

Every rule here (candidates, grid, protocol, nested selection, decision) is fixed by the
pre-registration committed before this script was first run. One interpretation it could not fix
in advance is recorded in the output: a GEOMETRIC midpoint is undefined for an exponent that can be
zero or negative, so a tie in p takes the arithmetic midpoint of the tied p range; c takes the
geometric midpoint, as pre-registered.

C0 REDEFINED 2026-10-05 (paper/evidence/PREREG_shared_constant_on_calculations.md). C0 is "the
shared constant the paper deploys". It used to be fitted on the ML model's blind-test predictions
with family F removed; it is now the deployed definition -- shared_constant.py, fitted on the
published calculation pairs -- with every compound of the held-out families removed
(shared_constant.shared_cp_without_family(F) in the outer leave-one-family-out). Its LOFO row is what the
paper quotes as "the shared constant on unseen families". The old positive control (C0 = 26.8%)
belonged to the old definition and is replaced by a consistency check against shared_constant.py.

Writes paper/evidence/shared_constant_test.json.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import contextlib
import io
import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import extend_blind_test as E
import family_calibration as FC
import shared_constant as SCN
from compute_seed_averaged import in_domain

OUT = "paper/evidence/shared_constant_test.json"
CS = np.round(np.arange(0.10, 0.995, 0.01), 2)
PS = np.round(np.arange(-1.50, 1.501, 0.05), 2)
P_FIXED = 0.80
CANDS = {  # id: (capped, p free, objective)
    "C1": (True, True, "logerr"),
    "C2": (False, True, "logerr"),
    "C3": (False, False, "logerr"),
    "C4": (True, False, "logerr"),
    "C5": (False, True, "share25"),
}
ALL_IDS = ["C0", *CANDS]


def _grids(d, capped, pfree):
    ps = PS if pfree else np.array([P_FIXED])
    T, kb, kr = d["T"].values.astype(float), d.k_pred.values, d.k_ref.values
    fac = CS[:, None, None] * (T[None, None, :] / 300.0) ** ps[None, :, None]
    if capped:
        fac = np.minimum(fac, 1.0)
    pred = kb[None, None, :] * fac
    loge = np.abs(np.log10(pred) - np.log10(kr)[None, None, :])
    ape = np.abs(pred - kr[None, None, :]) / kr[None, None, :]
    codes = pd.Categorical(d.compound).codes
    masks = [codes == k for k in range(codes.max() + 1)]
    lmed = np.stack([np.median(loge[:, :, m], axis=2) for m in masks], axis=2)
    amed = np.stack([np.median(ape[:, :, m], axis=2) for m in masks], axis=2)
    return ps, np.median(lmed, axis=2), np.mean(amed < 0.25, axis=2)


def _mid(ps, tied):
    ii, jj = np.nonzero(tied)
    c = float(np.sqrt(CS[ii].min() * CS[ii].max()))
    p = float((ps[jj].min() + ps[jj].max()) / 2.0)
    return round(c, 4), round(p, 4)


def fit(cid, d_pub, d_ml):
    """(c, p, capped) for one candidate fitted on the given training rows."""
    if cid == "C0":
        # the deployed shared constant, fitted by shared_constant.py on the calculation pairs
        # with every compound NOT in this training set removed (i.e. the held-out families)
        held = set(SCN.pairs().compound) - set(d_pub.compound)
        c, p = SCN.shared_cp(exclude=held)
        return float(c), float(p), True
    capped, pfree, obj = CANDS[cid]
    ps, score, share = _grids(d_pub, capped, pfree)
    if obj == "logerr":
        tied = score <= score.min() + 1e-12
    else:  # maximise the share under 25%, ties broken by the lower median log error
        best = share >= share.max() - 1e-12
        s2 = np.where(best, score, np.inf)
        tied = s2 <= s2.min() + 1e-12
    c, p = _mid(ps, tied)
    return c, p, capped


def predict(par, te):
    c, p, capped = par
    f = c * (te["T"].values / 300.0) ** p
    return te.k_pred.values * (np.minimum(f, 1.0) if capped else f)


def score_rows(rows):
    R = pd.DataFrame(rows)
    return dict(n=int(len(R)), median_ape=round(float(R.ape.median()), 1),
                under_25_pct=round(float((R.ape < 25).mean() * 100), 1),
                within_2x_pct=round(float(R.r.between(0.5, 2).mean() * 100), 1),
                per_family={f: round(float(g.ape.median()), 1) for f, g in R.groupby("fam")})


def lofo(cid, d, B, families):
    """Leave one family out over `families`; returns per-compound rows."""
    rows = []
    for F in families:
        par = fit(cid, d[(d.fam != F) & d.fam.isin(families)],
                  B[(B.fam != F) & B.fam.isin(families) | ~B.fam.isin(set(d.fam))])
        for cpd, te in d[d.fam == F].groupby("compound"):
            k = predict(par, te)
            rows.append(dict(compound=cpd, fam=F,
                             ape=float(np.median(100 * np.abs(k - te.k_ref.values) / te.k_ref.values)),
                             r=float(np.median(k / te.k_ref.values))))
    return rows


def main() -> int:
    FC.NO_DOMAIN_GATE = True
    with contextlib.redirect_stdout(io.StringIO()):
        d = FC.build_published()
    d["fam"] = d.compound.map(FC.yzfam)
    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    B = in_domain(B[B.klass == "half"])
    B["fam"] = B.compound.map(FC.yzfam)
    fams = sorted(d.fam.unique())
    print(f"published route: {d.compound.nunique()} compounds, {len(fams)} families")

    # C0's ML-route training set must lose family F exactly as the published side does; ML-route
    # compounds in families absent from the published route are always kept
    res = {"_generated_by": "test_shared_constant.py", "prereg": "paper/evidence/PREREG_shared_constant.md",
           "c0_definition": ("since 2026-10-05: shared_constant.py fitted on published calculations, "
                             "held-out families removed (PREREG_shared_constant_on_calculations.md)"),
           "interpretation": "p ties take the arithmetic midpoint (geometric undefined for p <= 0)",
           "lofo": {}, "full_fit": {}}
    for cid in ALL_IDS:
        res["lofo"][cid] = score_rows(lofo(cid, d, B, fams))
        res["full_fit"][cid] = dict(zip(("c", "p", "capped"), fit(cid, d, B)))
        s = res["lofo"][cid]
        print(f"  {cid}: median {s['median_ape']}%  under 25% {s['under_25_pct']}%  "
              f"within 2x {s['within_2x_pct']}%   full fit {res['full_fit'][cid]}")

    c0 = res["lofo"]["C0"]["median_ape"]
    # consistency control: this script's published frame is shared_constant's, and C0's folds are
    # exactly shared_cp_without_family / its full fit exactly shared_cp()
    assert len(d) == len(SCN.pairs()) and set(d.compound) == set(SCN.pairs().compound), (
        "published frame differs from shared_constant.pairs()")
    for F in fams:
        assert fit("C0", d[d.fam != F], None)[:2] == SCN.shared_cp_without_family(F), F
    assert tuple(res["full_fit"]["C0"][k] for k in ("c", "p")) == SCN.shared_cp()
    print(f"  consistency control: C0 = shared_constant.py ({SCN.shared_cp()}); "
          f"leave-one-family-out median {c0}%")

    # NESTED: the choice among the six is itself made without the outer family
    nested_rows, picks = [], {}
    for F in fams:
        inner = [f for f in fams if f != F]
        inner_med = {cid: float(pd.DataFrame(lofo(cid, d, B, inner)).ape.median()) for cid in ALL_IDS}
        pick = min(ALL_IDS, key=lambda c_: (inner_med[c_], c_))
        picks[F] = dict(pick=pick, inner_median={k: round(v, 1) for k, v in inner_med.items()})
        par = fit(pick, d[d.fam != F], B[(B.fam != F)])
        for cpd, te in d[d.fam == F].groupby("compound"):
            k = predict(par, te)
            nested_rows.append(dict(compound=cpd, fam=F,
                                    ape=float(np.median(100 * np.abs(k - te.k_ref.values) / te.k_ref.values)),
                                    r=float(np.median(k / te.k_ref.values))))
        print(f"  outer {F:<6} -> {pick}")
    res["nested"] = dict(score_rows(nested_rows), picks=picks)
    nm, ns = res["nested"]["median_ape"], res["nested"]["under_25_pct"]
    adopt = nm < c0 and ns >= res["lofo"]["C0"]["under_25_pct"]
    from collections import Counter
    choice = Counter(v["pick"] for v in picks.values()).most_common(1)[0][0]
    res["decision"] = dict(
        rule="adopt only if nested median < C0's AND nested share under 25% >= C0's",
        nested_median=nm, nested_under_25=ns, c0_median=c0,
        c0_under_25=res["lofo"]["C0"]["under_25_pct"],
        verdict="ADOPT" if adopt else "KEEP C0",
        candidate_if_adopted=choice if adopt else None,
        constant_if_adopted=res["full_fit"][choice] if adopt else None)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nNESTED: median {nm}%  under 25% {ns}%   vs C0 {c0}% / {res['lofo']['C0']['under_25_pct']}%"
          f"   => {res['decision']['verdict']}" + (f" ({choice}, {res['full_fit'][choice]})" if adopt else ""))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
