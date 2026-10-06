"""Does a prediction outside its family's measured range come out worse? (the range gate's evidence)

make_paper_predictions.py withheld every prediction falling outside the room-temperature span its
family's measured compounds cover. Section 3.8 said plainly that this had never been tested: "we
have not shown that falling outside it predicts a larger error". This tests it on the compounds
where the answer is known.

Each measured family member is held out, exactly as the deployed route scores it
(compute_deployed_route: its family constant fitted without it, with every reference re-chosen
inside the fold -- family_calibration.fold_pairs, PREREG amendment A). Its predicted value near 300 K
is compared with the span of its family-mates' measured values at 250-350 K -- the gate's own
definition, built here without the held-out compound -- and its held-out error is grouped by how
far outside that span it fell:
    inside          margin = 1
    marginal        1 < margin < LAB_FACTOR   (inside the 1.33x median disagreement between labs)
    clear           margin >= LAB_FACTOR

Writes paper/evidence/range_gate_test.json, which make_paper_predictions.py reads so the note on
every marginal prediction quotes these numbers rather than typed ones.
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
from scipy.stats import mannwhitneyu

import extend_blind_test as E
import family_calibration as FC
import shared_constant as SCN
import source_identity as SI

OUT = "paper/evidence/range_gate_test.json"
POOLED = "data/exports/kappa_v2/family_deployed_pooled.csv"
LAB_FACTOR = 1.33        # = make_paper_predictions' marginal threshold
FAMILY_ISSUE_MAX = 25.0  # the issuing bar the groups are counted against


def main() -> int:
    FC.NO_DOMAIN_GATE = True
    with contextlib.redirect_stdout(io.StringIO()):
        d = FC.build_published()
    P = pd.read_csv(POOLED)
    P = P[P.arm == "family"].set_index("compound")

    tr = SI.dedupe(pd.read_csv(FC.TRAIN, low_memory=False))
    tr["red"] = tr.formula.map(FC.red)
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr["t"] = pd.to_numeric(tr.method_tier, errors="coerce")
    m = tr[(tr.t == 0) & (tr.k > 0) & tr["T"].between(250, 350)]
    m300 = m.groupby("red").k.median()
    fam_of = {c: FC.yzfam(c) for c in m300.index}

    rows = []
    for c, r in P.iterrows():
        # the compound as it stands in the fold that scores it (references re-chosen inside the
        # fold, PREREG amendment A) -- the same frame compute_deployed_route scored it on
        fold = FC.fold_pairs(c)
        te = fold[fold.compound == c]
        if te.empty:
            continue
        i = (te["T"] - 300).abs().idxmin()
        # the exponent this compound was scored with when held out (compute_deployed_route
        # p_family_loo = shared_constant.shared_p_excluding([c])), not the deployed p
        p_c = (float(r.p_family_loo) if "p_family_loo" in P.columns and pd.notna(r.p_family_loo)
               else SCN.shared_p_in_fold(c))
        kpred = float(FC.apply_family(r.form, (r.c_family_loo, p_c),
                                      [te.loc[i, "T"]], [te.loc[i, "k_pred"]])[0])
        mates = [m300[x] for x in m300.index if fam_of[x] == r.family and x != c]
        if len(mates) < 2:          # the gate needs two measured compounds to define a span
            continue
        lo, hi = min(mates), max(mates)
        margin = max(lo / kpred, kpred / hi, 1.0)
        rows.append(dict(compound=c, family=r.family, kappa_pred_300=round(kpred, 2),
                         mates_lo=round(lo, 2), mates_hi=round(hi, 2), margin=round(margin, 3),
                         heldout_ape=round(float(r.ape), 1)))
    R = pd.DataFrame(rows)
    groups = {"inside": R[R.margin <= 1.0],
              "marginal": R[(R.margin > 1.0) & (R.margin < LAB_FACTOR)],
              "clear": R[R.margin >= LAB_FACTOR]}
    summ = {k: dict(n=int(len(g)),
                    median_ape=round(float(g.heldout_ape.median()), 1) if len(g) else None,
                    within_bar=int((g.heldout_ape < FAMILY_ISSUE_MAX).sum()),
                    max_margin=round(float(g.margin.max()), 2) if len(g) else None)
            for k, g in groups.items()}
    out_all = R[R.margin > 1.0]
    res = dict(
        _generated_by="test_range_gate.py",
        lab_factor=LAB_FACTOR, issue_bar_pct=FAMILY_ISSUE_MAX,
        groups=summ,
        mannwhitney_p_marginal_vs_inside=(round(float(mannwhitneyu(
            groups["marginal"].heldout_ape, groups["inside"].heldout_ape).pvalue), 3)
            if len(groups["marginal"]) and len(groups["inside"]) else None),
        mannwhitney_p_outside_vs_inside=(round(float(mannwhitneyu(
            out_all.heldout_ape, groups["inside"].heldout_ape).pvalue), 3)
            if len(out_all) and len(groups["inside"]) else None),
        per_compound=json.loads(R.sort_values("margin").to_json(orient="records")))
    json.dump(res, open(OUT, "w"), indent=2)
    for k, s in summ.items():
        print(f"  {k:<9} n={s['n']:>2}  median held-out error {s['median_ape']}%  "
              f"under {FAMILY_ISSUE_MAX:.0f}%: {s['within_bar']}/{s['n']}")
    print(f"  marginal vs inside p = {res['mannwhitney_p_marginal_vs_inside']};  "
          f"all outside vs inside p = {res['mannwhitney_p_outside_vs_inside']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
