"""Does 4f occupancy of the X site explain within-family scatter? Measured, not assumed.

WHY THIS TEST EXISTS. Three of the four issued predictions carry an open 4f shell at the X site
(HfCoBi is the exception), and the source calculations do not record whether the f electrons were
frozen into the core or treated explicitly. Section 3 bounds that exposure by noting that each
family constant is fitted across members with and without an open shell, so a systematic error in
the 4f treatment would show up as scatter within the family. This script tests that argument
directly: it partitions each family by 4f occupancy and asks whether the partition buys anything.

It does not. Under the 2026-09-19 architecture (MIN_MEMBERS_LOO = 2, so a two-member sub-family
IS scored, as a single pair), in the three families where the partition can be formed at all:

  * the fitted constant moves by at most 0.03 -- Ni-Sb 0.45 -> 0.45 (4f) / 0.48 (d),
    Sb-Pt 0.59 -> 0.58, Sb-Pd 0.30 -> 0.29 (4f) / 0.33 (d). That is the result that matters,
    because the constant is what is deployed. Worst issued change: TmSbPt 3.46 -> 3.40 (1.7%).
  * held-out error gets WORSE in every partition that can be scored:
    Ni-Sb 9.8 -> 11.4 (4f) / 12.6 (d), Sb-Pt 6.4 -> 16.6, Sb-Pd 32.7 -> 121.6.

THE TRAP THIS SCRIPT ALSO MEASURES. In an earlier run Sb-Pd's 4f pair scored 6.6% against 21.9%
for the family and looked like a discovery. Both members' references then came from ONE
paper (10.1088/1361-6463/aac567), so the pair was that laboratory agreeing with itself; with HoSbPd on its
lowest-disorder reference (Section 3.7) the same pair scores 121.6%. More generally, at two
members the "held-out error" is fit-on-one, predict-the-other, computed on a single pair -- and
within these same families that quantity ranges over an order of magnitude depending on which
pair is drawn (pair_protocol: Ni-Sb 7.5x, Sb-Pt 13.9x, Sb-Pd 21.7-176.6%). A sub-family that
scores well on one pair is the tight end of its own distribution. Selecting it is selecting the
pair that agrees -- the same contamination as the one-anchor family rule, where a cell selected
for its own conclusion. The numbers above are read back from split_4f_test.json, never from here.

Read-only with respect to the calibration: writes one artefact and changes nothing the paper
deploys.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import itertools
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")
import numpy as np

import family_calibration as FC
import shared_constant as SCN
from make_paper_predictions import yzfam, yzfam_split

OUT = "data/exports/kappa_v2/split_4f_test.json"
FAMILIES = ("Ni-Sb", "Sb-Pt", "Sb-Pd")

_CG = 0.0
_PG = 0.0


def _loo_and_c(g, c_g, p_g):
    """Held-out error under the pipeline's own gates, and the constant it would deploy."""
    out = FC.robust_outliers(g)
    sel = FC.select_form(g, c_g, p_g, exclude=set(out), fold=FC.fold_pairs)
    cp = FC.fit_form(FC.ADOPTED_FORM, g[~g.compound.isin(out)], c_g, p_g)
    return sel, (None if cp is None else float(cp[0]))


def _loo_per_compound(g, c_g, p_g) -> dict:
    """Leave-one-compound-out error for EACH member, under the pipeline's own fit gates.

    score_form() returns the family median; this returns the folds, so a split can be compared
    with the unsplit fit compound by compound, paired. A member whose fold cannot be fitted (fewer
    than four training rows, or no other member in its class) is absent from the result: under the
    split that is a compound the sub-family cannot predict at all, and it is counted as such.
    """
    out = {}
    for held in sorted(g.compound.unique()):
        te, trn = g[g.compound == held], g[g.compound != held]
        if len(trn) < 4 or trn.compound.nunique() < 1:
            continue
        # scored held out: the exponent is the shared p refitted without `held` as well
        cp = FC.fit_form(FC.ADOPTED_FORM, trn, c_g, SCN.shared_p_excluding([held]))
        if cp is None or not FC.downward_ok(FC.ADOPTED_FORM, cp, trn):
            continue
        k = FC.apply_family(FC.ADOPTED_FORM, cp, te["T"].values, te.k_pred.values)
        out[held] = float(np.median(100 * np.abs(k - te.k_ref.values) / te.k_ref.values))
    return out


def _pair_protocol(g):
    """Fit on ONE member, predict another, over every ordered pair -- the n=2 estimator.

    This is exactly what leave-one-compound-out degenerates to when a family has two members, and
    running it on families that have MORE than two is how its reliability is measured.
    """
    ms = sorted(g.compound.unique())
    vals = {}
    for a, b in itertools.permutations(ms, 2):
        trn, te = g[g.compound == a], g[g.compound == b]
        # `b` is scored: the exponent is the shared p refitted without it
        cp = FC.fit_form(FC.ADOPTED_FORM, trn, _CG, SCN.shared_p_excluding([b]))
        if cp is None:
            continue
        k = FC.apply_family(FC.ADOPTED_FORM, cp, te["T"].values, te.k_pred.values)
        vals[f"{a}->{b}"] = float(np.median(100 * np.abs(k - te.k_ref.values) / te.k_ref.values))
    return vals


def main() -> int:
    global _CG, _PG
    d = FC.build_published()
    d["fam"] = d.compound.map(yzfam)
    # the shared constant fitted on published calculations (shared_constant.py;
    # PREREG_shared_constant_on_calculations): sets the family form's p; the GLOBAL arm inside
    # select_form is refitted without each held-out compound
    import shared_constant as SCN
    c_g, p_g = SCN.shared_cp()
    _CG, _PG = float(c_g), float(p_g)
    sub = d.compound.map(yzfam_split)

    rep = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "global": {"c": round(float(c_g), 3), "p": round(float(p_g), 3)},
           "adopted_form": FC.ADOPTED_FORM,
           "min_members": FC.MIN_MEMBERS,
           "families": {}}

    print(f"global arm c={c_g:.3f} p={p_g:.3f}   form={FC.ADOPTED_FORM}\n")
    print(f"{'family':<12}{'n':>3}{'LOCO':>9}{'c':>8}   note")
    for fam in FAMILIES:
        g = d[d.fam == fam]
        sel, c_un = _loo_and_c(g, c_g, p_g)
        loo_un = sel["optimistic"]
        rec = {"n_members": int(g.compound.nunique()),
               "members": sorted(g.compound.unique()),
               "loo_ape": None if loo_un is None else round(loo_un, 1),
               "c": None if c_un is None else round(c_un, 3),
               "sub": {}}
        print(f"{fam:<12}{rec['n_members']:>3}{rec['loo_ape']:>8.1f}%{rec['c']:>8.3f}   unsplit")

        for tag in ("4f", "d"):
            gs = d[(d.fam == fam) & (sub == f"{fam}:{tag}")]
            if gs.empty:
                continue
            n = int(gs.compound.nunique())
            s = {"n_members": n, "members": sorted(gs.compound.unique())}
            if n >= FC.MIN_MEMBERS:
                sel2, c_sp = _loo_and_c(gs, c_g, p_g)
                v = sel2["optimistic"]
                s["loo_ape"] = None if v is None else round(v, 1)
                s["c"] = None if c_sp is None else round(c_sp, 3)
                s["testable"] = True
                shown = f"{s['loo_ape']:>8.1f}%{s['c']:>8.3f}" if v is not None else f"{'n/a':>16}"
            else:
                s["testable"] = False
                s["why_not"] = (f"{n} member(s): a held-out fit would rest on {n - 1} compound, "
                                f"below the two the pipeline requires")
                s["loo_ape"] = None
                s["c"] = None
                # What it WOULD score if the guard were relaxed, recorded precisely to show that
                # the figure is not usable rather than to use it. The constant is fitted for ONE
                # member too: leaving those out understated the worst-case move by more than 2x.
                if n == 2:
                    pairs = _pair_protocol(gs)
                    s["relaxed_pairs"] = {k: round(v, 1) for k, v in pairs.items()}
                    s["relaxed_median"] = round(float(np.median(list(pairs.values()))), 1)
                cp = FC.fit_form(FC.ADOPTED_FORM, gs, c_g, p_g)
                s["c_if_relaxed"] = None if cp is None else round(float(cp[0]), 3)
                # NOTE: this is NOT what the pipeline would deploy. A sub-family below MIN_MEMBERS
                # gets no family arm at all and falls through to the GLOBAL constant, so the move
                # actually deployed would be to c_g, which is larger still. Both are recorded.
                s["c_if_deployed"] = round(float(c_g), 3)
                shown = f"{'--':>8}{str(s.get('c_if_relaxed', '')):>8}"
            print(f"  {fam + ':' + tag:<10}{n:>3}{shown}   "
                  f"{'split' if s['testable'] else 'NOT TESTABLE'}")
            rec["sub"][tag] = s

        # The pair-variance evidence: how much does an n=2 estimate depend on WHICH pair?
        if rec["n_members"] >= 3:
            pv = _pair_protocol(g)
            rec["pair_protocol"] = {"values": {k: round(v, 1) for k, v in pv.items()},
                                    "median": round(float(np.median(list(pv.values()))), 1),
                                    "min": round(min(pv.values()), 1),
                                    "max": round(max(pv.values()), 1),
                                    "range_factor": round(max(pv.values()) / min(pv.values()), 1)}
        rep["families"][fam] = rec

    print("\nHow much does a two-member held-out error depend on WHICH pair you happen to have?")
    print(f"{'family':<12}{'trueLOCO':>10}{'pair med':>10}{'pair min':>10}{'pair max':>10}{'range':>8}")
    for fam, rec in rep["families"].items():
        p = rec.get("pair_protocol")
        if not p:
            continue
        print(f"{fam:<12}{rec['loo_ape']:>9.1f}%{p['median']:>9.1f}%{p['min']:>9.1f}%"
              f"{p['max']:>9.1f}%{p['range_factor']:>7.1f}x")

    # The headline of this script: the deployed constant is what matters, and it does not move.
    scoreable = [abs(r["c"] - r["sub"][t]["c"])
                 for r in rep["families"].values() for t in r["sub"]
                 if r["sub"][t].get("c") is not None and r["c"] is not None]
    unscoreable = [abs(r["c"] - r["sub"][t]["c_if_relaxed"])
                   for r in rep["families"].values() for t in r["sub"]
                   if r["sub"][t].get("c_if_relaxed") is not None and r["c"] is not None]
    # WHAT THE SPLIT WOULD DO TO THE ISSUED PREDICTIONS.
    #
    # An earlier version of this script asserted that no issued value changes. That is FALSE and a
    # referee could falsify it in one run: at 300 K the SHAPE form reduces to kappa = c * kappa_BTE,
    # so a prediction moves in exact proportion to its family constant. Three of the four move in
    # their printed digits. They move by ~2%, far inside a 50% interval, which is the honest claim
    # and is the one the paper now makes.
    import pandas as pd

    pred = pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv")
    pred = pred[pred.status == "ISSUED"]
    impact, worst = {}, 0.0
    for r in pred.itertuples():
        fam = str(r.family)
        sub4 = rep["families"].get(fam, {}).get("sub", {}).get("4f", {})
        c_new = sub4.get("c") if sub4.get("c") is not None else sub4.get("c_if_relaxed")
        if c_new is None:
            continue
        k_new = float(r.kappa_pred_300) * (c_new / float(r.c_used))
        rel = abs(k_new - float(r.kappa_pred_300)) / float(r.kappa_pred_300) * 100
        worst = max(worst, rel)
        impact[r.compound] = dict(family=fam, c_used=float(r.c_used), c_split=c_new,
                                  kappa_now=round(float(r.kappa_pred_300), 2),
                                  kappa_if_split=round(k_new, 2), rel_change_pct=round(rel, 2),
                                  inside_50pct_interval=bool(r.lo50 <= k_new <= r.hi50))
    rep["issued_prediction_impact"] = {
        "per_compound": impact,
        "worst_relative_change_pct": round(worst, 2),
        "worst_relative_change_pct_1dp": round(worst, 1),   # the figure quoted in the manuscript
        "all_inside_50pct_interval": all(v["inside_50pct_interval"] for v in impact.values()),
    }
    print("\nWhat the split would do to the four issued predictions:")
    print(f"{'compound':<10}{'family':<8}{'now':>7}{'if split':>10}{'change':>9}   in 50% band")
    for k, v in impact.items():
        print(f"{k:<10}{v['family']:<8}{v['kappa_now']:>7.2f}{v['kappa_if_split']:>10.2f}"
              f"{v['rel_change_pct']:>8.2f}%   {v['inside_50pct_interval']}")
    print(f"worst relative change: {worst:.2f}%")

    # THE DECIDING TEST, PAIRED AND COMPUTED -- not asserted. For every member of the three
    # families: its held-out error when predicted from the WHOLE family, against its held-out error
    # when predicted from its own 4f class alone. A member whose class holds no other member gets
    # no prediction under the split; that is a cost of splitting and is counted, not ignored.
    # An earlier version of this block hard-coded `held_out_error_worsens_where_testable: True`
    # and a prose verdict; the 2026-09-19 dedupe moved the numbers and the assertion went stale
    # without anything noticing. Everything below is derived from the folds.
    paired, unpredictable = {}, {}
    for fam in FAMILIES:
        g = d[d.fam == fam]
        whole = _loo_per_compound(g, c_g, p_g)
        for tag in ("4f", "d"):
            gs = d[(d.fam == fam) & (sub == f"{fam}:{tag}")]
            if gs.empty:
                continue
            part = _loo_per_compound(gs, c_g, p_g)
            for m in sorted(gs.compound.unique()):
                if m in whole and m in part:
                    paired[m] = dict(family=fam, cls=tag, whole=round(whole[m], 1),
                                     split=round(part[m], 1))
                elif m in whole:
                    unpredictable[m] = dict(family=fam, cls=tag, whole=round(whole[m], 1),
                                            why="no other member in its 4f class")
    w = np.array([v["whole"] for v in paired.values()])
    s_ = np.array([v["split"] for v in paired.values()])
    n_better = int((s_ < w).sum())
    n_worse = int((s_ > w).sum())
    med_w, med_s = float(np.median(w)), float(np.median(s_))
    # the split "wins" only if it lowers the paired median AND leaves no member unpredictable;
    # a partition that predicts fewer compounds is not a better calibration, it is a smaller one
    split_wins = bool(med_s < med_w and not unpredictable)
    rep["paired_test"] = {
        "protocol": ("per compound, held out: predicted from the whole family vs from its own 4f "
                     "class; members whose class holds no other member are unpredictable under "
                     "the split"),
        "per_compound": paired,
        "unpredictable_under_split": unpredictable,
        "n_paired": int(len(paired)),
        "median_whole": round(med_w, 1), "median_split": round(med_s, 1),
        "n_split_better": n_better, "n_split_worse": n_worse,
        "split_wins": split_wins,
    }
    print(f"\nPAIRED TEST over {len(paired)} compounds: whole-family {med_w:.1f}% vs own-4f-class "
          f"{med_s:.1f}%   split better in {n_better}, worse in {n_worse}; "
          f"unpredictable under the split: {sorted(unpredictable)}")
    for m, v in paired.items():
        print(f"    {m:<8}{v['family']:<6}{v['cls']:<3}{v['whole']:>7.1f}% -> {v['split']:>6.1f}%")

    cmax = round(max(scoreable + unscoreable), 3)
    rep["conclusion"] = {
        # quoted separately, because only the first is a number the pipeline would ever deploy
        "constant_moves_by_at_most": round(max(scoreable), 3),
        "constant_moves_by_at_most_incl_unscoreable": cmax,
        "issued_predictions_worst_relative_change_pct": round(worst, 2),
        "split_lowers_paired_median": bool(med_s < med_w),
        "split_leaves_members_unpredictable": sorted(unpredictable),
        "split_adopted": split_wins,
        "verdict": ((f"4f occupancy is NOT adopted as a sub-family rule: across {len(paired)} paired "
                     f"compounds the split gives a median held-out error of {med_s:.1f}% against "
                     f"{med_w:.1f}% for the whole family (better in {n_better}, worse in {n_worse})"
                     + (f", and leaves {len(unpredictable)} member(s) unpredictable "
                        f"({', '.join(sorted(unpredictable))})" if unpredictable else "")
                     + f". The deployed constant would move by up to {cmax} and an issued value by "
                       f"up to {round(worst, 1)}%.")
                    if not split_wins else
                    (f"4f occupancy LOWERS paired held-out error ({med_w:.1f}% -> {med_s:.1f}%, "
                     f"better in {n_better} of {len(paired)}) with no member left unpredictable; "
                     "SPLIT_FAMILIES should be revisited.")),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    cc = rep["conclusion"]
    print(f"\nlargest change in a deployed constant: {cc['constant_moves_by_at_most']} over the "
          f"splits the pipeline can score, {cc['constant_moves_by_at_most_incl_unscoreable']} "
          f"including those it cannot")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
