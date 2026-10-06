"""What the deployed calibration delivers, pooled: a family arm where one has been earned, the
global arm elsewhere.

WHY THIS SCRIPT EXISTS. Section 3 quotes the pooled accuracy of the route the paper actually
deploys -- a median error, a within-2x rate, and the split between compounds a family arm covers
and compounds left on the global arm. Those numbers had no committed producer: they were computed
once in a scratch script that no longer exists, then typed into the manuscript. When the
physical-domain tolerance was corrected and one more compound (LaBiPd) entered the in-domain set,
nothing could say what they had become. This is that producer.

WHAT IT COMPUTES. On the same paired frame family_calibration.py builds (published transport
calculations, one laboratory per compound, in-domain, gated), every compound receives:

    uncorrected   the published kappa_BTE as is
    global        kappa_BTE * min[c_g (T/300)^p_g, 1] with the shared constant refitted on the
                  published calculation pairs WITHOUT the scored compound (shared_constant.py;
                  PREREG_shared_constant_on_calculations rule 4)
    deployed      the family arm if the compound's family is adopted, else the global arm

The family arm is scored LEAVE-ONE-COMPOUND-OUT within the family, exactly as tab:famcal is: the
held-out compound never sees its own measurement in the fit -- and, since 2026-10-05 (PREREG
amendment A), never decides its siblings' references either: every fold re-runs the reference
ladder (family_calibration.fold_pairs) with the held-out compound absent from the family rung. Errors are per-compound medians, never
per-row, for the reason family_calibration gives -- a compound with 81 temperature points would
otherwise outvote one with a single point.

WHAT IT DOES NOT DO. It does not choose anything. The adopted families are read from the artefact
the paper commits to, and the shared constant comes from shared_constant.py, so this reports what
those imply and cannot quietly become a different calibration.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import extend_blind_test as E
import family_calibration as FC
import shared_constant as SCN

FAMCAL = "data/exports/kappa_v2/family_calibration.json"
OUT = "data/exports/kappa_v2/deployed_route.json"
POOLED = "data/exports/kappa_v2/family_deployed_pooled.csv"


def _ape(te: pd.DataFrame, k: np.ndarray) -> float:
    return float(FC.per_compound_ape(te, k).iloc[0])


def _within2(te: pd.DataFrame, k: np.ndarray) -> bool:
    r = float(np.median(k / te.k_ref.values))
    return 0.5 <= r <= 2.0


def main() -> int:
    fc = json.load(open(FAMCAL))
    adopted = {f for f, v in fc["families"].items() if v.get("adopted")}
    p_held = float(fc["global"]["p"]) if "global" in fc else None

    d = FC.build_published()
    d["fam"] = d.compound.map(FC.yzfam)
    # deployed shared constant (published calculations vs measurements); sets the family forms' p
    c_g, p_g = SCN.shared_cp()
    if p_held is not None and abs(p_held - p_g) > 1e-6:
        print(f"  WARNING: family_calibration.json holds p={p_held}, shared_cp() gives {p_g:.3f}")

    print(f"in-domain compounds {d.compound.nunique()}, families {d.fam.nunique()}, "
          f"adopted {sorted(adopted)}")
    print(f"global arm c={c_g:.3f} p={p_g:.3f}\n")

    rows = []
    for cpd, te_dep in d.groupby("compound"):
        fam = te_dep.fam.iloc[0]
        # REFERENCES RE-CHOSEN INSIDE THE FOLD (PREREG amendment A; family leave-one-out leak,
        # audit 2026-10-05). Every held-out score below is on the compound's fold reference (the
        # ladder WITHOUT the family-paper rung), its family-mates are fitted on their fold
        # references (family rung over members minus this compound), and the shared constant is
        # fitted on that fold's pairs without this compound. `te_dep` -- the deployed reference --
        # is used only for the in-sample "own paper" figures, which are not held-out claims.
        fold = FC.fold_pairs(cpd)
        te = fold[fold.compound == cpd]
        ref_fold = FC.fold_reference(cpd) or ("", "")
        ref_dep = FC.CHOSEN_REFERENCE.get(cpd, ("", ""))
        if te.empty:
            print(f"  {cpd}: its fold reference ({ref_fold[0]}) pairs with no calculation -- "
                  "not scorable held out; skipped")
            continue
        T = te["T"].values.astype(float)
        kb = te.k_pred.values
        # the global arm applied to a scored compound never saw that compound (prereg rule 4)
        c_h, p_h = SCN.shared_cp_in_fold(cpd)
        k_glob = kb * E.apply_cp(T, c_h, p_h)
        # EVERY COMPOUND IS SCORED ON ITS OWN FAMILY'S ARM (ruling of 2026-09-19). Held out where
        # the family has another member; in-sample where it does not, and labelled so. The global
        # arm is kept only as the comparison column.
        arm, k_dep, c_fam, p_fam = "global", k_glob, None, None
        if fam in adopted:
            # the family table drops its robust outliers from the FIT (never the score); the
            # scoreboard must fit the same way or its family medians drift from Table famcal
            # (Sb-Pt: 3.6% here against 4.0% there once YSbPt stopped being a harmless outlier)
            outl = set(fc["families"][fam].get("outliers") or {})
            mates = set(d.loc[(d.fam == fam) & (d.compound != cpd), "compound"])
            tr = fold[fold.compound.isin(mates) & ~fold.compound.isin(outl)]
            if tr.compound.nunique() >= 1 and len(tr) >= 4:
                # HELD OUT: the family form's exponent is the shared p refitted without this
                # compound too (on the fold's pairs; shared_constant.shared_p_in_fold)
                p_use = p_h
                par = FC.fit_form(FC.ADOPTED_FORM, tr, c_g, p_use)
                arm = "family"
            else:
                # single member: its own constant, in-sample, deployed p. For a family with no
                # other member the family rung cannot act, so the fold reference IS the deployed one.
                p_use = p_g
                par = FC.fit_form(FC.ADOPTED_FORM, te, c_g, p_use)
                arm = "family (in-sample)"
            if par is not None:
                c_fam, p_fam = float(par[0]), float(p_use)
                k_dep = FC.apply_family(FC.ADOPTED_FORM, (c_fam, p_fam), T, kb)
            else:
                arm = "global"
        # CALIBRATED ON ITS OWN PAPER. The constant fitted on this compound's own reference and the
        # in-sample error it gives -- the number for a compound whose only paper is the one in hand.
        # Reported beside the held-out figure, never in place of it.
        # On the DEPLOYED reference (the paper the compound is calibrated on), not the fold's.
        own = FC.fit_form(FC.ADOPTED_FORM, te_dep, c_g, p_g) if len(te_dep) >= 4 else None
        k_own = (FC.apply_family(FC.ADOPTED_FORM, own, te_dep["T"].values.astype(float),
                                 te_dep.k_pred.values) if own is not None else None)
        rows.append(dict(compound=cpd, family=fam, arm=arm, c_family_loo=c_fam,
                         ref_doi_heldout=ref_fold[0], ref_doi_deployed=ref_dep[0],
                         ref_moved_in_fold=bool(ref_fold[0] != ref_dep[0]),
                         p_family_loo=p_fam,
                         form=(FC.ADOPTED_FORM if arm != "global" else "GLOBAL"),
                         c_own=(round(float(own[0]), 3) if own is not None else None),
                         ape_anchored=(_ape(te_dep, k_own) if k_own is not None else None),
                         ape_uncorrected=_ape(te, kb), ape_global=_ape(te, k_glob),
                         ape_deployed=_ape(te, k_dep),
                         ratio_deployed=float(np.median(k_dep / te.k_ref.values)),
                         w2_uncorrected=_within2(te, kb), w2_global=_within2(te, k_glob),
                         w2_deployed=_within2(te, k_dep)))
    R = pd.DataFrame(rows).sort_values(["arm", "family", "compound"])

    def summ(sub: pd.DataFrame, col: str) -> tuple[float, float]:
        return float(sub[f"ape_{col}"].median()), float(100 * sub[f"w2_{col}"].mean())

    print(f"{'route':<34}{'n':>4}{'median APE':>12}{'within 2x':>11}")
    print("=" * 61)
    out = dict(n=int(len(R)), c_global=round(float(c_g), 3), p_global=round(float(p_g), 3),
               adopted=sorted(adopted))
    for lab, col in (("uncorrected published calculation", "uncorrected"),
                     ("global arm only", "global"),
                     ("deployed: family where earned + global", "deployed")):
        m, w = summ(R, col)
        out[col] = dict(median_ape=round(m, 1), within2x_pct=round(w, 1))
        print(f"  {lab:<32}{len(R):>4}{m:>11.1f}%{w:>10.0f}%")
    for arm in ("family", "family (in-sample)", "global"):
        sub = R[R.arm == arm]
        if sub.empty:
            continue
        m, w = summ(sub, "deployed")
        # the global arm scored on the SAME compounds, so the two medians are paired; the pooled
        # `global` block above includes the in-sample four and must not be quoted beside a
        # held-out figure
        mg, wg = summ(sub, "global")
        out[f"{arm.replace(' (in-sample)', '_insample')}_arm_compounds"] = dict(
            n=int(len(sub)), median_ape=round(m, 1), within2x_pct=round(w, 1),
            global_on_same_compounds=dict(median_ape=round(mg, 1), within2x_pct=round(wg, 1)),
            n_family_better=int((sub.ape_deployed < sub.ape_global).sum()),
            compounds=sorted(sub.compound))
        print(f"  {'  on the ' + arm + ' arm':<32}{len(sub):>4}{m:>11.1f}%{w:>10.0f}%")

    print("\nper compound  (anchored = fitted on its own paper, in-sample; deployed = held out):")
    print(R[["compound", "family", "arm", "c_own", "ape_anchored", "ape_uncorrected", "ape_global",
             "ape_deployed"]].round(1).to_string(index=False))
    A = R.ape_anchored.dropna()
    out["anchored_on_own_paper"] = dict(
        n=int(len(A)), median_ape=round(float(A.median()), 1),
        under_10_pct=int((A < 10).sum()), under_25_pct=int((A < 25).sum()),
        note="in-sample: each compound fitted on its own reference; not a validation figure")
    out["held_out_reference_rule"] = (
        "each held-out compound scored on the reference the ladder picks WITHOUT the family-paper "
        "rung, family-mates on references re-chosen without it (PREREG amendment A)")
    out["references_moved_in_fold"] = {
        r.compound: dict(deployed=r.ref_doi_deployed, held_out=r.ref_doi_heldout)
        for r in R.itertuples() if r.ref_moved_in_fold}

    Path(OUT).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")

    # THE PER-COMPOUND SCOREBOARD. family_deployed_pooled.csv existed with no producer, went stale
    # at 29 rows while the route had 30, and lost LaBiPd (38.9%). Everything that judges a compound
    # -- never a family mean -- reads this file, so it is written here, by the script that owns the
    # numbers, on every run. `ratio` is deployed/measured at the compound's median, kept for the
    # figure that plots it. `ape_global` is the same compound under the global constant, so the
    # family figure can pair the two arms compound by compound.
    P = (R[["compound", "family", "arm", "ape_deployed", "ratio_deployed", "c_own", "ape_anchored",
            "ape_global", "c_family_loo", "p_family_loo", "form", "ref_doi_heldout",
            "ref_doi_deployed", "ref_moved_in_fold"]]
         .rename(columns={"ape_deployed": "ape", "ratio_deployed": "ratio"})
         .sort_values("ape"))
    P.to_csv(POOLED, index=False)
    print(f"wrote {POOLED}  ({len(P)} compounds, {int((P.ape > 25).sum())} over 25%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
