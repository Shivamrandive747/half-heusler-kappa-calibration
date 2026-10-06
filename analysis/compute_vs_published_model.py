"""Score a PUBLISHED machine-learned model against measurement, on our held-out compounds.

WHY THIS EXISTS. The Introduction says surrogate models are "accurate against the quantity they
were trained on", and the Conclusions say their "agreement with measurement is incidental". Those
are empirical claims about other people's models, and until now the paper made them without
measuring one. Every numeric baseline in it was either our own surrogate, a statistical null, or a
published SEMI-EMPIRICAL estimate. A referee would ask, correctly, why the one comparison that
tests the paper's own framing is missing.

THE COMPARATOR. Elemental-SDNNFF (npj Comput. Mater. 9, 26, 2023; DOI 10.1038/s41524-023-00974-0)
is a neural-network force field whose lattice thermal conductivities are obtained through ShengBTE.
It is exactly the class of model the Introduction describes: trained to reproduce a first-principles
calculation, and fast enough to screen thousands of compounds. Its values sit in
data/external/kappa_ML_PREDICTED_not_training.csv, tagged "EXCLUDED from training labels", so
nothing here has ever seen them as a label.

THREE THINGS THIS IS CAREFUL ABOUT.

  1 TEMPERATURE. SDNNFF publishes one value per compound, at 300 K. Our measurements are curves
    from 200 to 1200 K. Comparing a 300 K number against a whole curve is the bug that once moved
    Ni-Sb from 31% to 9.7%, so only measured rows inside a window around 300 K are used, and the
    window is a parameter printed with the result.

  2 THE HOLDOUT IS ASYMMETRIC, AND IT FAVOURS THEM. Our arm is scored with the compound's entire
    chemistry withheld from the calibration fit. SDNNFF gets no holdout at all, and these compounds
    may well sit in its own training set. The comparison is therefore generous to the published
    model, in the same way the semi-empirical comparison in Section 3 is generous to Slack.

  3 WHAT LOSING WOULD MEAN. Against the published DFT it was trained on, a DFT surrogate should
    win, and an untracked file in this repo shows it doing so. That is not the question. The
    question is agreement with MEASUREMENT, which is the scale a screening user needs.

Read-only. Writes data/exports/kappa_v2/vs_published_model.json.
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
from scipy.stats import wilcoxon

import extend_blind_test as E
import shared_constant as SCN
from make_paper_predictions import structure_status, vec
from run_loco_chemistry import cluster_of

OUT = "data/exports/kappa_v2/vs_published_model.json"
SDNNFF = "data/external/kappa_ML_PREDICTED_not_training.csv"
SDNNFF_DOI = "10.1038/s41524-023-00974-0"
T_WINDOW = 50.0          # measured rows within +/- this of 300 K
T_REF = 300.0


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def _control(R: pd.DataFrame) -> dict:
    """Does the surrogate at least agree with the CALCULATION it is a surrogate for?

    Without this the headline could be a units or definition error rather than a scale offset: a
    number that disagrees with everything disagrees with everything for a reason, and "their model
    is wrong" is the least likely one. If SDNNFF sits close to the published calculation while both
    sit above measurement, the disagreement is the offset this paper is about, not a mistake in
    reading their file.

    THE COMPARATOR IS THE PUBLISHED FULL-BTE CALCULATION (FIXPASS S6, 2026-10-05). Until then the
    "published calculation" column was `k_pred` of the blind-test file -- OUR machine-learning
    model's prediction -- so the control compared one surrogate with another. It is now the
    published-route median calculation of each compound (family_calibration.published_calc_at: the
    median over published full-BTE sources at each temperature, contested temperatures resolved,
    carried to the measured temperature exactly as the published route pairs it), evaluated at the
    same measured rows inside the 300 K window, median per compound. The old comparison is kept,
    labelled as what it is.
    """
    import family_calibration as FC
    f = R.dropna(subset=["k_sdnnff"]).copy()
    f["k_bte"] = [FC.published_calc_at(c, T) for c, T in zip(f.red, f["T"])]
    f["k_bte"] = pd.to_numeric(f.k_bte, errors="coerce")
    g = (f.groupby("compound").agg(sd=("k_sdnnff", "median"), ml=("k_pred", "median"),
                                   bte=("k_bte", "median"), meas=("k_ref", "median")))
    gb = g.dropna(subset=["bte"])

    def cmp(a, b):
        if not len(a):
            return dict(n=0, median_ape=None, median_ratio=None)
        return dict(n=int(len(a)), median_ape=round(float(((a - b).abs() / b * 100).median()), 1),
                    median_ratio=round(float((a / b).median()), 2))
    return dict(
        n=int(len(g)),
        n_with_published_bte=int(len(gb)),
        published_bte_definition=("family_calibration.published_calc_at at each measured row "
                                  "inside the 300 K window, median per compound"),
        sdnnff_vs_published_bte_calculation=cmp(gb.sd, gb.bte),
        sdnnff_vs_measurement=cmp(g.sd, g.meas),
        sdnnff_vs_measurement_same_compounds_as_bte=cmp(gb.sd, gb.meas),
        published_bte_calculation_vs_measurement=cmp(gb.bte, gb.meas),
        sdnnff_vs_our_ml_surrogate=cmp(g.sd, g.ml),
        _meaning=("the surrogate should land NEAR the published calculation it reproduces and FAR "
                  "from measurement, by roughly the offset this paper measures. If it were far "
                  "from both, the comparison would be reading the wrong quantity. "
                  "sdnnff_vs_our_ml_surrogate is the comparison this block used to report under "
                  "the label 'published calculation'; it compares two surrogates."))


def _window(d: pd.DataFrame, sdk: pd.Series) -> dict:
    """The 300 K window is a free parameter, so show the answer does not depend on it."""
    out = {}
    for w in (25.0, 50.0, 75.0, 100.0):
        n = d[(d["T"] - T_REF).abs() <= w].copy()
        n["sd"] = n.red.map(sdk)
        n = n.dropna(subset=["sd"])
        g = n.groupby("compound").agg(sd=("sd", "median"), meas=("k_ref", "median"))
        out[f"plus_minus_{int(w)}K"] = dict(
            n=int(len(g)),
            sdnnff_median_ape=round(float(((g.sd - g.meas).abs() / g.meas * 100).median()), 1),
            median_ratio=round(float((g.sd / g.meas).median()), 2))
    return out


def main() -> int:
    # ---- the published model's predictions, one value per compound at 300 K -------------------
    sd = pd.read_csv(SDNNFF, low_memory=False)
    sd = sd[(sd.temperature_K == T_REF) & (sd.heusler_type.astype(str) == "ABC")]
    sd["red"] = sd.formula.map(red)
    sdk = sd.dropna(subset=["red"]).groupby("red").kappa_L_predicted.median()
    print(f"SDNNFF ternary predictions at {T_REF:.0f} K: {len(sdk)} compounds")

    # ---- our in-domain held-out set, exactly as the headline defines it ------------------------
    B = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
    d = B[B.klass == "half"].copy()
    d["_v"] = d.compound.map(vec)
    st = {x: structure_status(str(x)) for x in d.compound.unique()}
    d = d[(d._v == 18) & d.compound.map(lambda x: st[x][0] and not st[x][1])].copy()
    d["chem"] = d.compound.map(lambda c: cluster_of(c) or "?")
    d["red"] = d.compound.map(red)
    n_indomain = d.compound.nunique()

    # temperature match: only measurements near the one temperature SDNNFF reports
    near = d[(d["T"] - T_REF).abs() <= T_WINDOW].copy()
    print(f"in-domain compounds: {n_indomain}; with a measurement within "
          f"+/-{T_WINDOW:.0f} K of {T_REF:.0f} K: {near.compound.nunique()}")

    # ---- our arm: calibrated, with the compound's whole chemistry held out ---------------------
    rows = []
    for cl in sorted(set(d.chem)):
        te, tr = near[near.chem == cl], d[d.chem != cl]
        if not len(te) or len(tr) < 4:
            continue
        # shared constant fitted on published calculations without this cluster
        # (PREREG_shared_constant_on_calculations rule 4), not on the other clusters' predictions
        c, p = SCN.shared_cp_without_cluster(cl)
        if not np.isfinite(c):
            continue
        rows.append(te.assign(k_ours=te.k_pred.values * E.apply_cp(te["T"].values, c, p),
                              c_used=c, p_used=p))
    if not rows:
        print("no held-out rows near the reference temperature")
        return 1
    R = pd.concat(rows)
    R["k_sdnnff"] = R.red.map(sdk)
    R["k_sdnnff_cal"] = R.k_sdnnff * E.apply_cp(R["T"].values, R.c_used.values, R.p_used.values)

    # one figure per compound, so a compound with many measured rows does not dominate
    def per_compound(frame, col):
        f = frame.dropna(subset=[col]).copy()
        f["ape"] = (f[col] - f.k_ref).abs() / f.k_ref * 100.0
        return f.groupby("compound").ape.median()

    # WITHIN A FACTOR OF TWO IS A RATIO TEST, NOT AN ERROR TEST. This used "median error under
    # 100%", which is right only for over-prediction: an under-prediction leaves the factor-of-two
    # band at half the measurement, a 50% error. So "<100%" counted under-predictions between half
    # and zero as within 2x. The rule here is the one compute_deployed_route._within2 has always
    # used: the compound's median predicted/measured ratio lies in [0.5, 2].
    def within2(frame, col):
        f = frame.dropna(subset=[col]).copy()
        r = (f[col] / f.k_ref).groupby(f.compound).median()
        return r.between(0.5, 2.0)

    ours_all = per_compound(R, "k_ours")
    common = sorted(set(per_compound(R, "k_sdnnff").index))
    ours = per_compound(R, "k_ours").reindex(common)
    them = per_compound(R, "k_sdnnff").reindex(common)
    them_cal = per_compound(R, "k_sdnnff_cal").reindex(common)

    def w2(a, b):
        m = a.notna() & b.notna()
        if m.sum() < 5 or np.allclose(a[m].values, b[m].values):
            return None
        return round(float(wilcoxon(a[m].values, b[m].values).pvalue), 4)

    rep = dict(
        comparator=dict(name="Elemental-SDNNFF", doi=SDNNFF_DOI,
                        what="neural-network force field, lattice kappa via ShengBTE",
                        note="published predictions; EXCLUDED from our training labels"),
        protocol=dict(reference_temperature_K=T_REF, temperature_window_K=T_WINDOW,
                      our_arm="leave-one-chemistry-out, shared calibration refitted per fold on "
                              "published calculations without the held-out cluster",
                      their_arm="no holdout -- these compounds may be in its training set, "
                                "which favours the published model"),
        n_in_domain=int(n_indomain),
        n_scored_ours=int(ours_all.notna().sum()),
        n_common=len(common),
        compounds=common,
        ours_median_ape=round(float(ours.median()), 1),
        sdnnff_median_ape=round(float(them.median()), 1),
        sdnnff_calibrated_median_ape=round(float(them_cal.median()), 1),
        ours_within_2x_pct=round(float(within2(R, "k_ours").reindex(common).mean() * 100), 1),
        sdnnff_within_2x_pct=round(float(within2(R, "k_sdnnff").reindex(common).mean() * 100), 1),
        sdnnff_calibrated_within_2x_pct=round(float(
            within2(R, "k_sdnnff_cal").reindex(common).mean() * 100), 1),
        ours_closer_pct=round(float((ours < them).mean() * 100), 1),
        wilcoxon_p_ours_vs_sdnnff=w2(ours, them),
        wilcoxon_p_sdnnff_raw_vs_calibrated=w2(them, them_cal),
        median_ratio_sdnnff_to_measured=round(
            float((R.dropna(subset=["k_sdnnff"]).assign(r=lambda x: x.k_sdnnff / x.k_ref)
                   .groupby("compound").r.median()).median()), 2),
        seed=0,
        seed_note=("our arm is the seed-0 blind-test file (target_blind_test.csv); the SDNNFF "
                   "and published-BTE columns do not depend on the model seed"),
        positive_control=_control(R),
        window_sensitivity=_window(d, sdk),
        per_compound={c: dict(measured=round(float(R[R.compound == c].k_ref.median()), 2),
                              ours=round(float(ours[c]), 1),
                              sdnnff=round(float(them[c]), 1),
                              sdnnff_calibrated=round(float(them_cal[c]), 1))
                      for c in common},
        _meaning=("median absolute percentage error against measurement, per compound, at "
                  f"{T_REF:.0f} K. sdnnff_calibrated applies OUR transfer function to THEIR "
                  "prediction, which tests whether the correction transfers to an independent "
                  "calculation source."))

    print()
    print(f"  scored on {len(common)} compounds both cover")
    print(f"  ours              {rep['ours_median_ape']:5.1f}%   "
          f"within 2x {rep['ours_within_2x_pct']:.0f}%")
    print(f"  SDNNFF raw        {rep['sdnnff_median_ape']:5.1f}%   "
          f"within 2x {rep['sdnnff_within_2x_pct']:.0f}%   "
          f"median ratio to measurement {rep['median_ratio_sdnnff_to_measured']}x")
    print(f"  SDNNFF calibrated {rep['sdnnff_calibrated_median_ape']:5.1f}%")
    print(f"  ours closer on {rep['ours_closer_pct']:.0f}%, "
          f"paired Wilcoxon p={rep['wilcoxon_p_ours_vs_sdnnff']}")
    print(f"  their raw vs their calibrated: p={rep['wilcoxon_p_sdnnff_raw_vs_calibrated']}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
