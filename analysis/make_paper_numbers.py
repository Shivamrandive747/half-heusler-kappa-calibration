"""Every number the manuscript is allowed to state, in one file, regenerated from source.

The manuscript quotes `data/exports/kappa_v2/paper_numbers.json` and nothing else. Four headline
figures have already been withdrawn from this project because a number was typed into prose, the
code moved underneath it, and the prose did not follow. A registry does not prevent the code from
changing -- it makes the change visible, because regenerating this file changes the manuscript's
inputs in one place instead of in fourteen paragraphs.

Anything that cannot be derived from a committed artifact is not in here, which means it cannot go
in the paper. That is the point.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings
from datetime import datetime, timezone

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

from run_loco_chemistry import sites

EX = "data/exports/kappa_v2"
OUT = f"{EX}/paper_numbers.json"

# Figures published internally, then withdrawn. Regenerating must never reintroduce one.
BLOCKED = {"13.6": "hurdle gain -- seed reached the classifier only",
           "41.4": "single-seed headline",
           "36.5": "pooled set quoted as half-only",
           "79.5": "inter-lab ceiling, a literal computed nowhere",
           "18.0": "NbGaRu2 'measurement' fabricated from a title-only stub"}


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# Registry values that coincide with a withdrawn literal by chance (checked by hand, 2026-10-04): the
# guard skips these exact paths so a genuine reappearance anywhere else is still caught.
BLOCK_ALLOW = {"in_domain_table.one_parameter_ablation.within_2x_pct",
               # 2026-10-05 rerun: computed values that coincide with withdrawn literals
               # (79.5 = 31/39; 18.0 a per-compound error), checked by hand
               "in_domain_table.seed_medians.uncorrected_surrogate.within_2x.range[1]",
               "in_domain_table.seed_medians.one_parameter_c_only.within_2x.range[0]",
               "calibration.bands.0.80.empirical_coverage_pct",
               "reference_changes.moved_since_2026_09_18_snapshot.ScNiSb.ape_deployed_after",
               "split_4f_test.paired_test.per_compound.HoNiSb.split",
               "split_4f_test.paired_test.per_compound.ScNiSb.whole",
               # 2026-10-05: the c-only ablation's seed-median within-2x is 79.5 by coincidence (a
               # share of the in-domain compounds), not the withdrawn inter-lab literal
               "in_domain_table.seed_medians.one_parameter_c_only.within_2x.median",
               "split_4f_test.per_family.Ni-Sb.sub.4f.loo_ape",
               "split_4f_test.paired_test.per_compound.DyNiSb.split",
               "vs_published_model.positive_control.sdnnff_vs_published_calculation.median_ape"}


def _leaves(obj, path=""):
    """(dotted path, string value) for every scalar in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, str(obj)


def _assert_baked_constants(c: float, p: float) -> None:
    """FAIL LOUDLY when a CSV the registry quotes was computed with a constant that is no longer live.

    The structure-only, surrogate, calibrated-new, Bi-Pt twin and conditional tables each multiply a
    theory-scale value by the shared constant or a family constant and store only the product. If the
    constant moves (a data fix, a refit) and the producing script is not re-run, the registry would
    quote a stale product with nothing raising an error. Every stored product is recomputed here from
    the stored theory value and the LIVE constant (shared_constant.shared_cp(), family_calibration.json)
    and must agree to the 2-decimal rounding of the inputs (FIXPASS C2, 2026-10-05)."""
    import os
    from transfer_forms import apply_family
    TM = "data/Target_Materials"
    TOL = 0.012                      # two rounded 2-dp inputs/outputs (factor up to ~1.3 uncapped)
    bad: list[str] = []
    fc = jload(f"{EX}/family_calibration.json")
    if abs(fc["global"]["c"] - c) > 5e-4 or abs(fc["global"]["p"] - p) > 5e-4:
        bad.append(f"family_calibration.json global {fc['global']} != shared_cp() ({c}, {p}): "
                   "re-run family_calibration.py")
    fams = {f: r for f, r in fc["families"].items() if r.get("adopted")}

    def chk(name, comp, got, want):
        if not (np.isfinite(got) and np.isfinite(want)) or abs(got - want) > TOL:
            bad.append(f"{name} {comp}: stored {got} vs live {round(float(want), 3)}")

    def shared(k, T=300.0):
        return float(apply_family("GLOBAL", (c, p), [float(T)], [float(k)])[0])

    def rd(f):
        if not os.path.exists(f):
            print(f"  (constant check: {f} absent -- skipped; its registry block will say unavailable)")
            return None
        return pd.read_csv(f)

    d = rd(f"{TM}/STRUCTURE_ONLY_29_VERIFIED.csv")
    if d is not None:
        for r in d.itertuples():
            chk("STRUCTURE_ONLY_29_VERIFIED.kappa_exp_300", r.compound, float(r.kappa_exp_300), shared(r.kappa_theory_300))
    d = rd(f"{TM}/STRUCTURE_ONLY_12_PREDICTIONS.csv")
    if d is not None:
        for r in d.to_dict("records"):
            for T in (300, 600, 900):
                if f"kappa_exp_shared_{T}" in r:
                    chk(f"STRUCTURE_ONLY_12.kappa_exp_shared_{T}", r["compound"], float(r[f"kappa_exp_shared_{T}"]),
                        shared(r[f"kappa_theory_{T}"], T))
    d = rd(f"{TM}/SURROGATE_11_PREDICTIONS.csv")
    if d is not None:
        for r in d.dropna(subset=["kappa_theory_300"]).itertuples():
            chk("SURROGATE_11.kappa_exp_300", r.compound, float(r.kappa_exp_300), shared(r.kappa_theory_300))
    d = rd(f"{TM}/SERIES_GAP_PREDICTIONS.csv")
    if d is not None:
        for r in d.dropna(subset=["kappa_theory_300"]).itertuples():
            chk("SERIES_GAP.kappa_exp_300", r.compound, float(r.kappa_exp_300), shared(r.kappa_theory_300))
    d = rd(f"{TM}/CALIBRATED_NEW_PREDICTIONS.csv")
    if d is not None:
        for r in d.itertuples():
            fr = fams.get(r.family)
            if fr is None:
                bad.append(f"CALIBRATED_NEW {r.compound}: family {r.family} is no longer calibrated")
                continue
            want = f"c={fr['c']} p={fr['p']} {fr['form']}"
            if str(r.family_constant) != want:
                bad.append(f"CALIBRATED_NEW {r.compound}: constant '{r.family_constant}' vs live '{want}'")
            fam = lambda k: float(apply_family(fr["form"], (fr["c"], fr["p"]), [300.0], [float(k)])[0])  # noqa: E731
            chk("CALIBRATED_NEW.kappa_exp_route1_ML", r.compound, float(r.kappa_exp_route1_ML), fam(r.kappa_theory_ML_300))
            chk("CALIBRATED_NEW.kappa_exp_route2_series", r.compound, float(r.kappa_exp_route2_series),
                fam(r.kappa_theory_series_300))
    d = rd(f"{TM}/BIPT_THEORY_PREDICTIONS.csv")
    if d is not None:
        for r in d.itertuples():
            chk("BIPT.kappa_exp_twin_flat_300", r.compound, float(r.kappa_exp_twin_flat_300),
                shared(r.kappa_theory_twin_flat_300))
            chk("BIPT.kappa_exp_twin_family_ratio_300", r.compound, float(r.kappa_exp_twin_family_ratio_300),
                shared(r.kappa_theory_twin_family_ratio_300))
    d = rd(f"{TM}/CONDITIONAL_PREDICTIONS.csv")
    if d is not None:
        for r in d.itertuples():
            if str(r.c_basis).startswith("global"):
                form, cc, pp = "GLOBAL", c, p
            elif r.family in fams:
                fr = fams[r.family]
                form, cc, pp = fr["form"], fr["c"], fr["p"]
            else:
                bad.append(f"CONDITIONAL {r.compound}: basis '{r.c_basis}' but family {r.family} not calibrated")
                continue
            if abs(float(r.c_used) - float(cc)) > 5e-4:
                bad.append(f"CONDITIONAL {r.compound}: c_used {r.c_used} vs live {cc} ({r.c_basis})")
            fac = float(apply_family(form, (cc, pp), [float(r.quoted_at_K)], [1.0])[0])
            if abs(float(r.transfer_factor) - fac) > 0.006:
                bad.append(f"CONDITIONAL {r.compound}: transfer_factor {r.transfer_factor} vs live {fac:.3f}")
            chk("CONDITIONAL.kappa_pred_quoted", r.compound, float(r.kappa_pred_quoted),
                float(r.kappa_BTE_quoted) * fac)
    if bad:
        raise SystemExit("STALE CONSTANT in a CSV the registry quotes -- re-run its producer "
                         "(rerun_downstream.sh order):\n  " + "\n  ".join(bad))
    print("  constant check: every CSV-baked product matches the live shared / family constants")


def _new_results(N: dict) -> None:
    """The results added on 2026-10-03/04: the ML generator's accuracy against published calculations,
    the shared constant's error on families it never saw, the evidence behind the range gate and the
    family form, and the predictions made from structure alone. Every value is read from the artefact
    its script writes; nothing is typed here."""
    TM = "data/Target_Materials"
    # -- the ML model against published full calculations, whole X-site chemistry held out --------
    try:
        v = pd.read_csv(f"{EX}/bte_generator_validation.csv")
        g = v.groupby("compound").agg(ape=("ape", "median"), ratio=("ratio", "median"))
        N["ml_generator_validation"] = dict(
            n_compounds=int(len(g)), n_clusters=int(v.cluster.nunique()),
            median_ape=round(float(g.ape.median()), 1),
            within_2x_pct=round(100.0 * float(g.ratio.between(0.5, 2.0).mean()), 1),
            median_ratio=round(float(g.ratio.median()), 2),
            source="validate_bte_generator.py -> bte_generator_validation.csv",
            HOW_TO_STATE=("accuracy of the structure -> kappa_BTE step alone, against published full "
                          "calculations at VEC 18; never stacked onto the blind-test headline, which "
                          "already contains this step"))
    except Exception as e:  # noqa: BLE001
        N["ml_generator_validation"] = {"_unavailable": str(e)}
    # -- the shared constant on families it never saw (leave one family out, published calcs) ------
    try:
        sc = jload("paper/evidence/shared_constant_test.json")
        c0 = sc["lofo"]["C0"]
        N["shared_constant_unseen_family"] = dict(
            n_compounds=c0["n"], median_ape=c0["median_ape"], within_2x_pct=c0["within_2x_pct"],
            under_25_pct=c0["under_25_pct"], per_family=c0["per_family"],
            HOW_TO_STATE=("the validated error of the shared constant applied to a family with no "
                          "measured member; this is the error that travels with every shared-constant "
                          "estimate"))
    except Exception as e:  # noqa: BLE001
        N["shared_constant_unseen_family"] = {"_unavailable": str(e)}
    # -- how far independent calculations, and independent laboratories, disagree ------------------
    try:
        import source_identity as SI
        NOT_BTE = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"  # as make_paper_predictions
        import private_manifest as PM

        # Two pure helpers whose home modules are not redistributed. PM.helper returns the canonical
        # definition where that module exists (named in the local manifest only) and this identical
        # restatement everywhere else.
        def _is_hh(c):
            co = Composition(c)
            return len(co.elements) == 3 and len(set(co.values())) == 1

        def _red(f):
            try:
                return Composition(str(f).strip()).reduced_formula
            except Exception:  # noqa: BLE001
                return None
        is_hh, red = PM.helper("is_hh", _is_hh), PM.helper("red", _red)
        tr = SI.dedupe(pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv", low_memory=False))
        tr = tr[(pd.to_numeric(tr.method_tier, errors="coerce") == 1)
                & ~tr.method.astype(str).str.lower().str.contains(NOT_BTE, regex=True)]
        tr = tr.assign(T=pd.to_numeric(tr.temperature_K, errors="coerce"), k=pd.to_numeric(tr.kappa_L, errors="coerce"),
                       r=tr.formula.map(red), s=tr.source_doi.map(SI.canonical)).dropna(subset=["r", "s"])
        tr = tr[tr["T"].between(280, 320) & (tr.k > 0) & tr.r.map(lambda c: bool(c) and is_hh(c))]
        per = tr.groupby(["r", "s"]).k.median().groupby(level=0)
        spread = (per.max() / per.min())[per.size() >= 2]
        il = jload(f"{EX}/interlab_ceiling.json")
        N["source_disagreement"] = dict(
            calculations_300K=dict(n_compounds=int(len(spread)), median_factor=round(float(spread.median()), 2),
                                   p90_factor=round(float(spread.quantile(0.9)), 2), max_factor=round(float(spread.max()), 2),
                                   max_compound=str(spread.idxmax()),
                                   rule=("half Heuslers with full BTE calculations from >=2 distinct sources at 280-320 K; "
                                         "a preprint and its published version count as one source (source_identity)")),
            laboratories=dict(median_factor=round(float(il["median_disagreement_x"]), 2),
                              median_factor_per_compound=(round(float(il["median_disagreement_x_per_compound"]), 2)
                                                          if "median_disagreement_x_per_compound" in il else None),
                              p90_factor=round(float(il["p90_disagreement_x"]), 2),
                              within_2x=il.get("within_2x"), scope=il.get("scope"),
                              n_compounds=il["population"]["compounds"], n_pairs=il["n_pairs"]))
    except Exception as e:  # noqa: BLE001
        N["source_disagreement"] = {"_unavailable": str(e)}
    try:
        fn = jload(f"{EX}/family_null.json")
        N["family_null_baseline"] = dict(n=fn["head_to_head"]["n"], model_median_ape=fn["head_to_head"]["model"],
                                         family_null_median_ape=fn["head_to_head"]["family"],
                                         wilcoxon_p=fn["head_to_head"]["wilcoxon_p"],
                                         model_better_pct=fn["head_to_head"]["model_better_pct"],
                                         family_null_within_2x=fn["family_null"]["within_2x"],
                                         cannot_cover=fn["family_null_cannot_cover"])
    except Exception as e:  # noqa: BLE001
        N["family_null_baseline"] = {"_unavailable": str(e)}
    # -- range gate and family-form evidence ----------------------------------------------------
    try:
        rg = jload("paper/evidence/range_gate_test.json")
        N["range_gate"] = dict(groups=rg["groups"], issue_bar_pct=rg["issue_bar_pct"],
                               p_marginal_vs_inside=rg["mannwhitney_p_marginal_vs_inside"])
    except Exception as e:  # noqa: BLE001
        N["range_gate"] = {"_unavailable": str(e)}
    try:
        ff = jload("paper/evidence/family_form_nested.json")
        N["family_form_nested"] = dict(n_compounds=ff["n_compounds"], pooled_median=ff["pooled_median"],
                                       uncapped_chosen_in=ff["nocap_chosen_in"],
                                       HOW_TO_STATE=("the form choice made inside each fold; quote "
                                                     "beside the deployed held-out family figure"))
    except Exception as e:  # noqa: BLE001
        N["family_form_nested"] = {"_unavailable": str(e)}
    try:
        dr = jload(f"{EX}/deployed_route.json")["family_arm_compounds"]
        N["deployed_route"]["held_out"]["within_2x_pct"] = dr["within2x_pct"]
        N["deployed_route"]["held_out"]["global_within_2x_pct"] = dr["global_on_same_compounds"]["within2x_pct"]
    except Exception as e:  # noqa: BLE001
        print(f"  !! deployed held-out within-2x unavailable ({e})")
    try:
        P = pd.read_csv(f"{TM}/PAPER_PREDICTIONS.csv")
        iss = P[P.status == "ISSUED"]
        # Counted AT THE QUOTED TEMPERATURE. A value quoted at 500 K rests on that one calculation by
        # construction (it is quoted there because 300 K is missing or contested -- TiNiPb has two
        # 300 K calculations 5.6x apart and one uncontested 500 K one); counting n_src_300K missed it.
        _single = ((iss.quoted_at_K == 300) & (iss.n_src_300K <= 1)) | (iss.quoted_at_K != 300)
        N["predictions"]["issued_single_calculation_source"] = int(_single.sum())
        N["predictions"]["issued_with_calculated_curve"] = sorted(iss[iss.n_bte_temps > 2].compound)
        # STANDARD SIGN TEST: exact ties (family arm == shared arm, e.g. a member whose family
        # constant equals the shared one) carry no sign and are DROPPED, not counted as losses; the
        # number dropped is recorded. Counts come from the per-compound file, cross-checked against
        # deployed_route.json's family-arm compound list.
        from scipy.stats import binomtest as _bt
        _dr = jload(f"{EX}/deployed_route.json")["family_arm_compounds"]
        _pp = pd.read_csv(f"{EX}/family_deployed_pooled.csv")
        _fa = _pp[_pp.arm == "family"]
        assert sorted(_fa.compound) == sorted(_dr["compounds"]), \
            "family_deployed_pooled.csv family arm differs from deployed_route.json"
        _tie = np.isclose(_fa.ape.values, _fa.ape_global.values, rtol=0.0, atol=1e-9)
        _nb = int(((_fa.ape < _fa.ape_global) & ~_tie).sum())
        _nw = int(((_fa.ape > _fa.ape_global) & ~_tie).sum())
        N["deployed_route"]["held_out"]["sign_test_p_family_vs_shared"] = (
            round(float(_bt(_nb, _nb + _nw).pvalue), 3) if _nb + _nw else None)
        N["deployed_route"]["held_out"]["sign_test_counts"] = dict(
            n_family_better=_nb, n_family_worse=_nw, n_ties=int(_tie.sum()), n=int(len(_fa)),
            rule="two-sided exact binomial on better vs worse; exact ties dropped")
        N["predictions"]["issued_quoted_at_500K"] = sorted(iss[iss.quoted_at_K == 500].compound)
    except Exception as e:  # noqa: BLE001
        print(f"  !! issued provenance counts unavailable ({e})")
    # -- predictions from structure alone -------------------------------------------------------
    try:
        V = pd.read_csv(f"{TM}/STRUCTURE_ONLY_29_VERIFIED.csv")
        S12 = pd.read_csv(f"{TM}/STRUCTURE_ONLY_12_PREDICTIONS.csv").set_index("compound")
        SU = pd.read_csv(f"{TM}/SURROGATE_11_PREDICTIONS.csv")
        sh = N.get("shared_constant_unseen_family", {})
        main = {}
        for c in ("TmPbAu", "HfTeOs"):
            r = S12.loc[c]
            assert abs(float(r.kappa_theory_300) - float(V.set_index("compound").loc[c, "kappa_theory_300"])) < 0.011
            main[c] = dict(family=r.family, tier=r.tier, a_conv_A=float(r.a_conv_A),
                           structure_source=r.structure_source, cubic_is_lowest_3atom=bool(r.cubic_is_lowest_3atom),
                           kappa_theory_300=float(r.kappa_theory_300), seed_spread_300=float(r.seed_spread_300),
                           kappa_exp_shared_300=float(r.kappa_exp_shared_300),
                           family_members_heldout=str(r.family_members_heldout),
                           family_members_median_err_pct=float(r.family_members_median_err_pct))
        sut = SU.set_index("compound")
        zr = float(sut.loc["ZrTeOs", "kappa_SDNNFF_300"])
        twin = float(pd.read_csv(f"{EX}/chemical_twins.csv").ratio.median())
        main["HfTeOs"]["cross_check_ZrTeOs_SDNNFF_x_twin"] = round(zr * twin, 2)
        main["HfTeOs"]["cross_check_ratio_model_over_twin"] = round(main["HfTeOs"]["kappa_theory_300"] / (zr * twin), 2)
        surrogate_only = SU[SU.lit_kappa_papers == 0]
        well = surrogate_only[~surrogate_only.contains_Zn_Cd & ~surrogate_only.non_cubic_on_record.astype(bool)
                              & (surrogate_only.family_members_median_err_pct < 25)]
        supp = {c: dict(family=r.family, kappa_theory_300=float(r.kappa_theory_300), seed_spread_300=float(r.seed_spread_300),
                        kappa_exp_shared_300=float(r.kappa_exp_300), kappa_SDNNFF_300=float(r.kappa_SDNNFF_300),
                        model_over_SDNNFF=float(r.model_over_SDNNFF),
                        family_members_median_err_pct=float(r.family_members_median_err_pct))
                for c, r in well.set_index("compound").iterrows()}
        N["structure_only"] = dict(
            shared_constant=dict(c=N["calibration"]["c"], p=N["calibration"]["p"]),
            main_text=main, surrogate_only_well_supported=supp,
            counts=dict(listed_no_published_kappa=int(len(V)),
                        by_category={str(k): int(n) for k, n in V.category.value_counts().items()},
                        by_literature={str(k): int(n) for k, n in V.literature.value_counts().items()},
                        surrogate_only_candidates=int(len(SU)), surrogate_only=int(len(surrogate_only)),
                        surrogate_only_well_supported=int(len(well))),
            HOW_TO_STATE=("two separately validated steps, stated side by side and never merged: the model "
                          "against published calculations (ml_generator_validation) and the shared constant "
                          "on families never measured (shared_constant_unseen_family)"))
    except Exception as e:  # noqa: BLE001
        N["structure_only"] = {"_unavailable": str(e)}
    # -- new compounds inside the calibrated chemistry (two independent routes) --------------------
    try:
        from make_paper_predictions import FLUCT
        cf = jload(f"{EX}/conformal_indomain.json")["levels"]
        b50, b90 = float(cf["0.50"]["factor"]), float(cf["0.90"]["factor"])
        CN = pd.read_csv(f"{TM}/CALIBRATED_NEW_PREDICTIONS.csv")
        out = {}
        for _, r in CN.iterrows():
            lo, hi = (float(x) for x in str(r.family_measured_range_300K).split(" ")[0].split("-"))
            v1 = float(r.kappa_exp_route1_ML)
            margin = (lo / v1) if v1 < lo else (v1 / hi) if v1 > hi else 1.0
            els = {str(e) for e in Composition(r.compound).elements}
            out[r.compound] = dict(
                family=r.family, family_constant=r.family_constant, family_heldout_err_pct=float(r.family_heldout_err_pct),
                a_conv_A=float(r.a_conv_A), kappa_theory_ML_300=float(r.kappa_theory_ML_300),
                seed_spread_300=float(r.seed_spread_300), route1_ML_family=v1,
                route1_lo50=round(v1 / b50, 2), route1_hi50=round(v1 * b50, 2),
                route1_lo90=round(v1 / b90, 2), route1_hi90=round(v1 * b90, 2),
                kappa_theory_series_300=float(r.kappa_theory_series_300),
                series_is_interpolation=bool(r.series_is_interpolation),
                route2_series_family=float(r.kappa_exp_route2_series), routes_ratio=float(r.routes_ratio),
                family_measured_range_300K=[lo, hi], extrapolation_margin=round(margin, 2),
                range_label=("inside" if margin == 1.0 else "marginal" if margin < 1.33 else "clear"),
                element_holdout_X=str(r.element_holdout_X),
                valence_fluctuating=bool(els & set(FLUCT)),
                # a family constant that fails held out (above the issue bar) is not a validated
                # route, exactly as for the issued predictions (author, 2026-10-05: LuSbPd flagged)
                status=("flagged: valence-fluctuating X element" if els & set(FLUCT)
                        else "flagged: family fails held out" if float(r.family_heldout_err_pct) > 25.0
                        else "reported"))
        N["calibrated_new"] = out
    except Exception as e:  # noqa: BLE001
        N["calibrated_new"] = {"_unavailable": str(e)}
    # -- the chemical-twin METHOD only (author, 2026-10-05: the PrBiPt/NdBiPt estimates are dropped --
    # the twin route no longer beats the model on its own validation pairs, and Pr/Nd are thin in
    # training, so neither route is validated for them)
    try:
        tw = pd.read_csv(f"{EX}/chemical_twins.csv")
        N["twin_estimates"] = dict(
            method=dict(n_pairs=int(len(tw)), flat_ratio=round(float(tw.ratio.median()), 3),
                        loo_median_ape=round(float(tw.twin.median()), 1),
                        model_median_ape_same_pairs=round(float(tw.model.median()), 1)),
            HOW_TO_STATE=("method validation only; quote as a cross-check, not as a prediction route"))
    except Exception as e:  # noqa: BLE001
        N["twin_estimates"] = {"_unavailable": str(e)}
    # -- the ML side, as an ML paper reports it: regressors, attribution, training data --------------
    try:
        mc = jload(f"{EX}/model_comparison_production.json")
        N["model_comparison"] = {k: dict(r2_log=round(v["r2_log"], 3), rmse_log=round(v["rmse_log"], 3),
                                         mae_log=round(v["mae_log"], 3), median_ape_per_compound=round(v["median_ape"], 1),
                                         n_compounds=v["n"], n_rows=v["n_rows"])
                                 for k, v in mc.items() if isinstance(v, dict) and "r2_log" in v}
    except Exception as e:  # noqa: BLE001
        N["model_comparison"] = {"_unavailable": str(e)}
    for key, fn in (("shap_ranking", "shap_ranking.json"), ("training_data", "training_data_summary.json"),
                    ("curves_best3", "curves_best3.json"), ("corpus_figure", "corpus_figure.json")):
        try:
            N[key] = jload(f"{EX}/{fn}")
        except Exception as e:  # noqa: BLE001
            N[key] = {"_unavailable": str(e)}
    # -- significance at the level of the X-site chemistry clusters (compounds are not independent) --
    try:
        import compute_seed_averaged as _S
        from scipy.stats import wilcoxon as _wx, binomtest as _bt
        _files = {0: f"{EX}/target_blind_test.csv", **{s: f"paper/evidence/blind_d2_s{s}.csv" for s in (1, 2, 3, 4)}}
        per = {}
        for s, f in _files.items():
            raw = pd.read_csv(f)
            raw = raw[raw.klass == "half"] if "klass" in raw.columns else raw
            d = _S.in_domain(raw)
            M = _S.per_cmp(_S.nested(d))
            chem = d.groupby("compound").chem.first()
            row = {}
            for name in ("constant", "power"):
                Nl = _S.per_cmp(_S.null(d, name))
                ic = M.index.intersection(Nl.index)
                cl = pd.DataFrame({"m": M.a[ic], "n": Nl.a[ic], "chem": chem[ic]}).groupby("chem")[["m", "n"]].median()
                diff = cl.n - cl.m
                nz = diff[diff != 0]
                row[name] = dict(n_clusters=int(len(cl)), model_better=int((diff > 0).sum()),
                                 wilcoxon_p=round(float(_wx(cl.m, cl.n).pvalue), 4),
                                 sign_p=round(float(_bt(int((nz > 0).sum()), len(nz)).pvalue), 4))
            per[str(s)] = row
        # sensitivity: the headline without YNiBi, whose only measurement the supplement flags
        # (kappa_L rising with T: an electronic or bipolar term likely left in)
        _sens = []
        for s, f in _files.items():
            raw = pd.read_csv(f)
            raw = raw[raw.klass == "half"] if "klass" in raw.columns else raw
            d = _S.in_domain(raw)
            d = d[d.compound != "YNiBi"]
            M = _S.per_cmp(_S.nested(d))
            NC, NP = _S.per_cmp(_S.null(d, "constant")), _S.per_cmp(_S.null(d, "power"))
            ic = M.index.intersection(NC.index)
            _sens.append(dict(median=float(M.a.median()), within_2x=float(100 * M.r.between(0.5, 2).mean()),
                              p_constant=float(_wx(M.a[ic], NC.a[ic]).pvalue),
                              p_power=float(_wx(M.a[ic], NP.a[ic]).pvalue), n=int(len(M))))
        _sd = pd.DataFrame(_sens)
        N["headline_without_YNiBi"] = dict(
            n=int(_sd.n.iloc[0]), median_ape=round(float(_sd["median"].median()), 1),
            seed_range=[round(float(_sd["median"].min()), 1), round(float(_sd["median"].max()), 1)],
            within_2x_pct=round(float(_sd.within_2x.median()), 1),
            p_constant=round(float(_sd.p_constant.median()), 4), p_power=round(float(_sd.p_power.median()), 4),
            HOW_TO_STATE="sensitivity check: the headline with YNiBi (flagged measurement) removed")
        N["cluster_level_significance"] = dict(
            per_seed=per,
            wilcoxon_p_max={n: max(per[s][n]["wilcoxon_p"] for s in per) for n in ("constant", "power")},
            sign_p_range={n: [min(per[s][n]["sign_p"] for s in per), max(per[s][n]["sign_p"] for s in per)]
                          for n in ("constant", "power")},
            HOW_TO_STATE=("each chemistry cluster's median error counted once; quote beside the "
                          "compound-level p, which treats compounds as independent"))
    except Exception as e:  # noqa: BLE001
        N["cluster_level_significance"] = {"_unavailable": str(e)}


def _repair_outcome(t0) -> dict:
    """What the PDF repair did, and the fact that decides how much it mattered.

    The alarming reading of the wrong-PDF defect is that measurements were read out of the wrong
    paper and filed under the right DOI. For the analysed corpus that did not happen: every tier-0
    row under an affected DOI came from Starrydata's digitisation of the real publication, not from
    reading the local file. The defect was in the document store, not in the data. What was
    genuinely at risk is sample metadata read from a local PDF -- milling time, density -- which is
    why the reference ladder skips misidentified DOIs for exactly those rungs.
    """
    try:
        q = pd.read_csv("data/exports/kappa_v2/repair_queue.csv")
    except Exception:  # noqa: BLE001
        return {"_unavailable": "repair_queue.csv absent -- repair not run"}
    bad = {str(d).strip().lower() for d in q.doi}
    aff = t0[t0.d.isin(bad)]
    prov = aff.source.value_counts().to_dict() if len(aff) else {}
    return dict(
        dois_queued_for_refetch=int(len(q)),
        # the refetch outcome counts (13 recovered / 39 rejected / 241 unreachable) were typed literals
        # with no artefact behind them and are quoted nowhere in paper/*.tex: dropped (FIXPASS C2)
        tier0_rows_under_an_affected_doi=int(len(aff)),
        provenance_of_those_rows={str(k): int(v) for k, v in prov.items()},
        any_row_extracted_from_the_wrong_pdf=bool(
            any("starrydata" not in str(k).lower() for k in prov)),
        _meaning=("a row sourced from Starrydata was digitised from the real publication, so a "
                  "wrong local PDF under the same DOI never reached it. If every affected row is "
                  "Starrydata's, no measurement in the analysed corpus was touched by the defect."))


def main() -> int:
    N: dict = {"_generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "_source": "make_paper_numbers.py",
               "_rule": "the manuscript quotes this file; nothing is typed by hand"}

    # ---- the corpus -------------------------------------------------------------------------
    # a preprint and the journal article it became are ONE paper: without dedupe this counted 164
    # measured-side papers while every other file in the paper said 163
    import source_identity as SI
    tr = SI.dedupe(pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv",
                               low_memory=False))
    tr["red"] = tr.formula.map(red)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr = tr[tr.red.notna() & tr.k.notna() & (tr.k > 0)]
    tr["klass"] = tr.red.map(lambda r: (sites(r) or ["?"])[0])
    half = tr[tr.klass == "half"]
    N["corpus"] = dict(
        rows_all=int(len(tr)),
        compounds_all=int(tr.red.nunique()),
        half_heuslers=int(half.red.nunique()),
        half_measured_tier0=int(half[half.tier == 0].red.nunique()),
        half_with_dft_tier1=int(half[half.tier == 1].red.nunique()),
        source_dois_behind_measured=int(half[half.tier == 0].source_doi.nunique()))

    # ---- the blind test ---------------------------------------------------------------------
    B = pd.read_csv(f"{EX}/target_blind_test.csv")
    Bh = B[B.klass == "half"]
    N["blind_test"] = dict(compounds=int(Bh.compound.nunique()),
                           chemistry_clusters=int(Bh.cluster.nunique()),
                           temperature_points=int(len(Bh)),
                           holdout_unit="X-site chemistry cluster (cluster_of)")

    # ---- headline ---------------------------------------------------------------------------
    ns = jload("paper/evidence/nested_across_seeds.json")
    N["headline"] = {}
    for v in (ns if isinstance(ns, list) else [ns]):
        key = ("unconditional" if "threshold 0" in str(v.get("mode", ""))
               else "threshold_free_published_protocol")
        N["headline"][key] = dict(
            median_ape=round(float(v["median"]), 1),
            within_2x_pct=round(float(v["within_2x"]), 1),
            uncorrected_baseline_ape=round(float(v["uncorrected"]), 1),
            winners_curse_pp=round(float(v["curse_pp"]), 1),
            seed_range=[round(float(v["seed_lo"]), 1), round(float(v["seed_hi"]), 1)],
            n_seeds=int(v["n_seeds"]),
            note="seed-averaged; a single-seed nested figure is a draw, not an effect")

    # ---- the in-domain headline table (Table 2) and the offset (Section 3.1) -------------------
    # The abstract's 33.7 / 86.8 / 50.5 / 44.0 / 78.9 / 94.6-on-21 and Section 3.1's 1.55 /
    # 31 of 34 / 12 of 13 were quoted from seed_averaged_indomain.json, results_indomain.json,
    # baselines_ablation.json and fig02_offset.py's stdout without a registry entry, so the gate
    # that checks "every number is in the registry" could not see them.
    try:
        sa = jload(f"{EX}/seed_averaged_indomain.json")
        ri0 = jload(f"{EX}/results_indomain.json")
        ba = jload(f"{EX}/baselines_ablation.json")
        N["in_domain_table"] = dict(
            n=int(sa["per_seed"][0]["n"]) if isinstance(sa.get("per_seed"), list) and sa["per_seed"] and "n" in sa["per_seed"][0] else None,
            n_seeds=sa["n_seeds"],
            seed_medians=sa["table1_seed_medians"],
            # the nulls carry no model seed (a constant-kappa null has nothing to seed), so the
            # single results_indomain figures are the paired comparison for the seed median
            nulls={k: dict(median_ape=v["median_ape"], within_2x_pct=v["within_2x_pct"],
                           within_30_pct=v.get("within_30_pct"), bias=v.get("bias"))
                   for k, v in ri0["nulls"].items()},
            paired_p_seed_median=dict(constant=sa["p_constant_median"], power=sa["p_power_median"],
                                      n_seeds_clearing_both=sa["n_seeds_clearing_both"]),
            spearman=dict(median=sa["spearman_median"], range=sa["spearman_range"]),
            published_semi_empirical=ba["published_semi_empirical"],
            one_parameter_ablation=dict(**ba["one_parameter"], test=ba["exponent_earns_its_keep"],
                                        note="reference seed"),
            HOW_TO_STATE=("headline = seed_medians.calibrated (median over seeds, with its range); "
                          "uncorrected = seed_medians.uncorrected_surrogate; nulls are seed-free and "
                          "paired; the published semi-empirical figure is on its own compounds, never "
                          "beside the blind-test set as if paired"))
    except Exception as e:  # noqa: BLE001
        N["in_domain_table"] = {"_unavailable": str(e)}
        print(f"  !! in_domain_table UNAVAILABLE ({e})")
    try:
        # the per-family blind errors Section 3.6 quotes: a DIFFERENT protocol from the family
        # table (global constant, chemistry held out, the 38-compound blind set), so the two must
        # never be compared row by row
        N["blind_by_family"] = jload(f"{EX}/blind_by_family.json")
    except Exception as e:  # noqa: BLE001
        N["blind_by_family"] = {"_unavailable": str(e)}
        print(f"  !! blind_by_family UNAVAILABLE ({e}) -- run: python paper/fig04_blind.py")
    try:
        N["offset"] = jload(f"{EX}/offset_summary.json")
        N["offset"]["HOW_TO_STATE"] = ("Section 3.1: median_ratio_calc_over_meas over n_compounds / "
                                       "n_points; directional in families_with_median_ratio_above_1 "
                                       "of n_families and compounds_with_ratio_above_1 of n_compounds; "
                                       "the exceptions are compounds_with_ratio_below_1")
    except Exception as e:  # noqa: BLE001
        N["offset"] = {"_unavailable": str(e)}
        print(f"  !! offset UNAVAILABLE ({e}) -- run: python paper/fig02_offset.py")

    # ---- domain of applicability -------------------------------------------------------------
    # The two screens here are NOT new and NOT chosen to flatter the number: make_paper_predictions
    # applies both to every issued prediction and always has. They had simply never been applied to
    # the validation set. Both scopes are published so a reader can see the effect of the
    # restriction rather than take it on trust.
    try:
        dh = jload(f"{EX}/domain_headline.json")
        # the in/out-of-domain split Section 3.6 quotes (105.1% vs 31.0%, Mann-Whitney p) is the
        # chemistry-held-out scoring of results_indomain.json (seed 0, both sides scored the same
        # way) -- NOT domain_headline's in-sample 39.5% for the excluded set
        ri = jload(f"{EX}/results_indomain.json")
        N["domain_of_applicability"] = dict(
            definition="VEC = 18 (semiconducting) AND a cubic C1b record that is not polymorphic",
            screens_are_pre_existing=True,
            scopes=dh.get("rows"), excluded_compounds=dh.get("excluded"),
            chemistry_held_out_split=dict(
                in_domain=ri["in_domain"]["calibrated"],
                out_of_domain=ri["out_of_domain"]["calibrated"],
                test=ri["domain_split_test"],
                note=("single reference seed; a paired in/out comparison, so quote both sides "
                      "together and never the in-domain side alone as the headline (the headline is "
                      "in_domain_table.seed_medians.calibrated)")),
            HOW_TO_STATE=("quote the in-domain figure as the headline, with the unrestricted "
                          "figure and the excluded compounds reported alongside; never quote the "
                          "restricted number alone"))
    except Exception as e:  # noqa: BLE001
        N["domain_of_applicability"] = {"_unavailable": str(e)}

    # ---- nulls (must travel with the headline) ----------------------------------------------
    nb = jload(f"{EX}/null_baselines.json")
    N["nulls"] = dict(
        protocol=nb.get("protocol"),
        n_compounds=nb.get("n_compounds"), n_clusters=nb.get("n_clusters"),
        predictors=nb.get("predictors"), paired_tests=nb.get("paired_tests"),
        rank_signal=nb.get("rank_signal"),
        HOW_TO_STATE=("all half Heuslers, no domain screen, reference seed: quote the paired p-values "
                      "as they are (paired_tests) beside the in-domain seed-median ones; the "
                      "compounds are not independent, so pair them with the cluster-level test"))

    # ---- calibration ------------------------------------------------------------------------
    # FIT ON THE DOMAIN THE CONSTANTS ARE APPLIED TO.
    #
    # This is the third place the same mistake appeared: make_paper_predictions.py,
    # predict_novel_half_heuslers.py and here all fitted on every half Heusler in the blind test,
    # including the thirteen that fail the screens the paper declares. Each produced c=0.500,
    # p=0.950 while Methods and Results reported c=0.51, p=0.80. In this file it is the worst of
    # the three, because this is the registry the manuscript is supposed to quote FROM -- a wrong
    # value here is not one inconsistent artefact, it is the source of truth disagreeing with the
    # paper. Any future script that fits a calibration must filter to the domain first.
    #
    # SINCE 2026-10-05 THE SHARED CONSTANT IS FITTED ON PUBLISHED CALCULATIONS VS MEASUREMENTS
    # (paper/evidence/PREREG_shared_constant_on_calculations.md), through shared_constant.py, the one
    # place every script now takes it from -- no longer on the ML model's blind-test predictions.
    import shared_constant as SCN
    c, p = SCN.shared_cp()
    _pairs = SCN.pairs()
    # THE PRE-REGISTERED STABILITY BLOCK (leave-one-cluster-out, leave-one-family-out, compound
    # bootstrap 10-90%) comes from compute_shared_constant_stability.py. It replaces the typed
    # c_observed_range=[0.37, 0.54], which described run-to-run drift of the OLD blind-test fit.
    _stab = jload(f"{EX}/shared_constant_stability.json")
    if (abs(_stab["deployed"]["c"] - c) > 1e-9 or abs(_stab["deployed"]["p"] - p) > 1e-9):
        raise SystemExit(
            f"shared_constant_stability.json holds c={_stab['deployed']['c']}, "
            f"p={_stab['deployed']['p']} but shared_cp() gives c={c}, p={p}: re-run "
            "compute_shared_constant_stability.py before make_paper_numbers.py")
    _assert_baked_constants(float(c), float(p))
    cf = jload(f"{EX}/conformal_indomain.json")["levels"]
    N["calibration"] = dict(
        form="kappa_expt = kappa_BTE * min(c*(T/300)^p, 1)",
        c=round(float(c), 3), p=round(float(p), 3),
        fitted_on=(f"published calculation / reference measurement pairs: "
                   f"{_pairs.compound.nunique()} compounds, {len(_pairs)} rows "
                   "(shared_constant.py; PREREG_shared_constant_on_calculations.md)"),
        stability=_stab,
        # conformal_indomain.json, NOT interval_validation.json. The "half, support>=3" scope
        # gives 1.32 / 2.06 / 3.21 -- the bands conformal_indomain.json itself labels
        # superseded_bands, computed on 46 compounds with half and full calibration pooled.
        # This registry was regenerating the withdrawn numbers on every run while every
        # other consumer (make_paper_predictions, compute_conditional, fig05) already read
        # the in-domain file. Since PREREG amendment B the bands are residuals AT THE QUOTED
        # TEMPERATURE, median band over the five seeds, with the per-seed range beside it.
        bands={a: dict(multiply_divide=float(cf[a]["factor"]),
                       empirical_coverage_pct=float(cf[a]["empirical_coverage_pct"]),
                       factor_seed_range=cf[a]["factor_seed_range"],
                       factor_per_seed=cf[a]["factor_per_seed"],
                       factor_range_across_folds_seed0=cf[a]["factor_range_across_folds_seed0"])
               for a in ("0.50", "0.80", "0.90")},
        bands_500K={a: dict(multiply_divide=float(cfa[a]["factor"]),
                            empirical_coverage_pct=float(cfa[a]["empirical_coverage_pct"]),
                            factor_seed_range=cfa[a]["factor_seed_range"])
                    for a, cfa in ((a, jload(f"{EX}/conformal_indomain.json")["levels_500K"])
                                   for a in ("0.50", "0.80", "0.90"))})

    # ---- rejected interventions -------------------------------------------------------------
    # family_calibration_nested.json is NOT read any more (FIXPASS C2): no script produces it, it was
    # last written 2026-09-04 on the pre-pivot n=51 set with the withdrawn global c=0.50, p=0.95, and
    # the family route it described has since been rebuilt (family_calibration.py ->
    # family_calibration.json / deployed_route.json, both live below).
    N["rejected_interventions"] = {
        "hurdle_model": "-1.3 pp on experiment, p=0.84 (halves low-kappa error on DFT labels only)",
        "magnitude_calibration_beta": "design half picked it 47/60, lost out of sample",
        "tier0_experimental_data": "+5.5 pp but p=0.10; transition-metal subset worse",
        "doped_compound_data": "no gain, p>=0.10",
        # A POINTER, NOT A COPY. Supplementary Table S2 has five rows and this registry must hold
        # five entries or the two drift apart. The figures live under the top-level split_4f_test
        # key and are not restated here, because a second hand-written copy of a number is exactly
        # how the withdrawn headlines got written.
        "sub_family_split_4f": "see the top-level key 'split_4f_test' (Table S2, fifth row)",
        "per_family_calibration": ("superseded: the family route is live in 'family_calibration' and "
                                   "'deployed_route' (the old ML-route figures had no producer)"),
    }

    # ---- predictions ------------------------------------------------------------------------
    P = pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv")
    cols = ["compound", "family", "kappa_BTE_300", "kappa_pred_300", "quoted_at_K",
            "kappa_BTE_quoted", "kappa_pred_quoted", "extrapolation_margin",
            "lo50", "hi50", "lo90", "hi90", "n_bte_temps", "spacegroups", "note"]
    cols = [c_ for c_ in cols if c_ in P.columns]
    N["predictions"] = {
        st: P[P.status == st][cols].round(2).to_dict(orient="records")
        for st in ("ISSUED", "FLAGGED", "REFUSED")}
    N["predictions"]["counts"] = {k: int(v) for k, v in P.status.value_counts().items()}
    # Section 3 says what the naive pipeline would have issued for TiNiPb (its higher calculation
    # times its family constant) against the highest half-Heusler kappa_L ever measured near room
    # temperature. Both were literals; the 55.6 was the OLD global constant times 109.0.
    _h300 = half[(half.tier == 0) & pd.to_numeric(half.temperature_K, errors="coerce").between(280, 320)]
    _hmax = _h300.loc[_h300.k.idxmax()]
    _tn = P[P.compound == "TiNiPb"]
    # only while TiNiPb is still refused on a naive 300 K value: since 2026-10-02 it is issued at
    # 500 K from its uncontested calculation, and the sentence this fed leaves the manuscript
    if len(_tn) and _tn.status.iloc[0] == "REFUSED" and pd.notna(_tn.kappa_pred_300.iloc[0]):
        N["predictions"]["tinipb_naive_vs_max_measured"] = dict(
            tinipb_naive_300K=round(float(_tn.kappa_pred_300.iloc[0]), 1),
            note=str(_tn.note.iloc[0]),
            max_measured_half_heusler_280_320K=round(float(_hmax.k), 1),
            max_measured_compound=str(_hmax.red), max_measured_source=str(_hmax.source_doi),
            ratio=round(float(_tn.kappa_pred_300.iloc[0]) / float(_hmax.k), 1))

    # the conditional tier (supplementary Table S1): the constant each row was multiplied by is
    # recorded per row, because since 2026-09-19 the anchored families use their OWN constant here
    # and Ni-Bi alone falls back to the global one
    C = pd.read_csv("data/Target_Materials/CONDITIONAL_PREDICTIONS.csv")
    ccols = ["compound", "family", "quoted_at_K", "c_used", "c_basis", "cap_K", "transfer_factor",
             "kappa_BTE_quoted", "kappa_pred_quoted", "lo50", "hi50", "lo90", "hi90",
             "n_sources", "source_spread", "caveat", "unlocked_by"]
    N["conditional"] = dict(
        n=int(len(C)),
        per_family={f: int(n) for f, n in C.family.value_counts().items()},
        n_at_500K=int((C.quoted_at_K == 500).sum()),
        rows=C[ccols].round(3).to_dict(orient="records"),
        HOW_TO_STATE=(f"{len(C)} conditional, all in families with no usable measured member; "
                      "they use the shared constant as a stated proxy."))

    # NOVEL_HALF_HEUSLER_PREDICTIONS.csv (predict_novel_half_heuslers.py, the legacy global constant)
    # is no longer read: no paper/*.tex quotes novel_compounds, and the novel candidates the paper does
    # report come from the structure-only scripts below (FIXPASS C2, 2026-10-05).

    # ---- per-family calibration ---------------------------------------------------------------
    # Families whose measured members agree with each other carry their own transfer function.
    # This is registered separately from N["calibration"] because it answers a DIFFERENT question
    # from the headline: the headline holds out a whole chemistry and asks whether an unseen family
    # can be predicted (it cannot be helped by a family arm, and the eleven conditional predictions
    # all sit there); this asks whether an unseen MEMBER of a measured family can be, which is the
    # situation of every issued prediction. Quoting one figure for the other would repeat exactly
    # the scope error that withdrew three earlier headlines.
    try:
        fc = jload(f"{EX}/family_calibration.json")
        _rcj = jload(f"{EX}/reference_changes.json")
        ad = {f: r for f, r in fc["families"].items() if r.get("adopted")}
        N["family_calibration"] = dict(
            rule=fc["rule"], protocol=fc["protocol"], source=fc.get("source"),
            global_arm=fc["global"],
            adopted={f: dict(tier=r.get("tier"), c=r["c"], p=r["p"], n_members=r["n_members"],
                             ratio_spread=r["ratio_spread"], members=r["members"],
                             validation=r.get("validation"),
                             loo_family_ape=r.get("loo_family_ape"),
                             insample_ape=r.get("insample_ape"),
                             loo_global_ape=r.get("loo_global_ape"),
                             # each member calibrated on its own paper (in-sample), and the spread
                             # of those constants -- the family's honest uncertainty
                             members_own=r.get("members_own"),
                             c_members_min=r.get("c_members_min"), c_members_max=r.get("c_members_max"),
                             c_members_spread=r.get("c_members_spread")) for f, r in ad.items()},
            refused={f: r.get("why_not") for f, r in fc["families"].items()
                     if not r.get("adopted")},
            # per family: how many laboratories have measured ANY member (any tier-0 source), and how
            # many distinct papers the CHOSEN references come from -- Bi-Pd's two members are one
            # paper, which the table must show
            laboratories={f: int(tr[(tr.tier == 0) & tr.red.isin(r["members"])].source_doi.nunique())
                          for f, r in ad.items()},
            reference_papers={f: len({_rcj["all_choices"][c]["doi"] for c in r["members"] if c in _rcj["all_choices"]})
                              for f, r in ad.items()},
            n_validated_under_25=sum(1 for r in ad.values() if r.get("loo_family_ape") is not None and r["loo_family_ape"] < 25),
            n_validated=sum(1 for r in ad.values() if r.get("loo_family_ape") is not None),
            n_anchored=sum(1 for r in ad.values() if r.get("loo_family_ape") is None),
            HOW_TO_STATE=("every family carries its own constant (ruling 2026-09-19). Report each "
                          "family's VALIDATION type: 'leave-one-out (n=k)' families quote loo_family_ape; "
                          "single-member families are ANCHORED and quote insample_ape, never as a "
                          "validation figure. Report c_members_spread beside every constant, and the "
                          "number of distinct papers behind the members (Bi-Pd's two members are ONE "
                          "paper; Sb-Pt's three of four are one paper). Where loo_global_ape is "
                          "equal to or lower than loo_family_ape, say so. Never quote a family figure "
                          "as the headline, which is leave-one-CHEMISTRY-out and a different claim"))
    except Exception as e:  # noqa: BLE001
        N["family_calibration"] = {"_unavailable": str(e)}

    # ---- the deployed route, per compound ---------------------------------------------------
    # compute_deployed_route.py scores every in-domain compound on its own family's arm (held out
    # where the family has another member, in-sample where it does not) and writes the pooled
    # figures Section 3 quotes. Until 2026-09-19 those figures had no registry entry.
    try:
        dr = jload(f"{EX}/deployed_route.json")
        pool = pd.read_csv(f"{EX}/family_deployed_pooled.csv")
        held = pool[~pool.arm.str.contains("in-sample")]
        N["deployed_route"] = dict(
            n=dr["n"], deployed=dr["deployed"], global_only=dr["global"], uncorrected=dr["uncorrected"],
            held_out=dict(n=int(len(held)), median_ape=round(float(held.ape.median()), 1),
                          under_25=int((held.ape < 25).sum()), under_30=int((held.ape < 30).sum()),
                          # the global constant scored on the SAME held-out compounds -- the only
                          # figure that may be quoted beside the held-out median; `global_only`
                          # above includes the four in-sample compounds and is NOT paired with it
                          global_on_same_compounds=dr["family_arm_compounds"].get("global_on_same_compounds"),
                          n_family_better=dr["family_arm_compounds"].get("n_family_better")),
            in_sample=dr.get("family_insample_arm_compounds"),
            anchored_on_own_paper=dr.get("anchored_on_own_paper"),
            # per family, the median over members of the constant fitted on each member's OWN deployed
            # reference (in-sample fit quality, Table famcal's "own-paper fit" column -- never a
            # validation figure) beside the held-out median of the same family
            own_paper_fit_by_family={f: dict(n=int(len(g)),
                                             own_paper_fit=round(float(g.ape_anchored.median()), 1),
                                             held_out=(round(float(g[g.arm == "family"].ape.median()), 1)
                                                       if (g.arm == "family").any() else None))
                                     for f, g in pool.groupby("family") if g.ape_anchored.notna().any()},
            HOW_TO_STATE=(f"three different numbers, never interchangeable: DEPLOYED ({dr['n']}, mixes "
                          f"held-out and {dr['family_insample_arm_compounds']['n']} in-sample), HELD-OUT "
                          f"({len(held)}, the transfer test; compare ONLY with held_out."
                          f"global_on_same_compounds, never with global_only), ANCHORED-ON-OWN-PAPER "
                          f"({dr['anchored_on_own_paper']['n']}, in-sample fit quality -- NEVER a "
                          "validation figure). Quote which."))
    except Exception as e:  # noqa: BLE001
        N["deployed_route"] = {"_unavailable": str(e)}
        print(f"  !! deployed_route UNAVAILABLE ({e}) -- run compute_deployed_route.py")

    # ---- references moved by the nanostructured-sample demotion -------------------------------
    # reference_choice.py ranks a ball-milled / sub-micron sample below any bulk alternative,
    # because the calculation is a bulk crystal. audit_reference_changes.py runs the ladder with
    # that rule off and on and records every compound that moved, with its rung and its error on
    # both references. Section 3 quotes those two compounds and their before/after figures.
    try:
        rc = jload(f"{EX}/reference_changes.json")
        N["reference_changes"] = dict(
            rule=rc["rule"], n_moved=len(rc["moved"]), n_annotated=len(rc["annotated"]),
            moved={m["compound"]: dict(before_doi=m["before_doi"], after_doi=m["after_doi"],
                                       after_rule=m["after_rule"],
                                       ape_before=m["ape_before"], ape_after=m["ape_after"],
                                       ape_deployed_before=m.get("ape_deployed_before"),
                                       ape_deployed_after=m.get("ape_deployed_after"))
                   for m in rc["moved"]},
            moved_since_2026_09_18_snapshot={
                m["compound"]: dict(before_doi=m["before_doi"], after_doi=m["after_doi"],
                                    after_rule=m["after_rule"],
                                    ape_deployed_after=m.get("ape_deployed_after"))
                for m in rc.get("moved_since_2026_09_18_snapshot", [])},
            # family leave-one-out with the snapshot references vs the current ones: this is where
            # "Sb-Pd went from 21.9% to 32.7%" comes from, and the only place it may be quoted from
            family_loo_before_after=rc.get("family_loo_before_after", {}),
            HOW_TO_STATE=("`moved` is the nano rule alone (global-arm AND deployed error before/"
                          "after); `moved_since_2026_09_18_snapshot` is every reference that moved "
                          "for any reason; `family_loo_before_after` is the family figure before "
                          "and after those moves -- quote the family move from HERE, never from "
                          "memory. Say which compounds moved and WHY, quote both errors, and say "
                          "that no other compound's reference changed"))
    except Exception as e:  # noqa: BLE001
        N["reference_changes"] = {"_unavailable": str(e)}
        print(f"  !! reference_changes UNAVAILABLE ({e}) -- run: python audit_reference_changes.py")

    # ---- does 4f occupancy explain within-family scatter? -------------------------------------
    # Section 3 argues that a systematic error in the treatment of the 4f sublattice would appear as
    # scatter within families that mix open and closed shells. test_4f_split.py tests that argument
    # by partitioning on 4f occupancy directly, which is a stronger check than the absence of
    # scatter. Recorded here so the sentences quoting it are not typed by hand.
    try:
        sp = jload(f"{EX}/split_4f_test.json")
        N["split_4f_test"] = dict(
            conclusion=sp["conclusion"],
            # the deciding test: per compound, predicted from the whole family vs from its own 4f
            # class; computed, never asserted (the old verdict flag was a hard-coded True)
            paired_test=sp.get("paired_test"),
            issued_prediction_impact=sp["issued_prediction_impact"],
            per_family={f: dict(n_members=r["n_members"], loo_ape=r["loo_ape"], c=r["c"],
                                sub={t: dict(n_members=s["n_members"], testable=s["testable"],
                                             loo_ape=s.get("loo_ape"), c=s.get("c"),
                                             c_if_relaxed=s.get("c_if_relaxed"),
                                             relaxed_median=s.get("relaxed_median"))
                                     for t, s in r["sub"].items()},
                                pair_protocol=r.get("pair_protocol"))
                        for f, r in sp["families"].items()},
            HOW_TO_STATE=("Since 2026-09-19 (dedupe + calculation-disagreement gate) the 4f split "
                          "is NOT a 'nothing moves' result: it would move Ni-Sb's lanthanide "
                          "constant by conclusion.constant_moves_by_at_most and GdNiSb/TbNiSb by "
                          "issued_prediction_impact.worst_relative_change_pct_1dp. The reason it is "
                          "not adopted is paired_test: median_split vs median_whole over n_paired "
                          "compounds, n_split_better/worse, and the members it leaves unpredictable. "
                          "Quote THAT, never a bound alone. Never quote Sb-Pd's historical 6.6%: one "
                          "pair from ONE laboratory; the same pair now scores per_family['Sb-Pd']"
                          "['sub']['4f']['loo_ape']."))
    except Exception as e:  # noqa: BLE001
        # Announced, not swallowed. Section 3 and supplementary Table S2 quote sixteen numbers that
        # live only in this artefact, and no gate reads it, so a silent miss would leave the paper
        # citing figures nothing produces -- the same defect that withdrew four earlier headlines.
        N["split_4f_test"] = {"_unavailable": str(e)}
        print(f"  !! split_4f_test UNAVAILABLE ({e})\n"
              f"     run: python test_4f_split.py\n"
              f"     until then Section 3 and supplementary Table S2 quote numbers with no artefact")

    # ---- the single measured anchor in each conditional family -------------------------------
    # Supplementary S3 states how the calibrated calculation did on YNiBi, the one measured Ni-Bi
    # compound, as evidence that the conditional tier is not unreasonable even though that
    # measurement is disqualified as validation. Until 2026-09-11 the figure it quoted (21.7%,
    # ratio 0.86) was typed into the prose and produced by no script -- the same defect as the
    # 79.5% in BLOCKED above, and internally inconsistent besides (a ratio of 0.86 is 14% error,
    # not 21.7%). It is computed here by the route the conditional tier actually uses -- the same
    # c, p fitted above and the same semi-empirical filter as compute_conditional.py -- so the
    # prose can only ever say what the code says. The nearest-to-300 K point is the headline
    # because Table S1 reports at 300 K; the all-temperature median travels with it.
    # FIXPASS C2 (2026-10-05): the calculation is the PIPELINE's -- deduplicated corpus
    # (source_identity.dedupe), one value per calculated temperature by
    # family_calibration.resolve_calculations (median over sources, contested temperatures out) and
    # carried to the measurement temperature by family_calibration.bte_at (log-log interpolation,
    # else 1/T capped at 2.5x). The old block took the FIRST row within a flat 150 K on the raw CSV:
    # for YNiBi at 300 K that was 5.91 (a published value AND its arXiv duplicate) over 9.08.
    import family_calibration as _FC
    _tr = SI.dedupe(pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv", low_memory=False))
    _tr["red"] = _tr.formula.map(red)
    _tr["t"] = pd.to_numeric(_tr.method_tier, errors="coerce")
    _tr["TK"] = pd.to_numeric(_tr.temperature_K, errors="coerce")
    _tr["k"] = pd.to_numeric(_tr.kappa_L, errors="coerce")
    _m = _tr.method.astype(str).str.lower()
    _NOT = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"
    _bte_all = _tr[(_tr.t == 1) & ~_m.str.contains(_NOT, regex=True) & (_tr.k > 0)]
    N["conditional_anchors"] = {}
    # ZrCoBi's two laboratories near 300 K, so Section 3 can say how far apart they are without
    # typing the factor: the anchored HfCoBi prediction inherits that single-laboratory basis.
    _z = _tr[(_tr.red == "ZrCoBi") & (_tr.t == 0) & _tr.TK.between(280, 320)].groupby("source_doi").k.median()
    N["zrcobi_laboratories_300K"] = dict(values={str(k): round(float(v), 2) for k, v in _z.items()},
                                         ratio_high_over_low=round(float(_z.max() / _z.min()), 2) if len(_z) >= 2 else None)
    for _cp, _fam in (("YNiBi", "Ni-Bi"), ("ZrCoBi", "Co-Bi")):
        _meas = _tr[(_tr.red == _cp) & (_tr.t == 0) & _tr.k.notna() & _tr.TK.notna()]
        _bte = _bte_all[_bte_all.red == _cp].dropna(subset=["TK"])
        _calc, _picks = _FC.resolve_calculations(_bte, [_cp]) if len(_bte) else ({}, [])
        if _picks:
            # every calculated temperature contested: resolve_calculations would pick against family-
            # mates, which are not passed here -- no uncontested calculation to anchor on
            N["conditional_anchors"][_cp] = {"_unavailable": "every calculated temperature contested",
                                             "contested": {str(k): v for k, v in _FC.CALC_DISAGREE.get(_cp, {}).items()}}
            continue
        _bagg = pd.Series(_calc.get(_cp, {}), dtype=float).sort_index()
        _pts = []
        for _r in _meas.itertuples():
            _kd = _FC.bte_at(_bagg, float(_r.TK)) if len(_bagg) else None
            if _kd is None:
                continue
            _kp = float(_kd) * min(c * (_r.TK / 300.0) ** p, 1.0)
            _pts.append((_r.TK, _kp / _r.k, 100.0 * abs(_kp - _r.k) / _r.k))
        if not _pts:
            N["conditional_anchors"][_cp] = {"_unavailable": "no calculation reachable by bte_at (>= 200 K, <= 2.5x carry)"}
            continue
        _n300 = min(_pts, key=lambda x: abs(x[0] - 300.0))
        N["conditional_anchors"][_cp] = dict(
            family=_fam, n_points=len(_pts),
            nearest_300K=dict(T_K=round(float(_n300[0])), ratio=round(_n300[1], 2),
                              error_pct=round(_n300[2], 1)),
            median_all_T=dict(ratio=round(float(np.median([x[1] for x in _pts])), 2),
                              error_pct=round(float(np.median([x[2] for x in _pts])), 1)),
            all_within_2x=bool(all(0.5 <= x[1] <= 2.0 for x in _pts)),
            calculation_by_T={str(int(k)): round(float(v), 2) for k, v in _bagg.items()},
            route=("calibrated calculation: family_calibration.bte_at(resolved median calculation, T) * "
                   "min(c (T/300)^p, 1), same c,p as above; deduplicated corpus"),
            HOW_TO_STATE=(("YNiBi: quote nearest_300K, and say in the same sentence that the "
                           "measurement is disqualified as validation (bipolar leak, kappa rises "
                           "with T). One anchor is consistency, never validation.")
                          if _cp == "YNiBi" else
                          ("ZrCoBi: this compound ANCHORS the Co-Bi family (single member; quote "
                           "family_calibration.adopted['Co-Bi'].insample_ape); its reference is the 98.1%-dense "
                           "Zhao 2018 sample. The other ZrCoBi laboratory reads lower by the "
                           "ratio in zrcobi_laboratories_300K (ball-milled 20 h) -- HfCoBi's "
                           "anchored prediction inherits that single-laboratory basis and must "
                           "say so.")))

    # ---- evidence tiering for the worked examples -------------------------------------------
    # paper/evidence/evidence_chain.csv (worked_examples) is no longer read (FIXPASS C2): no script in
    # the repository produces it (hand-made 2026-09-06, "13 worked examples" of a superseded set) and
    # no paper/*.tex quotes worked_examples.

    # ---- semi-empirical vs full BTE (supplementary section sec:tiergap) -----------------------
    # Was quoted from a producer-less CSV of 2026-09-04 (pre tier correction); now recomputed from the
    # current tiers by compute_semiempirical_vs_bte.py.
    try:
        N["semiempirical_vs_bte"] = jload(f"{EX}/semiempirical_vs_bte.json")
        N["semiempirical_vs_bte"]["HOW_TO_STATE"] = (
            "Supplementary S10: n_compounds half Heuslers with both labels in a matched 100 K bin; "
            "median disagreement median_factor; within_2x_pct within a factor of two; worst max_factor")
    except Exception as e:  # noqa: BLE001
        N["semiempirical_vs_bte"] = {"_unavailable": str(e)}
        print(f"  !! semiempirical_vs_bte UNAVAILABLE ({e}) -- run: python compute_semiempirical_vs_bte.py")

    # ---- the exponent used to carry a single-temperature calculation -------------------------
    #
    # Methods carries a lone calculated value to the measurement temperature as kappa ~ 1/T. That
    # used to be justified in the text by naming the Umklapp form, which is an appeal to a textbook
    # rather than to anything in this corpus. It does not need to be: the calculations that DO
    # report several temperatures state the exponent themselves, so it is measured here instead.
    try:
        from family_calibration import NOT_BTE
        # deduped, for the same reason the paper count is: a preprint and the article it became
        # would otherwise contribute the same curve twice and inflate n_curves.
        _ce = SI.dedupe(pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv",
                                    low_memory=False))
        _ce["TK"] = pd.to_numeric(_ce.temperature_K, errors="coerce")
        _ce["k"] = pd.to_numeric(_ce.kappa_L, errors="coerce")
        _ce["t"] = pd.to_numeric(_ce.method_tier, errors="coerce")
        _me = _ce.method.astype(str).str.lower()
        _b = _ce[(_ce.t == 1) & (_ce.k > 0) & ~_me.str.contains(NOT_BTE, regex=True)
                 & _ce.TK.between(200, 1300)]
        _sl = []
        for (_f, _d), _g in _b.groupby([_b.formula.astype(str), _b.source_doi.astype(str)]):
            _c = _g.dropna(subset=["TK", "k"]).groupby("TK").k.median()
            if len(_c) >= 3 and _c.index.min() > 0:
                _sl.append(float(np.polyfit(np.log(_c.index.values), np.log(_c.values), 1)[0]))
        _s = np.array(_sl)
        N["carry_exponent"] = dict(
            n_curves=int(_s.size),
            min_temperatures_per_curve=3,
            median_loglog_slope=round(float(np.median(_s)), 2),
            slope_min=round(float(_s.min()), 2), slope_max=round(float(_s.max()), 2),
            max_abs_deviation_from_minus_one=round(float(np.abs(_s + 1).max()), 2),
            frac_within_0p1_of_minus_one=round(float((np.abs(_s + 1) < 0.1).mean()), 3),
            _meaning=("log-log slope of every published BTE curve in the corpus reporting >=3 "
                      "temperatures. This is what licenses the 1/T carry, not the Umklapp form."))
    except Exception as e:  # noqa: BLE001
        N["carry_exponent"] = {"_unavailable": str(e)}

    # ---- provenance: the source counts the Data-availability paragraph quotes -----------------
    #
    # "198 publications ... a further 20 ... two more" were typed into manuscript.tex by hand, and
    # this project has withdrawn four headline numbers for exactly that. They are computed here so
    # the paragraph can be checked. The gap between distinct tier-0 DOIs and the number that can
    # actually be cited is one unresolvable DOI (10.36410/jcpr.2020.21.3.319, 8 rows), which
    # build_bibliography.py already reports as a data-quality problem.
    try:
        import glob as _glob
        import io as _io
        import re as _re
        _cite = _re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
        _used: set = set()
        for _f in sorted(_glob.glob("paper/*.tex")):
            _t = _re.sub(r"(?m)^\s*%.*$", "",
                         _io.open(_f, encoding="utf-8", errors="replace").read())
            for _m in _cite.finditer(_t):
                _used |= {x.strip() for x in _m.group(1).split(",") if x.strip()}
        _bib = _io.open("paper/references.bib", encoding="utf-8").read()
        _doi2key = {}
        for _k, _b in _re.findall(r"@\w+\{([^,]+),(.*?)\n\}", _bib, _re.S):
            _m = _re.search(r"doi\s*=\s*\{([^}]*)\}", _b)
            if _m:
                _doi2key[_m.group(1).lower()] = _k
        import source_identity as _SI
        _pt = tr.copy()
        _pt["d"] = _pt.source_doi.map(_SI.canonical).astype(str).str.strip().str.lower().str.replace(
            r"^https?://(dx\.)?doi\.org/", "", regex=True).str.rstrip(".,;)")
        # an arXiv-only source is a source: counting only "10." ids hid six analysed preprints that
        # were never cited. arxiv:NNNN.NNNNNvK -> its DataCite DOI 10.48550/arxiv.NNNN.NNNNN
        _pt["d"] = _pt.d.str.replace(r"^arxiv:(\d{4}\.\d{4,5})(v\d+)?$", r"10.48550/arxiv.\1", regex=True)
        _out = {}
        for _tier, _name in ((0, "measurement"), (1, "calculation")):
            _s = sorted(set(_pt[(_pt.tier == _tier) & _pt.d.str.startswith("10.")].d))
            _no_entry = [d for d in _s if d not in _doi2key]
            _unc = [d for d in _s if d in _doi2key and _doi2key[d] not in _used]
            _out[f"{_name}_source_dois"] = len(_s)
            _out[f"{_name}_sources_in_reference_list"] = len(_s) - len(_no_entry) - len(_unc)
            _out[f"{_name}_sources_with_no_bib_entry"] = sorted(_no_entry)
            _out[f"{_name}_sources_never_cited"] = sorted(_unc)
        _out["_meaning"] = (
            "Deduped distinct source DOIs by method tier, and how many of them actually reach the "
            "reference list. measurement_sources_in_reference_list is the number the Data "
            "availability paragraph should quote; a non-empty *_never_cited is a provenance gap.")
        N["provenance_sources"] = _out
    except Exception as e:  # noqa: BLE001
        N["provenance_sources"] = {"_unavailable": str(e)}

    # ---- how far the wrong-PDF defect reaches into the reported numbers ----------------------
    #
    # verify_pdf_identity found that a large minority of stored PDFs match neither their own DOI
    # nor their recorded title: the fetch path stored whatever came back. The corpus-wide count is
    # alarming on its own and says nothing about the results, so what is computed here is the
    # narrow question -- whether any CHOSEN REFERENCE, the only source a reported number is
    # computed from, rests on one, and where the defect can still act indirectly through the
    # lab-count that ranks references.
    try:
        _pi = jload(f"{EX}/pdf_identity.json")
        _rc = jload(f"{EX}/reference_changes.json")

        def _nz(x):
            return str(x).strip().lower().replace("https://doi.org/", "").rstrip(".,;)")

        # Only the DOIs the triage calls misfiled. The raw "mismatch" list also holds arXiv
        # versions of the right article, which fail an exact first-eight-words test on one
        # inserted word, and scans with no extractable text, which cannot be judged at all.
        _tri = _pi["mismatch_triage"]["dois"]
        _mis = {_nz(x) for x in _tri["another_paper"] + _tri["access_or_error_interstitial"]}
        _fp = {_nz(x) for x in _tri["same_paper_fuzzy_title"]}
        _wrn = {_nz(e.get("doi") if isinstance(e, dict) else e) for e in _pi["warn"]}
        _ch = {c: _nz(v["doi"]) for c, v in _rc["all_choices"].items()}
        _t0 = tr[tr.tier == 0].copy()
        _t0["d"] = _t0.source_doi.map(_nz)
        _aff = _t0[_t0.d.isin(_mis)]
        _expo = {}
        for _c in sorted(set(_aff.formula) & set(_ch)):
            _g = _t0[_t0.formula == _c]
            _srcs = set(_g.d)
            _expo[_c] = dict(sources=len(_srcs), on_a_mismatched_pdf=len(_srcs & _mis),
                             chosen_ref_rule=_rc["all_choices"][_c]["rule"])
        N["pdf_identity_exposure"] = dict(
            stored_pdfs=int(_pi["n_pdfs"]),
            flagged_by_exact_test=int(_pi["mismatch_triage"]["n_flagged"]),
            false_positives_same_paper=len(_fp),
            unverifiable_no_text=int(_pi["mismatch_triage"]["counts"]["no_extractable_text"]),
            mismatched=len(_mis),
            mismatched_pct=round(100.0 * len(_mis) / max(int(_pi["n_pdfs"]), 1), 1),
            repair=_repair_outcome(_pt),
            chosen_references=len(_ch),
            chosen_refs_on_a_mismatch=sorted(c for c, d in _ch.items() if d in _mis),
            chosen_refs_on_a_warn=sorted(c for c, d in _ch.items() if d in _wrn),
            tier0_rows_from_a_mismatched_pdf=int(len(_aff)),
            analysed_compounds_with_an_affected_alternative_source=_expo,
            _meaning=("chosen_refs_on_a_mismatch is the one that would invalidate a reported "
                      "number; the per-compound block is the indirect route, where a phantom "
                      "source still counts toward the lab centrality that ranks references."))
    except Exception as e:  # noqa: BLE001
        N["pdf_identity_exposure"] = {"_unavailable": str(e)}

    # ---- a PUBLISHED machine-learned model, scored against measurement -----------------------
    #
    # The Introduction's claim that surrogates are accurate against calculation and not against
    # experiment was, until now, an inference from the 1.55 offset rather than a measurement of any
    # real surrogate. compute_vs_published_model.py measures one.
    try:
        N["vs_published_model"] = jload(f"{EX}/vs_published_model.json")
    except Exception as e:  # noqa: BLE001
        N["vs_published_model"] = {"_unavailable": str(e)}

    # ---- is the cap right for the family constants, whose caps bind much earlier? ------------
    try:
        N["cap_test"] = jload(f"{EX}/cap_test.json")
    except Exception as e:  # noqa: BLE001
        N["cap_test"] = {"_unavailable": str(e)}

    _new_results(N)

    # ---- guard: no withdrawn literal may reappear -------------------------------------------
    import re
    hits = []
    for path, leaf in _leaves(N):
        if path in BLOCK_ALLOW:
            continue
        for lit, why in BLOCKED.items():
            # only flag a standalone number, not a coincidental substring of a longer one
            if re.search(rf"(?<![\d.]){re.escape(lit)}(?![\d])", leaf):
                hits.append(f"{lit} at {path} ({why})")
    # PRIVACY: no identifier or list-membership column from a non-redistributed input may reach the
    # registry (the column names come from the local manifest; "uuid" alone without it)
    import private_manifest as PM
    _blob = json.dumps(N).lower()
    assert not PM.column_pattern().search(_blob), "a non-redistributed input reached the registry"
    N["_blocklist_check"] = ("clean" if not hits
                             else "REVIEW THESE: " + "; ".join(hits))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(N, fh, indent=2)

    print(f"wrote {OUT}\n")
    print(f"  corpus            {N['corpus']['compounds_all']} Heuslers, "
          f"{N['corpus']['half_heuslers']} half, "
          f"{N['corpus']['half_measured_tier0']} measured from "
          f"{N['corpus']['source_dois_behind_measured']} papers")
    print(f"  blind test        {N['blind_test']['compounds']} compounds, "
          f"{N['blind_test']['chemistry_clusters']} chemistries")
    h = N["in_domain_table"]["seed_medians"]["calibrated"]
    print(f"  headline          in domain, n={N['in_domain_table']['n']}: {h['median_ape']['median']}% median "
          f"(seeds {h['median_ape']['range'][0]}-{h['median_ape']['range'][1]}), {h['within_2x']['median']}% within 2x")
    h = N["headline"]["unconditional"]
    print(f"  all {h.get('n', '?')}, no screen {h['median_ape']}% median, {h['within_2x_pct']}% within 2x "
          f"(uncorrected {h['uncorrected_baseline_ape']}%)")
    print(f"  calibration       c={N['calibration']['c']}, p={N['calibration']['p']}")
    print(f"  predictions       {N['predictions']['counts']}")
    print(f"  blocklist         {N['_blocklist_check']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
