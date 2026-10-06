"""Theoretical kappa_L for PrBiPt and NdBiPt -- the two cubic VEC-18 half Heuslers inside the model's
training chemistry with no published kappa of any kind (a candidate enumeration + literature check).

WHAT IS PREDICTED. The model's THEORETICAL (bulk-crystal, BTE-like) kappa_L. Bi-Pt has no measured
member, so there is no family calibration; the experimental-scale value is the SHARED-constant estimate,
labelled as such. A neighbour family's constant is not used: the pre-registered test
(test_neighbour_family_transfer.py) found it worse than the shared constant, 47.8% vs 25.2%.

INPUTS, each checked:
  lattice constant  lanthanide-contraction fit over the Bi-Pt lanthanide members' training lattice
                    constants (neither compound is in DXMag); leave-one-out error printed
  model             CatBoost, 5 seeds, the paper's training pool; seed spread printed
CHECKS:
  1. the model's held-out error on the Bi-Pt siblings themselves (bte_generator_validation.csv,
     whole chemistry cluster hidden)
  2. the family's published calculations across the lanthanide series (model-independent)

Writes data/Target_Materials/BIPT_THEORY_PREDICTIONS.csv.
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
from pymatgen.core import Composition, Element

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import s19_kappa_dataset as ds
from pipeline.s20_kappa_train import make_model
from transfer_forms import apply_family
from make_paper_predictions import yzfam

TARGETS = ["PrBiPt", "NdBiPt"]
FAMILY = "Bi-Pt"
GAP = "data/Target_Materials/HEUSLER_KAPPA_TIER3_GAP_ROWS.csv"
TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
VAL = "data/exports/kappa_v2/bte_generator_validation.csv"
FAMCAL = "data/exports/kappa_v2/family_calibration.json"
SHARED = "paper/evidence/shared_constant_test.json"
OUT = "data/Target_Materials/BIPT_THEORY_PREDICTIONS.csv"
NOT_BTE = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"
SEEDS = [0, 1, 2, 3, 4]
TEMPS = (300, 600, 900)
COORDS = "[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]"


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def xsite(c):
    return str(sorted(Composition(c).elements, key=lambda e: (e.X if e.X else 99.0))[0])


def main() -> int:
    fc = json.load(open(FAMCAL))
    # the flat twin ratio is MEASURED by test_chemical_twins.py (median over the published twin pairs),
    # not typed: it was the literal 0.972
    TWIN_RATIO = float(pd.read_csv("data/exports/kappa_v2/chemical_twins.csv").ratio.median())
    assert FAMILY not in fc["families"], "Bi-Pt now has a calibration -- use the family route instead"
    c_g, p_g = fc["global"]["c"], fc["global"]["p"]
    sh = json.load(open(SHARED))["lofo"]["C0"]

    bb = ds.build(tier_max=1, tier_min=1, group_by="element_system", extra_csv=GAP, verbose=False)
    X, y, w, meta = bb["X"], bb["y"], bb["weights"], bb["meta"].copy()
    meta["red"] = meta.formula.map(red)
    leak = [c for c in TARGETS if c in set(meta.red)]
    assert not leak, f"in the training pool, so not a prediction: {leak}"
    a_train = meta.groupby("red").struct_a_A.median().dropna()

    # ---- lattice constant: lanthanide contraction across the family ---------------------------
    mates = {m: float(a_train[m]) for m in a_train.index
             if m and yzfam(m) == FAMILY and Element(xsite(m)).is_lanthanoid}
    zz = np.array([Element(xsite(m)).Z for m in mates])
    aa = np.array(list(mates.values()))
    b1, b0 = np.polyfit(zz, aa, 1)
    loo = []
    for i in range(len(zz)):
        k = np.ones(len(zz), bool)
        k[i] = False
        q1, q0 = np.polyfit(zz[k], aa[k], 1)
        loo.append(100 * abs(q1 * zz[i] + q0 - aa[i]) / aa[i])
    print(f"lattice: linear fit over {len(mates)} lanthanide {FAMILY} members "
          f"{dict(sorted(((m, round(v, 3)) for m, v in mates.items()), key=lambda t: Element(xsite(t[0])).Z))}")
    print(f"  slope {b1:.4f} A per Z; leave-one-out error median {np.median(loo):.2f}%, max {max(loo):.2f}%")

    # ---- check 1: the model on the siblings, chemistry hidden -----------------------------------
    V = pd.read_csv(VAL)
    sib = V[V.fam == FAMILY]
    sib300 = sib[sib["T"] == 300].groupby("compound").agg(pub=("k_published", "median"), gen=("k_generated", "median"))
    sib300["err_pct"] = 100 * (sib300.gen - sib300.pub).abs() / sib300.pub
    print(f"\ncheck 1 -- model vs published, {FAMILY} siblings with their chemistry hidden (300 K):")
    print(sib300.round(2).to_string())
    print(f"  median error {sib300.err_pct.median():.1f}%, worst {sib300.err_pct.max():.1f}%")

    # ---- check 2: the family's published calculations across the series ------------------------
    tr = pd.read_csv(TRAIN, low_memory=False)
    tr["red"] = tr.formula.map(red)
    t = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    full = (t == 1) & ~tr.method.astype(str).str.lower().str.contains(NOT_BTE, regex=True)
    pub = tr[full & tr.red.map(lambda s: bool(s) and yzfam(s) == FAMILY) & tr["T"].between(280, 320) & (tr.k > 0)]
    series = pub.groupby("red").k.median()
    series = series[[Element(xsite(m)).is_lanthanoid for m in series.index]]
    series = series.sort_index(key=lambda s: s.map(lambda m: Element(xsite(m)).Z))
    print(f"\ncheck 2 -- published {FAMILY} calculations across the lanthanide series (300 K): "
          f"{series.round(2).to_dict()}")

    # ---- predict ------------------------------------------------------------------------------
    models = []
    for s in SEEDS:
        m = make_model("catboost", {"random_seed": s})
        m.fit(X, y, sample_weight=w)
        models.append(m)
    rows = []
    for c in TARGETS:
        z = Element(xsite(c)).Z
        a = float(b1 * z + b0)
        els = sorted(Composition(c).elements, key=lambda e: (e.X if e.X else 99.0))
        feat = ds._per_formula_features(c, a, None, str([str(e) for e in els]), COORDS)
        assert feat is not None, f"{c}: features could not be built"
        rec = dict(compound=c, family=FAMILY, a_conv_A=round(a, 4),
                   a_source=f"lanthanide-contraction fit over {len(mates)} {FAMILY} members (LOO max {max(loo):.2f}%)")
        for T in TEMPS:
            q = pd.Series({k: feat.get(k, np.nan) for k in X.columns})
            q["temperature_K"], q["inv_T"], q["log_T"] = float(T), 1000.0 / T, np.log10(T)
            ps = [float(10 ** m.predict(pd.DataFrame([q])[X.columns])[0]) for m in models]
            rec[f"kappa_theory_{T}"] = round(float(np.median(ps)), 2)
            rec[f"seed_spread_{T}"] = round(max(ps) / min(ps), 3)
            # the shared-constant value of the ML number is kept for the record but is NOT the one to
            # quote: the model is unreliable for Pr/Nd (below). Quote kappa_exp_twin_* instead.
            rec[f"kappa_exp_of_ML_value_DO_NOT_QUOTE_{T}"] = round(float(apply_family(
                "GLOBAL", (c_g, p_g), [float(T)], [float(np.median(ps))])[0]), 2)
        lo = series[[Element(xsite(m)).Z < z for m in series.index]]
        hi = series[[Element(xsite(m)).Z > z for m in series.index]]
        rec.update(series_neighbour_below=(f"{lo.index[-1]} {lo.iloc[-1]:.2f}" if len(lo) else None),
                   series_neighbour_above=(f"{hi.index[0]} {hi.iloc[0]:.2f}" if len(hi) else None),
                   model_err_on_siblings_pct=round(float(sib300.err_pct.median()), 1),
                   shared_estimate_label=(f"SHARED-CONSTANT ESTIMATE, family never measured: shared constant on "
                                          f"unseen families {sh['median_ape']}% median, {sh['within_2x_pct']}% within 2x"))
        # RELIABILITY (2026-10-03). Pr and Nd each sit in only 3 training compounds. With one of their
        # compounds hidden the model misses it by 22-63% (PrBiPd +63%, NdBiPd -49%), against 6-15% for
        # Gd/Tb/Lu; along the series it zigzags La 2.4 -> Pr 4.4 -> Nd 1.9 -> Gd 5.9. The model value is
        # therefore NOT the estimate to quote. The chemical-twin route (published RBiPd x the Pd->Pt ratio)
        # is model-independent: flat ratio 0.972 (chemical-twin method, LOO 9.1% on 23 pairs) and this
        # family's own ratio, interpolated in Z between LaBiPt/LaBiPd and GdBiPt/GdBiPd.
        x = xsite(c)
        pdc = f"{x}BiPd"
        kpd = tr[full & (tr.red == pdc) & tr["T"].between(280, 320) & (tr.k > 0)].k.median()
        rat = {r: series.get(f"{r}BiPt") / tr[full & (tr.red == f"{r}BiPd") & tr["T"].between(280, 320)].k.median()
               for r in ("La", "Gd")}
        zr = {"La": Element("La").Z, "Gd": Element("Gd").Z}
        ratio_z = rat["La"] + (rat["Gd"] - rat["La"]) * (z - zr["La"]) / (zr["Gd"] - zr["La"])
        rec.update(element_training_compounds=int(sum(1 for m in set(meta.red) if m and x in
                                                      [str(e) for e in Composition(m).elements])),
                   model_reliability="UNRELIABLE: X element in 3 training compounds; one-out test 22-63% error",
                   twin_source=f"{pdc} published {kpd:.2f}",
                   twin_flat_ratio=round(TWIN_RATIO, 4),
                   kappa_theory_twin_flat_300=round(float(kpd * TWIN_RATIO), 2),
                   kappa_theory_twin_family_ratio_300=round(float(kpd * ratio_z), 2),
                   family_pt_over_pd_ratio=round(float(ratio_z), 3),
                   # experimental-scale estimate of the TWIN values: shared constant, family never measured
                   kappa_exp_twin_flat_300=round(float(apply_family("GLOBAL", (c_g, p_g), [300.0],
                                                                    [float(kpd * TWIN_RATIO)])[0]), 2),
                   kappa_exp_twin_family_ratio_300=round(float(apply_family("GLOBAL", (c_g, p_g), [300.0],
                                                                            [float(kpd * ratio_z)])[0]), 2))
        rows.append(rec)
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print("\n" + R.T.to_string())
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
