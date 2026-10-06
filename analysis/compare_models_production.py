"""Compare the four regressors on the descriptor set the paper actually uses.

WHY THIS EXISTS. `s20_kappa_train compare` restricts to the 17 descriptors chosen by the feature
selection run, but the production model -- the one that generates every k_pred in the blind test,
and the one Section 3 describes -- is fitted on all 45. So the stored comparison described a model
the paper does not use, and quoting it beside a 45-feature SHAP decomposition would have put two
different models in one figure.

It is worth recording why the 17-descriptor selection was never adopted, because the selection
history looks like it should have been (R2 0.910 / 17.7% at 17 features against 0.886 / 20.5% at
40). The curve is not monotonic and not smooth: it reads 17.4% at 8 features, 22.2% at 7, 17.7% at
11 and 24.0% at 5. Differences of that size appearing and disappearing as one descriptor is added
or removed are selection noise, not signal, and picking the argmin of such a curve is a way of
overfitting the selection itself. Keeping all 45 is the conservative choice.

Writes model_comparison_production.json alongside the existing file rather than overwriting it,
so both experiments remain on record.

PRODUCTION BUILD (FIXPASS, 2026-10-05). The comparison used ds.build(tier_max=1, tier_min=1) with the
default formula grouping and WITHOUT the semi-empirical gap rows, while every production fit uses
group_by="element_system" and extra_csv=HEUSLER_KAPPA_TIER3_GAP_ROWS.csv. It now builds exactly that
pool (production_build), with the same weights and the same groups for the folds. METRICS: r2_log
(and mae/rmse_log) stay per ROW; median_ape (and mean_ape, within_*) are per COMPOUND -- each
compound's median row error, then the median over compounds; the per-row median is kept as
median_ape_per_row. Each model's "units" field says which.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from pymatgen.core import Composition

from pipeline.s19_kappa_dataset import build
from pipeline.s20_kappa_train import grouped_oof, metrics

import pandas as pd  # noqa: E402

OUT = Path("data/exports/kappa_v2/model_comparison_production.json")
GAP = "data/Target_Materials/HEUSLER_KAPPA_TIER3_GAP_ROWS.csv"


def production_build() -> dict:
    """The pool, weights and CV groups every production fit uses (run_target_blind_test.py):
    tier-1 labels plus the semi-empirical gap rows (their own weight), grouped by element system.
    paper/fig07_model.py imports this so its SHAP panel describes the same model."""
    return build(tier_max=1, tier_min=1, group_by="element_system", extra_csv=GAP, verbose=False)


def compound_of(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return str(f)


def main() -> int:
    b = production_build()
    X, y, g, w = b["X"], b["y"], b["groups"], b["weights"]
    comp = b["meta"].formula.map(compound_of).values
    print(f"MODEL COMPARISON on the production pool: {X.shape[1]} descriptors, {len(y)} rows, "
          f"{len(set(comp))} compounds, {len(set(g))} element-system CV groups\n")
    out = {}
    # the out-of-fold predictions themselves, so the per-regressor parity panels (supplement) are
    # drawn from exactly the predictions these metrics score
    preds = pd.DataFrame({"compound": comp, "temperature_K": X["temperature_K"].values,
                          "log10_kappa_true": y})
    for name in ("catboost", "lightgbm", "krr", "gpr"):
        try:
            oof = grouped_oof(name, X, y, g, w)
            preds[f"log10_kappa_{name}"] = oof
            m = metrics(y, oof, groups=comp)          # per-compound errors; R2 stays per row
            m["median_ape_per_row"] = metrics(y, oof)["median_ape"]
            m["units"] = dict(r2_log="row", mae_log="row", rmse_log="row",
                              median_ape="compound (median over compounds of each compound's median row APE)",
                              mean_ape="compound", within_20pct="compound", within_30pct="compound",
                              median_ape_per_row="row")
            m["build"] = ("ds.build(tier_max=1, tier_min=1, group_by='element_system', extra_csv=gap rows); "
                          "production weights; GroupKFold(5) on element systems")
            out[name] = m
            print(f"  {name:<12}R2 {m['r2_log']:.4f}   median err/compound {m['median_ape']:5.2f}%   "
                  f"(per row {m['median_ape_per_row']:5.2f}%)   within20% {m['within_20pct']:5.1f}%   "
                  f"n={m['n']} compounds, {m['n_rows']} rows")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<12}FAILED {str(exc)[:70]}")
    assert len({(v["n"], v["n_rows"]) for v in out.values()}) == 1, "models scored on different rows"
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    preds.to_csv(OUT.with_name("model_comparison_oof.csv"), index=False)
    print(f"wrote {OUT.with_name('model_comparison_oof.csv')}")

    best_r2 = max(out, key=lambda k: out[k]["r2_log"])
    best_ape = min(out, key=lambda k: out[k]["median_ape"])
    print(f"\n  best R2        : {best_r2}")
    print(f"  best median err: {best_ape}")
    if best_r2 != best_ape:
        print("  -> the two metrics disagree; the figure must show both rather than declare "
              "a winner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
