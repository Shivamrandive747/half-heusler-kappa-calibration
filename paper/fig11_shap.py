"""Figure -- what the regression model has learned: SHAP attribution of log10 kappa_BTE.

The production model (CatBoost, seed 0) is fitted on the production pool (compare_models_production.
production_build: tier-1 labels + the reduced-weight semi-empirical rows for full Heuslers, production
weights) and explained with exact tree SHAP values (CatBoost's own ShapValues). The target is
log10 kappa_BTE, so a SHAP value of +0.1 multiplies the predicted calculation by 10^0.1 = 1.26.
  (a) beeswarm: each dot is one training row; horizontal position = that descriptor's contribution;
      colour = the descriptor's value within its range (low blue, high red).
  (b) mean |SHAP| for the same descriptors.
Nothing is typed: descriptors are ranked by the computed attribution; the counts are printed.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import figlib as F

TOP = 12
LABEL = {
    "gamma_mass": "mass disorder $\\Gamma_M$", "gamma_radius": "size disorder $\\Gamma_r$",
    "M_X": "mass, X site", "M_Y": "mass, Y site", "M_Z": "mass, Z site",
    "r_X": "radius, X site", "r_Y": "radius, Y site", "r_Z": "radius, Z site",
    "X_X": "electronegativity, X", "X_Y": "electronegativity, Y", "X_Z": "electronegativity, Z",
    "M_mean": "mean atomic mass", "r_mean": "mean atomic radius", "X_mean": "mean electronegativity",
    "dM_mean_X": "mass contrast, X", "dM_mean_Y": "mass contrast, Y", "dM_mean_Z": "mass contrast, Z",
    "dr_mean_X": "size contrast, X", "dr_mean_Y": "size contrast, Y", "dr_mean_Z": "size contrast, Z",
    "M_sum": "formula mass", "r_sum": "sum of radii", "M_spread": "mass spread", "r_spread": "radius spread",
    "M_ratio_XZ": "mass ratio X/Z", "r_ratio_XZ": "radius ratio X/Z", "dX_XZ": "electroneg. difference X$-$Z",
    "X_spread": "electronegativity spread", "volume_per_atom": "volume per atom", "density_g_cm3": "density",
    "bond_scale": "bond-length scale", "n_atoms_conv": "atoms per cell", "is_full_heusler": "full Heusler (flag)",
    "is_quaternary": "quaternary (flag)", "bond_min": "shortest bond", "bond_mean": "mean bond length",
    "bond_std": "bond-length spread", "bond_ratio": "bond-length ratio", "slack_geom": "Slack geometric term",
    "temperature_K": "temperature", "inv_T": "1/T", "log_T": "log T",
}


def main() -> int:
    import matplotlib.pyplot as plt
    from catboost import Pool
    from compare_models_production import production_build
    from pipeline.s20_kappa_train import make_model

    b = production_build()
    X, y, w = b["X"], b["y"], b["weights"]
    mdl = make_model("catboost", {"random_seed": 0})
    mdl.fit(X, y, sample_weight=w)
    sv = mdl.get_feature_importance(Pool(X, y, weight=w), type="ShapValues")[:, :-1]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:TOP]
    cols = [X.columns[i] for i in order]

    F.use_style()
    fig = plt.figure(figsize=(F.DOUBLE, F.DOUBLE * 0.46))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.12)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    for row, j in enumerate(order):
        v = X.iloc[:, j].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        col = np.clip((v - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
        jit = rng.normal(0, 0.10, len(v))
        rgba = cmap(np.nan_to_num(col))
        rgba[np.isnan(v)] = (0.6, 0.6, 0.6, 1.0)          # descriptor missing for that row: grey
        axa.scatter(sv[:, j], np.full(len(v), row) + jit, c=rgba, s=4, lw=0, alpha=0.8, zorder=3)
    axa.axvline(0, color="#888888", lw=0.7, zorder=1)
    axa.set_yticks(range(TOP))
    axa.set_yticklabels([LABEL.get(c, c) for c in cols], fontsize=6.8)
    axa.set_ylim(TOP - 0.5, -0.5)
    axa.set_xlabel("SHAP value (contribution to log$_{10}\\,\\kappa_L^{\\mathrm{BTE}}$)")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=axa, fraction=0.035, pad=0.015, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"], fontsize=6.5)
    cb.set_label("descriptor value", fontsize=6.8)
    F.panel_label(axa, "a", loc="lower right")

    axb.barh(range(TOP), imp[order], color=F.OURS, height=0.62, zorder=3)
    for row, j in enumerate(order):
        axb.text(imp[j] * 1.02, row, f"{imp[j]:.3f}", va="center", fontsize=6.0, color="#333333")
    axb.set_yticks(range(TOP))
    axb.set_yticklabels([])
    axb.set_ylim(TOP - 0.5, -0.5)
    axb.set_xlim(0, imp[order].max() * 1.28)
    axb.set_xlabel("mean |SHAP value|")
    F.panel_label(axb, "b", loc="lower right")

    F.save(fig, "fig11_shap")
    import json as _json
    rank = []
    for c, j in zip(cols, order):
        v = X.iloc[:, j].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        rank.append(dict(descriptor=c, label=LABEL.get(c, c), mean_abs_shap=round(float(imp[j]), 6),  # 6 dp: printing a 4-dp value at 3 dp double-rounds (0.0335 -> 0.034)
                         corr_value_shap=round(float(np.corrcoef(v[ok], sv[ok, j])[0, 1]), 2),
                         n_missing=int((~ok).sum())))
    Path("data/exports/kappa_v2/shap_ranking.json").write_text(_json.dumps(dict(
        _generated_by="paper/fig11_shap.py", model="CatBoost seed 0, production pool",
        target="log10 kappa_BTE", n_rows=int(len(y)), n_descriptors=int(X.shape[1]), top=rank), indent=2))
    print(f"  SHAP on {len(y)} rows x {X.shape[1]} descriptors; top {TOP}:")
    for c, j in zip(cols, order):
        v = X.iloc[:, j].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        corr = np.corrcoef(v[ok], sv[ok, j])[0, 1]
        print(f"    {c:<18} mean|SHAP| {imp[j]:.4f}   corr(value, SHAP) {corr:+.2f}   missing {int((~ok).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
