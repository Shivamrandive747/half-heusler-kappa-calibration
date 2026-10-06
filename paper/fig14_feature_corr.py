"""Supplementary figure -- Pearson correlation between the model's descriptors on the production pool.

Descriptors that do not vary across the pool are left out (their correlation is undefined) and named in
the printout. Pairwise-complete correlation, so the 81 rows without bond descriptors do not drop the
others. Order: the descriptor groups of the model (disorder, site, mean/contrast, geometry, bonds,
temperature).
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import figlib as F


def main() -> int:
    import matplotlib.pyplot as plt
    from compare_models_production import production_build
    from fig11_shap import LABEL

    X = production_build()["X"].astype(float)
    const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    Xv = X.drop(columns=const)
    C = Xv.corr(method="pearson", min_periods=50).to_numpy()
    n = C.shape[0]
    F.use_style()
    fig, ax = plt.subplots(figsize=(F.DOUBLE * 0.78, F.DOUBLE * 0.70))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    labels = [LABEL.get(c, c) for c in Xv.columns]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.2)
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Pearson correlation", fontsize=7)
    F.save(fig, "fig14_feature_corr")
    hi = [(Xv.columns[i], Xv.columns[j], C[i, j]) for i in range(n) for j in range(i + 1, n) if abs(C[i, j]) > 0.95]
    print(f"  {n} varying descriptors (constant, left out: {const}); pairs with |r| > 0.95: {len(hi)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
