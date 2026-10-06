"""Figure -- the data the regression model learns from (the production training pool).

  (a) kappa_L of the training labels on a linear axis: strongly skewed.
  (b) log10 kappa_L: close to symmetric, which is why the model is fitted on the logarithm.
  (c) temperature coverage of the labels.
  (d) training compounds by valence-electron count, half Heuslers (the target class) and full Heuslers
      (which enter only through reduced-weight semi-empirical estimates).
Read from compare_models_production.production_build (the exact pool the model and the blind test use).
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import figlib as F


def main() -> int:
    import matplotlib.pyplot as plt
    from compare_models_production import production_build
    from make_paper_predictions import vec
    from pymatgen.core import Composition
    import private_manifest as PM

    # a pure helper whose home module is not redistributed: PM.helper returns the canonical
    # definition where it exists and this identical restatement everywhere else
    def _is_hh(c):
        co = Composition(c)
        return len(co.elements) == 3 and len(set(co.values())) == 1
    is_hh = PM.helper("is_hh", _is_hh)

    b = production_build()
    m = b["meta"].copy()
    m["k"] = 10 ** np.asarray(b["y"], dtype=float)
    m["red"] = m.formula.map(lambda f: Composition(str(f)).reduced_formula)
    m["hh"] = m.red.map(lambda c: bool(is_hh(c)))
    m["T"] = pd.to_numeric(m.temperature_K, errors="coerce")

    F.use_style()
    fig, axs = plt.subplots(1, 4, figsize=(F.DOUBLE, F.DOUBLE * 0.27))
    a, bx, cx, dx = axs
    a.hist(m.k, bins=40, color=F.OURS, alpha=0.9)
    a.set_xlabel("$\\kappa_L$ (W m$^{-1}$ K$^{-1}$)")
    a.set_ylabel("labels")
    bx.hist(np.log10(m.k), bins=30, color=F.OURS, alpha=0.9)
    bx.set_xlabel("log$_{10}\\,\\kappa_L$")
    cx.hist(m["T"], bins=np.arange(150, 1350, 100), color="#6E8FA8", alpha=0.95)
    cx.set_xlabel("temperature (K)")
    comp = m.drop_duplicates("red")
    comp = comp.assign(vec=comp.red.map(vec))
    rng = range(int(comp.vec.min()), int(comp.vec.max()) + 1)
    hh = [int(((comp.vec == v) & comp.hh).sum()) for v in rng]
    fh = [int(((comp.vec == v) & ~comp.hh).sum()) for v in rng]
    dx.bar(list(rng), hh, color=F.OURS, label="half Heusler")
    dx.bar(list(rng), fh, bottom=hh, color=F.REFUSED, label="full Heusler")
    dx.set_xlabel("valence electrons per formula")
    dx.set_ylabel("compounds")
    dx.legend(fontsize=6, loc="upper right", frameon=False)
    for ax_, l in zip(axs, "abcd"):
        F.panel_label(ax_, l, loc="upper right" if l in "ab" else "upper left")
    F.save(fig, "fig12_dataset")
    import json as _json
    tb = m["T"].round(-2).value_counts().sort_index()
    Path("data/exports/kappa_v2/training_data_summary.json").write_text(_json.dumps(dict(
        _generated_by="paper/fig12_dataset.py", n_labels=int(len(m)), n_compounds=int(comp.red.nunique()),
        n_half_heusler=int(comp.hh.sum()), n_half_heusler_vec18=int(((comp.vec == 18) & comp.hh).sum()),
        T_min=float(m["T"].min()), T_max=float(m["T"].max()),
        labels_by_T={str(int(k)): int(v) for k, v in tb.items()},
        kappa_min=round(float(m.k.min()), 2), kappa_max=round(float(m.k.max()), 1),
        kappa_median=round(float(m.k.median()), 2)), indent=2))
    print(f"  {len(m)} labels, {comp.red.nunique()} compounds ({int(comp.hh.sum())} half Heuslers); "
          f"T {m['T'].min():.0f}-{m['T'].max():.0f} K; VEC 18 half Heuslers {int(((comp.vec == 18) & comp.hh).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
