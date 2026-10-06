"""Calibrating the published calculation directly, against calibrating the surrogate: which route
is better, scored on the same compounds.

WHY THIS SCRIPT EXISTS. Section 3 compares the two routes the paper could deploy -- multiply the
PUBLISHED transport calculation by the transfer function, or multiply the regression model's
estimate of it -- and quotes median errors and biases for both, corrected and uncorrected. Those
numbers rested on surrogate_vs_published.json, whose recorded producer (scratchpad/verify_surrogate.py)
no longer exists anywhere, and whose calibrated figures do not reproduce under the paper's current
constants. The uncorrected half of the comparison did reproduce, which is how the artefact was
dated rather than dismissed. This is the producer, and it writes the artefact it reads from.

POPULATION. Exactly Figure 2's: in-domain half Heuslers carrying both a laboratory measurement and
a published transport calculation, measured and calculated values matched within 50 K, aggregated
to one point per compound per 100 K bin so ZrNiSn's 646 rows cannot outvote a compound with one.
The surrogate's estimate at each of those points is taken from the blind test's per-row predictions
at the reference seed, matched on compound and temperature bin.

SCORING. Per-compound medians, never per-row -- the same reason as everywhere else in this paper.
Bias is the median over compounds of the per-compound median ratio.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import extend_blind_test as E
import fig02_offset as F2
from make_paper_predictions import structure_status
from pymatgen.core import Composition, Element

BLIND = "data/exports/kappa_v2/target_blind_test.csv"
OUT = "data/exports/kappa_v2/surrogate_vs_published.json"


def vec(f):
    return int(sum((Element(str(e)).group if Element(str(e)).group <= 12
                    else Element(str(e)).group - 10) for e in Composition(f).elements))


def in_domain(c):
    st = structure_status(str(c))
    return vec(c) == 18 and st[0] and not st[1]


def per_compound(d: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """(median APE %, median bias, within-2x %) over compounds, each compound a median of its bins."""
    g = d.assign(_r=d[col] / d.k_exp, _a=100 * (d[col] / d.k_exp - 1).abs()).groupby("compound")
    ape, bias = g._a.median(), g._r.median()
    return float(ape.median()), float(bias.median()), float(100 * bias.between(0.5, 2.0).mean())


def main() -> int:
    raw = F2.pairs()
    raw = raw[raw.compound.map(in_domain)]
    raw["Tbin"] = (raw["T"] / 100).round() * 100
    d = (raw.groupby(["compound", "Tbin"], as_index=False)
            .agg(k_exp=("k_exp", "median"), k_pub=("k_dft", "median")))

    B = pd.read_csv(BLIND)
    B = B[B.klass == "half"].copy()
    B["Tbin"] = (pd.to_numeric(B["T"], errors="coerce") / 100).round() * 100
    S = (B.groupby(["compound", "Tbin"], as_index=False).k_pred.median()
          .rename(columns={"k_pred": "k_sur"}))
    d = d.merge(S, on=["compound", "Tbin"], how="inner")
    print(f"population: {d.compound.nunique()} compounds, {len(d)} compound-temperature points "
          f"(surrogate available at every one)")

    Bh = B.copy()
    Bh["ind"] = Bh.compound.map(in_domain)
    c, p = E.fit_cp(Bh[Bh.ind])
    mult = np.minimum(c * (d.Tbin / 300.0) ** p, 1.0)
    d["k_pub_cal"] = d.k_pub * mult
    d["k_sur_cal"] = d.k_sur * mult
    print(f"global constants c={c:.3f} p={p:.3f}\n")

    out = dict(n_compounds=int(d.compound.nunique()), n_points=int(len(d)),
               c=round(float(c), 3), p=round(float(p), 3),
               _generated_by="compute_surrogate_vs_published.py")
    print(f"{'route':<36}{'median APE':>12}{'bias':>8}{'within 2x':>11}")
    print("=" * 67)
    for lab, col, key in (("uncorrected published calculation", "k_pub", "published_raw"),
                          ("uncorrected surrogate", "k_sur", "surrogate_raw"),
                          ("calibrated published calculation", "k_pub_cal", "published_cal"),
                          ("calibrated surrogate", "k_sur_cal", "surrogate_cal")):
        a, b, w = per_compound(d, col)
        out[key] = dict(median_ape=round(a, 1), bias=round(b, 2), within2x_pct=round(w, 1))
        print(f"  {lab:<34}{a:>11.1f}%{b:>8.2f}{w:>10.0f}%")

    diff = (d.assign(_x=100 * (d.k_sur / d.k_pub - 1).abs()).groupby("compound")._x.median())
    out["surrogate_vs_published_median_pct"] = round(float(diff.median()), 1)
    print(f"\n  surrogate differs from the published calculation by a median of "
          f"{diff.median():.1f}% per compound")

    Path(OUT).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
