"""Does the chemical-twin trick that fixed hafnium generalise to the other lanthanide-contraction pairs?

THE OBSERVATION IT COMES FROM. Hafnium is the worst-predicted element in the corpus (33.8% median,
and every one of its 17 compounds under-predicted) because it holds 2.7 training rows per compound
against zirconium's 23. With little to learn from, the model falls back on a general "heavier conducts
less" trend -- and hafnium is exactly where that trend breaks. Hf and Zr have nearly identical radii,
so their bonding is the same and only the mass differs; the published Hf/Zr kappa ratio is 0.92, not
the 1/sqrt(2) = 0.71 that naive mass scaling implies. Predicting Hf from its Zr twin scored 12%
against the model's 32%, on five pairs.

THE QUESTION HERE. Five pairs is too thin to publish and too suggestive to ignore. The lanthanide
contraction produces the same size coincidence for every 4d/5d pair in the transition block --
Y/La, Zr/Hf, Nb/Ta, Mo/W, Ru/Os, Rh/Ir, Pd/Pt, Ag/Au -- so if the mechanism is real the trick should
work for all of them, and if hafnium was a fluke it will not.

WHAT IS CHECKED, AND WHY EACH CHECK IS THERE:

  the size premise is VERIFIED, not assumed -- the radii are printed per pair, and a pair whose
      radii differ materially is reported as such rather than quietly included;
  twins are built through pymatgen Composition and re-reduced, never by string replacement, because
      "ZrNiSn".replace("Zr","Hf") happens to work while many substitutions do not;
  both members must be cubic half Heuslers with a published 300 K calculation, so the comparison is
      between like and like;
  the ratio is fitted LEAVE-ONE-OUT -- the pair being predicted never contributes to the ratio used
      on it -- otherwise the test would be circular;
  two variants are scored: a ratio fitted per pair, and one global ratio shared by all pairs. The
      global one is the stronger claim and the one that would generalise to a pair we have no data
      for, so both are reported;
  the model's own held-out error on the very same compounds is the comparison, so the two methods
      are judged on identical ground.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from make_paper_predictions import vec, structure_status, yzfam

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
GENVAL = "data/exports/kappa_v2/bte_generator_validation.csv"
NOT_BTE = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"
PAIRS = [("Y", "La"), ("Zr", "Hf"), ("Nb", "Ta"), ("Mo", "W"),
         ("Ru", "Os"), ("Rh", "Ir"), ("Pd", "Pt"), ("Ag", "Au")]


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def swap(formula: str, a: str, b: str):
    """Replace element a with b through a Composition, never by string surgery."""
    try:
        d = Composition(formula).get_el_amt_dict()
        if a not in d or b in d:
            return None
        d[b] = d.pop(a)
        return Composition(d).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    tr = pd.read_csv(TRAIN, low_memory=False)
    tr["red"] = tr.formula.map(red)
    t = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["TK"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    meth = tr.method.astype(str).str.lower()
    b = tr[(t == 1) & (tr.k > 0) & ~meth.str.contains(NOT_BTE, regex=True)
           & tr.TK.between(280, 320)].dropna(subset=["red"])
    K = b.groupby("red").k.median()
    nsrc = b.groupby("red").source_doi.nunique()
    G = pd.read_csv(GENVAL).groupby("compound").agg(ape=("ape", "median"), ratio=("ratio", "median"))

    def ok(c):
        try:
            s = structure_status(c)
            return vec(c) == 18 and s[0] and not s[1]
        except Exception:  # noqa: BLE001
            return False

    print("STEP 1 -- IS THE SIZE PREMISE TRUE? (the trick only makes sense if the twins are the same size)")
    print(f"  {'pair':<10}{'radius light':>14}{'radius heavy':>14}{'size diff':>11}"
          f"{'mass light':>12}{'mass heavy':>12}{'mass ratio':>12}")
    good_pairs = []
    for a, c in PAIRS:
        ea, ec = Element(a), Element(c)
        ra, rc = (ea.atomic_radius or 0) * 100, (ec.atomic_radius or 0) * 100
        dsz = 100 * abs(rc - ra) / ra if ra else 99
        print(f"  {a + '/' + c:<10}{ra:>13.0f}p{rc:>13.0f}p{dsz:>10.1f}%"
              f"{ea.atomic_mass:>12.1f}{ec.atomic_mass:>12.1f}{ec.atomic_mass / ea.atomic_mass:>12.2f}")
        if dsz <= 6.0:
            good_pairs.append((a, c))
        else:
            print(f"           ^ radii differ by {dsz:.0f}% -- not a true size twin, excluded")
    print()

    print("STEP 2 -- FIND THE TWIN COMPOUNDS (both cubic, VEC 18, both with a published 300 K calculation)")
    rows = []
    for a, c in good_pairs:
        for light in sorted(K.index):
            if a not in {str(e) for e in Composition(light).elements}:
                continue
            heavy = swap(light, a, c)
            if not heavy or heavy not in K.index:
                continue
            if not (ok(light) and ok(heavy)):
                continue
            rows.append(dict(pair=f"{a}/{c}", light=light, heavy=heavy,
                             k_light=float(K[light]), k_heavy=float(K[heavy]),
                             ratio=float(K[heavy] / K[light]),
                             src_light=int(nsrc.get(light, 0)), src_heavy=int(nsrc.get(heavy, 0)),
                             model_ape=float(G.ape[heavy]) if heavy in G.index else np.nan))
    T = pd.DataFrame(rows)
    if T.empty:
        print("  no twin pairs found at all -- the experiment cannot run")
        return 1
    print(f"  twin pairs found: {len(T)}   across {T.pair.nunique()} element pairs")
    print()
    print(f"  {'pair':<8}{'light':<10}{'heavy':<10}{'k light':>9}{'k heavy':>9}{'ratio':>8}"
          f"{'srcs':>7}{'model err':>11}")
    for _, r in T.sort_values(["pair", "ratio"]).iterrows():
        print(f"  {r.pair:<8}{r.light:<10}{r.heavy:<10}{r.k_light:>9.2f}{r.k_heavy:>9.2f}"
              f"{r.ratio:>8.3f}{f'{r.src_light}/{r.src_heavy}':>7}"
              f"{(f'{r.model_ape:.0f}%' if np.isfinite(r.model_ape) else '-'):>11}")
    print()

    print("STEP 3 -- IS THE RATIO CONSISTENT? (per pair, and overall)")
    print(f"  {'pair':<8}{'n':>3}{'median ratio':>14}{'spread':>9}{'range':>16}")
    for p, g in T.groupby("pair"):
        print(f"  {p:<8}{len(g):>3}{g.ratio.median():>14.3f}{g.ratio.max() / g.ratio.min():>8.2f}x"
              f"{f'{g.ratio.min():.2f}-{g.ratio.max():.2f}':>16}")
    print(f"  {'ALL':<8}{len(T):>3}{T.ratio.median():>14.3f}{T.ratio.max() / T.ratio.min():>8.2f}x"
          f"{f'{T.ratio.min():.2f}-{T.ratio.max():.2f}':>16}")
    naive = {f"{a}/{c}": np.sqrt(Element(a).atomic_mass / Element(c).atomic_mass)
             for a, c in good_pairs}
    print()
    print("  what naive mass scaling kappa ~ 1/sqrt(M) would predict, per pair:")
    print("    " + "   ".join(f"{k} {v:.2f}" for k, v in naive.items()))
    print(f"  observed overall: {T.ratio.median():.3f}")
    print()

    print("STEP 4 -- LEAVE-ONE-OUT TEST. Predict each heavy compound from its light twin.")
    print("  The ratio used on a pair is fitted WITHOUT that pair.")
    print(f"  {'heavy':<10}{'from':<10}{'true':>8}{'per-pair':>10}{'err':>7}"
          f"{'global':>9}{'err':>7}{'model err':>11}")
    per, glo, mod = [], [], []
    for i, r in T.iterrows():
        others_all = T.drop(index=i)
        others_pair = others_all[others_all.pair == r.pair]
        rp = others_pair.ratio.median() if len(others_pair) else np.nan
        rg = others_all.ratio.median()
        pp = r.k_light * rp if np.isfinite(rp) else np.nan
        pg = r.k_light * rg
        ep = 100 * abs(pp - r.k_heavy) / r.k_heavy if np.isfinite(pp) else np.nan
        eg = 100 * abs(pg - r.k_heavy) / r.k_heavy
        per.append(ep); glo.append(eg); mod.append(r.model_ape)
        print(f"  {r.heavy:<10}{r.light:<10}{r.k_heavy:>8.2f}"
              f"{(f'{pp:.2f}' if np.isfinite(pp) else '-'):>10}"
              f"{(f'{ep:.0f}%' if np.isfinite(ep) else '-'):>7}{pg:>9.2f}{eg:>6.0f}%"
              f"{(f'{r.model_ape:.0f}%' if np.isfinite(r.model_ape) else '-'):>11}")
    per, glo, mod = np.array(per, float), np.array(glo, float), np.array(mod, float)
    print()
    print("=" * 74)
    print("RESULT")
    print("=" * 74)
    m = np.isfinite(mod)
    print(f"  compounds with a model score to compare against : {int(m.sum())} of {len(T)}")
    print(f"  twin method, ONE global ratio  : median {np.nanmedian(glo[m]):>5.1f}%")
    print(f"  twin method, ratio per pair    : median {np.nanmedian(per[m]):>5.1f}%")
    print(f"  the ML model on the same ones  : median {np.nanmedian(mod[m]):>5.1f}%")
    better = int(np.sum(glo[m] < mod[m]))
    print(f"  twin (global) beats the model on {better} of {int(m.sum())}")
    from scipy import stats
    if m.sum() >= 6:
        w = stats.wilcoxon(glo[m], mod[m])
        print(f"  paired Wilcoxon: p = {w.pvalue:.4f}")
    print()
    print("  per pair, twin (global) vs model:")
    T2 = T.assign(twin=glo, model=mod)
    for p, g in T2.groupby("pair"):
        gg = g[np.isfinite(g.model)]
        if not len(gg):
            continue
        print(f"    {p:<8} n={len(gg):<3} twin {gg.twin.median():>5.1f}%   model {gg.model.median():>5.1f}%"
              f"   {'twin wins' if gg.twin.median() < gg.model.median() else 'model wins'}")
    T2.to_csv("data/exports/kappa_v2/chemical_twins.csv", index=False)
    print("\nwrote data/exports/kappa_v2/chemical_twins.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
