"""A measured lattice conductivity may not exceed the perfect-crystal calculation.

THE PHYSICS. kappa_BTE is computed for an infinite, defect-free, isotopically pure single crystal
with only phonon-phonon scattering. Every feature of a real sample -- porosity, grain boundaries,
point defects, antisite disorder, second phases -- adds a scattering channel, and by Matthiessen's
rule adding a channel can only shorten the mean free path. A polycrystal therefore cannot conduct
heat better than the perfect crystal of the same compound. The transfer function encodes exactly
this: kappa_expt = kappa_BTE * min[c(T/300)^p, 1] is capped at 1 and can only correct DOWNWARD.

So a compound measuring above its calculation is not a compound needing a large correction, it is a
compound the model cannot represent at all. Leaving it in does not merely add error: it drags the
family constant toward 1 for everyone else, and the cap means even a perfect per-compound constant
cannot fit it. Sn-Pt is the demonstration -- giving each member its own ideal constant scores 32.6%,
WORSE than the single family constant at 32.3%, which is the signature of a capped form being asked
to go somewhere it cannot.

THE TOLERANCE IS MEASURED, NOT CHOSEN. Independent calculations of the same compound disagree by a
median factor of 1.36 across the 62 in-domain compounds calculated more than once, so a measurement
within 1.36x of exceeding its calculation is inside the calculation's own uncertainty and is kept. A
measurement beyond it cannot be explained that way. Sources are compared only where they report at
the SAME temperature: kappa ~ 1/T, so a 300 K value over a 900 K value from one paper is a factor of
three that has nothing to do with two groups disagreeing.

THE TOLERANCE WAS 1.18, AND THAT NUMBER WAS WRONG. Three arXiv preprints sat in the corpus under a
different source_doi from the journal article each later became, so one calculation was counted as
two independent groups -- see `source_identity.py`. A paper agrees with itself exactly, so each
self-duplicate contributed a ratio of 1.00 and dragged the median toward the one value it can never
legitimately take. Collapsing them moves the median from 1.18 (over 107 apparent compounds) to 1.36
(over 62 real ones). THE GATE'S VERDICT MOVES BY ONE COMPOUND, WITHOUT CONSEQUENCE. TiSnPt and
ZrSnPt sit at 1.46 and are dropped under either tolerance -- they remain the catch this gate exists
for. LaBiPd lies between the two values and is now retained, which returns Bi-Pd to two measured
members; that is still one short of the three a family calibration requires, so Bi-Pd stays on the
global arm. The five adopted families and every fitted constant are unchanged.

An intermediate value of 1.29 was briefly carried here and is wrong. It came from a dedupe that
mapped a MISSING source_doi to the string "nan", which pandas groups where it drops a true NaN, so
4092 rows of unknown provenance collapsed into one phantom source and inflated the very counts this
was meant to deflate. Both arms now exclude missing provenance identically.

WHAT IT CATCHES, and why those are the right catches. TiSnPt at 1.48x and ZrSnPt at 1.46x, both from
one 2006 conference proceedings, both Pt-based and strongly metallic. Their reported kappa_L is
kappa_total minus a Wiedemann-Franz electronic term; in a metallic sample that term is most of the
total, so a Lorenz number taken at the degenerate limit under-subtracts and inflates what is left.
An inflated kappa_L rising above the perfect crystal is the expected signature of that error, not a
property of the material. DySbPt sits at exactly 1.00 and is kept -- it is not evidence of anything.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import numpy as np
import pandas as pd

CALC_DISAGREEMENT = 1.36   # measured: median max/min, matched temperature, 62 in-domain compounds


def inverted(d: pd.DataFrame, tol: float = CALC_DISAGREEMENT) -> dict[str, float]:
    """compound -> median measured/calculated, for those exceeding the calculation beyond `tol`.

    `d` is the paired frame: columns compound / k_ref (measured) / k_pred (calculated).
    """
    r = d.groupby("compound").apply(lambda g: float(np.median(g.k_ref / g.k_pred)))
    return {c: float(v) for c, v in r.items() if v > tol}


def apply(d: pd.DataFrame, tol: float = CALC_DISAGREEMENT, verbose: bool = True) -> pd.DataFrame:
    bad = inverted(d, tol)
    if not bad:
        if verbose:
            print("physical-domain gate: no compound measures above its calculation")
        return d
    if verbose:
        print(f"physical-domain gate: {len(bad)} compound(s) measure above the perfect crystal "
              f"(tolerance {tol:.2f}x, the median calculation-to-calculation disagreement)")
        for c, v in sorted(bad.items(), key=lambda x: -x[1]):
            print(f"    {c:<9} measured/calculated = {v:.2f}x -- outside what the model can represent")
    return d[~d.compound.isin(bad)]
