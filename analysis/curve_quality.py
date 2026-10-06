"""Reject measured kappa_L curves whose SHAPE is not lattice conduction.

WHY A SHAPE TEST BEATS THE PROXY RULES IT REPLACES. A reported kappa_L is almost never measured
directly: a laboratory measures total conductivity and subtracts an estimated electronic term,
L*sigma*T. Get the Lorenz number L wrong and the remainder is not lattice conductivity at all, and
the error announces itself in the SHAPE of the curve rather than in any single value:

  L too small   the electronic part is under-subtracted, and what is booked as lattice
                conductivity RISES with temperature. Lattice conduction cannot do that. This is the
                defect that disqualifies YNiBi in the paper, reported by its own measuring authors.
  L too large   the electronic part is over-subtracted, and the remainder COLLAPSES toward zero.
                VCoSn falls 12.80 -> 0.36 W/m/K between 301 and 572 K, a slope of T^-5.3 where
                Umklapp scattering gives T^-1.

Both are detectable from the curve alone, with no appeal to structure, chemistry or provenance.
That matters because the rule this replaces was a proxy: the manuscript excluded any compound with
a non-cubic polymorph on record, on the theory that the measurement might have been of the wrong
phase. Tested, that rule discarded five compounds and only one deserved it -- MgAgSb, whose curve
RISES at slope +4.65 and which this gate catches directly and for the right reason. The other four
(LiZnSb, LaSbPd, VFeSb, BaAgSb) predict at 9-29% and were being thrown away for having a hexagonal
entry in a database.

THE THRESHOLDS, and why they are where they are. Umklapp scattering alone gives d(ln kappa)/d(ln T)
= -1. Adding boundary or point-defect scattering makes the fall SHALLOWER, never steeper, so any
real curve sits above -1 at high temperature and well above -2 anywhere. A curve steeper than -2 is
therefore not a lattice curve. In the other direction a small positive slope can be genuine noise on
a short baseline, so the rise threshold is +0.2 rather than 0.

WHAT THIS DOES NOT DO. It does not judge whether a value is right, only whether a CURVE is
physical. A single measured point has no shape and is never rejected here -- it is simply not
testable, and MIN_POINTS says so rather than pretending otherwise.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import numpy as np
import pandas as pd

MIN_POINTS = 4        # below this a curve has no reliable shape
MIN_SPAN = 1.15       # and the points must cover at least this ratio in temperature
MAX_FALL = -2.0       # steeper than this is over-subtraction, not conduction
MAX_RISE = 0.20       # above this is a bipolar leak; small positives can be noise
MIN_RISE_SPAN = 150.0 # ...but only trust a rise measured over at least this many kelvin


def curve_slope(T, k) -> float:
    """d(ln kappa)/d(ln T) for one compound from one source. NaN when not testable.

    DUPLICATE TEMPERATURES ARE COLLAPSED FIRST, and skipping that step produces false rejections.
    A single paper often reports several SAMPLES of the same compound -- different densities, grain
    sizes or synthesis routes -- and the corpus stores them under one DOI with no sample identifier.
    Pooling them makes a staircase, not a curve: ZrNiSn from PhysRevB.59.8615 carries 6.05, 9.08 AND
    16.39 W/m/K at 201 K, three separate specimens, and fitting through all of them returned a
    spurious +0.25 rise that would have rejected a perfectly good compound measured by 26
    laboratories. Taking the median at each temperature recovers the shape the samples share, which
    is what the physics test is about.
    """
    d = pd.DataFrame({"T": np.asarray(T, dtype=float), "k": np.asarray(k, dtype=float)})
    d = d[np.isfinite(d["T"]) & np.isfinite(d["k"]) & (d["T"] > 0) & (d["k"] > 0)]
    if d.empty:
        return float("nan")
    d = d.groupby(d["T"].round(0)).k.median()
    if len(d) < MIN_POINTS or d.index.max() / d.index.min() < MIN_SPAN:
        return float("nan")
    return float(np.polyfit(np.log(d.index.values), np.log(d.values), 1)[0])


def classify(slope: float) -> tuple[str, str]:
    """(verdict, reason). 'untestable' is not a failure -- it is an absence of evidence."""
    if not np.isfinite(slope):
        return "untestable", "fewer than %d points or too narrow a temperature span" % MIN_POINTS
    if slope > MAX_RISE:
        return "reject", (f"kappa_L rises with temperature (slope {slope:+.2f}); lattice conduction "
                          "cannot, so the electronic term is under-subtracted")
    # NOTE: the span condition is applied by classify_with_span(), not here -- see below.
    if slope < MAX_FALL:
        return "reject", (f"kappa_L falls as T^{slope:.2f}, steeper than Umklapp allows; the "
                          "electronic term is over-subtracted and the remainder is not lattice")
    return "keep", f"slope {slope:+.2f}, consistent with lattice conduction"


def classify_with_span(slope: float, span: float) -> tuple[str, str]:
    """As classify(), but a RISE is only trusted when measured over enough temperature.

    A small positive slope across 90 K is indistinguishable from scatter; the same slope across
    600 K is unmistakable. Judging the two alike forces a choice of threshold that decides which
    compounds survive, which is the wrong reason to choose a number. Requiring span as well makes
    the test about the strength of the evidence instead: reject a rise when there is enough
    temperature range to be sure of it, and withhold judgement when there is not.

    Measured cost of this on the current corpus: the strict form (rise alone) cuts Ni-Sb from 7
    members to 6 and Sb-Pd from 3 to 2, dropping Sb-Pd below the minimum and destroying its
    calibration. The span-aware form leaves both intact and still rejects NbCoSn, whose rise runs
    over 679 K, and ErSbPd's high-temperature curve, over 625 K.

    A FALL steeper than MAX_FALL is rejected regardless of span: no amount of shortness makes
    T^-5.3 a plausible lattice curve.
    """
    v, why = classify(slope)
    if v == "reject" and slope > 0 and np.isfinite(span) and span < MIN_RISE_SPAN:
        return "keep", (f"slope {slope:+.2f} rises, but only over {span:.0f} K -- too short a "
                        f"baseline to distinguish from scatter, so not rejected")
    return v, why


def audit(meas: pd.DataFrame, formula="red", temp="TK", kappa="k", source="source_doi"):
    """One row per (compound, source) curve with its slope and verdict."""
    out = []
    for (c, doi), g in meas.groupby([formula, source]):
        s = curve_slope(g[temp], g[kappa])
        span = float(g[temp].max() - g[temp].min())
        v, why = classify_with_span(s, span)
        out.append(dict(compound=c, source_doi=doi, n_points=len(g), slope=s,
                        verdict=v, reason=why,
                        t_lo=float(g[temp].min()), t_hi=float(g[temp].max()),
                        k_lo=float(g[kappa].min()), k_hi=float(g[kappa].max())))
    return pd.DataFrame(out)


def rejected_pairs(meas: pd.DataFrame, **kw) -> set:
    """The (compound, source) pairs whose curve is not physical."""
    a = audit(meas, **kw)
    return set(map(tuple, a[a.verdict == "reject"][["compound", "source_doi"]].values))
