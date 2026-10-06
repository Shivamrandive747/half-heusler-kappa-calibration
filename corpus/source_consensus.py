"""Disqualify a laboratory that contradicts a consensus of three or more others.

A reported kappa_L is a subtraction of an electronic term from a total, and a laboratory that gets
that subtraction wrong is wrong by roughly the same factor on every compound it publishes. The
curve-shape gate in curve_quality catches a wrong Lorenz number when it distorts the SHAPE of a
curve; it cannot catch one that is wrong by a constant factor, because a uniformly scaled curve has
the correct shape. This gate catches that case, and it is the only quality test the corpus can
support at source level.

WHY NOT THE OBVIOUS TEST. Judging a laboratory by its average offset across the compounds it shares
with others was tried and abandoned: 82 of 86 in-domain sources overlap with anyone else on only one
or two compounds, so there is nothing to average and no way to tell a biased laboratory from an
unusual compound. But eight compounds have been measured four or more times, and on those a single
source can be checked against a genuine consensus it had no part in setting.

THE RULE, fixed before it was run:

    a compound is a TEST BED only if MIN_SOURCES independent sources measured it near 300 K;
    on a test bed, a source is disqualified if it sits further than FACTOR from the geometric mean
        of the others -- which, with 4+ sources, is a majority the source cannot itself swing;
    a source disqualified on ANY test bed is dropped EVERYWHERE, including on compounds where no
        consensus exists to check it. That last clause is the entire point: a source is tested where
        the literature is thick and excluded where it is thin, which is where it does damage.

It never reads a calculation, so it cannot be tuned to reduce the calculation-to-measurement error
it is meant to improve. On the current corpus it disqualifies four sources, caught on four different
compounds in four different families, and costs two compounds their only measurement.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import numpy as np
import pandas as pd
from scipy import stats

MIN_SOURCES = 4     # below this there is no consensus, only a second opinion
FACTOR = 2.0        # a factor-of-two disagreement between laboratories is not sample variation
T_LO, T_HI = 280.0, 320.0   # the window the test beds are compared in


def disqualified(meas: pd.DataFrame, domain=None) -> dict[str, list[tuple[str, float]]]:
    """Sources to drop, mapped to the (compound, ratio) evidence that disqualified each.

    `meas` needs columns red / TK / k / source_doi. `domain` optionally restricts the test beds to
    compounds the calibration actually applies to, so an out-of-scope compound cannot disqualify a
    source that is only ever used in scope.
    """
    d = meas[(meas.k > 0) & meas.TK.between(T_LO, T_HI)].dropna(subset=["red", "source_doi"])
    if domain is not None:
        d = d[d.red.map(domain)]
    if d.empty:
        return {}
    per = d.groupby(["red", "source_doi"]).k.median().reset_index()
    n = per.groupby("red").source_doi.nunique()
    beds = set(n[n >= MIN_SOURCES].index)

    out: dict[str, list[tuple[str, float]]] = {}
    for c in sorted(beds):
        g = per[per.red == c]
        for _, r in g.iterrows():
            others = g[g.source_doi != r.source_doi].k.values
            ratio = float(r.k) / float(stats.gmean(others))
            if ratio > FACTOR or ratio < 1.0 / FACTOR:
                out.setdefault(str(r.source_doi), []).append((c, ratio))
    return out


# A CROSS-COMPOUND CONSISTENCY TEST WAS TRIED HERE AND REMOVED (2026-09-18).
#
# The idea: a laboratory beyond FACTOR from every partner on every compound it shares, on two or
# more compounds, is a property of the laboratory and should be demoted even where no single
# compound has the four sources the test above needs. On the raw training set it fired on one
# laboratory, 10.1088/1361-6463/aac567, which reads HoSbPd at 0.34x Mastronardi 1999 and appeared
# to read ErSbPd at 0.46x Sekimoto 2006. But Sekimoto's ErSbPd curve RISES with temperature (4.8
# to 6.5 W/m/K over 364-989 K), the signature of an unsubtracted electronic term, and the curve
# shape gate already rejects it. On gated data aac567 is contradicted on one compound only, and the
# rule fires on zero laboratories. A gate that catches nothing is not kept. What remains is a
# genuine two-laboratory disagreement on HoSbPd, 2.9x, that no measurement-only rule can settle;
# reference_choice's declared tiebreak (wider temperature span) decides it, and the paper says so.
