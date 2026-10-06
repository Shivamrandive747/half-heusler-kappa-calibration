"""The transfer-function forms, in one place with no project imports.

WHY THIS MODULE IS SEPARATE AND DELIBERATELY DEPENDENCY-FREE. Both the fitter
(family_calibration.py) and the consumer (make_paper_predictions.py) need to turn a form plus its
parameters into a number, and family_calibration already imports from make_paper_predictions. Put
the shared function in either of them and the two import each other -- which is exactly what
happened: `ImportError: cannot import name 'structure_status' from partially initialized module`.
Keeping the forms here, importing nothing from the project, breaks the cycle by construction rather
than by import ordering, which is the kind of fix that survives someone adding an import later.

There must be exactly ONE place a form becomes a number. The forms are no longer all of the shape
c*(T/300)^p: MATTHIESSEN is a series resistance and mis-applying it as a rescaling would produce a
wrong prediction with nothing raising an error. Every consumer routes through apply_family.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import numpy as np

FORMS = ("GLOBAL", "CONSTANT", "SHAPE", "SHAPE_NOCAP", "CURVE", "MATTHIESSEN")


def apply_family(form, par, T, k_bte):
    """Apply one family's fitted form to a calculated conductivity.

    GLOBAL / CONSTANT / SHAPE / CURVE all read par as (c, p) and rescale by min(c (T/300)^p, 1);
    they differ only in which of c and p was fitted and on what. SHAPE_NOCAP is SHAPE without the
    cap, c (T/300)^p: the family arm adopted 2026-10-01, because in Sn-Pt and Bi-Pd the measured
    conductivity keeps falling more slowly than the calculation above the temperature where the cap
    stops all correction. MATTHIESSEN reads par as (f, K0) and puts a boundary resistance in series
    with the intrinsic conductivity, which is not a rescaling and cannot be expressed as one.
    """
    T = np.asarray(T, dtype=float)
    k_bte = np.asarray(k_bte, dtype=float)
    if form == "MATTHIESSEN":
        f, k0 = par
        return 1.0 / (1.0 / np.maximum(float(f) * k_bte, 1e-9) + 1.0 / max(float(k0), 1e-9))
    c, p = par
    factor = float(c) * (T / 300.0) ** float(p)
    return k_bte * (factor if form == "SHAPE_NOCAP" else np.minimum(factor, 1.0))


def reduces_everywhere(form, par, T, k_bte) -> bool:
    """Does this form actually lower the calculated value on the data it was fitted to?

    The premise of the whole method is that a real sintered pellet conducts less than the perfect
    crystal the calculation models. For a rescaling that is c < 1. For MATTHIESSEN the prefactor
    alone does not decide it -- the series term drags the result below f*k, so an f above 1 can
    still reduce -- and testing f as though it were c silently removed the form from every
    candidate list. The honest test is the premise itself, evaluated on the data.
    """
    pred = apply_family(form, par, T, k_bte)
    return bool(np.all(pred < np.asarray(k_bte, dtype=float) * (1.0 - 1e-9)))
