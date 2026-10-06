"""Was choosing the uncapped family form itself a held-out decision? (record only, not in the paper)

The family arm moved from SHAPE (c (T/300)^p capped at 1) to SHAPE_NOCAP (the same, uncapped) on
2026-10-01, by the author's decision, after both had been scored on the same compounds. Choosing
between two forms after seeing both scores is a selection, so this measures what that selection is
worth when it too is held out: for each compound, the form is chosen using the OTHER compounds only
(pooled median of their leave-one-out errors, with the outer compound removed from the data), and
the outer compound is then scored with the chosen form fitted on its own family-mates.

Writes paper/evidence/family_form_nested.json.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import contextlib
import io
import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np

import extend_blind_test as E
import family_calibration as FC
import shared_constant as SCN

OUT = "paper/evidence/family_form_nested.json"
FORMS = ("SHAPE", "SHAPE_NOCAP")


def loo(d, form, p_g, c_g, drop=None):
    """compound -> held-out error inside its family, the outer compound `drop` removed entirely."""
    out = {}
    dd = d if drop is None else d[d.compound != drop]
    for fam, g0 in dd.groupby("fam"):
        for held in g0.compound.unique():
            # THE FOLD'S REFERENCES (PREREG amendment A): the siblings' references are re-chosen
            # without `held`, and `held` is scored on its own fold reference -- the deployed pairs
            # let the held-out compound pick its siblings' laboratory (the Sb-Pd leak).
            g = FC.fold_pairs(held)
            g = g[(g.fam == fam) & (g.compound != drop)] if drop is not None else g[g.fam == fam]
            te, tr = g[g.compound == held], g[g.compound != held]
            if len(tr) < 4 or not len(te):
                continue
            # the exponent is the shared p of that fold, refitted without `held`
            # (shared_constant.shared_p_in_fold). The outer compound `drop` is NOT also removed
            # from that p fit -- doing so would need one fit per (held, drop) pair (~1 h)
            cp = FC.fit_form(form, tr, c_g, SCN.shared_p_in_fold(held))
            k = FC.apply_family(form, cp, te["T"].values, te.k_pred.values)
            out[held] = float(np.median(100 * np.abs(k - te.k_ref.values) / te.k_ref.values))
    return out


def main() -> int:
    FC.NO_DOMAIN_GATE = True
    with contextlib.redirect_stdout(io.StringIO()):
        d = FC.build_published()
        # the shared constant (shared_constant.py, fitted on published calculations;
        # PREREG_shared_constant_on_calculations); every score uses the held-out p, see loo()
        c_g, p_g = SCN.shared_cp()
    d["fam"] = d.compound.map(FC.yzfam)
    full = {f: loo(d, f, p_g, c_g) for f in FORMS}
    ks = sorted(set(full["SHAPE"]) & set(full["SHAPE_NOCAP"]))
    picks, nested = {}, {}
    for k in ks:
        inner = {f: loo(d, f, p_g, c_g, drop=k) for f in FORMS}
        common = sorted(set(inner["SHAPE"]) & set(inner["SHAPE_NOCAP"]))
        med = {f: float(np.median([inner[f][x] for x in common])) for f in FORMS}
        picks[k] = min(FORMS, key=lambda f: (med[f], f))
        nested[k] = full[picks[k]][k]
    res = dict(
        _generated_by="test_family_form_nested.py",
        n_compounds=len(ks),
        pooled_median=dict(SHAPE=round(float(np.median([full["SHAPE"][k] for k in ks])), 1),
                           SHAPE_NOCAP=round(float(np.median([full["SHAPE_NOCAP"][k] for k in ks])), 1),
                           nested_choice=round(float(np.median(list(nested.values()))), 1)),
        nocap_chosen_in=sum(v == "SHAPE_NOCAP" for v in picks.values()),
        per_compound={k: dict(SHAPE=round(full["SHAPE"][k], 1),
                              SHAPE_NOCAP=round(full["SHAPE_NOCAP"][k], 1), chosen=picks[k])
                      for k in ks})
    json.dump(res, open(OUT, "w"), indent=2)
    pm = res["pooled_median"]
    print(f"{len(ks)} held-out compounds: SHAPE {pm['SHAPE']}%   SHAPE_NOCAP {pm['SHAPE_NOCAP']}%   "
          f"form chosen held-out {pm['nested_choice']}%   (uncapped chosen in "
          f"{res['nocap_chosen_in']} of {len(ks)} folds)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
