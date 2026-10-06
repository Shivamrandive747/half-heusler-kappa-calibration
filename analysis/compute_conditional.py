"""Predicted values for the conditional candidates (fourteen since 2026-09-19).

These are NOT issued predictions. Each sits in a bonding family that cannot yet license it: Ni-Bi
has no usable measured member, and the Co-Bi / Ni-Pb / Sb-Ir candidates carry their only surviving
calculation at 500 K, where make_paper_predictions.py does not issue. What this computes is what the
prediction WOULD be, at the family's own constant where the family has one (Co-Bi 0.75, Ni-Pb 0.41,
Sb-Ir 0.16 -- anchored on one member each) and at the global constant only where no family constant
exists (Ni-Bi). HfCoBi was conditional until 2026-09-19 and is now ISSUED through the anchored
Co-Bi constant; it is removed from this tier automatically.

Every candidate here has already passed, in scratchpad/audit_unlock.py:
  VEC = 18 | cubic 216 on record, not polymorphic | no actinide | a real transport calculation |
  an X site represented in training | independent calculations agreeing within 2x

Three candidates were removed by those checks and are deliberately absent: YbNiBi (no cubic record,
no transport calculation, X site unseen), TiNiPb (two sources 5.6x apart), HoNiBi (2.37x apart).

ALL FIFTEEN ARE NOW WRITTEN, four of them at 500 K rather than 300 K. TiCoBi, HfNiPb, HfSbIr and
TiSbIr each carry exactly ONE row that survives the semi-empirical filter, and it sits at 500 K.
Until 2026-09-12 that dropped them entirely, on the reasoning that a 500 K calculation cannot
support a 300 K number -- true, but the wrong conclusion: the transfer function is defined at every
temperature, so the right move is to quote them AT 500 K rather than to withhold them. Each row
records `quoted_at_K`, and the 300 K columns are left empty for those four so that no consumer can
read a 500 K value as a room-temperature one. Their raw tier-1 counts (4, 2, 4, 3) still look
adequate only until the Slack/Debye-Callaway rows are removed, which is what that filter is for.

ScNiBi was added 2026-09-11 after the DXMag candidate screen surfaced it: VEC 18, cubic 216 with no
competing polymorph, Sc well represented in training, and real BTE at 300 and 500 K -- the same
data profile as HfCoBi. It is the tenth Ni-Bi member and changes no existing number.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

import extend_blind_test as E
from transfer_forms import apply_family
from make_paper_predictions import FAMILY_ISSUE_MAX, FLUCT, RADIO, structure_status, vec
from run_loco_chemistry import cluster_of, sites

OUT = "data/Target_Materials/CONDITIONAL_PREDICTIONS.csv"
BTE = "bte|boltz|phono3py|shengbte|almabte|iterative|rta"
NOT = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"

COND = {
    "HfCoBi": "Co-Bi", "TiCoBi": "Co-Bi",
    "HfNiPb": "Ni-Pb",
    "HfSbIr": "Sb-Ir", "TiSbIr": "Sb-Ir",
    "DyNiBi": "Ni-Bi", "ErNiBi": "Ni-Bi", "GdNiBi": "Ni-Bi", "LuNiBi": "Ni-Bi",
    "NdNiBi": "Ni-Bi", "PrNiBi": "Ni-Bi", "ScNiBi": "Ni-Bi", "SmNiBi": "Ni-Bi",
    "TbNiBi": "Ni-Bi", "TmNiBi": "Ni-Bi",
}
UNLOCKED_BY = {# Co-Bi and Ni-Pb are ANCHORED since 2026-09-19 (one measured member, in-sample
               # 6.8% / 6.7%). Their remaining candidates sit here only because their single
               # calculation is at 500 K; a second measured member would make them VALIDATED.
               "Co-Bi": "a second measured Co-Bi half Heusler (the family is anchored on ZrCoBi)",
               "Ni-Pb": "a second measured Ni-Pb half Heusler (the family is anchored on ZrNiPb)",
               "Sb-Ir": "one further measured Sb-Ir half Heusler",
               # TWO, not one. YNiBi is broken, so the family sits at zero qualifying members;
               # a single new measurement takes it to one, still short of the >=2 rule enforced
               # in make_paper_predictions.py. This string previously said "one" and contradicted
               # both that rule and Section 5, and it ships in the supplementary CSV.
               "Ni-Bi": ("two sound measurements: a re-measurement of YNiBi below its bipolar "
                         "onset plus one further Ni-Bi compound, or two new compounds")}


def caveat_for(cp_: str) -> str:
    """Apply the SAME element screens the issued predictions get.

    make_paper_predictions.py refuses radioactive elements and marks valence-fluctuating ones
    FLAGGED. This script applied neither: it worked from the hardcoded COND list, so SmNiBi was
    listed beside the other eight with no hint that Sm can be di- or trivalent, which makes the
    VEC=18 premise unsafe for it. A screen that runs in one script and not its sibling is not a
    screen. Importing FLUCT/RADIO rather than restating them is deliberate -- a copied constant
    is how the two drifted apart in the first place.
    """
    els = [str(e) for e in Composition(cp_).elements]
    v = vec(cp_)
    if v != 18:
        return f"OFF-DOMAIN: VEC {v} != 18"
    rad = [e for e in els if e in RADIO]
    if rad:
        return f"REFUSE: radioactive/actinide ({','.join(rad)})"
    fl = [e for e in els if e in FLUCT]
    if fl:
        return f"FLAGGED: {FLUCT[fl[0]]}"
    return ""


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    import source_identity as SI
    # A COMPOUND THE PAPER ISSUES OR FLAGS IS NOT CONDITIONAL. Since 2026-09-19 every family carries
    # its own constant, and a single-member family is ANCHORED and may issue (labelled as such).
    # HfCoBi is issued that way; it must not also appear here. The candidate list above is the
    # historical one; whatever PAPER_PREDICTIONS.csv now decides takes precedence.
    try:
        _issued = set(pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv").compound)
    except FileNotFoundError:
        _issued = set()
    for _c in [c for c in COND if c in _issued]:
        print(f"  {_c}: now handled by make_paper_predictions.py ({COND[_c]} carries a constant); "
              "removed from the conditional tier")
        COND.pop(_c)
    # A preprint and the journal article it became are ONE source. Without this the n_sources
    # column counted the Miyazaki preprint and its published version as two independent
    # calculations agreeing to the last decimal, which passed three conditional compounds through
    # the agreement check vacuously.
    tr = SI.dedupe(pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv",
                               low_memory=False))
    tr["red"] = tr.formula.map(red)
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr["t"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr = tr.dropna(subset=["red", "k", "T"])
    m = tr.method.astype(str).str.lower()
    bte = tr[(tr.t == 1) & m.str.contains(BTE, regex=True) & ~m.str.contains(NOT, regex=True)
             & (tr.k > 0)]

    # the deployed shared constant: fitted on published calculations vs measurements, not on the
    # ML model's blind-test predictions (PREREG_shared_constant_on_calculations.md, 2026-10-05)
    import shared_constant as SCN
    c, p = SCN.shared_cp()
    # bands at the quoted temperature (PREREG amendment B): "levels" = 300 K, "levels_500K" = 500 K
    _cfj = json.load(open("data/exports/kappa_v2/conformal_indomain.json"))
    cf = _cfj["levels"]
    b50, b90 = cf["0.50"]["factor"], cf["0.90"]["factor"]
    _cf5 = _cfj.get("levels_500K") or cf
    b50_500, b90_500 = _cf5["0.50"]["factor"], _cf5["0.90"]["factor"]
    import family_calibration as FC   # bte_at (FIXPASS S7)
    from make_paper_predictions import source_key
    print(f"calibration c={c:.3f}, p={p:.3f}   bands x/div {b50} (50%), {b90} (90%)\n")
    # EVERY FAMILY WITH A MEASURED MEMBER CARRIES ITS OWN CONSTANT (ruling 2026-09-19). A conditional
    # estimate is "what the method would issue", and for Co-Bi, Ni-Pb and Sb-Ir that is the family's
    # own constant, not the global one -- HfCoBi is issued at c=0.75 and TiCoBi must not be quoted at
    # c=0.51 in the same paper. Ni-Bi has no member and keeps the global constant as a stated proxy.
    # `c_used` / `c_basis` record which, so the supplementary table can say so per row.
    fam_c: dict[str, tuple[float, str]] = {}
    fam_form: dict[str, str] = {}   # the family arm's form; the global fallback keeps the cap
    try:
        _fams = json.load(open("data/exports/kappa_v2/family_calibration.json"))["families"]
        for _f, _v in _fams.items():
            _loo, _ins = _v.get("loo_family_ape"), _v.get("insample_ape")
            _err = _loo if _loo is not None else _ins
            if _loo is not None:
                _b = f"family, validated leave-one-out {_loo:.1f}%"
            else:
                _b = f"family, anchored on one member (in-sample {_ins:.1f}%)"
            if float(_err) >= FAMILY_ISSUE_MAX:
                _b += f" -- does not clear the {FAMILY_ISSUE_MAX:.0f}% bar"
            fam_c[_f] = (float(_v["c"]), _b)
            fam_form[_f] = _v.get("form", "SHAPE")
    except FileNotFoundError:
        print("no family_calibration.json -- every conditional estimate uses the global constant")
    for _f in sorted(set(COND.values())):
        _cu, _b = fam_c.get(_f, (c, "global (family has no measured member)"))
        print(f"  {_f}: c={_cu:.2f}  {_b}")
    print()

    rows = []
    for cp_, fam in COND.items():
        s = bte[bte.red == cp_]
        if not len(s):
            print(f"  {cp_}: no BTE calculation at any temperature -- skipped")
            continue
        # THE MEDIAN CALCULATION PER TEMPERATURE, CARRIED BY family_calibration.bte_at (FIXPASS S7).
        # This took the FIRST row nearest 300 K and, when it lay within 150 K, used its value as if
        # it were at 300 K. Now: median over sources at each calculated temperature (row order no
        # longer matters); a 300 K value only where a calculation lies within 150 K of 300 K (the
        # availability rule is unchanged), carried to 300 K by log-log interpolation or 1/T; else the
        # compound is quoted at its calculated temperature nearest 300 K, at that temperature's value.
        bagg = s.groupby(s["T"].round(-1)).k.median().sort_index()
        Tq = float(bagg.index.values[np.abs(bagg.index.values - 300).argmin()])
        at300 = abs(Tq - 300) <= 150
        kb300 = FC.bte_at(bagg, 300.0) if at300 else None
        if at300 and kb300 is None:
            at300 = False
        kb = float(kb300) if at300 else float(bagg.loc[Tq])

        # QUOTE AT THE TEMPERATURE THE CALCULATION ACTUALLY EXISTS AT.
        #
        # Four candidates -- TiCoBi, HfNiPb, HfSbIr, TiSbIr -- were previously dropped entirely
        # because their only surviving BTE row sits at 500 K rather than near room temperature.
        # That is a bookkeeping reason, not a physical one: the transfer function is defined at
        # every temperature, and a 500 K calculation supports a 500 K statement perfectly well.
        # Withholding them made the method look blinder than it is. They are now quoted at their
        # own temperature, with `quoted_at_K` recording it, and their 300 K columns left empty so
        # that nothing downstream can mistake a 500 K number for a room-temperature one.
        c_used, c_basis = fam_c.get(fam, (c, "global (family has no measured member)"))
        form_used = fam_form.get(fam, "GLOBAL")
        fac = float(apply_family(form_used, (c_used, p), [300.0 if at300 else Tq], [1.0])[0])
        k3 = kb * fac
        # count independent sources AT THE QUOTED TEMPERATURE, not always near 300 K: a compound
        # quoted at 500 K has no rows in a 250-350 K window and would otherwise report n_sources=0,
        # which reads as "no provenance" when the truth is "provenance at a different temperature".
        # one source = DOI, else dataset URL (PhononDB has no DOI; FIXPASS S8)
        _w = s[s["T"].between(Tq - 50, Tq + 50)]
        srcs = _w.groupby(source_key(_w)).k.median()
        spread = float(srcs.max() / srcs.min()) if len(srcs) > 1 else 1.0
        rows.append(dict(compound=cp_, family=fam, x_site=(cluster_of(cp_) or "?").replace("half:X=", ""),
                         quoted_at_K=int(round(300 if at300 else Tq)),
                         c_used=round(c_used, 2), c_basis=c_basis,
                         # temperature above which min(c (T/300)^p, 1) is 1: the constant stops acting.
                         # The uncapped family form has no such temperature.
                         calibration_form=form_used,
                         cap_K=(None if form_used == "SHAPE_NOCAP"
                                else int(round(300.0 * (1.0 / c_used) ** (1.0 / p))) if c_used < 1 else 300),
                         transfer_factor=round(fac, 3),
                         kappa_BTE_quoted=round(kb, 2), kappa_pred_quoted=round(k3, 2),
                         # the 300 K columns stay EMPTY for a compound quoted elsewhere, so a
                         # consumer that reads them cannot silently treat a 500 K value as one
                         kappa_BTE_300=round(kb, 2) if at300 else np.nan,
                         kappa_pred_300=round(k3, 2) if at300 else np.nan,
                         lo50=round(k3 / (b50 if at300 else b50_500), 2),
                         hi50=round(k3 * (b50 if at300 else b50_500), 2),
                         lo90=round(k3 / (b90 if at300 else b90_500), 2),
                         hi90=round(k3 * (b90 if at300 else b90_500), 2),
                         band_at_K=300 if at300 else 500,
                         n_bte_rows=len(s), n_sources=len(srcs),
                         source_spread=round(spread, 2),
                         status="CONDITIONAL",
                         caveat=" | ".join(x for x in (caveat_for(cp_),
                                                       ("FAMILY CONSTANT FAILS: " + c_basis
                                                        if "does not clear" in c_basis else ""))
                                           if x),
                         unlocked_by=UNLOCKED_BY[fam]))
    R = pd.DataFrame(rows).sort_values(["family", "kappa_pred_quoted"])
    R.to_csv(OUT, index=False)

    print(f"  {'compound':<9}{'family':<8}{'X':<4}{'at K':>6}{'BTE':>8}{'pred':>8}"
          f"{'50% range':>13}{'90% range':>14}{'src':>5}{'spread':>8}")
    for r in R.itertuples():
        print(f"  {r.compound:<9}{r.family:<8}{r.x_site:<4}{r.quoted_at_K:>6}"
              f"{r.kappa_BTE_quoted:>8.2f}"
              f"{r.kappa_pred_quoted:>8.2f}{f'{r.lo50}-{r.hi50}':>13}"
              f"{f'{r.lo90}-{r.hi90}':>14}{r.n_sources:>5}{r.source_spread:>7.2f}x")
    print(f"\n  {len(R)} conditional predictions across {R.family.nunique()} families")
    for f, g in R.groupby("family"):
        print(f"    {f:<7}{len(g)} compounds, released by {UNLOCKED_BY[f]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
