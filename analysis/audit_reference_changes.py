"""Which compounds changed reference laboratory when a ladder rule was added, and by which rule.

WHY THIS EXISTS. A calibration is only as honest as its choice of reference, and a rule that moves
references must be able to show it moved them for the reason it states -- never because the new
reference happened to agree better with the calculation. This script runs the ladder with the
nanostructured demotion switched OFF and ON, on identical inputs, and records every compound whose
chosen laboratory differs, with the rung that decided each. It is the artefact a referee asks for
when told "references were re-selected", and it is the falsification test of the rule: if it moves
compounds that were already accurate, the rule is doing something other than what it claims.

WHAT IT FOUND (first run, 2026-09-18). Exactly two compounds move, TaFeSb and ZrCoBi, each from a
ball-milled sample to a bulk laboratory; two more keep their laboratory and gain a note. Under the
deployed route TaFeSb goes 101.8% -> 21.4% and ZrCoBi 49.2% -> 17.2%, and no other compound's error
changes by more than 0.05 pp. Zero compounds got worse.

Read-only with respect to the corpus; writes data/exports/kappa_v2/reference_changes.json.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import io
import contextlib
import json
import sys
import warnings
from datetime import datetime, timezone

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np

import extend_blind_test as E
import family_calibration as FC
import reference_choice as RC
import shared_constant as SCN

OUT = "data/exports/kappa_v2/reference_changes.json"


def _choices(with_nano: bool) -> dict:
    """Run the whole published-route build with the demotion on or off; return its choices.

    Going through build_published() rather than re-deriving the measured frame guarantees the two
    runs see identical inputs -- same curve gate, same consensus gate, same in-domain set.
    """
    real = RC.nanostructured_sources
    if not with_nano:
        RC.nanostructured_sources = lambda *a, **k: set()
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d = FC.build_published()
        return {c: tuple(v) for c, v in FC.CHOSEN_REFERENCE.items()}, d
    finally:
        RC.nanostructured_sources = real


_FOLDS: dict = {}


def _fold(with_nano: bool, held: str):
    """The published pairs in the leave-one-out fold that scores `held`, under the same ladder
    setting as the arm being audited (PREREG amendment A: references re-chosen inside the fold --
    `held` by the ladder WITHOUT the family rung, its siblings with the family rung over members
    minus `held`). Cached per (setting, compound)."""
    key = (with_nano, held)
    if key not in _FOLDS:
        real = RC.nanostructured_sources
        if not with_nano:
            RC.nanostructured_sources = lambda *a, **k: set()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _FOLDS[key] = FC.build_published(held_out=(held,))
        finally:
            RC.nanostructured_sources = real
    return _FOLDS[key]


def _nano_evidence(doi: str, compound: str) -> dict:
    """What the samples table records for the demoted specimen -- the reason it was demoted."""
    import sqlite3
    try:
        con = sqlite3.connect(RC.DB)
        row = con.execute("SELECT milling_time_h, grain_size_um, synthesis_method FROM samples "
                          "WHERE lower(doi)=? AND reduced_formula=? ORDER BY milling_time_h DESC "
                          "LIMIT 1", (doi.lower(), compound)).fetchone()
        con.close()
    except Exception:  # noqa: BLE001
        row = None
    if not row:
        return {}
    return dict(milling_time_h=row[0], grain_size_um=row[1], synthesis=str(row[2] or "")[:80])


def _lab_ratio_300K(d_off, d_on, compound: str) -> float | None:
    """How much higher the new reference reads than the old one, near 300 K, from the measured
    values themselves (k_ref) -- never from the calculation."""
    import numpy as np
    a = d_off[(d_off.compound == compound) & (d_off["T"].between(280, 340))].k_ref
    b = d_on[(d_on.compound == compound) & (d_on["T"].between(280, 340))].k_ref
    if a.empty or b.empty:
        return None
    return round(float(np.median(b) / np.median(a)), 2)


def _deployed_ape(d, compound: str, c_g: float, p_g: float, fold=None) -> float | None:
    """The compound's error on ITS FAMILY'S ARM as compute_deployed_route scores it: held out
    from its family-mates where it has any, else fitted on itself (anchored). This is the number
    the manuscript's per-compound claims rest on, so both sides of every move are scored this way.

    `fold` is the pairs frame of the fold that scores `compound` (_fold): the compound and its
    family-mates are then taken from it, so their references are re-chosen inside the fold
    (PREREG amendment A); family membership is still that of `d`."""
    from make_paper_predictions import yzfam
    if d[d.compound == compound].empty:
        return None
    fam = yzfam(compound)
    mates = set(d.loc[(d.compound.map(yzfam) == fam) & (d.compound != compound), "compound"])
    src_frame = d if fold is None else fold
    te = src_frame[src_frame.compound == compound]
    if te.empty:
        return None
    tr = src_frame[src_frame.compound.isin(mates)]
    held_out = tr.compound.nunique() >= 1 and len(tr) >= 4
    src = tr if held_out else te
    # held out: the family form's exponent is the shared p refitted without the compound too
    # (on the fold's pairs, shared_constant.shared_p_in_fold); anchored in-sample: the deployed p
    p_use = ((SCN.shared_p_in_fold(compound) if fold is not None
              else SCN.shared_p_excluding([compound])) if held_out else p_g)
    cp = FC.fit_form(FC.ADOPTED_FORM, src, c_g, p_use)
    if cp is None:
        return None
    T = te["T"].values.astype(float)
    k = FC.apply_family(FC.ADOPTED_FORM, cp, T, te.k_pred.values)
    return round(float(FC.per_compound_ape(te, k).iloc[0]), 1)


def _global_arm_ape(d, compound: str, c_g: float, p_g: float) -> float | None:
    """The compound's error on the global arm, per-compound median, as compute_deployed_route
    scores it. Both moved compounds sit on the global arm, so this is their deployed error.

    The shared constant applied is the one fitted WITHOUT the compound (shared_constant.shared_cp;
    PREREG_shared_constant_on_calculations rule 4), on the CURRENT reference set in both arms, so a
    before/after difference isolates the compound's own reference change. c_g, p_g are unused for
    scoring and kept for callers."""
    te = d[d.compound == compound]
    if te.empty:
        return None
    T = te["T"].values.astype(float)
    c_h, p_h = SCN.shared_cp(exclude=[compound])
    k = te.k_pred.values * E.apply_cp(T, c_h, p_h)
    return round(float(FC.per_compound_ape(te, k).iloc[0]), 1)


def main() -> int:
    # the deployed shared constant (published calculations vs measurements), and its pairs built
    # NOW, before anything below monkeypatches reference_choice -- shared_constant caches them
    with contextlib.redirect_stdout(io.StringIO()):
        c_g, p_g = SCN.shared_cp()
    off, d_off = _choices(with_nano=False)
    on, d_on = _choices(with_nano=True)
    compounds = sorted(set(off) | set(on))
    nano_pairs = RC.nanostructured_sources()
    changes, notes = [], []
    for c in compounds:
        a, b = off.get(c), on.get(c)
        if a is None or b is None:
            continue
        if a[0] != b[0]:
            changes.append(dict(compound=c, before_doi=a[0], before_rule=a[1],
                                after_doi=b[0], after_rule=b[1],
                                before_was_nanostructured=(a[0].lower(), c) in nano_pairs,
                                ape_before=_global_arm_ape(d_off, c, c_g, p_g),
                                ape_after=_global_arm_ape(d_on, c, c_g, p_g),
                                ape_deployed_before=_deployed_ape(d_off, c, c_g, p_g,
                                                                  fold=_fold(False, c)),
                                ape_deployed_after=_deployed_ape(d_on, c, c_g, p_g,
                                                                 fold=_fold(True, c)),
                                demoted_specimen=_nano_evidence(a[0], c),
                                demotion_signal=RC.NANO_REASON.get((a[0].lower(), c)),
                                new_over_old_ratio_300K=_lab_ratio_300K(d_off, d_on, c)))
        elif a[1] != b[1]:
            notes.append(dict(compound=c, doi=a[0], before_rule=a[1], after_rule=b[1]))

    # ALSO against the pre-campaign snapshot, so every reference that moved for ANY reason since
    # then is on the record -- the nano rule is one of several changes (tiebreak rounding, new
    # sample records reaching rung 1, misidentified PDFs excluded). A referee asks "which
    # references changed and why", not "which changed because of rule X".
    base_path = "data/exports/kappa_v2/reference_choice_BEFORE_nano_rung.json"
    since_start = []
    try:
        base = json.load(open(base_path))
        for c in sorted(on):
            b = base.get(c)
            if b and b[0] != on[c][0]:
                since_start.append(dict(compound=c, before_doi=b[0], before_rule=b[1],
                                        after_doi=on[c][0], after_rule=on[c][1],
                                        ape_global_after=_global_arm_ape(d_on, c, c_g, p_g),
                                        ape_deployed_after=_deployed_ape(d_on, c, c_g, p_g,
                                                                         fold=_fold(True, c))))
    except FileNotFoundError:
        pass

    # FAMILY-LEVEL before/after for every family that contains a compound whose reference moved
    # since the snapshot. Section 3 says "Sb-Pd went from X to Y when HoSbPd's reference was
    # corrected"; both X and Y must come from a script, and this is where X is computed: the whole
    # published route rebuilt with the SNAPSHOT's reference choices forced, then the adopted form
    # scored leave-one-compound-out exactly as family_calibration.py scores it.
    family_moves = {}
    if since_start:
        real_pc = RC.per_compound
        snap = {c: tuple(v) for c, v in base.items()}
        # A compound that has joined the family step since the snapshot (VFeSb once its hypothetical
        # hexagonal record stopped counting; TiSnPt, ZrSnPt, ErBiPd, TmNiSb once they stopped being
        # dropped) has no snapshot reference. It takes its CURRENT reference in both arms, so the
        # before/after below still isolates what it exists to measure -- a change of reference --
        # rather than mixing in a change of membership. Without this the rebuild raised KeyError.
        RC.per_compound = lambda meas, compounds, *a, **k: {
            **real_pc(meas, compounds, *a, **{**k, "verbose": False}),
            **{c: snap[c] for c in compounds if c in snap}}
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                d_snap = FC.build_published()
        finally:
            RC.per_compound = real_pc
        _snap_cache: dict = {}

        def _snap_fold(h):
            if h not in _snap_cache:
                RC.per_compound = lambda meas, compounds, *a, **k: {
                    **real_pc(meas, compounds, *a, **{**k, "verbose": False}),
                    **{c: snap[c] for c in compounds if c in snap}}
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _snap_cache[h] = FC.build_published(held_out=(h,))
                finally:
                    RC.per_compound = real_pc
            return _snap_cache[h]
        from make_paper_predictions import yzfam
        d_snap["fam"], d_on["fam"] = d_snap.compound.map(yzfam), d_on.compound.map(yzfam)
        for f in sorted({yzfam(r["compound"]) for r in since_start}):
            g0, g1 = d_snap[d_snap.fam == f], d_on[d_on.fam == f]
            if min(g0.compound.nunique(), g1.compound.nunique()) < FC.MIN_MEMBERS_LOO:
                continue
            # BOTH arms re-choose references inside each fold (PREREG amendment A): the snapshot
            # arm forces the snapshot choices for every compound it holds, the current arm runs the
            # live ladder; in each, the held-out compound gets no family rung.
            e0 = FC.score_form(FC.ADOPTED_FORM, g0, c_g, p_g, fold=_snap_fold)
            e1 = FC.score_form(FC.ADOPTED_FORM, g1, c_g, p_g, fold=lambda h: _fold(True, h))
            family_moves[f] = dict(
                members=sorted(g1.compound.unique()),
                moved=[r["compound"] for r in since_start if yzfam(r["compound"]) == f],
                loo_ape_before=None if e0 is None else round(e0, 1),
                loo_ape_after=None if e1 is None else round(e1, 1))
    print(f"ladder run on {len(on)} compounds, demotion OFF vs ON")
    print(f"\nREFERENCE MOVED ({len(changes)}):")
    for r in changes:
        print(f"  {r['compound']:<9} {r['before_doi'][:36]:<38} [{r['before_rule']}]")
        print(f"        ->   {r['after_doi'][:36]:<38} [{r['after_rule']}]")
        print(f"             signal: {r['demotion_signal']}")
        print(f"             global-arm error {r['ape_before']}% -> {r['ape_after']}%   "
              f"demoted specimen: {r['demoted_specimen']}   new/old at 300 K: {r['new_over_old_ratio_300K']}x")
    print(f"\nSAME REFERENCE, RULE ANNOTATED ({len(notes)}):")
    for r in notes:
        print(f"  {r['compound']:<9} {r['doi'][:36]:<38} [{r['after_rule']}]")

    rep = dict(generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               rule="nanostructured_sources() demotion: milling_time_h >= "
                    f"{RC.NANO_MILL_H} or grain_size_um < {RC.NANO_GRAIN_UM} or 'nano' in synthesis",
               n_compounds=len(on), moved=changes, annotated=notes,
               moved_since_2026_09_18_snapshot=since_start,
               family_loo_before_after=family_moves,
               all_choices={c: dict(doi=v[0], rule=v[1]) for c, v in on.items()})
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    if family_moves:
        print("\nFAMILY LEAVE-ONE-OUT, snapshot references -> current references:")
        for f, v in family_moves.items():
            print(f"  {f:<6} {v['loo_ape_before']}% -> {v['loo_ape_after']}%   "
                  f"moved: {', '.join(v['moved'])}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
