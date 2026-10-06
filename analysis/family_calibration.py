"""Per-(Y,Z)-family transfer functions, for families whose measured members agree with each other.

WHY THIS IS NOT THE INTERVENTION THAT WAS ALREADY REJECTED. The project tested a per-family
calibration before and found nothing: 31.0% -> 31.1%, p=0.22 (family_calibration_indomain.json).
That test held out the whole CHEMISTRY CLUSTER, which for a family-keyed correction removes the
family itself, so it answered "can we predict a family we have never measured". The answer to that
is still no, and nothing here changes it -- every one of the paper's eleven conditional predictions
sits in such a family and keeps the global calibration.

The question this script answers is the other one: can we predict a NEW MEMBER of a family we HAVE
measured? That is the situation of every issued prediction -- when ErSbPt is predicted, DySbPt,
LuSbPt, TbSbPt and YSbPt are all measured, and that information genuinely exists. The protocol is
therefore leave-one-COMPOUND-out within the family.

WHICH kappa_BTE THE ARM IS FITTED ON, and why it is not the blind test. Fitting on the blind test's
`k_pred` -- the per-cluster surrogate -- was tried first, for consistency with
extend_blind_test.fit_cp, and it finds almost nothing: only Sn-Pt qualifies. The reason is
measurable. The surrogate differs from the published BTE it estimates by a factor spanning 1.98x
across the 32 compounds that have both, and that error is independent of family, so it washes the
family structure out:

    family   ratio spread from PUBLISHED BTE   from SURROGATE
    Ni-Sn                             1.33             2.40
    Sb-Pt                             1.30             1.61
    Ni-Sb                             2.57             2.64

The deployed route does not use the surrogate. make_paper_predictions.py applies the transfer
function to a PUBLISHED tier-1 BTE value, and the manuscript says so ("this model is a fallback,
not the deployed route", section2_methods). A calibration must be fitted on the same kind of input
it is applied to, so this script pairs published BTE against measurement directly. Pass
--source surrogate to reproduce the blind-test view for comparison.

WHEN A FAMILY QUALIFIES -- fixed BEFORE looking at which families win, because with fourteen
families and one threshold it would otherwise be trivial to select the answer:

  >= MIN_MEMBERS measured members          a factor fitted on one or two compounds is noise
  ratio spread < MAX_SPREAD                the members must agree; a family whose BTE/measured
                                           ratios span more than this has no single factor to learn
  fitted c < MAX_C                         the arm must correct downward; a family fitting c >= 1
                                           has reference data that contradicts the premise
  >= MIN_TEMPS distinct T, MIN_SPAN K,     only then is the exponent p identifiable; otherwise the
  and >= MIN_MEMBERS_P members             family gets a pure constant (p = 0) valid over the range
                                           its own members were measured in

Ni-Sb was once the case that proved the rule was a rule (most members, refused on ratio spread);
since 2026-09-19 every family carries a constant and Ni-Sb validates at 9.8%. History kept for the record.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy import stats

import curve_quality as CQ
import source_consensus as SC
import reference_choice as RC
import physical_domain as PD
import extend_blind_test as E
import shared_constant as SCN
import transfer_forms as TF
from transfer_forms import apply_family  # noqa: F401  -- re-exported for consumers
from compute_seed_averaged import fit_c_only
from make_paper_predictions import (CALC_DISAGREE_MAX, source_key, structure_status, vec, yzfam,
                                    yzfam_split)

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
BLIND = "data/exports/kappa_v2/target_blind_test.csv"
OUT = "data/exports/kappa_v2/family_calibration.json"
NOT_BTE = ("not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx")

# EVERY FAMILY CARRIES ITS OWN CONSTANT. Ruling of 2026-09-19: a family's correction comes from
# that family's own members, never from a global average, however few members it has. What
# changes with member count is not WHETHER a constant is fitted but HOW HONESTLY it can be tested:
#
#   n >= 2   leave-one-compound-out. At n = 2 that is fit-on-one, predict-the-other -- a single
#            pair, and the record says so (`validation = "leave-one-out (n=2)"`).
#   n = 1    the constant IS that compound's own. There is no held-out test; the record carries the
#            in-sample form error and `validation = "in-sample (single member)"`. The paper calls
#            these ANCHORED, never VALIDATED.
#
# The earlier floor of three existed because a constant on one or two compounds is noisy. It still
# is. The ruling is that a noisy constant from the right family beats a precise constant from the
# wrong population, and that the noise is disclosed rather than hidden behind a global fallback.
MIN_MEMBERS = 1       # every family with a measured member gets a constant
MIN_MEMBERS_LOO = 2   # below this there is nothing to hold out; validation is in-sample
MIN_MEMBERS_SPLIT = 2 # a DECLARED sub-family may go to two; see SPLIT_FAMILIES

# Families whose members are split by 4f occupancy of the X site.
#
# EMPTY, AND PERMANENTLY SO. THE EXPERIMENT IS CLOSED AND THE RESULT IS NEGATIVE.
# The evidence is produced by test_4f_split.py and reported in the paper. Do not reopen this
# without reading that script's docstring.
#
# WHAT WAS MEASURED (re-run 2026-09-19 under MIN_MEMBERS = 1, so every partition is scoreable).
# Partitioning by 4f occupancy moves the deployed constant by at most 0.03 (Ni-Sb:d 0.45 -> 0.48)
# and no issued prediction by more than 1.7%. Held-out error gets WORSE everywhere it can be
# recomputed: Ni-Sb 9.8 -> 11.4% (4f) / 12.6% (d), Sb-Pt 6.4 -> 16.6%, and Sb-Pd 32.7 -> 121.6%.
#
# WHY IT ONCE LOOKED PROMISING. Under the old rules Sb-Pd's 4f pair (ErSbPd, HoSbPd) appeared to
# reach 6.6% -- because both references then came from ONE laboratory (Mukhopadhyay 2018), and a
# lab agrees with itself. With HoSbPd on Mastronardi 1999 (its lowest-disorder sample, by rung 2)
# the same pair disagrees 2.9x and scores 121.6%. The apparent gain was one laboratory's two
# samples, not a 4f effect. test_4f_split.py records all of it; Table S2 quotes it.
SPLIT_FAMILIES: tuple[str, ...] = ()
MAX_SPREAD = 1.4      # BTE/measured ratios must agree to better than this
MIN_MEMBERS_P = 4     # two parameters need more than three compounds -- see below
MAX_C = 1.0           # a family arm must actually correct DOWNWARD -- see below
MIN_TEMPS = 5         # distinct temperatures needed before p is identifiable
MIN_SPAN = 300.0      # K -- and they must cover this much range
NO_CONSENSUS_GATE = False   # set by --no-consensus-gate, to measure what the gate is worth
# THE PHYSICAL-DOMAIN GATE IS OFF BY DEFAULT (author's decision, 2026-09-30). It dropped any compound
# measuring above its own calculation beyond 1.36x (TiSnPt, ZrSnPt, ErBiPd). Its criterion is the
# sign of the effect this paper measures, which the Methods themselves concede "could in principle
# manufacture that effect". Measured on identical code (test_inclusions.py): keeping the three
# changes none of the four issued predictions, does not hurt the 25 compounds already scored
# (18.3% -> 16.3%, p = 0.87), and turns Sn-Pt from a one-member fit into a two-paper test. Run with
# --domain-gate to restore it and measure the difference.
NO_DOMAIN_GATE = True
PICKS = []                  # calculations kept where every calculated temperature was contested
PICK_WHEN_ALL_CONTESTED = True   # False restores the old drop, for test_inclusions' "before" arm
REFERENCE_MODE = "single"   # "single" = one laboratory per compound; "geomean" = the old blend
CHOSEN_REFERENCE = {}       # compound -> (doi, why), filled by build_published for the report
CALC_DISAGREE = {}          # compound -> {T: spread}: temperatures where its calculations disagree > 2x
# HOW FAR A CALCULATED POINT MAY SIT FROM THE MEASUREMENT IT IS PAIRED WITH.
#
# This was a flat 100 K, which is not a tolerance but a licence: kappa_BTE falls roughly as 1/T, so
# 100 K at 300 K is a 33% error in the reference -- twice the 1.18x that independent calculations of
# the same compound disagree by. The damage was not diffuse. Every Ni-Sb compound has a calculation
# at 300 K only, so all 7 to 11 points of its measured 200-400 K curve paired to that ONE number,
# and the "temperature dependence of the correction" became the measurement's own fall divided by a
# constant. Fitting c(T/300)^p to that forces p negative by arithmetic, and the family's apparently
# anomalous exponent of -1.1 was an artefact of the pairing, not a property of the lattice.
#
# The replacement is relative and derived rather than chosen: allow a gap only while the calculation's
# own variation across it stays inside the calculation-to-calculation disagreement. With kappa ~ 1/T
# that makes the admissible fractional gap the same 0.15-0.18, so 15% of the measurement temperature
# -- +-45 K at 300 K, +-135 K at 900 K -- where the old rule allowed +-100 K at both.
T_MATCH_FRAC = 0.15   # unused by the published route now that the calculation is carried properly
EXTRAP_MAX = 2.5      # how far a single-temperature calculation may be carried, either way
T_MATCH = 100.0       # retained only for the surrogate route, which pairs on a shared grid


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def in_domain(c) -> bool:
    try:
        s = structure_status(c)
        return vec(c) == 18 and s[0] and not s[1]
    except Exception:  # noqa: BLE001
        return False


def bte_at(bagg: pd.Series, T: float):
    """The calculated kappa_L at temperature T, carried there the way the physics says it goes.

    THE PROBLEM THIS SOLVES. Published BTE results are overwhelmingly a single number at 300 K, while
    measurements are curves over hundreds of kelvin. The old code paired anything within a flat 100 K
    of a calculated point, so all 7 to 11 points of a Ni-Sb compound's 200-400 K curve were compared
    against the SAME 300 K number. The correction factor's temperature dependence then became the
    measurement's own fall divided by a constant, which forces the fitted exponent negative by
    arithmetic; Ni-Sb's apparently anomalous -1.1 was that, not a property of the lattice. Simply
    tightening the window does not fix it either -- it deletes the curves, and ErBiPd loses every
    point it has.

    THE PHYSICS. Above roughly half the Debye temperature, three-phonon Umklapp scattering dominates
    and the BTE solution goes as kappa_L ~ 1/T. That is not an assumption bolted on here, it is the
    regime these calculations are reported in, so carrying a calculated point to a nearby temperature
    by 1/T is how the calculation itself behaves. Where the calculation supplies two or more
    temperatures nothing is assumed at all: it is interpolated in log-log, which reproduces any
    power law the calculation actually has, 1/T or otherwise.

    THE BOUNDS. Extrapolation is capped at EXTRAP_MAX in either direction and never below 200 K,
    where the 1/T law fails as boundary and point-defect scattering take over and the calculation
    would have to be redone rather than stretched.
    """
    ts = bagg.index.values.astype(float)
    if T < 200.0:
        return None
    if len(ts) >= 2 and ts.min() <= T <= ts.max():
        return float(np.exp(np.interp(np.log(T), np.log(ts), np.log(bagg.values.astype(float)))))
    i = int(np.abs(ts - T).argmin())
    t0 = float(ts[i])
    if not (1.0 / EXTRAP_MAX) <= (T / t0) <= EXTRAP_MAX:
        return None
    return float(bagg.iloc[i]) * t0 / T          # Umklapp: kappa ~ 1/T


def resolve_calculations(bte: pd.DataFrame, keep) -> tuple[dict, list]:
    """One calculated value per compound and temperature.

    THE RULE. The median over every calculated row at each temperature. A temperature at which
    independent calculations -- one per source (make_paper_predictions.source_key: the DOI, else the
    dataset URL, so PhononDB counts) -- disagree by more than CALC_DISAGREE_MAX is
    contested and not used: a median of two numbers that far apart is not a calculation. That is
    the rule this pipeline has always applied.

    THE ONE CHANGE (author's decision, 2026-09-30). A compound whose EVERY calculated temperature
    is contested used to be dropped. It is now kept, with the one candidate value closest to the
    median calculation of its own family-mates at that temperature, read from their uncontested
    values only (a snapshot taken before any pick, so the order compounds are visited cannot matter).
    Only TmNiSb is affected: 9.85 kept, 0.54 rejected, the outlier an automated high-throughput
    database. A contested temperature inside an otherwise usable curve is still left out rather
    than filled -- filling it would splice one source's point into another source's curve. The rule
    reads calculations only, never a measurement.

    Returns ({compound: {T: kappa}}, [picks]) and records every contested temperature in
    CALC_DISAGREE for the artefact.
    """
    fam = {c: yzfam(c) for c in keep}
    med, contested, rows_by = {}, {}, {}
    for c in keep:
        bb = bte[bte.red == c]
        rows_by[c] = bb
        # ONE SOURCE = source_doi, else source_url, else source (FIXPASS S8). Grouping on source_doi
        # alone dropped every PhononDB calculation (no DOI; NaN keys vanish in a groupby), so a
        # PhononDB value that disagreed with a DOI'd calculation was never seen as a disagreement.
        per_src = bb.groupby([bb.TK.round(-1), source_key(bb)]).k.median()
        spread = per_src.groupby(level=0).agg(lambda s: s.max() / s.min() if s.min() > 0 else np.inf)
        bad = {float(T) for T in spread[spread > CALC_DISAGREE_MAX].index}
        if bad:
            CALC_DISAGREE[c] = {int(T): round(float(spread[T]), 2) for T in sorted(bad)}
            contested[c] = sorted(bad)
        allmed = bb.groupby(bb.TK.round(-1)).k.median()
        med[c] = {float(T): float(v) for T, v in allmed.items() if float(T) not in bad}
    base = {c: dict(v) for c, v in med.items()}
    picks = []
    for c, Ts in contested.items():
        if base[c] or not PICK_WHEN_ALL_CONTESTED:
            continue          # the compound still has usable temperatures: leave the contested out
        bb = rows_by[c]
        for T in Ts:
            at = bb[bb.TK.round(-1) == T]
            g = at.groupby(source_key(at)).k.median()
            mates = [base[o][T] for o in keep if o != c and fam[o] == fam[c] and T in base[o]]
            basis = "median calculation of its family-mates"
            if not mates:
                mates = [base[o][T] for o in keep if o != c and T in base[o]]
                basis = "median calculation of all in-domain compounds"
            target = float(np.median(mates))
            src = min(g.index, key=lambda s_: abs(np.log(g[s_] / target)))
            med[c][T] = float(g[src])
            picks.append(dict(compound=c, T=int(T), kept_source=src, kept_value=round(float(g[src]), 2),
                              alternatives={k: round(float(v), 2) for k, v in g.items() if k != src},
                              target=round(target, 2), basis=basis))
    return med, picks


_PREP_CACHE: dict = {}


def _prepare() -> dict:
    """The reference-independent half of build_published, computed once per process and setting.

    Everything here -- the deduplicated corpus, the curve-shape gate, the source-consensus gate, the
    in-domain compound set and the resolved calculations -- is independent of which laboratory each
    compound is referenced to, so the leave-one-out folds (fold_pairs) share it and only re-run the
    reference ladder. Keyed on the switches that change it. Callers never mutate the frames.
    """
    key = (NO_CONSENSUS_GATE, PICK_WHEN_ALL_CONTESTED)
    global PICKS
    if key in _PREP_CACHE:
        PICKS = list(_PREP_CACHE[key]["picks"])
        return _PREP_CACHE[key]
    # A PREPRINT AND THE PAPER IT BECAME ARE ONE CALCULATION. The calculation side below takes the
    # median over sources at each temperature; an undeduplicated pair counts one study twice and
    # drags that median toward it. make_paper_predictions.py has deduplicated since 2026-09-19;
    # the fit must see the same corpus the predictions are made from.
    import source_identity as SI
    tr = SI.dedupe(pd.read_csv(TRAIN, low_memory=False))
    tr["red"] = tr.formula.map(red)
    tr["t"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["TK"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    meth = tr.method.astype(str).str.lower()
    meas = tr[(tr.t == 0) & (tr.k > 0) & tr.TK.between(200, 1300)].dropna(subset=["red"])
    bte = tr[(tr.t == 1) & (tr.k > 0) & ~meth.str.contains(NOT_BTE, regex=True)
             & tr.TK.between(200, 1300)].dropna(subset=["red"])
    # REJECT MEASURED CURVES WHOSE SHAPE IS NOT LATTICE CONDUCTION.
    #
    # A reported kappa_L is a subtraction, not a direct measurement, and a wrong Lorenz number
    # shows up in the SHAPE of the curve. curve_quality tests exactly that and nothing else. It
    # replaces the blanket polymorph exclusion, which was a proxy for the same worry and a bad one:
    # that rule discarded five compounds and only MgAgSb deserved it, which this gate catches
    # directly at slope +4.65. Co-Sn's 153.9% error under the global arm traced to a single curve,
    # VCoSn falling as T^-5.3 from 12.80 to 0.36 W/m/K, which this removes.
    meas_ungated = meas
    rej = CQ.rejected_pairs(meas.rename(columns={"TK": "TK"}))
    if rej:
        before = len(meas)
        bad = np.array([(r, d) in rej for r, d in zip(meas.red, meas.source_doi)])
        meas = meas[~bad]
        print(f"curve-shape gate: dropped {len(rej)} unphysical curve(s), "
              f"{before - len(meas)} rows; compounds affected: "
              f"{sorted({c for c, _ in rej})}")
    # DISQUALIFY A LABORATORY THAT CONTRADICTS A CONSENSUS OF THREE OR MORE OTHERS.
    #
    # The curve-shape gate above cannot catch a Lorenz number that is wrong by a constant
    # factor, because scaling a curve leaves its shape correct. This does. See source_consensus
    # for why the test bed must have 4+ sources and why disqualification propagates to the thin
    # compounds where the source cannot be checked.
    # A DISQUALIFIED SOURCE IS RANKED LAST, NOT DELETED.
    #
    # Deleting it cost ScSbPd and YSbPd their only measurement and took the whole Sb-Pd family below
    # the three-member minimum. Demoting it instead keeps every compound that has any measurement at
    # all, while guaranteeing the source is never chosen anywhere an alternative exists. A compound
    # resting on a doubtful number is more use than a compound resting on none, provided the doubt
    # is recorded -- and it is, in the artifact, per compound.
    dq = {} if NO_CONSENSUS_GATE else SC.disqualified(meas, domain=in_domain)
    if dq:
        print(f"source-consensus gate: {len(dq)} source(s) ranked last, not deleted")
        for src, hits in dq.items():
            print("    " + src + " -- caught on "
                  + ", ".join(f"{c} at {r:.2f}x consensus" for c, r in hits))
    keep = {c for c in set(meas.red) & set(bte.red) if in_domain(c)}
    MED, PICKS = resolve_calculations(bte, keep)
    for p_ in PICKS:
        print(f"calculation kept for {p_['compound']} at {p_['T']} K: {p_['kept_value']} "
              f"(rejected {list(p_['alternatives'].values())}; {p_['basis']} = {p_['target']})")
    _PREP_CACHE[key] = dict(meas=meas, meas_ungated=meas_ungated, bte=bte, dq=dq, keep=keep,
                            med=MED, picks=list(PICKS))
    return _PREP_CACHE[key]


FOLD_CHOSEN: dict = {}      # (held, settings) -> {compound: (doi, why)} of that fold's ladder run


def build_published(held_out=()) -> pd.DataFrame:
    """Pair every in-domain measurement with the nearest published BTE point.

    Columns are named k_ref / k_pred / T / compound so the frame is a valid argument to
    extend_blind_test.fit_cp unchanged -- that function reads only those four.

    `held_out` (default none = the DEPLOYED pairs): compounds scored in a leave-one-out fold. Their
    references are chosen by the ladder WITHOUT the family-paper rung and they do not count as
    family members when the rung chooses their siblings' references (reference_choice.per_compound
    `held_out`; PREREG amendment A). A fold build never overwrites CHOSEN_REFERENCE. Use
    fold_pairs(held) rather than calling this with `held_out` directly: it caches per compound.
    """
    P = _prepare()
    meas, bte, dq, keep, MED = P["meas"], P["bte"], P["dq"], P["keep"], P["med"]
    held = tuple(sorted({str(h) for h in held_out}))
    chosen = (RC.per_compound(meas, keep, disqualified=dq, family_of=yzfam, held_out=held,
                              verbose=not held)
              if REFERENCE_MODE == "single" else {})
    if held:
        FOLD_CHOSEN[(held, _fold_settings())] = chosen
    else:
        global CHOSEN_REFERENCE
        CHOSEN_REFERENCE = chosen
    rows = []
    for c in sorted(keep):
        mm, bb = meas[meas.red == c], bte[bte.red == c]
        if REFERENCE_MODE == "single":
            # ONE LABORATORY, CHOSEN ON SAMPLE QUALITY BEFORE ANY CALCULATION IS READ.
            # Blending laboratories builds a curve nobody measured: each contributed a different
            # temperature range, so the blend's T-dependence is an artefact of who overlapped where,
            # and the calibration inherits every contributor's uncertainty at once. See
            # reference_choice for the ladder and why no rung of it may read kappa_BTE.
            src, _why = chosen[c]
            mm = mm[mm.source_doi.astype(str) == src]
            agg = mm.groupby(mm.TK.round(-1)).k.median()
        else:
            # the superseded blend, kept runnable so the change can be measured
            agg = (mm.groupby([mm.TK.round(-1), "source_doi"]).k.median()
                     .groupby(level=0).apply(stats.gmean))
        # THE CALCULATION SIDE GETS THE SAME CARE AS THE MEASUREMENT SIDE.
        #
        # This previously took whichever calculated row happened to sort first at the nearest
        # temperature, so the reference depended on row order. TiSnPt has EIGHT independent
        # calculations spanning 11.18 to 27.23 W/m/K at 300 K, and the arbitrary pick made it look
        # as though the compound had been measured at 1.48x its own calculation -- an inversion the
        # transfer function cannot represent, since it may only correct downward. Collapsing the
        # calculations to a median per temperature first makes the reference independent of row
        # order and robust to one broken calculation, which is the same protection the measured side
        # gets from choosing a single well-characterised laboratory.
        # A MEDIAN OF CALCULATIONS THAT DISAGREE BY MORE THAN 2x IS NOT A CALCULATION. The issuing
        # script refuses such a compound ("CALCULATIONS DISAGREE ... no value to calibrate"); the
        # same rule applies here, at each temperature, or the family would be fitted on a number the
        # paper elsewhere declines to use. Dedupe exposed this: TmNiSb has independent calculations
        # of 9.85 and 0.54 W/m/K at 300 K, and a duplicated preprint had been out-voting the
        # 0.54 by accident. The rule reads calculation against calculation only, never a
        # measurement, so it cannot select for agreement with the reference. Temperatures where the
        # calculations disagree contribute no pair; a compound left with none is reported.
        # See resolve_calculations: identical to the old inline rule except that a compound whose
        # EVERY calculated temperature is contested keeps one calculation instead of being dropped.
        bagg = pd.Series(MED.get(c, {}), dtype=float).sort_index()
        if bagg.empty:
            continue   # no calculated temperature at all
        for T, kref in agg.items():
            kp = bte_at(bagg, float(T))
            if kp is None:
                continue
            rows.append(dict(compound=c, T=float(T), k_ref=float(kref), k_pred=float(kp)))
    out = pd.DataFrame(rows, columns=["compound", "T", "k_ref", "k_pred"])
    # A REAL POLYCRYSTAL CANNOT BEAT THE PERFECT CRYSTAL -- see physical_domain.
    return out if NO_DOMAIN_GATE else PD.apply(out)


def _fold_settings() -> tuple:
    return (NO_CONSENSUS_GATE, PICK_WHEN_ALL_CONTESTED, NO_DOMAIN_GATE, REFERENCE_MODE)


_FOLD_CACHE: dict = {}


def fold_pairs(held: str) -> pd.DataFrame:
    """The published-route pairs AS THEY STAND IN THE LEAVE-ONE-OUT FOLD THAT SCORES `held`.

    THE LEAK THIS CLOSES (audit 2026-10-05; PREREG amendment A). The family-paper rung qualifies a
    paper only if it covers two or more family members, so on the deployed pairs the held-out
    compound's own membership can pull a sibling onto the held-out compound's laboratory -- and the
    sibling is then the training data the held-out compound is scored against. Sb-Pd: HoSbPd is
    referenced to 10.1088/1361-6463/aac567 only because ErSbPd is also in that paper; without ErSbPd
    the ladder picks 10.1063/1.123596, 2.85x higher. Here the ladder runs as if `held` had not been
    measured for the family rung: its siblings' references are chosen on (members - held) WITH the
    family rung, and `held`'s own reference by the ladder WITHOUT it.

    Every compound's rows are returned (a frame like build_published()), with `fam` added. Cached per
    held-out compound and setting; a copy is returned so callers may add columns.
    """
    key = (str(held), _fold_settings())
    if key not in _FOLD_CACHE:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            f = build_published(held_out=(str(held),))
        f["fam"] = f.compound.map(yzfam)
        _FOLD_CACHE[key] = f
    return _FOLD_CACHE[key].copy()


def fold_reference(held: str):
    """(doi, why) the fold that scores `held` references `held` to (ladder without the family rung)."""
    fold_pairs(held)
    return FOLD_CHOSEN.get(((str(held),), _fold_settings()), {}).get(str(held))


def published_calc_at(compound: str, T: float):
    """The published-route median calculation of an in-domain compound, carried to T (bte_at), or
    None -- the same calculated value build_published pairs with a measurement at T."""
    med = _prepare()["med"].get(compound)
    if not med:
        return None
    return bte_at(pd.Series(med, dtype=float).sort_index(), float(T))


def measured_frame(gated: bool = True):
    """(measured rows, disqualified sources) after the same eligibility gates the published route
    applies: deduplicated corpus, tier 0, kappa > 0, 200-1300 K, and (gated=True) the curve-shape
    gate; consensus disqualification returned for the ladder to rank last. All compounds, not only
    in-domain ones. For the blind test's single-reference rule (run_target_blind_test; PREREG
    amendment A). gated=False returns the rows before the curve-shape gate, for a caller that ranks
    rejected curves last instead of deleting the compound."""
    P = _prepare()
    return (P["meas"] if gated else P["meas_ungated"]).copy(), dict(P["dq"])


def build_surrogate() -> pd.DataFrame:
    B = pd.read_csv(BLIND)
    d = B[B.klass == "half"].copy()
    d = d[d.compound.map(in_domain)]
    return d[["compound", "T", "k_ref", "k_pred"]].copy()


def per_compound_ape(d: pd.DataFrame, k: np.ndarray) -> pd.Series:
    """Median APE per compound. Per compound, never per row -- a compound with 81 temperature
    points would otherwise outvote one with a single point."""
    return (d.assign(_a=100 * np.abs(k - d.k_ref.values) / d.k_ref.values)
             .groupby("compound")._a.median())


def loo_within_family(g: pd.DataFrame, c_g: float, p_g: float, fit_p: bool):
    """Leave-one-COMPOUND-out inside one family: fit on the others, score the held-out compound.

    The global arm is scored on the SAME held-out compounds, so the two lists are paired and their
    difference is meaningful. The global arm applied to `held` is the shared constant refitted
    without `held` (shared_constant.shared_cp(exclude=[held]); PREREG_shared_constant_on_calculations
    rule 4); c_g, p_g are kept in the signature for callers but no longer used for scoring.
    """
    fam_e, glob_e = [], []
    for held in sorted(g.compound.unique()):
        te, tr = g[g.compound == held], g[g.compound != held]
        if len(tr) < 4 or tr.compound.nunique() < 2:
            continue
        c, p = (E.fit_cp(tr) if fit_p else (fit_c_only(tr), 0.0))
        if not np.isfinite(c):
            continue
        fam_e.append(float(per_compound_ape(te, te.k_pred.values
                                            * E.apply_cp(te["T"].values, c, p)).iloc[0]))
        cg_h, pg_h = SCN.shared_cp(exclude=[held])
        glob_e.append(float(per_compound_ape(te, te.k_pred.values
                                             * E.apply_cp(te["T"].values, cg_h, pg_h)).iloc[0]))
    return fam_e, glob_e



FORMS = ("GLOBAL", "CONSTANT", "SHAPE", "SHAPE_NOCAP", "CURVE", "MATTHIESSEN")

# MATTHIESSEN grid. Coarse deliberately: this is fitted on three to seven compounds, and a fine
# grid would buy precision the sample cannot support while making the fit slower to audit.
_MATT_F = np.linspace(0.30, 1.20, 19)
_MATT_K = (2.0, 4.0, 6.0, 9.0, 13.0, 18.0, 25.0, 40.0, 70.0, 150.0, 400.0)


def downward_ok(form, par, trn=None) -> bool:
    """The correction must still reduce the calculated value; otherwise the premise is inverted.

    For the rescaling forms that is simply c < 1. Sn-Pt fitting c = 1.000 is what this catches: its
    one member with both sides, ZrSnPt, has a calculation BELOW its measurement, which is a
    reference defect rather than a family whose physics differs.

    MATTHIESSEN needs a different test, and applying the c < 1 rule to it was a bug that removed the
    form from every candidate list without saying so. Its prefactor is not the total correction --
    the series term drags the result below f*k -- so the honest test is the premise itself,
    evaluated on the family's own data. transfer_forms.reduces_everywhere does exactly that.
    """
    if form != "MATTHIESSEN":
        return float(par[0]) < MAX_C - 1e-9
    if trn is None or not len(trn):
        return True
    return TF.reduces_everywhere(form, par, trn["T"].values, trn.k_pred.values)


def _per_compound_logerr(trn, pred):
    """Median over compounds of each compound's median |log10 error| -- the project's objective."""
    e = np.abs(np.log10(np.maximum(pred, 1e-9)) - np.log10(trn.k_ref.values))
    u = trn.compound.values
    return float(np.median([np.median(e[u == x]) for x in np.unique(u)]))


def _grid_argmin_mid(cs, scores) -> float:
    """The grid value minimising `scores`, with exact ties resolved to the MIDDLE of the tied range.

    A three-member family fits each held-out constant on two compounds, and the per-compound median
    of two errors is their mean: flat for every c between the two members' own constants. np.argmin
    then returned whichever grid point a 1e-16 rounding residue favoured, so the SAME data gave
    HoSbPd's held-out error as 9.4% or 25.6% depending on whether the cap's min() was in the
    expression. The median of two numbers is their midpoint; the tie is resolved the same way, to
    the geometric centre of the tied range. That centre is returned itself, not snapped back to the
    grid: two adjacent tied grid points (Fe-Sb's 0.39 and 0.40) are exactly equidistant from it, so
    snapping would reintroduce the coin-flip. A single best grid point is returned unchanged.
    """
    cs, scores = np.asarray(cs, dtype=float), np.asarray(scores, dtype=float)
    tied = cs[scores <= np.nanmin(scores) + 1e-9]
    return float(np.round(np.sqrt(tied.min() * tied.max()), 4))


def fit_form(form, trn, c_g, p_g):
    """Fit one candidate form on `trn`. Returns its parameters, or None if it cannot be fitted.

    The five forms, and why each is offered:
      GLOBAL       the shared constants (shared_constant.py, fitted on published calculations;
                   refitted without the held-out compound when scored) -- what a family arm must beat
      CONSTANT     family magnitude, no temperature term; one parameter
      SHAPE        family magnitude on the GLOBAL temperature exponent; also one parameter, but it
                   keeps the temperature behaviour of the shared fit across all families, which
                   is why it wins for families whose own span is too short to fit p themselves
      CURVE        family magnitude AND exponent; two parameters, needs real temperature coverage
      MATTHIESSEN  1/k = 1/(f*k_BTE) + 1/K0 -- boundary scattering in series with the intrinsic
                   conductivity, the standard physical picture for a sintered pellet. It is the
                   only form here that is not a pure rescaling, and it wins where a family's high
                   and low conductivity members need different effective factors.
    """
    if form == "GLOBAL":
        return c_g, p_g
    if form in ("CONSTANT", "SHAPE", "SHAPE_NOCAP"):
        # one parameter on the c grid; CONSTANT holds p at zero, the SHAPE forms at the global exponent
        p = 0.0 if form == "CONSTANT" else p_g
        T, kb = trn["T"].values.astype(float), trn.k_pred.values
        if not len(trn):
            return None
        # FITTED INSIDE THE PHYSICAL RANGE, c < MAX_C. Scanning past it let the cap make every
        # c >= 1 predict the same (the calculation itself), so the flat tie ran up to 1.20 and its
        # midpoint landed at or above 1, which downward_ok then rejected -- Sn-Pt lost its family
        # arm to a tie, not to its data. A correction that must reduce is fitted where it reduces.
        cs = E.CS[E.CS < MAX_C - 1e-9]
        scores = [_per_compound_logerr(trn, apply_family(form, (c, p), T, kb)) for c in cs]
        return _grid_argmin_mid(cs, scores), p
    if form == "CURVE":
        c, p = E.fit_cp(trn)
        return (float(c), float(p)) if np.isfinite(c) else None
    if form == "MATTHIESSEN":
        T, kb = trn["T"].values.astype(float), trn.k_pred.values
        best, bp = np.inf, None
        for f in _MATT_F:
            for k0 in _MATT_K:
                s = _per_compound_logerr(trn, apply_family("MATTHIESSEN", (f, k0), T, kb))
                if s < best:
                    best, bp = s, (float(f), float(k0))
        return bp
    raise ValueError(form)


def curve_identifiable(g) -> bool:
    """Is there enough temperature range to fit an exponent at all?

    Fitting p on a 200 K span is fitting noise: Ni-Sb spans 200-400 K, Sb-Pd 200-360 K. The form is
    simply not offered to such a family, rather than offered and then regretted.
    """
    return (g.compound.nunique() >= MIN_MEMBERS_P
            and int(g["T"].round(-1).nunique()) >= MIN_TEMPS
            and float(g["T"].max() - g["T"].min()) >= MIN_SPAN)


def score_form(form, g, c_g, p_g, exclude=(), fold=None):
    """Leave-one-compound-out error for one form inside one family.

    `exclude` names members dropped from the FIT (robust outliers). They are never dropped from the
    SCORE: a family must still predict its own outlier, and the cost of doing so is reported.

    GLOBAL is scored leave-one-out too: the shared constant applied to `held` is refitted on the
    published calculation pairs without `held` (shared_constant.shared_cp(exclude=[held]);
    PREREG_shared_constant_on_calculations rule 4). The family forms scored on `held` carry the shared
    exponent refitted without `held` as well (shared_constant.shared_p_excluding([held])); the
    `p_g` argument is the deployed p and is used only for the deployed fit, never for a score.

    `fold` (held -> pairs frame, normally fold_pairs): REFERENCES RE-CHOSEN INSIDE EACH FOLD
    (PREREG amendment A). The held-out compound is scored on its own fold's reference (ladder without
    the family rung) and the family-mates are fitted on theirs (family rung over members - held);
    the shared constant / exponent then come from shared_constant.shared_cp_in_fold(held), fitted on
    that fold's pairs. Members are those of `g`. fold=None scores `g` as given (the old, leaky
    behaviour, kept for callers that pass frames whose references were forced, e.g. a snapshot).
    """
    errs = []
    members = set(g.compound.unique())
    for held in sorted(members):
        if fold is not None:
            gf = fold(held)
            gf = gf[gf.compound.isin(members)]
            te = gf[gf.compound == held]
            trn = gf[(gf.compound != held) & (~gf.compound.isin(exclude))]
            cp_glob = SCN.shared_cp_in_fold(held)
        else:
            te = g[g.compound == held]
            trn = g[(g.compound != held) & (~g.compound.isin(exclude))]
            cp_glob = SCN.shared_cp(exclude=[held])
        if te.empty:
            continue
        # A held-out fit may rest on ONE compound when the family has exactly two. That fold is a
        # single pair and is labelled as such upstream; the rows floor keeps a fit from resting on
        # fewer points than it has parameters worth of freedom.
        if len(trn) < 4 or trn.compound.nunique() < 1:
            continue
        cp = (cp_glob if form == "GLOBAL"
              else fit_form(form, trn, c_g, float(cp_glob[1])))
        if cp is None or not downward_ok(form, cp, trn):
            return None
        k = apply_family(form, cp, te["T"].values, te.k_pred.values)
        errs.append(float(np.median(100 * np.abs(k - te.k_ref.values) / te.k_ref.values)))
    return float(np.median(errs)) if errs else None


# SHAPE_NOCAP since 2026-10-01 (author's decision), SHAPE before. Still one parameter per family on
# the global exponent; only the cap is gone. Sn-Pt and Bi-Pd measured curves fall more slowly than
# the calculation (T^-0.77) or not at all, and the cap stopped all correction above ~370-510 K, so
# their members agreed on c and still missed by 30%. Without it: Sn-Pt 32 -> 21%, Bi-Pd 30 -> 25%,
# Fe-Sb 26 -> 21%; Co-Sb 10.5 -> 19%; pooled unchanged (test_family_form_nested.py).
ADOPTED_FORM = "SHAPE_NOCAP"


def select_form(g, c_g, p_g, exclude=(), fold=None):
    """Fit the ONE pre-chosen form. Every candidate is still scored, but only for the record.

    THE FORM IS NOT CHOSEN PER FAMILY, AND THAT DECISION IS THE RESULT OF MEASURING WHAT CHOOSING
    COSTS. Selecting the best of five forms per family looks strong on the numbers you can see and
    performs worse on the compounds you actually want to predict. Ni-Sb is the demonstration: its
    best-of-five scores 24.5% when the form has seen the compound it is judged on, and 40.0% under
    nested selection, against 32.2% for doing nothing at all. With three to seven members per
    family, which form "wins" is mostly which compounds happened to be measured, so choosing buys
    noise and pays for it out of sample.

    SHAPE is fixed for every qualifying family instead: the family's own magnitude carried on the
    GLOBAL temperature exponent. One parameter per family. The physical claim is narrow and
    testable -- families differ in how much conductivity their microstructure costs, while how that
    cost varies with temperature is common to all of them -- and the exponent it borrows is the
    best-supported number in the model, fitted across every family's calculations rather than three
    compounds (since 2026-10-05 on published calculations vs measurements, not the blind test).

    Measured over the seven families with three or more members, against the global arm:
        Ni-Sn 41.2 -> 17.1    Sb-Pd 43.0 -> 19.4    Bi-Pd 29.1 -> 19.4    Sb-Pt 23.7 -> 19.6
        Sn-Pt 37.4 -> 32.3    Ni-Sb 32.2 -> 32.4    Co-Sb 16.8 -> 20.7
        median 32.2 -> 19.6, better in five of seven, Wilcoxon p = 0.078

    The two families it does not help are kept and reported rather than dropped. A rule that
    improved every family would be a rule fitted to the families.
    """
    cands = [f for f in FORMS if f != "CURVE" or curve_identifiable(g)]
    scores = {f: score_form(f, g, c_g, p_g, exclude, fold=fold) for f in cands}
    scores = {f: v for f, v in scores.items() if v is not None}
    if ADOPTED_FORM not in scores:
        return dict(form="GLOBAL", optimistic=None, nested=None,
                    global_arm=scores.get("GLOBAL"), scores=scores)
    return dict(form=ADOPTED_FORM, optimistic=scores[ADOPTED_FORM],
                # no nested figure is reported because nothing is selected: with the form fixed in
                # advance there is no choice for an inner loop to make, and quoting one would imply
                # a selection step that does not happen.
                nested=None, global_arm=scores.get("GLOBAL"), scores=scores,
                best_available=min(scores, key=scores.get),
                best_available_ape=min(scores.values()))


def _members_own(g, c_g, p_g) -> dict:
    """Each member's constant fitted on ITS OWN reference, and the in-sample form error it gives.

    This is the 'calibrated on its own paper' number. It is not a held-out figure and is never
    quoted as one; it answers 'how well does the form fit this compound's own measurement', which
    is the question for a compound whose only paper is the one in hand. Beside it the record keeps
    the held-out figure, which answers 'how well do its siblings predict it'. Both are reported.
    """
    out = {}
    for c in sorted(g.compound.unique()):
        te = g[g.compound == c]
        cp = fit_form(ADOPTED_FORM, te, c_g, p_g) if len(te) >= 4 else None
        if cp is None:
            out[c] = dict(own_c=None, own_ape=None, n_rows=int(len(te)), reference=CHOSEN_REFERENCE.get(c, ("", ""))[0])
            continue
        k = apply_family(ADOPTED_FORM, cp, te["T"].values, te.k_pred.values)
        out[c] = dict(own_c=round(float(cp[0]), 3),
                      own_ape=round(float(np.median(100 * np.abs(k - te.k_ref.values) / te.k_ref.values)), 1),
                      n_rows=int(len(te)), reference=CHOSEN_REFERENCE.get(c, ("", ""))[0])
    return out


def robust_outliers(g):
    """Members more than 3 robust deviations from the family's median ratio, in log space."""
    r = g.loc[g.assign(_x=(g["T"] - 300).abs()).groupby("compound")._x.idxmin()]
    ratio = (r.k_pred / r.k_ref).values
    names = r.compound.values
    if len(ratio) < 4:
        return {}
    lr = np.log(ratio)
    med = np.median(lr)
    mad = np.median(np.abs(lr - med)) or 1e-9
    z = np.abs(lr - med) / (1.4826 * mad)
    return {n: float(zz) for n, zz, in zip(names, z) if zz > 3.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("published", "surrogate"), default="published",
                    help="which kappa_BTE to calibrate; 'published' is the deployed route")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--domain-gate", action="store_true",
                    help="restore the physical-domain gate (off by default), to measure it")
    ap.add_argument("--reference-mode", choices=("single", "geomean"), default="single",
                    help="one laboratory per compound, or the superseded blend across labs")
    ap.add_argument("--no-consensus-gate", action="store_true",
                    help="skip the source-consensus gate, to measure what it is worth")
    a = ap.parse_args()
    global NO_CONSENSUS_GATE
    NO_CONSENSUS_GATE = a.no_consensus_gate
    global REFERENCE_MODE
    REFERENCE_MODE = a.reference_mode
    global NO_DOMAIN_GATE
    NO_DOMAIN_GATE = not a.domain_gate

    d = build_published() if a.source == "published" else build_surrogate()
    d["fam"] = d.compound.map(yzfam)

    # THE DECLARED SPLIT, APPLIED WHERE IT WAS SHOWN TO WORK AND NOWHERE ELSE.
    #
    # Within a bonding family the X site can be a lanthanide or a group-3 transition metal (Sc, Y),
    # and those two carry different 4f occupancy. Sb-Pd is the case where that difference dominates:
    # its lanthanides ErSbPd and HoSbPd agree to 1.10x (ratios 0.234 and 0.213) while ScSbPd sits
    # outside at 0.315, and the family's prediction target TmSbPd is itself a lanthanide.
    #
    # THE SPLIT WAS TESTED AND REJECTED. NOTHING IS ADOPTED; SPLIT_FAMILIES IS EMPTY.
    # Splitting removes members, and a family constant's error scales with 1/(members), so a split
    # could only pay where the homogeneity gained beats the sample lost. Measured by
    # test_4f_split.py, which writes data/exports/kappa_v2/split_4f_test.json:
    #
    #     family     c unsplit -> split     held-out error     verdict
    #     Ni-Sb        0.450 -> 0.440       9.7% -> 11.4%      worse
    #     Sb-Pt        0.590 -> 0.580       6.4% -> 16.6%      worse
    #     Sb-Pd        0.260 -> 0.260       cannot be scored   two members; see SPLIT_FAMILIES
    #
    # The constant is what gets deployed, and it does not move. An earlier version of this comment
    # recorded Ni-Sb as 9.7% -> 19.0% and Sb-Pd as ADOPTED. Neither reproduced: the run gives 11.4%,
    # and nothing was ever adopted because the tuple above has always been empty.
    if SPLIT_FAMILIES:
        sub = d.compound.map(yzfam_split)
        d["fam"] = np.where(d.fam.isin(SPLIT_FAMILIES) & sub.notna(), sub, d.fam)
        print(f"declared sub-family split applied to: {', '.join(SPLIT_FAMILIES)}")

    # THE GLOBAL (SHARED) ARM IS FITTED ON PUBLISHED CALCULATIONS VS MEASUREMENTS.
    #
    # Since 2026-10-05 (paper/evidence/PREREG_shared_constant_on_calculations.md) the shared (c, p)
    # is fitted on the published-route pairs of build_published() with no model in it -- it used to
    # be fitted on the ML model's blind-test predictions, where p moved with the model's seed. One
    # module supplies it (shared_constant.py). The deployed pair goes into the json `global` entry and
    # sets the family forms' exponent; inside the family leave-one-out the global arm applied to a
    # held-out compound is refitted without that compound (score_form).
    c_g, p_g = SCN.shared_cp()
    print(f"source={a.source}   in-domain compounds {d.compound.nunique()}   "
          f"families {d.fam.nunique()}   paired rows {len(d)}")
    print(f"global arm (shared constant, published calculations vs measurements)  "
          f"c={c_g:.3f}  p={p_g:.3f}")
    if (a.source == "published" and not a.domain_gate and a.reference_mode == "single"
            and not a.no_consensus_gate):
        # shared_constant builds its pairs through its own import of this module; under the default
        # rules they must be exactly the pairs this run built, or the two disagree on the data.
        c_r, p_r = E.fit_cp(d)
        assert abs(c_r - c_g) < 1e-9 and abs(p_r - p_g) < 1e-9, (c_r, p_r, c_g, p_g)
    print()

    fams = {}
    for fam, g in d.groupby("fam"):
        n = g.compound.nunique()
        temps = int(g["T"].round(-1).nunique())
        t_lo, t_hi = float(g["T"].min()), float(g["T"].max())
        ratio = g.loc[g.assign(_d=(g["T"] - 300).abs())
                       .groupby("compound")._d.idxmin()].set_index("compound")
        ratio = ratio.k_pred / ratio.k_ref
        spread = float(ratio.max() / ratio.min()) if len(ratio) > 1 else float("inf")
        rec = dict(n_members=int(n), members=sorted(g.compound.unique()),
                   ratio_spread=round(spread, 2), n_temps=temps,
                   t_lo=round(t_lo), t_hi=round(t_hi), t_span=round(t_hi - t_lo),
                   form="GLOBAL", c=round(float(c_g), 3), p=round(float(p_g), 3), adopted=False)

        # A DECLARED sub-family may go to two members; every other family still needs three. At two,
        # leave-one-out fits on ONE compound and the score is a single number, so the weakness is
        # recorded on the record itself rather than left for a reader to infer from n_members.
        is_split = ":" in str(fam)
        floor = MIN_MEMBERS_SPLIT if is_split else MIN_MEMBERS
        if is_split:
            rec["declared_split"] = True
            rec["split_of"] = str(fam).split(":")[0]
        if n < floor:
            rec["why_not"] = f"only {n} measured member(s); nothing to fit"
            fams[fam] = rec
            continue

        # A SINGLE-MEMBER FAMILY IS ANCHORED, NOT VALIDATED. Its constant is fitted on its one
        # compound and there is nothing to hold out. The record carries the in-sample form error --
        # how well the form fits the compound with its own best constant -- and says plainly that no
        # held-out figure exists. It is adopted because the ruling is that a family's own member is
        # the right anchor for that family; it is never quoted as a validation result.
        if n < MIN_MEMBERS_LOO:
            cp = fit_form(ADOPTED_FORM, g, c_g, p_g)
            if cp is None or not downward_ok(ADOPTED_FORM, cp, g):
                rec["why_not"] = ("single member and its own constant does not correct downward; "
                                  "suspect reference data")
                fams[fam] = rec
                continue
            k = apply_family(ADOPTED_FORM, cp, g["T"].values, g.k_pred.values)
            ins = float(np.median(100 * np.abs(k - g.k_ref.values) / g.k_ref.values))
            rec.update(form=ADOPTED_FORM, params=[round(float(x), 4) for x in cp],
                       c=round(float(cp[0]), 3), p=round(float(cp[1]), 3), adopted=True,
                       validation="in-sample (single member)", loo_family_ape=None,
                       insample_ape=round(ins, 1), valid_t_lo=round(t_lo), valid_t_hi=round(t_hi),
                       weakness="one member: the constant is this compound's own; nothing is held out")
            fams[fam] = rec
            continue
        rec["validation"] = f"leave-one-out (n={n})"
        if n == 2:
            rec["weakness"] = ("two members: each held-out fit rests on ONE compound, so the quoted "
                               "error is the median of two numbers from a single pair")
        if is_split and n < 3:
            rec["weakness"] = (f"{n} members: leave-one-out fits on {n - 1} compound(s), so the "
                               f"quoted error is the median of {n} numbers")

        # Robust outliers are reported and dropped from the FIT, never from the SCORE.
        out = robust_outliers(g)
        if out:
            rec["outliers"] = {k: round(v, 1) for k, v in out.items()}

        # THE FORM IS SELECTED, NOT ASSUMED. The previous rule admitted a family only if its
        # ratios spanned less than 1.4x -- a range statistic one member destroys, and one that says
        # nothing about whether a family calibration actually predicts better. Measured properly, a
        # family arm beat the global one in seven of eight families while that rule adopted two.
        # references re-chosen inside every leave-one-out fold (PREREG amendment A); the surrogate
        # source has no reference ladder, and the blend mode has no family rung, so they score g
        sel = select_form(g, c_g, p_g, exclude=set(out),
                          fold=(fold_pairs if a.source == "published" else None))
        rec["candidates"] = {k: round(v, 1) for k, v in sel["scores"].items()}
        rec["loo_optimistic"] = None if sel["optimistic"] is None else round(sel["optimistic"], 1)
        rec["loo_nested"] = None if sel["nested"] is None else round(sel["nested"], 1)
        rec["loo_global_ape"] = None if sel["global_arm"] is None else round(sel["global_arm"], 1)

        if sel["form"] == "GLOBAL":
            rec["why_not"] = "the pre-chosen form could not be fitted for this family"
            fams[fam] = rec
            continue
        # A family whose own arm is WORSE than global is still adopted and still reported. The form
        # is fixed in advance, so keeping it only where it wins would reintroduce exactly the
        # selection this design exists to avoid -- and the families it does not help are the
        # evidence the rule was applied rather than tuned.
        rec["beats_global"] = (sel["global_arm"] is None
                               or sel["optimistic"] <= sel["global_arm"])

        cp = fit_form(sel["form"], g[~g.compound.isin(out)], c_g, p_g)
        if cp is None or not downward_ok(sel["form"], cp, g[~g.compound.isin(out)]):
            rec["why_not"] = (f"fitted c = {cp[0]:.3f} >= {MAX_C}: the calculation does not sit "
                              "above the measurement for this family, which inverts the premise "
                              "of the transfer function -- suspect reference data")
            fams[fam] = rec
            continue

        rec["best_available"] = sel.get("best_available")
        rec["best_available_ape"] = (None if sel.get("best_available_ape") is None
                                     else round(sel["best_available_ape"], 1))
        rec.update(form=sel["form"], params=[round(float(x), 4) for x in cp],
                   c=round(float(cp[0]), 3), p=round(float(cp[1]), 3), adopted=True,
                   # a family's arm is valid only over the temperatures its own members cover
                   valid_t_lo=round(t_lo), valid_t_hi=round(t_hi),
                   loo_family_ape=rec["loo_optimistic"],
                   members_own=_members_own(g, c_g, p_g))
        # THE SPREAD OF THE MEMBERS' OWN CONSTANTS IS THE FAMILY'S HONEST UNCERTAINTY. Ruling of
        # 2026-09-19: a compound with one paper is calibrated on that paper and the record says so.
        # Each member therefore carries its own constant (fitted on its own reference) and the
        # in-sample form error that constant achieves. The family constant is what a NEW member
        # gets; the spread of the members' own constants is how far that can be wrong. Sb-Pd's
        # members want 0.26 / 0.33 / 0.75 -- a 2.9x spread -- and any prediction from the family
        # constant must carry it.
        own = [m["own_c"] for m in rec["members_own"].values() if m["own_c"] is not None]
        if own:
            rec["c_members_min"], rec["c_members_max"] = round(min(own), 3), round(max(own), 3)
            rec["c_members_spread"] = round(max(own) / min(own), 2)
        fams[fam] = rec

    print(f"{'family':<9}{'n':>3}{'form':>10}{'c':>8}{'p':>8}{'valid T':>12}"
          f"{'nested':>9}{'optim':>8}{'global':>8}   note")
    print("-" * 108)
    for f, r in sorted(fams.items()):
        vt = (f"{r['valid_t_lo']}-{r['valid_t_hi']}" if r["adopted"] else "-")
        def _n(x):
            return f"{x:.1f}" if isinstance(x, (int, float)) else "-"
        print(f"{f:<9}{r['n_members']:>3}{r['form']:>10}{r['c']:>8.3f}{r['p']:>8.3f}{vt:>12}"
              f"{_n(r.get('loo_nested')):>9}{_n(r.get('loo_optimistic')):>8}"
              f"{_n(r.get('loo_global_ape')):>8}   "
              f"{'ADOPTED' if r['adopted'] else str(r.get('why_not', ''))[:44]}")
        if r.get("outliers"):
            print(f"{'':>12}outlier(s) dropped from the fit, still scored: {r['outliers']}")

    gap = [(f, r['loo_nested'], r['loo_optimistic']) for f, r in fams.items()
           if r.get('loo_nested') is not None and r.get('loo_optimistic') is not None]
    if gap:
        print()
        print("  nested vs optimistic -- the size of the self-selection:")
        for f, nn, oo in sorted(gap):
            print(f"     {f:<9} nested {nn:>5.1f}%   optimistic {oo:>5.1f}%   gap {nn - oo:+.1f} pp")

    worse = [f for f, r in fams.items() if r["adopted"]
             and r.get("loo_family_ape") is not None and r.get("loo_global_ape") is not None
             and r["loo_family_ape"] > r["loo_global_ape"]]
    if worse:
        print()
        print(f"  WARNING: adopted but WORSE than global under leave-one-out: {worse}")

    R = {"_generated_by": "family_calibration.py",
         "_generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "source": a.source,
         "rule": dict(min_members=MIN_MEMBERS, max_ratio_spread=MAX_SPREAD,
                      min_temps_for_p=MIN_TEMPS, min_span_for_p=MIN_SPAN, t_match_K=T_MATCH),
         "protocol": "leave-one-COMPOUND-out within the family; the global arm is scored on the "
                     "same held-out compounds so the two are paired, with the shared constant "
                     "refitted without each held-out compound",
         "global_definition": "shared (c, p) fitted on published calculations vs measurements "
                              "(shared_constant.py; PREREG_shared_constant_on_calculations.md)",
         "global": dict(c=round(float(c_g), 3), p=round(float(p_g), 3)),
         "tiers": {"A": "family c and p", "B": "family c, global p", "C": "global c and p"},
         # compounds whose independent calculations disagree by more than CALC_DISAGREE_MAX at some
         # temperature: those temperatures contributed no pair. A compound absent from every family
         # above because of this is listed here so the paper can say why it is not scored.
         "calc_disagree_max": CALC_DISAGREE_MAX,
         "calc_disagree": {c: dict(temperatures=v,
                                   scored=bool(c in set(d.compound)),
                                   family=yzfam(c))
                           for c, v in sorted(CALC_DISAGREE.items())},
         # compounds kept because EVERY calculated temperature was contested: which value was used
         "calculation_picks": PICKS,
         "physical_domain_gate": "off" if NO_DOMAIN_GATE else "on",
         "families": fams}
    json.dump(R, open(a.out, "w"), indent=2)
    if CALC_DISAGREE:
        print("\ncalculations disagree > %.1fx (temperature: spread); those temperatures contribute "
              "no pair:" % CALC_DISAGREE_MAX)
        for c, v in sorted(CALC_DISAGREE.items()):
            print(f"  {c:<8} {yzfam(c):<6} {v}   {'still scored at other T' if c in set(d.compound) else 'NOT SCORED'}")
    ad = sorted(f for f, r in fams.items() if r["adopted"])
    print(f"\n{len(ad)} of {len(fams)} families adopt a family calibration: {ad}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
