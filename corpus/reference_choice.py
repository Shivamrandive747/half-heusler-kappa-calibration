"""Pick ONE laboratory per compound, on sample quality, before any calculation is looked at.

WHY ONE AND NOT AN AVERAGE. Averaging four laboratories is not averaging four measurements of the
same thing. Each measured a different temperature range on a different sample, so the blended curve's
temperature dependence is set by which laboratories happened to overlap where -- it is a curve nobody
measured. Worse, a calibration fitted to a blend inherits every contributing laboratory's uncertainty
at once, which is the opposite of what a calibration is for. One laboratory's curve is a real,
internally consistent measurement of one real sample: same furnace, same Lorenz subtraction, same
instrument, across every temperature it reports.

WHY THE CHOICE CANNOT BE MADE ON THE ANSWER. Picking whichever value sits closest to the calculation
would guarantee a small error and mean nothing. So the ladder below never reads kappa_BTE. Every rung
is a property of the SAMPLE or of the measurement's coverage, all of it knowable before the compound
is ever compared to a calculation.

THE LADDER, fixed in advance and applied to every compound identically:

    0. WHERE FOUR OR MORE LABORATORIES MEASURED THE COMPOUND, take the one at the centre of them.
       What is being predicted is what an experiment measures, and the best-characterised single
       sample is by construction atypical. One real laboratory's curve is still what comes out --
       the consensus chooses which laboratory, it is never averaged into the value.
    0b. COVERAGE IS A TIE-BREAK, NOT A PRECONDITION. Making it one was tried and was wrong on the
       physics: it picked TiCoSb's wider-ranging source over its better-characterised one, and that
       wider source is 93% dense, multiphase, and reports no 300 K point at all. A reference that
       stops short does not need to be excluded, because the calibration already records the range
       its own members cover and refuses to act outside it -- insufficient coverage narrows a
       family's stated validity window rather than corrupting its fit.
    1. RECORDED SAMPLE QUALITY. Among sources that report a relative density for this compound,
       take the densest -- porosity is the single largest microstructural contributor to a depressed
       lattice conductivity, and it is the variable the whole calibration residual is attributed to.
       Ties break to single-phase over multiphase, then to the widest measured temperature range.
    2. VERIFIABLE METHODOLOGY. Failing that, among sources whose full text we hold and whose samples
       we recorded, take the one covering the widest temperature range.
    3. CENTRALITY. Failing that -- which is the common case, since most measurements are digitised
       curves carrying no sample metadata -- take the source sitting closest to the median of all
       sources for this compound, in log space. This is a measurement-only criterion: it asks which
       laboratory the others agree with, never which one the calculation agrees with.
    4. Ties break to the widest temperature range, then alphabetically by DOI, so the result is
       deterministic and reproduces exactly on re-run.

A source disqualified by source_consensus is ranked LAST at every rung rather than deleted. That
keeps a compound whose only measurement comes from a disqualified laboratory -- a compound with a
doubtful number is still more use than a compound with none, provided the doubt is recorded -- while
guaranteeing such a source is never chosen wherever any alternative exists.

A NANOSTRUCTURED SAMPLE IS DEMOTED THE SAME WAY. The calculation is a perfect crystal; a sample
ball-milled for tens of hours to sub-micron grains is a deliberate departure from that, and it sits
far below an ordinary pellet of the same compound. Before this rule, rung 2 preferred such samples
BECAUSE they were the ones with a sample record: TaFeSb's reference (milled 35 h, 0.25 um) and
ZrCoBi's (milled 20 h) were each chosen over a bulk laboratory reading 1.5-1.7x higher whose paper
had never been read. See nanostructured_sources() for the fixed criteria. Like disqualification it
never deletes -- a compound whose only measurement is nanostructured keeps it, and the reason
string says so.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import functools
import json
import re
import sqlite3

import numpy as np
import pandas as pd

DB = "data/heusler.sqlite"
CONSENSUS_N = 4   # matches source_consensus.MIN_SOURCES: below this there is no consensus

# THE PUBLIC RELEASE HAS NO SAMPLE DATABASE (98 MB, not shipped). Without it the three readers below
# returned empty sets, every compound fell through to a different rung of the ladder, and the release
# scored the five shipped seeds at 28.9 % instead of the paper's 33.1 % -- silently. The three
# DB-derived tables are therefore frozen by freeze_release_inputs.py into this file, and read ONLY
# when the database is absent or empty (sqlite3.connect creates an empty file where none exists).
# In the working copy the database is present and nothing below changes.
FROZEN = "data/external/REFERENCE_EVIDENCE_frozen.json"


def _frozen(db: str):
    """The frozen tables, if the database is unusable and the frozen file exists; else None."""
    import os
    if os.path.exists(db) and os.path.getsize(db) > 0:
        return None
    try:
        with open(FROZEN, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def sample_quality(db: str = DB) -> dict[tuple[str, str], tuple[float, int]]:
    """(doi, reduced_formula) -> (relative_density_pct, is_single_phase). Empty dict if unreadable."""
    out: dict[tuple[str, str], tuple[float, int]] = {}
    fz = _frozen(db)
    if fz is not None:
        return {(d, f): (float(r), int(s)) for d, f, r, s in fz["sample_quality"]}
    try:
        con = sqlite3.connect(db)
        q = ("SELECT lower(doi) d, reduced_formula f, relative_density_pct r, phase_purity p "
             "FROM samples WHERE reduced_formula IS NOT NULL")
        bad = misidentified_pdfs()
        for d, f, r, p in con.execute(q):
            if r is None or str(d) in bad:
                continue
            single = 1 if (p and "ingle" in str(p) and "multi" not in str(p).lower()) else 0
            key = (str(d), str(f))
            if key not in out or float(r) > out[key][0]:
                out[key] = (float(r), single)
        con.close()
    except Exception:  # noqa: BLE001 -- a missing DB must not break the calibration
        pass
    return out


# WHAT COUNTS AS A NANOSTRUCTURED SAMPLE. Fixed before any outcome was looked at, from the
# corpus's own distribution and the literature convention, not from which compounds it flags:
#   milling_time_h >= 10   the corpus splits at ~5 h between powder preparation (1-5 h) and
#                          high-energy milling for grain refinement (10-40 h); 10 h is the
#                          conventional lower bound for the latter and sits above the median
#   grain_size_um  <  1    sub-micron grains, the regime where boundary scattering dominates
#                          for half-Heuslers (phonon mean free paths ~0.1-1 um)
#   an explicit "nano" in the synthesis text
# Changing 10 h to 5 h flags exactly the same paired compounds, so the threshold is not fitted.
NANO_MILL_H = 10.0
NANO_GRAIN_UM = 1.0
MIN_REF_POINTS = 4    # = curve_quality.MIN_POINTS: below this a source is a point, not a curve
# A title that declares a nanostructuring process or outcome. NOT "spark plasma" or "grain size"
# alone -- the former is ordinary consolidation, the latter can mean large grains.
NANO_TITLE_KEYWORDS = ("melt spun", "melt-spun", "melt spinning", "nanostructur", "nanocrystal",
                       "nano-", "nanocomposite", "ball mill", "ball-mill", "mechanical alloy",
                       "reduced grain", "refined grain", "grain refin", "fine-grain", "fine grain",
                       "ultrafine")


NANO_REASON: dict[tuple[str, str], str] = {}   # (doi, formula) -> which signal fired; for the audit


def nanostructured_sources(db: str = DB) -> set[tuple[str, str]]:
    """(doi, reduced_formula) pairs whose recorded sample is nanostructured.

    WHY THIS IS A RUNG. The calculation being calibrated is a perfect, infinite crystal. A sample
    that was ball-milled for tens of hours to sub-micron grains is a deliberate departure from
    that -- grain refinement is the standard engineering tool for suppressing lattice conductivity,
    and such a sample sits far below where an ordinary bulk pellet of the same compound sits. Scoring
    a bulk-crystal calculation against it, with a constant fitted on bulk pellets, is comparing two
    different things. TaFeSb (milled 35 h, 0.25 um grains) and ZrCoBi (milled 20 h) were both chosen
    over a bulk laboratory that read 1.5-1.7x higher -- and they were chosen BECAUSE the nanostructured
    paper had a sample record and the bulk one did not, which is rung 2 rewarding paperwork.

    This is a DEMOTION, applied at every rung exactly as source_consensus disqualification is: a
    nanostructured source is used only when nothing else exists. It never deletes a compound, and it
    reads only the sample's recorded synthesis -- never kappa_BTE, never the error.
    """
    out: set[tuple[str, str]] = set()
    fz = _frozen(db)
    if fz is not None:
        for d, f, why in fz["nanostructured"]:
            out.add((d, f)); NANO_REASON[(d, f)] = why
        return out
    try:
        con = sqlite3.connect(db)
        q = ("SELECT lower(doi) d, reduced_formula f, milling_time_h m, grain_size_um g, "
             "synthesis_method sm, synthesis_route sr FROM samples WHERE reduced_formula IS NOT NULL")
        bad = misidentified_pdfs()
        rows = [r for r in con.execute(q) if str(r[0]) not in bad]
        # GRAIN SIZE OVERRIDES MILLING TIME, at the level of the paper. The ZrCoBi reference
        # (10.3390/ma11050728) records 10.0 h of milling -- exactly the threshold -- and 15 um grains
        # on a sibling sample made by the same route. Ten hours that leave 15 um grains is powder
        # preparation, not grain refinement. Where a paper reports a grain size for any of its
        # samples, that size decides for every sample it made the same way; milling time is only
        # consulted when no grain size was reported at all.
        paper_grain: dict[str, float] = {}
        for d, f, m, g, sm, sr in rows:
            if g is not None:
                paper_grain[str(d)] = min(float(g), paper_grain.get(str(d), 1e9))
        for d, f, m, g, sm, sr in rows:
            txt = f"{sm or ''} {sr or ''}".lower()
            pg = paper_grain.get(str(d))
            if pg is not None:
                nano = pg < NANO_GRAIN_UM or "nano" in txt
                why = f"recorded grain {pg} um" if pg < NANO_GRAIN_UM else "'nano' in synthesis text"
            else:
                nano = (m is not None and float(m) >= NANO_MILL_H) or "nano" in txt
                why = f"recorded milling {m} h" if (m is not None and float(m) >= NANO_MILL_H) else "'nano' in synthesis text"
            if nano:
                out.add((str(d), str(f))); NANO_REASON[(str(d), str(f))] = why

        # THE PAPER'S OWN TITLE, where no sample record exists. HfNiSn's reference is titled
        # "Reduced Grain Size and Improved Thermoelectric Properties of Melt Spun (Hf,Zr)NiSn" --
        # the authors declare a grain-refined sample in the title, and the pipeline had no way to
        # know because that paper was never read. This reads the declaration. The keyword list is
        # deliberately narrow: it names a nanostructuring PROCESS or OUTCOME, never a consolidation
        # step ("spark plasma sintering" is how most bulk pellets are made). Across the corpus it
        # flags 10 measuring (paper, compound) pairs in 7 papers, all of which name melt spinning,
        # nanocomposites, fine grains or mechanical alloying. Recorded per pair, never per formula.
        titles = dict((str(x).lower(), str(y or "").lower())
                      for x, y in con.execute("SELECT doi, title FROM papers"))
        con.close()
        try:
            import pandas as pd
            tr = pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv",
                             low_memory=False, usecols=["formula", "source_doi", "method_tier"])
            tr = tr[(tr.method_tier == 0)].dropna(subset=["source_doi"])
            for d, f in tr.groupby(["source_doi", "formula"]).size().index:
                dl = str(d).lower()
                if dl in bad or (dl, str(f)) in out:
                    continue
                hit = [k for k in NANO_TITLE_KEYWORDS if k in titles.get(dl, "")]
                if hit:
                    out.add((dl, str(f))); NANO_REASON[(dl, str(f))] = f"title: '{hit[0]}' (no sample record)"
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001 -- a missing DB must not break the calibration
        pass
    return out


PDF_IDENTITY = "data/exports/kappa_v2/pdf_identity.json"


@functools.lru_cache(maxsize=4)
def misidentified_pdfs(path: str = PDF_IDENTITY) -> frozenset[str]:
    """DOIs whose stored PDF is a DIFFERENT paper, per verify_pdf_identity.py.

    31% of the corpus's PDFs fail a content check against their own DOI and title. Whatever was
    extracted from such a PDF describes some other paper's samples, so the DOI must not count as
    "methodology on record" -- that rung was choosing ZrCoBi's alternative laboratory on the
    strength of NbCoSn samples read from a Nb-Co-Sn paper stored under the ZrCoBi DOI. Announced
    loudly if the artefact is absent, because a silent empty set would restore the defect.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
        return frozenset(str(m["doi"]).lower() for m in rep.get("mismatch", []))
    except FileNotFoundError:
        print(f"  !! {path} absent -- run verify_pdf_identity.py; every PDF is being trusted")
        return frozenset()


def documented_sources(db: str = DB) -> set[str]:
    """DOIs whose full text we hold, whose samples we recorded, AND whose PDF is the right paper."""
    fz = _frozen(db)
    if fz is not None:
        return set(fz["documented"])
    try:
        con = sqlite3.connect(db)
        s = {str(r[0]) for r in con.execute("SELECT DISTINCT lower(doi) FROM samples")}
        con.close()
        return s - misidentified_pdfs()
    except Exception:  # noqa: BLE001
        return set()


def _dist(m: float, grand: float) -> float:
    """Distance from the consensus, ROUNDED so a floating-point residue can never decide.

    With exactly two laboratories the two distances are equal by construction, and the ladder's
    declared tiebreak is the wider temperature span. In practice they differed by 1e-16 -- and
    that residue, not the span, chose HoSbPd's reference: the lab reading 2.0 W/m/K over the lab
    reading 6.0, three-fold apart. Rounding to nine places makes a genuine tie a tie, so the
    declared tiebreaks actually run. This is a correctness fix, not a rule change.
    """
    return round(abs(m - grand), 9) if np.isfinite(m) else 9e9


# A data deposit is not a laboratory. ONE definition, in source_identity (DEPOSIT / is_deposit);
# NOT_A_PAPER is kept as an alias for callers of this module. The separator after the repository
# prefix is "." OR "/": real figshare and Zenodo DOIs are 10.6084/m9.figshare.29438735 and
# 10.5281/zenodo.123456, so the old local copy, which required "/", never matched a single real
# deposit and the Starrydata snapshot qualified as a family paper.
import source_identity as _SI  # noqa: E402

NOT_A_PAPER = _SI.DEPOSIT


def is_deposit(src) -> bool:
    """A data-repository DOI (figshare / Zenodo / 10.17188): a deposit, never a laboratory.

    Used at every rung of choose() and by the family rung (PREREG amendment C, 2026-10-05): a
    deposit is ranked LAST, exactly like a disqualified source, and it does not count towards the
    CONSENSUS_N laboratories or the consensus median -- the Starrydata snapshot re-publishes curves
    from papers already in the corpus, so counting it would count those laboratories twice.
    """
    return _SI.is_deposit(src)


def choose(mm: pd.DataFrame, compound: str, qual, documented, disqualified=(),
           nano=()) -> tuple[str, str]:
    """Return (chosen source_doi, the rung of the ladder that chose it) for one compound.

    `mm` is every measured row for this compound, with columns TK / k / source_doi.
    `nano` is the set of (doi, compound) pairs from nanostructured_sources(); such a source is
    ranked after every non-nanostructured alternative at every rung, exactly as a disqualified
    source is. The reason string records when that demotion decided the outcome.
    """
    srcs = sorted(set(mm.source_doi.astype(str)))
    nn = {s: (1 if (s.lower(), compound) in nano else 0) for s in srcs}
    dep = {s: (1 if is_deposit(s) else 0) for s in srcs}
    labs = [s for s in srcs if not dep[s]]     # laboratories: data deposits are not counted
    if len(srcs) == 1:
        # kept, never deleted -- but the record says what kind of sample it is
        return srcs[0], ("sole source" + (" (nanostructured)" if nn[srcs[0]] else "")
                         + (" (data deposit)" if dep[srcs[0]] else ""))

    span = {s: float(mm.loc[mm.source_doi.astype(str) == s, "TK"].max()
                     - mm.loc[mm.source_doi.astype(str) == s, "TK"].min()) for s in srcs}
    dq = {s: (1 if s.lower() in disqualified else 0) for s in srcs}
    # A SOURCE WITH TOO FEW POINTS TO BE A CURVE IS DEMOTED TOO. curve_quality already declares a
    # curve untestable below MIN_POINTS; the same floor applies here, because a single point at
    # 774 K (a figshare deposit) was chosen as HfNiSn's reference on "centrality" the moment the
    # melt-spun source was demoted. A reference carries a temperature dependence or it is not one.
    npts = {s: int((mm.source_doi.astype(str) == s).sum()) for s in srcs}
    thin = {s: (1 if npts[s] < MIN_REF_POINTS else 0) for s in srcs}
    # the three demotions together form the sort prefix; `key` is used at every rung below so the
    # ordering rule is stated once and cannot drift between rungs
    # A DATA DEPOSIT IS DEMOTED like a disqualified source (amendment C): it is chosen only where no
    # laboratory measured the compound.
    key = lambda s: (dq[s], dep[s], nn[s], thin[s])  # noqa: E731

    def _grand(med: dict) -> float:
        """The consensus: median over LABORATORIES (deposits excluded unless nothing else exists)."""
        base = [med[s_] for s_ in (labs or srcs)]
        return float(np.nanmedian(base))

    def _why(best: str, base: str) -> str:
        """Append a note when a nanostructured alternative was passed over for this choice."""
        passed = [s for s in srcs if nn[s] and not nn[best]]
        return f"{base}; nanostructured source demoted" if passed else base

    # RUNG 0 -- WHERE A CONSENSUS EXISTS, TAKE THE LABORATORY AT ITS CENTRE.
    #
    # The transfer function predicts what an EXPERIMENT will measure, so where many laboratories
    # have measured a compound the representative one is the central one. The best-documented single
    # sample is by construction atypical: TiCoSb's densest recorded sample reads 11.67 against a
    # 24-laboratory consensus of 14.41, which is about the 0.8x a 5%-porous sample should read, and
    # density is recorded for only 2 of those 24 -- so "take the densest" was choosing the better of
    # an arbitrary pair, not the best of twenty-four. This is still ONE laboratory's real curve, with
    # its own internally consistent temperature dependence; the consensus only decides WHICH
    # laboratory, it is never averaged into the reference. The threshold is the one source_consensus
    # already uses, for the same reason: below four sources there is no consensus, only a second
    # opinion, and the sample-quality rungs below take over.
    if len(labs) >= CONSENSUS_N:
        med = {}
        for s_ in srcs:
            d = mm[mm.source_doi.astype(str) == s_]
            med[s_] = float(np.log(d.k[d.k > 0]).median()) if (d.k > 0).any() else np.nan
        grand = _grand(med)
        best = sorted(srcs, key=lambda s_: (*key(s_),
                                            _dist(med[s_], grand),
                                            -span[s_], s_))[0]
        return best, _why(best, f"central of {len(labs)} labs")

    # rung 1 -- recorded relative density
    withdens = [s for s in srcs if (s.lower(), compound) in qual]
    if withdens:
        best = sorted(withdens, key=lambda s: (*key(s), -qual[(s.lower(), compound)][0],
                                               -qual[(s.lower(), compound)][1], -span[s], s))[0]
        return best, _why(best, f"density {qual[(best.lower(), compound)][0]:.1f}%")

    # rung 2 -- full text held and samples recorded.
    #
    # A nanostructured source is demoted here too, and this is the rung where it matters most: the
    # nanostructured paper is precisely the one that HAS a sample record (its milling time is why we
    # know), so without the demotion this rung systematically prefers it over a bulk laboratory
    # whose paper was never read. Demoted below the undocumented sources, a nanostructured source is
    # chosen here only if every alternative is also nanostructured or disqualified.
    doc = [s for s in srcs if s.lower() in documented and not nn[s] and not thin[s] and not dep[s]]
    if doc:
        best = sorted(doc, key=lambda s: (dq[s], -span[s], s))[0]
        return best, _why(best, "methodology on record")
    # if the only documented sources are nanostructured, this rung declines to choose and the
    # decision falls to centrality below, where those sources are demoted as everywhere else

    # rung 3 -- the laboratory the others agree with
    med = {}
    for s in srcs:
        d = mm[mm.source_doi.astype(str) == s]
        med[s] = float(np.log(d.k[d.k > 0]).median()) if (d.k > 0).any() else np.nan
    grand = _grand(med)
    best = sorted(srcs, key=lambda s: (*key(s), _dist(med[s], grand),
                                       -span[s], s))[0]
    return best, _why(best, "closest to consensus")


def family_paper(meas: pd.DataFrame, members, qual, doc, dq, nano):
    """The ONE paper a bonding family is referenced to, or None. (author's ruling, 2026-10-01)

    WHY A FAMILY RUNG. A family constant is a property of a kind of sample: it says how far a real
    pellet sits below the perfect crystal. Members referenced to different laboratories mix kinds of
    sample, and their constants disagree for that reason alone -- Sb-Pd's members each fit their own
    paper to ~4% but span 3x between papers. One laboratory that measured several members made them
    the same way and measured them the same way, so its members are comparable with each other.

    Eligible: a primary paper (not a data deposit), not disqualified, with at least MIN_REF_POINTS
    points on each member it covers, and not nanostructured for that member; it must cover two or
    more members. Chosen by: most members covered, then full text held, then how many of its
    members carry a recorded density, then the widest temperature span it covers on every member,
    then DOI. Like every rung here it never reads kappa_BTE.
    """
    cover: dict[str, list[str]] = {}
    for c in members:
        mm = meas[meas.red == c]
        for s, g in mm.groupby(mm.source_doi.astype(str)):
            if (is_deposit(s) or s.lower() in dq or (s.lower(), c) in nano
                    or len(g) < MIN_REF_POINTS):
                continue
            cover.setdefault(s, []).append(c)
    elig = {s: sorted(v) for s, v in cover.items() if len(v) >= 2}
    if not elig:
        return None, []

    def span(s):
        g = meas[meas.source_doi.astype(str) == s]
        return min(float(g[g.red == c].TK.max() - g[g.red == c].TK.min()) for c in elig[s])

    best = sorted(elig, key=lambda s: (-len(elig[s]), -int(s.lower() in doc),
                                       -sum((s.lower(), c) in qual for c in elig[s]),
                                       -span(s), s))[0]
    return best, elig[best]


def n_laboratories(mm: pd.DataFrame) -> int:
    """Distinct measuring sources that are laboratories (data deposits not counted)."""
    return sum(1 for s in set(mm.source_doi.astype(str)) if not is_deposit(s))


def per_compound(meas: pd.DataFrame, compounds, disqualified=(), db: str = DB, verbose=True,
                 family_of=None, held_out=()):
    """compound -> (chosen doi, reason). `meas` needs red / TK / k / source_doi.

    `family_of` (compound -> bonding family) switches on the family-paper rung: a member with fewer
    than CONSENSUS_N laboratories is referenced to its family's paper where one exists. A member
    with a consensus keeps the laboratory at its centre -- overriding a many-laboratory consensus
    with one paper was measured to worsen Ni-Sn (20.7% -> 25.1%) and is worse physics: the central
    laboratory is what "an experiment measures" for that compound.

    `held_out` names compounds being SCORED in a leave-one-out fold (PREREG amendment A; the family
    leave-one-out leak found by the 2026-10-05 audit). Each keeps its plain-ladder choice -- the
    family-paper rung is never applied to it -- and it is not counted as a member of its family, so
    its own measurement can neither qualify a family paper (which needs >= 2 members) nor pull a
    sibling onto the held-out compound's laboratory. Without this, Sb-Pd's HoSbPd was referenced to
    10.1088/1361-6463/aac567 only because ErSbPd (the held-out compound) was also in that paper.
    The family rung therefore runs on (compounds - held_out) exactly as if the held-out compounds
    had never been measured, and the held-out compounds get the ladder WITHOUT the family rung.
    """
    qual, doc, nano = sample_quality(db), documented_sources(db), nanostructured_sources(db)
    dq = {str(d).lower() for d in disqualified}
    held = {str(c) for c in held_out}
    out = {}
    for c in sorted(compounds):
        mm = meas[meas.red == c]
        if mm.empty:
            continue
        out[c] = choose(mm, c, qual, doc, dq, nano)
    if family_of is not None:
        fams: dict[str, list[str]] = {}
        for c in out:
            if c in held:
                continue          # scored in this fold: no family rung, not a family member
            f = family_of(c)
            if f:
                fams.setdefault(f, []).append(c)
        for f, mem in sorted(fams.items()):
            src, covered = family_paper(meas, mem, qual, doc, dq, nano)
            for c in covered:
                if n_laboratories(meas[meas.red == c]) < CONSENSUS_N and out[c][0] != src:
                    out[c] = (src, f"family paper ({len(covered)} {f} members)")
    if verbose:
        rung = pd.Series([r for _, r in out.values()]).str.replace(r"density .*", "density", regex=True)
        print(f"single-reference selection: {len(out)} compounds")
        for k, v in rung.value_counts().items():
            print(f"    chosen by {k}: {v}")
    return out
