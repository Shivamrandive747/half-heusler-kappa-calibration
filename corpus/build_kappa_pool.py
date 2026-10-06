"""Step 2 checkpoint — merge every mined kappa source into ONE Heusler training pool.

WHY THIS FILE EXISTS. Nine harvests now write nine CSVs with nine different column names and nine
different levels of theory. Pooling them naively would let a Slack-model estimate outvote a
four-phonon BTE calculation on the same compound, which is exactly how a training set quietly
becomes worse than the smaller one it replaced (this project has done that once already, with CMEG).

SO EVERY ROW KEEPS TWO THINGS:
  method_tier   0 experimental / 1 full BTE / 2 anharmonic reduced model / 3 semi-empirical
  source_doi + source_url + source_location   -- where the number physically came from

The tier is not cosmetic. Tier 3 (AGL Debye-Gruneisen, Slack) runs 38% median error against our own
44 measured Heuslers and reads ~23% high; tier 1 is a real Boltzmann-transport solution. They are
kept in the same file but never averaged together, and `weight` reflects the difference.

HEUSLERS ONLY -- the user's standing constraint. A row survives only if its formula is a Heusler
stoichiometry (1:1:1, 2:1:1, or 1:1:1:1 quaternary).

QUARANTINE, NOT DELETION. Rows failing a physics sanity check are written to a separate file with
the reason, so the exclusion is auditable rather than invisible.

THE LOW-TEMPERATURE ROWS ARE NOT ERRORS -- READ THIS BEFORE "FIXING" THEM
------------------------------------------------------------------------
A first pass flagged 112 rows as impossible because kappa ran to 13,409 W/m/K (AlFe2Nb), far above
diamond. Checking the temperature killed that reading: EVERY one of them sits at T <= 70 K, and the
curve is textbook --

    AlFe2Nb   10 K  13409  |  30 K  4417  |  50 K  765  |  100 K  147  |  300 K  ~10

That is not a parsing bug. It is what a Boltzmann-transport calculation on a PERFECT INFINITE
CRYSTAL must do as temperature falls: Umklapp scattering freezes out, phonon mean free paths
diverge, and only the (absent) sample boundary would limit them. A real specimen is capped by its
own grain size long before this.

So these rows are correct and useless at the same time. They are excluded by a TEMPERATURE FLOOR,
not by a kappa ceiling, and they are labelled `below_T_floor` rather than `bad` -- the distinction
matters because a kappa ceiling would also have thrown away genuinely high-conductivity compounds at
room temperature. Thermoelectrics operate at 300-1200 K; nothing below `T_FLOOR` trains anything.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import re
import sys
import warnings
from math import gcd
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import pandas as pd
from pymatgen.core import Composition

OUT = "data/external/KAPPA_POOL_MASTER.csv"
QUAR = "data/external/KAPPA_POOL_quarantined.csv"
# Below this, perfect-crystal BTE diverges (see module docstring) and no real sample follows it.
T_FLOOR = 200.0
KAPPA_LO = 0.05

TIERS = {0: "experimental", 1: "full BTE", 2: "anharmonic / ML-potential BTE", 3: "semi-empirical",
         9: "ML regressor output (EXCLUDED)"}
WEIGHT = {0: 1.00, 1: 1.00, 2: 0.60, 3: 0.25, 9: 0.0}


# A neural network can enter this pipeline in two very different roles, and conflating them would
# be the most damaging mistake available here:
#
#   ML FORCE FIELD  -- the network predicts atomic FORCES; phonons and kappa still come out of a
#                      real Boltzmann-transport solve (Elemental-SDNNFF, MLIP, MACE, GAP).
#                      This is a genuine calculation, cheaper and less accurate than DFT-BTE.
#                      -> tier 2, weight 0.6. Kept.
#   ML REGRESSOR    -- the network predicts kappa DIRECTLY from descriptors, with no phonon physics
#                      in between. That is another model's opinion, not data. Training on it teaches
#                      our model to imitate its errors.
#                      -> EXCLUDED (see kappa_ML_PREDICTED_not_training.csv).
ML_FORCEFIELD = ("sdnnff", "mlip", "ml potential", "machine-learning potential", "mace", "gap ",
                 "neural network force field", "nep ", "moment tensor")
ML_REGRESSOR = ("regression", "descriptor model", "random forest", "gradient boost", "xgboost",
                "surrogate", "predicted ltc", "gnn prediction", "graph neural network prediction")


# A MEASUREMENT whose method string never says "experiment" (FIXPASS D6/D5, 2026-10-05): a bare
# "Exp" / "measured" label, or a lattice value written as total minus electronic kappa
# ("kappa - kappa_e", "κ - κe"). Anchored at the start so a calculation that merely mentions a
# measured quantity ("BTE with measured lattice constant") is not read as one.
_MEASURED = re.compile(r"^\s*(exp\.?|measured|measurement)\s*$"
                       r"|^\s*(κ|kappa)(_?tot(al)?)?\s*[-−–]\s*(κ|kappa)_?e\b", re.I)
# FULL BTE spelled without the letters "bte" (FIXPASS D6): three(-and-four)-phonon solutions
# ("3ph", "3ph+4ph", "3+4ph", "SCP+3,4ph"), self-consistent-phonon BTE ("SCP ..."), the relaxation
# time approximation ("RTA") and "anharmonic lattice dynamics". Read only AFTER every disclaimer.
_FULL_BTE = re.compile(r"\b3-?ph\b|\b3\s*[+,]\s*4\s*-?ph\b|\bscph?\b|\brta\b"
                       r"|anharmonic lattice dynamics", re.I)
# MACHINE-LEARNED FORCE CONSTANTS (FIXPASS D7): phono3py on pypolymlp force constants is an
# ML-potential BTE -- tier 2 by the TIERS definition above, whatever else the string says.
_ML_FC = re.compile(r"pypolymlp|machine[- ]learned force constants|\bmlp force constants", re.I)


def tier_of(method: str) -> int:
    """0 measured / 1 full BTE / 2 approximate or ML-potential BTE / 3 semi-empirical / 9 excluded.

    ORDER IS THE RULE. Every disclaimer (semi-empirical / Slack / NOT BTE / approx / ML potential)
    is read before any keyword that would claim a full BTE, so "Slack model on MLIP elastics" is
    tier 3 and "phono3py on pypolymlp force constants" is tier 2.
    """
    m = str(method).lower()
    if any(k in m for k in ML_REGRESSOR):
        return 9                                   # excluded below, never trained on
    if "experiment" in m or "wiedemann" in m or _MEASURED.search(m):
        return 0
    # A semi-empirical model evaluated on ML-potential inputs is still semi-empirical: this test
    # used to come AFTER the ML-force-field one, so "Slack model on MLIP elastics" read as tier 2.
    if any(k in m for k in ("semi-empirical", "semi empirical", "slack", "debye-callaway",
                            "debye callaway")):
        return 3
    if _ML_FC.search(m) or any(k in m for k in ML_FORCEFIELD):
        return 2
    # A DISCLAIMER MUST BE READ BEFORE THE KEYWORD IT DISCLAIMS.
    #
    # These three lines used to sit AFTER the BTE test below, and the BTE test asks whether "bte"
    # appears anywhere in the string. It appears inside "NOT BTE". So every row labelled
    # "semi-empirical (Slack / Debye-Callaway-type, NOT BTE)" was tiered 1 = full BTE at weight 1.0
    # and trained on as a first-principles calculation: 429 rows across 363 compounds, 8.7% of the
    # tier-1 pool, from arXiv:2501.11644 (332), PhysRevX 4:011019 "BTE-approx" (77) and an empirical
    # Slack model on ML-potential elastics (17). Prediction references were never affected, because
    # make_paper_predictions, compute_conditional and family_calibration all filter on the method
    # STRING as well as the tier -- only the model's training labels were. Measured reach on a real
    # prediction: removing those rows moved ScAsPt by 14.2% and HfAsIr by 4.9%.
    #
    # Slack and Debye-Callaway are semi-empirical models, so they belong at tier 3 with the rest of
    # that class; an approximated or reduced BTE is still a transport solve and stays at tier 2.
    # (the semi-empirical test now sits above the ML-force-field one; see the top of the function)
    #
    # "compressed" (FIXPASS D6): arXiv:2503.05913 compresses the 3-ph/4-ph force constants to a
    # low-rank (PCP) form and scans the rank -- an approximated transport solve, so tier 2 like the
    # other approximations, not tier 1 although the string contains "3-ph".
    if ("not full bte" in m or "not bte" in m or "bte-approx" in m or "bte approx" in m
            or "reduced-model" in m or "reduced model" in m or "compressed" in m):
        return 2
    if any(k in m for k in ("bte", "shengbte", "phono3py", "tdep", "phonon")) or _FULL_BTE.search(m):
        return 1
    return 3


def num(s):
    return pd.to_numeric(s, errors="coerce")


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def is_heusler_stoich(f) -> bool:
    """1:1:1 (half), 2:1:1 (full/inverse) or 1:1:1:1 (quaternary Y). Nothing else.

    THE ROUNDING BUG THIS GUARDS. An earlier version did `int(round(x))` with no tolerance, so
    `Ti1Sb1Ru1.1` became 1:1:1 and entered the pool as a half-Heusler. It is not one -- it is an
    off-stoichiometric alloy with 10% excess ruthenium, and a whole series of them
    (Ti1Sb1Ru1.1 ... Ti1Sb1Ru1.9, Zr0.88Ni1Bi1) came in the same way. The pool is stoichiometric
    Heuslers only, so a composition must be integer to within 2% before its ratio is even
    considered."""
    try:
        c = Composition(str(f)).reduced_composition
    except Exception:  # noqa: BLE001
        return False
    raw = [float(x) for x in c.values()]
    if not raw or any(abs(x - round(x)) > 0.02 for x in raw):
        return False
    v = [int(round(x)) for x in raw]
    if not v or any(x <= 0 for x in v) or len(v) not in (3, 4):
        return False
    g = 0
    for x in v:
        g = gcd(g, x)
    r = tuple(sorted(x // g for x in v))
    return r in {(1, 1, 1), (1, 1, 2), (1, 1, 1, 1)}


NOT_HEUSLER_PROTOTYPE = {
    # SrMnBi2 is 1:1:2, so the ratio test above calls it a "full Heusler" with Bi on the doubled
    # site. It is not one: it is a layered ZrCuSiAs-type Dirac semimetal (strongly non-cubic,
    # a = 4.64 A, c = 11.56 A) with no interpenetrating-fcc Heusler motif. It reached tier 0 with 27
    # experimental rows because the structure registry accepts ANY spacegroup match against
    # {216, 225, 119, 139} and ignores JARVIS's own `is_heusler_prototype` flag.
    # A composition ratio is not a prototype -- see also the cF12 fluorite trap for binary sg225.
    "SrMnBi2": "layered ZrCuSiAs-type Dirac semimetal, not a Heusler prototype",
}


def frame(df, *, kappa, temp, method, src, doi=None, url=None, loc=None,
          formula="formula", sg=None, a=None) -> pd.DataFrame:
    g = pd.DataFrame()
    g["formula_raw"] = df[formula]
    g["kappa_L"] = num(df[kappa])
    g["temperature_K"] = num(df[temp]) if temp in df else pd.NA
    g["method"] = df[method].astype(str) if method in df else method
    g["source"] = src
    for name, col in (("source_doi", doi), ("source_url", url), ("source_location", loc),
                      ("spacegroup", sg), ("a_A", a)):
        g[name] = df[col] if (col and col in df) else pd.NA
    # HOW THE LATTICE VALUE WAS OBTAINED, carried through automatically wherever a source states it.
    # This matters: 87% of the experimental rows are `wf_subtracted`, i.e. kappa_total minus an
    # ESTIMATED electronic part (Wiedemann-Franz with the Kim/Snyder Lorenz number), and only ~12%
    # are values the authors themselves reported as lattice kappa. Those two carry very different
    # uncertainty, and until now the distinction was destroyed here -- `frame()` kept a fixed column
    # list, so `kappa_basis` never reached the pool or the training set and accuracy could not be
    # broken down by it.
    g["kappa_basis"] = (df["kappa_basis"] if "kappa_basis" in df else pd.NA)
    return g


def main() -> int:
    parts: list[pd.DataFrame] = []

    def add(path, **kw):
        p = Path(path)
        if not p.exists():
            print(f"  MISSING {path}")
            return
        d = pd.read_csv(p)
        if "confidence" in d:                    # agent-flagged bad rows never enter
            d = d[d.confidence != "suspect"]
        if "provenance" in d:
            # A value a paper QUOTES from another paper is not that paper's result. Letting it in
            # would double-count the original and inflate agreement between "independent" sources.
            # A quoted value double-counts a source we already hold; a figure estimate is a
            # number read off a plot by eye. Neither is evidence.
            prov = d.provenance.astype(str)
            sec = (prov.str.startswith("secondary") | prov.str.startswith("figure_estimate")
                   | prov.str.startswith("transcription_error"))
            if int(sec.sum()):
                print(f"    {p.name}: dropped {int(sec.sum())} quoted/figure-estimate rows")
            d = d[~sec]
        g = frame(d, **kw)
        parts.append(g)
        print(f"  {p.name:<44}{len(g):>6} rows")

    print("READING SOURCES")
    add("data/external/kappa_bte_heusler_MASTER.csv", kappa="kappa_L", temp="temperature_K",
        method="method_class", src="BTE master (PhononDB/Carrete/npj/SciRep)",
        doi="source_doi", url="source_url")
    add("data/external/kappa_repos_mined.csv", kappa="kappa_L", temp="temperature_K",
        method="method_class", src="repository mining", doi="source_doi", url="source_url")
    add("data/external/kappa_papers_mined.csv", kappa="kappa_L", temp="temperature_K",
        method="method_class", src="paper mining", doi="source_doi", url="source_url",
        loc="source_location", sg="spacegroup_number", a="a_A")
    add("data/external/kappa_from_pdf_hh130.csv", kappa="kappa_L", temp="temperature_K",
        method="method", src="HH130 PDF", doi="source_doi", loc="location")
    add("data/external/kappa_pdf_batch2.csv", kappa="kappa_L", temp="temperature_K",
        method="method", src="paper mining (PDF, batch 2)", doi="source_doi", loc="location",
        sg="space_group", a="lattice_parameter_A")
    add("data/external/kappa_aflow_gap.csv", kappa="kappa_L", temp="temperature_K",
        method="method", src="AFLOW AGL (gap-targeted)", doi="source_doi", url="source_url",
        sg="spacegroup_number", a="a_A")
    # Starrydata: EXPERIMENTAL kappa vs T, digitised from published plots. Structure-confirmed
    # against MP/OQMD/JARVIS/AFLOW before entry. Rows holding only TOTAL kappa carry a null
    # kappa_L and are dropped by the numeric gate below.
    add("data/external/kappa_starrydata_heusler.csv", kappa="kappa_L", temp="temperature_K",
        method="method", src="Starrydata (experimental)", doi="source_doi", url="source_url",
        loc="source_location")
    # Phonix: auto-kappa (VASP+ALAMODE) three-phonon BTE, 300 K quoted from arXiv:2504.21245.
    add("data/external/kappa_phonix_heusler.csv", kappa="kappa_L", temp="temperature_K",
        method="method_class", src="Phonix (first-principles BTE)", doi="source_doi",
        url="source_url", loc="source_location", sg="spacegroup_number", a="a_A")

    p = Path("data/external/aflow_agl_heusler.csv")
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.is_heusler == True]  # noqa: E712
        d["_T"] = 300.0
        d["_m"] = "AGL Debye-Gruneisen (quasi-harmonic)"
        g = frame(d, kappa="kappa_300K", temp="_T", method="_m", src="AFLOW AGL",
                  url="aurl", formula="reduced_formula", sg="spacegroup", a="a")
        parts.append(g)
        print(f"  {p.name:<44}{len(g):>6} rows")

    d = pd.concat(parts, ignore_index=True)
    print(f"\n  raw total: {len(d)} rows")

    d["formula"] = d.formula_raw.map(red)
    d = d[d.formula.notna()]

    # ---- corpus rules that must survive a rebuild (corpus_rules.py, FIXPASS 2026-10-05) -------
    # D7 before tiering: the NIMS MDR rows arrive with method_class "BTE" and must tier as
    # ML-force-constant BTE (2). D4: Starrydata typo ZnNiSn (title-gated) and named non-Heuslers.
    import corpus_rules as CR
    _audit = []
    d, _ch = CR.relabel_nims_mlp(d)
    if len(_ch):
        print(f"  D7 relabelled {len(_ch)} NIMS MDR rows as phono3py on ML force constants")
    for _u in CR.unverified_nims(d):
        print(f"  WARNING: NIMS MDR dataset not verified as MLP or DFT -- read it: {_u}")
    d, _a = CR.fix_typos_and_non_heuslers(d)
    _audit.append(_a)

    # ---- Heuslers only -------------------------------------------------------
    before = len(d)
    d["heusler"] = d.formula.map(is_heusler_stoich)
    nonh = d[~d.heusler]
    d = d[d.heusler].copy()
    print(f"  non-Heusler stoichiometry dropped: {before - len(d)} rows "
          f"({nonh.formula.nunique()} formulae)")

    # ---- physics sanity ------------------------------------------------------
    d["temperature_K"] = num(d.temperature_K)
    d["_bad"] = None
    d.loc[d.kappa_L.isna(), "_bad"] = "no numeric kappa"
    d.loc[d.kappa_L <= 0, "_bad"] = "kappa <= 0"
    d.loc[(d.kappa_L > 0) & (d.kappa_L < KAPPA_LO), "_bad"] = f"kappa < {KAPPA_LO} (unphysical)"
    # correct physics, wrong regime -- kept out of training, not called an error
    d.loc[d.temperature_K < T_FLOOR, "_bad"] = f"below_T_floor (T < {T_FLOOR:.0f} K)"
    d.loc[d.temperature_K.isna(), "_bad"] = "no stated temperature"
    # Named non-Heuslers that PASS the ratio test -- quarantined with the reason, never silently.
    # APPEND rather than overwrite: every assignment above is unconditional, so writing this one
    # last clobbered whatever an earlier check had recorded. For SrMnBi2, 80 of its 107 rows were
    # already below the temperature floor and would have lost that reason entirely. Both facts are
    # true and the audit file should say so -- this is a prototype disqualification that applies
    # regardless of temperature, not a replacement for it.
    for _f, _why in NOT_HEUSLER_PROTOTYPE.items():
        _m = d.formula.astype(str) == _f
        _new = f"not a Heusler prototype: {_why}"
        d.loc[_m, "_bad"] = [(_new if (x is None or pd.isna(x)) else f"{x}; also {_new}")
                             for x in d.loc[_m, "_bad"]]
    q = d[d._bad.notna()]
    if len(q):
        print(f"  QUARANTINED {len(q)} rows:")
        for why, gg in q.groupby("_bad"):
            print(f"    {why:<40}{len(gg):>5} rows  {sorted(set(gg.formula))[:6]}")
        q.assign(reason=q._bad).drop(columns=["_bad", "heusler"]).to_csv(QUAR, index=False)
    d = d[d._bad.isna()].drop(columns=["_bad", "heusler"])

    d["method_tier"] = d.method.map(tier_of)
    ml = d[d.method_tier == 9]
    if len(ml):
        print(f"  EXCLUDED {len(ml)} rows whose kappa is an ML REGRESSOR output, not a "
              f"calculation: {sorted(set(ml.method))[:3]}")
        ml.assign(reason="ML regressor output, not first-principles or measured").to_csv(
            "data/external/KAPPA_POOL_ml_excluded.csv", index=False)
    d = d[d.method_tier != 9]
    d["tier_name"] = d.method_tier.map(TIERS)
    d["weight"] = d.method_tier.map(WEIGHT)
    d = d.drop_duplicates(subset=["formula", "kappa_L", "temperature_K", "source_doi"])
    # D3 deposit copies dropped, the rest marked as a deposit; then D1 one copy per Starrydata
    # sample (D3 first: the deposit copies our WF rows that D1 removes -- see apply_corpus_rules)
    d, _a = CR.handle_deposits(d)
    _audit.append(_a)
    d, _a = CR.drop_duplicate_samples(d)
    _audit.append(_a)
    _audit = [x for x in _audit if len(x)]
    if _audit:
        _r = pd.concat(_audit, ignore_index=True)
        print(f"  corpus rules removed/relabelled {len(_r)} rows: "
              f"{_r.groupby(['rule', 'action']).size().to_dict()}")
        _r.to_csv("data/external/KAPPA_POOL_corpus_rules_audit.csv", index=False)

    print(f"\n{'=' * 72}\nHEUSLER kappa_L POOL\n{'=' * 72}")
    print(f"  rows {len(d)}   DISTINCT HEUSLER COMPOUNDS {d.formula.nunique()}\n")
    print(f"  {'tier':<32}{'rows':>8}{'compounds':>12}{'weight':>9}")
    for t in sorted(d.method_tier.unique()):
        g = d[d.method_tier == t]
        print(f"  {t} {TIERS[t]:<30}{len(g):>8}{g.formula.nunique():>12}{WEIGHT[t]:>9.2f}")

    print(f"\n  {'source':<44}{'rows':>8}{'compounds':>12}")
    for s, g in d.groupby("source"):
        print(f"  {str(s)[:42]:<44}{len(g):>8}{g.formula.nunique():>12}")

    T = num(d.temperature_K)
    d["temperature_K"] = T
    withT = d[T.notna()]
    print(f"\n  WITH A STATED TEMPERATURE : {len(withT)} rows, {withT.formula.nunique()} compounds")
    print(f"    T range {T.min():.0f} - {T.max():.0f} K")
    nT = withT.groupby("formula").temperature_K.nunique()
    print(f"    multi-temperature compounds (a real kappa-vs-T curve): {int((nT > 1).sum())}")
    print(f"    near 300 K (250-350 K): "
          f"{withT[withT.temperature_K.between(250, 350)].formula.nunique()} compounds")

    best = d.sort_values("method_tier").groupby("formula").first()
    print("\n  BEST AVAILABLE METHOD PER COMPOUND")
    for t, n in best.method_tier.value_counts().sort_index().items():
        print(f"    {TIERS[t]:<32}{n:>5} compounds")

    Path("data/external").mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
