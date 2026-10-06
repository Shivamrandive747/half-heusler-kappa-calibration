"""A value a paper QUOTES from another paper is not a measurement. Keep such rows out of tier 0.

WHY THIS EXISTS. `build_training_set.py` states the rule ("a paper citing another paper's number is
not evidence") and drops quotations upstream by `provenance`. Four first-principles papers slipped
through anyway: each compares its calculation with an experimental value taken from the literature,
and the extraction filed that literature value as the paper's own tier-0 measurement. The blind
test scores every compound against the median of ALL its measured rows, so a second-hand number
enters the reference it is scored against -- and, being a copy of a value already in the corpus or
of one we never vetted, it counts the same sample twice or admits one with no provenance at all.

HOW THEY WERE FOUND. Not by looking for the one we already suspected. `candidates()` lists every
source that carries a tier-0 row AND a calculated row (tier >= 1) for the same compound -- the
fingerprint of a paper setting its calculation beside an experiment. Each was then read:

    source                           compounds         evidence (read in the source)
    arxiv:2604.00775v1               NbCoSn            first-principles study of Nb2Co2InSb-type
                                                       compounds; NbCoSn 13.25 W/m/K quoted on p. 1
    10.1016/j.jallcom.2016.06.263    TiNiSn            title: "Lattice thermal conductivity of NiTiSn
                                                       ... from first-principles calculations"; 8.0
                                                       quoted on p. 2
    10.1088/1361-648x/abcc0f         VFeSb             title: "First-principles electronic structure,
                                                       phonon properties, lattice thermal
                                                       conductivity ..."; 12.1 / 7.0 quoted on p. 7
    10.1021/acsami.1c15955           ZrCoSb, ZrNiSn    Table 1 column "kappa_L Exp. 300" carries
      (+ its preprint arxiv:2111.00186v1)              citation superscripts 36, 34, 35: all three
                                                       values are literature quotes

Two candidates are GENUINE and stay (REVIEWED_GENUINE): the Nature Physics NbFeSb paper measured its
own polycrystalline sample (Methods: kappa measured 2-1000 K, kappa_L by Wiedemann-Franz), and the
AlVFe2 rows come from Starrydata curves of the paper's own samples.

WIDENED 2026-10-05 (FIXPASS D2). Two more quotes were found by reading, not by the detector:
TiNiSn 8.0 (10.1063/1.4939887, a DFT paper whose calculations are of OTHER compounds) and TiCoSb
20.0 (10.1016/j.jallcom.2024.178078, an experimental paper quoting a cited value). `candidates()`
now raises any tier-0 source that carries any calculated row, has a computational title, or adds a
lone value (<= 2 rows) to a compound measured by >= 3 other sources. Every source it raised was
read and filed below with its evidence; two more quotes came out of it (the chem. mater. quaternary
paper's reference NiTiSn curve, the Rev. Mex. Fis. comparison table).

The rows are removed from tier 0 only; the same sources' calculated rows are untouched.
`verify()` returns any candidate in neither list, so a new one cannot pass silently.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

SECONDARY_QUOTE_SOURCES: dict[str, str] = {
    "arxiv:2604.00775v1": "first-principles paper; NbCoSn value quoted from the literature (p. 1)",
    "10.1016/j.jallcom.2016.06.263": "first-principles paper; TiNiSn value quoted (p. 2)",
    "10.1088/1361-648x/abcc0f": "first-principles paper; VFeSb values quoted (p. 7)",
    "10.1021/acsami.1c15955": "Table 1 experimental column carries citations 34-36",
    "arxiv:2111.00186v1": "preprint of 10.1021/acsami.1c15955; same Table 1 quotes",
    # widened detector, 2026-10-05 (each read in the PDF in data/pdfs)
    "10.1063/1.4939887": "DFT paper on MPtBi; p. 5 'it was experimentally reported that half-Heusler "
                         "TiNiSn has a high kl of 8 W/m K' -- TiNiSn 8.0 is a quote",
    "10.1016/j.jallcom.2024.178078": "(Sc,V)CoSb paper; p. 1 'TiCoSb (~20 W m-1 K-1) [19,34]' -- "
                                     "TiCoSb 20.0 is a cited value, not its own sample",
    "10.1021/acs.chemmater.8b01096": "DFT quaternary-Heusler design paper; Fig. 5 caption: 'The "
                                     "experimental (Exp.) ... values of lattice thermal conductivity "
                                     "of NiTiSn are shown for reference' -- digitised literature curve",
    "10.31349/revmexfis.67.060501": "first-principles Ba2AgZ paper; its comparison table quotes "
                                    "Fe2VAl 'Exp' values with citations ([8], [10])",
}

REVIEWED_GENUINE: dict[str, str] = {
    "10.1038/s41567-023-02188-z": "NbFeSb measured on the authors' own polycrystalline sample",
    "10.15541/jim20250041": "AlVFe2 Starrydata curves of the paper's own samples",
    # widened detector, 2026-10-05
    "10.1016/j.jallcom.2023.171050": "Ru2TiGe ingot synthesised by the authors (Sec. 2); Fig. 9 "
                                     "kappa_L of their sample, 11.2 W/m-K in the text",
    "10.1088/1361-648x/ac30b5": "ZrCo1-xIrxSb series; abstract 'For ZrCoSb, kappa_L is found to be "
                                "15.13 W m-1 K-1 at 300 K' -- the authors' own x = 0 sample",
    "10.1021/acsaem.7b00203": "Zr1-xVxNiSn series; the ZrNiSn row is the x = 0 sample (Starrydata "
                              "sample 82833 carries its own Seebeck and conductivity curves). The "
                              "PDF stored under this DOI is a different paper (10.1063/1.3238363); "
                              "judged from the Starrydata record and the Crossref title",
    "10.1063/1.3091267": "Toberer et al., p-type LiZnSb: measured on the authors' samples; every row "
                         "is a Wiedemann-Franz row needing the same sample's own resistivity",
    "10.1103/physrevb.81.064404": "Barth et al., Co2TiZ 'ab initio and experiment': measured "
                                  "samples; every row is a Wiedemann-Franz row on the same "
                                  "sample's own resistivity",
}


def is_secondary(source_doi, method_tier) -> bool:
    """True for a tier-0 row whose source only quotes the value."""
    try:
        t = int(float(method_tier))
    except (TypeError, ValueError):
        return False
    return t == 0 and str(source_doi) in SECONDARY_QUOTE_SOURCES


def drop_secondary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(kept, removed). Removes tier-0 rows from the listed sources; nothing else."""
    if "source_doi" not in df.columns or "method_tier" not in df.columns:
        return df, df.iloc[0:0]
    m = [is_secondary(s, t) for s, t in zip(df.source_doi, df.method_tier)]
    m = pd.Series(m, index=df.index)
    return df[~m].copy(), df[m].copy()


# Title words of a computational paper (Crossref cache). A computational paper that carries a
# tier-0 row is the fingerprint of a quoted experiment.
CALC_TITLE = re.compile(r"first[- ]principles|ab[- ]initio|density[- ]functional|\bDFT\b|"
                        r"high[- ]throughput|machine[- ]learning", re.I)
CROSSREF = "data/external/crossref_cache.json"
THIN_ROWS = 2          # a source contributing <= this many tier-0 rows for a compound ...
THIN_OTHERS = 3        # ... that has >= this many other sources is a candidate


@lru_cache(maxsize=1)
def _titles() -> dict:
    try:
        cr = json.loads(Path(CROSSREF).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for k, r in cr.items():
        t = (r or {}).get("title")
        out[str(k).lower()] = (t[0] if isinstance(t, list) and t else (t or "")) or ""
    return out


def candidates(df: pd.DataFrame) -> list[str]:
    """Tier-0 sources that look like they might only QUOTE a measurement. A source is raised if
    (widened 2026-10-05, FIXPASS D2):
      a. it also carries ANY calculated row (tier >= 1), for any compound -- the first version
         required the SAME compound and missed a DFT paper quoting TiNiSn beside its own MPtBi
         calculations;
      b. its Crossref title reads as computational (CALC_TITLE);
      c. it contributes <= THIN_ROWS tier-0 rows for a compound that has >= THIN_OTHERS other
         sources -- a lone number beside a well-measured compound is how a quote looks.
    Data deposits are not raised: they are not laboratories, and corpus_rules.handle_deposits (D3)
    deals with them. Source ids are compared in canonical form (source_identity)."""
    import source_identity as SI
    t = pd.to_numeric(df.method_tier, errors="coerce")
    src = df.source_doi.map(SI.canonical)
    calc = {str(s) for s in src[t >= 1].dropna()}
    t0 = df[t == 0].assign(_s=src[t == 0])
    t0 = t0[t0._s.notna()]
    t0 = t0[~t0._s.map(SI.is_deposit)]
    titles = _titles()
    nsrc = t0.groupby("formula")._s.nunique()
    hit = set()
    for s, g in t0.groupby("_s"):
        s = str(s)
        if s in calc or CALC_TITLE.search(titles.get(s.lower(), "")):
            hit.add(s)
            continue
        for f, h in g.groupby("formula"):
            if len(h) <= THIN_ROWS and nsrc.get(f, 0) - 1 >= THIN_OTHERS:
                hit.add(s)
                break
    return sorted(hit - {"nan"})


def verify(df: pd.DataFrame) -> list[str]:
    """Candidates neither removed nor reviewed as genuine. Empty is the gate passing."""
    known = set(SECONDARY_QUOTE_SOURCES) | set(REVIEWED_GENUINE)
    return [s for s in candidates(df) if s not in known]
