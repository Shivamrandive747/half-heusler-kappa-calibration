"""A preprint and its published version are ONE source. Resolve source ids to a canonical form.

WHY THIS EXISTS. The corpus keys provenance on `source_doi`, and an arXiv preprint carries a
different id from the journal article it later became. Nothing upstream reconciled the two, so a
single calculation entered the pool twice under two ids and every statistic that counts *distinct
sources* treated it as two independent groups.

That is not a cosmetic miscount. A paper agrees with itself exactly, so each self-duplicate
contributes a disagreement ratio of 1.00 to any spread statistic -- dragging a median that is meant
to measure how far independent groups diverge toward the one value it can never legitimately take.
Two of those medians are live gate tolerances, and one evidence gate in `compute_replication_tier`
admits a compound for having "two independent calculations that agree within 1.5x" when both rows
came from the same paper.

    corpus stat                                with duplicates   deduplicated
    half Heuslers calculated >=2x at 300 K     162 / median 1.15   93 / median 1.36
    same, all temperatures                     175 / median 1.20  110 / median 1.66

HOW THE PAIRS WERE FOUND. Not by looking for the one we already suspected. `verify()` below scores
every pair of sources sharing three or more compounds and flags any pair whose shared values are
byte-identical -- which is a fingerprint no two independent calculations produce. Both pairs it
finds share 100% of their compounds and agree to the last decimal on 99-100% of them. The
measurement side of the corpus is clean: zero duplicate pairs.

The map is deliberately tiny and evidence-bearing rather than a rule about id prefixes. Not every
arXiv id in the corpus is a duplicate -- a preprint with no published counterpart is a legitimate
sole source and must stay.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import math
import re
from itertools import combinations

import numpy as np
import pandas as pd

# redundant id -> the version of record it collapses into.
# Both verified by `verify()`: identical compound set, identical values.
DUPLICATE_SOURCES: dict[str, str] = {
    # Miyazaki, Ota, Sekiguchi et al. -- 141 half Heuslers, 99% of values byte-identical
    "arxiv:2010.12467v1": "10.1038/s41598-021-92030-4",
    # 119 half Heuslers, 100% byte-identical
    "arxiv:2107.03735v2": "10.1016/j.commatsci.2021.110938",
    # 3 compounds, 100% byte-identical
    "arxiv:2111.00186v1": "10.1021/acsami.1c15955",
    # ZrRuTe (J. Phys.: Condens. Matter, 2020): one compound, both rows byte-identical (9.97 and
    # 9.83 W/m/K at 800 K, same page). Below verify()'s three-compound threshold, so found by
    # resolving the preprint's published version (2026-10-04), not by the fingerprint scan.
    "arxiv:1912.03883v3": "10.1088/1361-648x/ab9d49",
    # Carrete et al., PRX 4, 011019 (2014), and the Zenodo deposit of the same article -- DataCite
    # records the deposit as IsIdenticalTo the DOI above. 98 of its 99 compounds are in the PRX
    # rows with byte-identical values, but under different method strings ("reconstructed IFCs",
    # "full ab-initio anharmonic calculation") that the tiering routed to tier 3, so ~99 BTE
    # compounds were being counted as semi-empirical AND compared against their own originals.
    "10.5281/zenodo.10562671": "10.1103/PhysRevX.4.011019",
}


# DATA DEPOSITS ARE NOT LABORATORIES (FIXPASS D3, 2026-10-05). DataCite record prefixes for data
# repositories (DataCite test prefix 10.17188, Zenodo, figshare). A deposit can republish another
# laboratory's numbers -- 10.6084/m9.figshare.29438735 is the "20250701_starrydata2" snapshot -- so
# no rule that counts or chooses laboratories may treat one as a source of its own. The separator
# class is "[./]": real deposit DOIs read "10.5281/zenodo.123" and "10.6084/m9.figshare.123"; the
# older pattern demanded "/" after zenodo/figshare and therefore matched none of them.
DEPOSIT = re.compile(r"^10\.(17188|5281/zenodo|6084/m9\.figshare)[./]", re.I)


def is_deposit(doi) -> bool:
    """True for a data-repository DOI (a deposit), False for a paper or missing provenance."""
    if doi is None or (isinstance(doi, float) and math.isnan(doi)):
        return False
    return bool(DEPOSIT.match(str(doi).strip()))


def canonical(doi):
    """The version-of-record id for a source id. Missing provenance stays missing.

    The guard is not defensive tidiness. 4092 of the corpus's 10242 rows carry no source_doi, and
    `str(nan)` is the string "nan" -- which pandas groups, where it drops a true NaN. Without this,
    every row of unknown provenance collapses into ONE phantom source and INFLATES the independent
    source count for 46 compounds: the exact failure this module exists to prevent, introduced by
    the module itself. It put 103 in a manuscript sentence whose correct value is 93.
    """
    if doi is None:
        return doi
    if isinstance(doi, float) and math.isnan(doi):
        return doi
    s = str(doi)
    return DUPLICATE_SOURCES.get(s, s)


def dedupe(df: pd.DataFrame, col: str = "source_doi") -> pd.DataFrame:
    """Rewrite `col` to canonical ids and drop rows the collapse makes redundant.

    Redundant means: same compound, same temperature, same value, same canonical source. A genuine
    disagreement between the preprint and the published version is KEPT -- the published number does
    not erase the fact that a revision happened, and the spread statistics should see it.
    """
    if col not in df.columns:
        return df
    d = df.copy()
    d[col] = d[col].map(canonical)
    # The source key falls back to source_url where there is no DOI, and the method tier is part of
    # the key (audit 2026-10-05): without both, a tier-1 and a tier-3 row of the same compound,
    # temperature and value with an empty source_doi collapsed into one, silently dropping 12
    # compounds from the semi-empirical comparison. A preprint/published pair is unaffected: it is
    # keyed on the canonical DOI, which both carry.
    d["_src_key"] = d[col]
    if "source_url" in d.columns:
        d["_src_key"] = d["_src_key"].where(d["_src_key"].notna(), d["source_url"])
    keys = [k for k in ("temperature_K", "kappa_L", "_src_key", "method_tier") if k in d.columns]
    if "formula" in d.columns:
        # Redundancy is judged on the REDUCED formula: the same compound spelled "LaPtSb" in one
        # copy and "LaSbPt" in the other is one row, not two.
        from pymatgen.core import Composition

        def _red(f):
            try:
                return Composition(str(f)).reduced_formula
            except Exception:  # noqa: BLE001
                return str(f)
        d["_red_key"] = d.formula.map(_red)
        keys = ["_red_key"] + keys
    out = d.drop_duplicates(subset=keys) if keys else d
    return out.drop(columns=["_red_key", "_src_key"], errors="ignore")


def verify(df: pd.DataFrame, value: str = "kappa_L", by: str = "formula",
           min_shared: int = 3, rel_tol: float = 1e-3, abs_tol: float = 1e-4,
           dT: float = 1.0) -> list[dict]:
    """Re-detect duplicate sources. Returns any pair NOT already in the map.

    Call this after any corpus rebuild. An empty return is the gate passing; a non-empty return
    means a new preprint/published pair entered the corpus and `DUPLICATE_SOURCES` is now stale.

    VALUES ARE COMPARED WITH A TOLERANCE (FIXPASS D3). The first version rounded both sides to 6
    decimal places and asked for equality, so a copy that had been rounded to 4 places on the way
    -- the Starrydata snapshot deposit stores 3.7974 where the curve has 3.80164 -- never matched
    its original. Two points now match when |dT| <= `dT` K and |dk| <= max(abs_tol, rel_tol * k).
    """
    d = df.dropna(subset=[by, value, "source_doi"]).copy()
    d["_v"] = pd.to_numeric(d[value], errors="coerce")
    d = d[d._v > 0]
    d["_s"] = d.source_doi.map(canonical)
    # Key on the REDUCED formula, not the string as written: two copies of one dataset can spell
    # the same compound "LaPtSb" and "LaSbPt", and a string match would never see them meet.
    if by == "formula":
        from pymatgen.core import Composition

        def _red(f):
            try:
                return Composition(str(f)).reduced_formula
            except Exception:  # noqa: BLE001
                return str(f)
        d["_key"] = d[by].map(_red)
        by = "_key"
    # Compare ROW-LEVEL (compound, temperature, value) triples, never a per-compound median. A
    # source that reports two methods for one compound at one temperature -- PRX 4, 011019 does,
    # "BTE" beside "BTE-approx" -- has no single value there, and a median of the two matches
    # nothing. That is exactly how its Zenodo copy hid from the first version of this test.
    d["_T"] = (pd.to_numeric(d.temperature_K, errors="coerce").fillna(-1.0)
               if "temperature_K" in d.columns else 0.0)
    # per source, per compound: arrays of (T, value)
    pts = {s: {c: (h._T.to_numpy(dtype=float), h._v.to_numpy(dtype=float))
               for c, h in g.groupby(by)}
           for s, g in d.groupby("_s")}
    cpds = {s: set(v) for s, v in pts.items()}

    def _hits(pa, pb):
        """points of pa with a tolerance match in pb"""
        n = 0
        for t, v in zip(*pa):
            ok = (np.abs(pb[0] - t) <= dT) & (np.abs(pb[1] - v) <= np.maximum(abs_tol, rel_tol * v))
            n += bool(ok.any())
        return n

    out = []
    for a, b in combinations(sorted(pts), 2):
        shared_c = cpds[a] & cpds[b]
        if len(shared_c) < min_shared:
            continue
        na = sum(len(pts[a][c][0]) for c in shared_c)
        nb = sum(len(pts[b][c][0]) for c in shared_c)
        # count from the SMALLER source's side, so a small copy of a large dataset is caught
        small, big = (a, b) if na <= nb else (b, a)
        hit = sum(_hits(pts[small][c], pts[big][c]) for c in shared_c)
        # matched (compound, T, value) points, as a fraction of the SMALLER source's points on
        # the shared compounds
        ident = hit / max(1, min(na, nb))
        if hit >= min_shared and ident >= 0.5:
            out.append(dict(a=a, b=b, shared=int(len(shared_c)), frac_identical=round(ident, 3)))
    return out


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
    tr = pd.read_csv(TRAIN, low_memory=False)
    print(f"{TRAIN}: {len(tr)} rows, {tr.source_doi.nunique()} raw source ids")
    dd = dedupe(tr)
    print(f"after canonicalisation: {len(dd)} rows, {dd.source_doi.nunique()} sources "
          f"({len(tr) - len(dd)} redundant rows removed)")
    left = verify(tr)
    if left:
        print(f"\nGATE FAILED -- {len(left)} unmapped duplicate pair(s):")
        for h in left:
            print(f"  {h['a']} == {h['b']}  ({h['shared']} shared, "
                  f"{h['frac_identical']:.0%} identical)")
        raise SystemExit(1)
    print("\ngate passed: no unmapped duplicate source pair in the corpus")
    raise SystemExit(0)
