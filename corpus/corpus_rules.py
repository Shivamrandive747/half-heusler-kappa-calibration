"""Data-quality rules for the measured corpus, in code (fix pass of 2026-10-05).

WHY A MODULE. A hand-removed row came back three times before: every correction made by editing a
CSV was undone by the next rebuild. Each rule below is therefore a function over a DataFrame,
called by the builders (`harvest_starrydata.py`, `build_kappa_pool.py`, `build_training_set.py`) for
future builds and by `apply_corpus_rules.py` for the files as they stand. Every function returns
(kept, removed) or (frame, changed) so the removed rows can be written to an audit CSV.

THE RULES (FIXPASS D1, D3, D4, D5; D2 lives in `secondary_quotes.py`, D6 in
`build_kappa_pool.tier_of`):

  D1 one copy per sample. Starrydata digitises some samples twice: the authors' published
     lattice kappa_L curve AND the total-kappa curve, from which our harvest derives a second
     kappa_L by Wiedemann-Franz (or by subtracting a measured kappa_e). Both rows describe the same
     sample at the same temperature (NbFeSb, 10.1016/j.actamat.2019.11.010, 422.88 K: published
     12.19 and 3.17 W/m/K beside our 12.18 and 3.14). Where a sample has a published kappa_L
     curve, our derived copy of THAT sample is dropped. The sample is identified by Starrydata's
     sample_id, recovered from the harvest file by (source_doi, temperature, value); a row the
     harvest cannot identify falls back to doi + compound + temperature within 0.5 K.

  D3 a data deposit is not a laboratory. 10.6084/m9.figshare.29438735 is the
     "20250701_starrydata2" snapshot, re-entered via a GitHub repository: 111 tier-0 rows with no
     primary DOI. A deposit row matching a primary-source row (same compound, |dT| <= 1 K,
     |dk|/k <= 1%) is a copy and is dropped; the rest are kept with `source` marked as a deposit.
     The identity test (`source_identity.is_deposit`) is the one place the deposit prefixes live.

  D4 non-Heuslers and one typo. ZnNiSn is Starrydata's typo for ZrNiSn: re-labelled ONLY where the
     source's own title (Crossref cache) names ZrNiSn, otherwise removed. CuAgTe is a two-phase
     (Cu2Te)50(Ag2Te)50 composite; five Phonix ABO oxides are not Heuslers; AgSbSe2 is a rocksalt
     chalcogenide. BaAgSb (hexagonal), LiZnSb (hexagonal) and MgAgSb (tetragonal) are real
     compounds outside the cubic domain and are KEPT -- the structure screen excludes them, and
     `is_half_heusler_in_domain` is the only test a count should use.

  D5 the semi-empirical gap file holds semi-empirical estimates. A row whose method tiers as a
     measurement (`tier_of` == 0: "measured", "kappa - kappa_e") or whose compound is a named
     non-Heusler is removed from HEUSLER_KAPPA_TIER3_GAP_ROWS.csv.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import source_identity as SI

HARVEST = "data/external/kappa_starrydata_heusler.csv"
# rows the harvest gate itself dropped (D1 at harvest time); read back so a corpus row whose copy
# the harvest no longer carries can still be identified by sample
HARVEST_DUPES = "data/external/kappa_starrydata_heusler_duplicate_samples.csv"
CROSSREF = "data/external/crossref_cache.json"

PUBLISHED_BASIS = "reported_lattice"
DERIVED_BASES = ("wf_subtracted", "total_minus_measured_ke")
FALLBACK_DT = 0.5             # K, D1 fallback when no sample id can be recovered
DEPOSIT_DT = 1.0              # K, D3
DEPOSIT_DK = 0.01             # relative, D3
DEPOSIT_MARK = "data deposit, not a laboratory"

# ---------------------------------------------------------------------------------------------
# D4 -- named non-Heuslers. Each passes the composition-ratio test, which is why it must be named.
NON_HEUSLER: dict[str, str] = {
    "CuAgTe": "two-phase (Cu2Te)50(Ag2Te)50 composite, not a single-phase Heusler",
    "CsAgO": "Phonix ABO oxide, not a Heusler",
    "KAgO": "Phonix ABO oxide, not a Heusler",
    "NaLiO": "Phonix ABO oxide, not a Heusler",
    "RbAgO": "Phonix ABO oxide, not a Heusler",
    "RbAuO": "Phonix ABO oxide, not a Heusler",
    "AgSbSe2": "rocksalt-derived I-V-VI2 chalcogenide, not a Heusler",
}

# Real compounds outside the cubic domain: kept, never counted as in-domain half Heuslers.
NON_CUBIC_KEPT: dict[str, str] = {
    "BaAgSb": "hexagonal (ZrBeSi-type), quasi-2D",
    "LiZnSb": "hexagonal (LiGaGe-type)",
    "MgAgSb": "tetragonal (alpha-MgAgSb)",
}

# Starrydata composition typos: wrong formula -> (intended formula, word the title must contain)
TYPO_RELABEL: dict[str, tuple[str, str]] = {
    "ZnNiSn": ("ZrNiSn", "ZrNiSn"),
}


@lru_cache(maxsize=1)
def _crossref() -> dict:
    try:
        return json.loads(Path(CROSSREF).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _title(doi) -> str:
    cr = _crossref()
    r = cr.get(str(doi)) or cr.get(str(doi).lower()) or {}
    t = r.get("title")
    return (t[0] if isinstance(t, list) and t else (t or "")) or ""


def is_half_heusler_in_domain(formula) -> bool:
    """The in-domain test every half-Heusler count should use: 1:1:1 ternary, VEC 18, recorded in a
    cubic spacegroup and in no non-cubic one (make_paper_predictions.vec / structure_status)."""
    import make_paper_predictions as MPP
    from pymatgen.core import Composition
    try:
        c = Composition(str(formula)).reduced_composition
    except Exception:  # noqa: BLE001
        return False
    if len(c) != 3 or any(abs(v - 1) > 1e-6 for v in c.values()):
        return False
    r = c.reduced_formula
    if MPP.vec(r) != 18:
        return False
    cub, noncub, _ = MPP.structure_status(r)
    return bool(cub and not noncub)


# ---------------------------------------------------------------------------------------------
# D4
def fix_typos_and_non_heuslers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel title-confirmed typos, remove named non-Heuslers. Returns (kept, audit rows)."""
    if "formula" not in df.columns:
        return df, df.iloc[0:0]
    d = df.copy()
    audit = []
    for bad, (good, word) in TYPO_RELABEL.items():
        m = d.formula.astype(str) == bad
        if not m.any():
            continue
        dois = d.loc[m, "source_doi"] if "source_doi" in d.columns else pd.Series(index=d.index[m])
        ok = pd.Series([word.lower() in _title(s).lower() for s in dois], index=dois.index)
        rel = ok[ok].index
        gone = ok[~ok].index
        if len(rel):
            audit.append(d.loc[rel].assign(rule="D4", action="relabelled",
                                           reason=f"Starrydata typo {bad} -> {good}; source title names {good}"))
            d.loc[rel, "formula"] = good
            if "parent_formula" in d.columns:
                d.loc[rel, "parent_formula"] = good
            # the typo's structure fields describe the wrong compound: take the target's
            sc = [c for c in d.columns if c.startswith("struct_") or c == "lattice_convention"]
            donor = d[(d.formula == good) & ~d.index.isin(rel)]
            if sc and len(donor):
                if "method_tier" in donor.columns and (pd.to_numeric(donor.method_tier, errors="coerce") == 0).any():
                    donor = donor[pd.to_numeric(donor.method_tier, errors="coerce") == 0]
                row = donor[sc].astype(str).agg("|".join, axis=1).value_counts().index[0]
                drow = donor[donor[sc].astype(str).agg("|".join, axis=1) == row].iloc[0]
                for c in sc:
                    d.loc[rel, c] = drow[c]
            for c in ("spacegroup", "a_A"):          # pool columns
                if c in d.columns:
                    don = d[(d.formula == good) & ~d.index.isin(rel)][c].dropna()
                    d.loc[rel, c] = don.mode().iloc[0] if len(don) else np.nan
        if len(gone):
            audit.append(d.loc[gone].assign(rule="D4", action="removed",
                                            reason=f"{bad}: Starrydata typo not confirmed by the source title"))
            d = d.drop(index=gone)
    m = d.formula.astype(str).isin(NON_HEUSLER)
    if m.any():
        audit.append(d[m].assign(rule="D4", action="removed",
                                 reason=d.loc[m, "formula"].map(NON_HEUSLER)))
        d = d[~m]
    return d, (pd.concat(audit) if audit else df.iloc[0:0].assign(rule=None, action=None, reason=None))


# ---------------------------------------------------------------------------------------------
# D1
def _sample_lookup() -> dict:
    """(source_doi, T rounded 4 dp, kappa_L rounded 5 dp) -> (sample_id, kappa_basis)."""
    parts = []
    for p in (HARVEST, HARVEST_DUPES):
        if Path(p).exists():
            h = pd.read_csv(p, low_memory=False,
                            usecols=lambda c: c in ("source_doi", "sample_id", "temperature_K",
                                                    "kappa_L", "kappa_basis"))
            parts.append(h)
    if not parts:
        return {}
    h = pd.concat(parts, ignore_index=True)
    h = h[pd.to_numeric(h.kappa_L, errors="coerce").notna()]
    key = list(zip(h.source_doi.astype(str), pd.to_numeric(h.temperature_K).round(4),
                   pd.to_numeric(h.kappa_L).round(5)))
    out: dict = {}
    for k, s, b in zip(key, h.sample_id, h.kappa_basis):
        out.setdefault(k, set()).add((s, b))
    # an ambiguous key (two samples, same doi/T/value) identifies nothing
    return {k: next(iter(v)) for k, v in out.items() if len({x[0] for x in v}) == 1}


def drop_duplicate_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D1: drop our derived kappa_L copy of a sample that also has a published kappa_L curve."""
    need = {"source_doi", "temperature_K", "kappa_L", "kappa_basis", "formula"}
    if not need <= set(df.columns):
        return df, df.iloc[0:0]
    d = df.copy()
    lk = _sample_lookup()
    T = pd.to_numeric(d.temperature_K, errors="coerce")
    K = pd.to_numeric(d.kappa_L, errors="coerce")
    sid = [lk.get((str(s), round(t, 4) if pd.notna(t) else t, round(k, 5) if pd.notna(k) else k),
                  (None, None))[0]
           for s, t, k in zip(d.source_doi, T, K)]
    d["_sid"] = sid
    basis = d.kappa_basis.astype(str)
    pub = d[basis == PUBLISHED_BASIS]
    pub = pub[pub._sid.notna()]
    pub_samples = set(zip(pub.source_doi.astype(str), pub._sid))
    der = basis.isin(DERIVED_BASES)
    drop = pd.Series(False, index=d.index)
    by_sample = der & d._sid.notna()
    drop[by_sample] = [(str(s), i) in pub_samples
                       for s, i in zip(d.loc[by_sample, "source_doi"], d.loc[by_sample, "_sid"])]
    # fallback for a derived row the harvest cannot identify: doi + compound + T within 0.5 K
    fb = der & d._sid.isna()
    pub_all = d[basis == PUBLISHED_BASIS]
    if fb.any() and len(pub_all):
        pg = {k: g for k, g in pub_all.groupby([pub_all.source_doi.astype(str),
                                                         pub_all.formula.astype(str)])}
        for i in d.index[fb]:
            g = pg.get((str(d.at[i, "source_doi"]), str(d.at[i, "formula"])))
            if g is not None and (abs(pd.to_numeric(g.temperature_K) - T[i]) <= FALLBACK_DT).any():
                drop[i] = True
    gone = d[drop].assign(rule="D1", action="removed",
                          reason="derived (WF / total-minus-kappa_e) copy of a sample whose "
                                 "published kappa_L curve is in the corpus",
                          sample_id=d.loc[drop, "_sid"])
    return d[~drop].drop(columns="_sid"), gone.drop(columns="_sid")


def duplicate_sample_mask_harvest(h: pd.DataFrame) -> pd.Series:
    """D1 at harvest time, where sample_id is a column: True for derived rows to drop."""
    if not {"sample_id", "source_doi", "kappa_basis"} <= set(h.columns):
        return pd.Series(False, index=h.index)
    pub = set(zip(h.loc[h.kappa_basis == PUBLISHED_BASIS, "source_doi"].astype(str),
                  h.loc[h.kappa_basis == PUBLISHED_BASIS, "sample_id"]))
    der = h.kappa_basis.isin(DERIVED_BASES)
    return der & pd.Series([(str(s), i) in pub for s, i in zip(h.source_doi, h.sample_id)],
                           index=h.index)


# ---------------------------------------------------------------------------------------------
# D3
def handle_deposits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop deposit rows that copy a primary-source row; mark the rest as a deposit."""
    if not {"source_doi", "formula", "temperature_K", "kappa_L", "method_tier"} <= set(df.columns):
        return df, df.iloc[0:0]
    d = df.copy()
    dep = d.source_doi.map(SI.is_deposit)
    t0 = pd.to_numeric(d.method_tier, errors="coerce") == 0
    prim = d[t0 & ~dep & d.source_doi.notna()]
    T = pd.to_numeric(d.temperature_K, errors="coerce")
    K = pd.to_numeric(d.kappa_L, errors="coerce")
    pg = {f: (pd.to_numeric(g.temperature_K).to_numpy(), pd.to_numeric(g.kappa_L).to_numpy(),
              g.source_doi.astype(str).to_numpy())
          for f, g in prim.groupby(prim.formula.astype(str))}
    drop = pd.Series(False, index=d.index)
    match = {}
    for i in d.index[dep & t0]:
        p = pg.get(str(d.at[i, "formula"]))
        if p is None:
            continue
        ok = (np.abs(p[0] - T[i]) <= DEPOSIT_DT) & (np.abs(p[1] - K[i]) <= DEPOSIT_DK * p[1])
        if ok.any():
            drop[i] = True
            match[i] = p[2][ok][0]
    gone = d[drop].assign(rule="D3", action="removed",
                          reason="data-deposit row copying a primary-source row "
                                 f"(|dT| <= {DEPOSIT_DT:g} K, |dk|/k <= {DEPOSIT_DK:.0%})")
    gone["matched_source"] = pd.Series(match)
    d = d[~drop]
    # marked on the measured side only: a deposit of CALCULATIONS (the Zenodo copy of PRX 4,
    # 011019) is already collapsed onto its paper by source_identity.canonical
    keep_dep = d.source_doi.map(SI.is_deposit) & (pd.to_numeric(d.method_tier, errors="coerce") == 0)
    if "source" in d.columns and keep_dep.any():
        s = d.loc[keep_dep, "source"].astype(str)
        d.loc[keep_dep, "source"] = [x if x.startswith(DEPOSIT_MARK) else f"{DEPOSIT_MARK} ({x})"
                                     for x in s]
    return d, gone


# ---------------------------------------------------------------------------------------------
# D5
def clean_gap_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only semi-empirical estimates for Heuslers in the gap file."""
    from build_kappa_pool import tier_of
    d = df.copy()
    meas = d.method.map(tier_of) == 0
    nonh = d.formula.astype(str).isin(NON_HEUSLER)
    gone = d[meas | nonh].copy()
    gone["rule"] = "D5"
    gone["action"] = "removed"
    gone["reason"] = np.where(meas[meas | nonh], "measured value, not a semi-empirical estimate", "")
    nh = nonh[meas | nonh]
    gone.loc[nh[nh].index, "reason"] = [
        (r + "; " if r else "") + "non-Heusler: " + NON_HEUSLER[f]
        for r, f in zip(gone.loc[nh[nh].index, "reason"], gone.loc[nh[nh].index, "formula"])]
    return d[~(meas | nonh)], gone


# ---------------------------------------------------------------------------------------------
# D7 -- the NIMS MDR "PhononDB" rows are ML-force-constant BTE, i.e. tier 2 (author, 2026-10-05).
#
# 3807 rows (47 compounds x 81 temperatures) entered as source "BTE master (PhononDB/Carrete/npj/
# SciRep)", method "BTE", no DOI, source_url https://mdr.nims.go.jp/datasets/<uuid>. They are
# phono3py kappa computed on MACHINE-LEARNED force constants (pypolymlp), published by A. Togo on
# NIMS MDR (2026-01-24); the upstream method string says so ("fc3 from pypolymlp MLP"), but the pool
# builder took the coarse `method_class` ("BTE") and the rows were trained on as tier 1. By the
# pipeline's own TIERS definition (2 = "anharmonic / ML-potential BTE", weight 0.60) they are tier 2.
# The lead checked all 47 dataset pages (pypolymlp / "_mlp_" force constants, compound matches).
# A NIMS MDR row is relabelled when its dataset is one of these 47, or when its own method text
# names pypolymlp / an MLP; any other NIMS MDR dataset is reported by `unverified_nims` instead of
# being guessed at.
NIMS_MDR = "https://mdr.nims.go.jp/datasets/"
MLP_METHOD = "BTE (phono3py on pypolymlp machine-learned force constants; NIMS MDR, Togo 2026)"
MLP_SOURCE = "NIMS MDR phono3py-MLP (Togo 2026)"
NIMS_MLP_VERIFIED: frozenset[str] = frozenset({
    "02fa4c75-ed5f-4897-b727-67d05b922c4a",
    "0327bb7f-ceb9-471d-9003-127a418f749f",
    "0ae95029-24af-4b2c-a790-eca8dc15388d",
    "0d868dba-d91c-48d7-8a91-1122edd9aa7c",
    "1086c60e-40bd-476c-8970-8f5cfb73f696",
    "1a11145d-47ce-45b8-b62a-8e2ffee9f983",
    "1a3f79d1-ece3-4293-a5b1-338829b7a528",
    "1eb327cc-dc11-44e5-a784-d67c25bcd4c5",
    "2038afb1-540b-4d93-a777-c56ca85316fb",
    "2551586e-beb8-4506-bbf9-6c674a2592c2",
    "27cba6d6-d095-4e5a-97b3-8c7e146cf1a4",
    "27fe5b43-a99a-45cd-a930-ec62f0223583",
    "2ce6cc9c-bfab-4546-889a-ac0568c00399",
    "2d8de8a1-ca42-4e70-b0dd-7223034e028b",
    "374b23f4-ecf9-4888-a89c-ada524ef86b3",
    "3b394b86-5644-4b17-aeab-baa1d52f58cb",
    "42a6a8fd-8666-43a5-b105-52528d0d7ad0",
    "458af2d6-471b-47ca-95d0-7d0b975c0252",
    "458e4be7-182d-4b9d-9c61-b689def75f29",
    "4bf0b23c-07e5-4122-96f5-5cc8c9b97d6f",
    "4fc8ba73-3c18-4e17-b1df-38841789e4db",
    "51859f36-3733-407e-bc25-4245e181ee8f",
    "5523608c-846e-43cc-a676-583ba7158609",
    "62e1e29d-e7f4-41f7-b9ab-535f21521813",
    "6542e971-cde2-47e8-82dc-17a0f6644490",
    "6827be79-288c-486e-9457-229624b1fdbd",
    "6d01f60e-e19c-4936-a0c5-bd40f71a641c",
    "7d3e6bb9-087e-41b3-9519-aebe1a90983f",
    "91b0cbde-c115-4664-8ad8-7bf7ace3137f",
    "9bb5db2d-8712-47ae-86b0-0ddc6d274141",
    "a0b4db6a-be3a-480a-8ec3-fa64481cc7df",
    "a192430b-84c8-4334-80ee-b526dd044142",
    "a1efbf1b-d290-4265-9db0-b309c252b605",
    "a7a13051-4ea5-4f5e-9eaf-b621af1584fc",
    "adf654a5-af66-44c0-b887-311321593403",
    "b325defd-6776-4fb4-9b01-97763ac959b2",
    "ba81e27e-c781-4da0-88cf-2716f0299bfb",
    "c250b833-43cc-4571-b62a-68b4ee640cdf",
    "c5e13710-9f41-4d82-a6b0-1043ab287a89",
    "c8aa5222-79ee-4d0d-9567-7c44d1c7814c",
    "cce17a7e-45d7-4886-b84b-48766c80aeea",
    "cebadd9d-1b46-42dc-9edb-601219c8fec4",
    "df2910b6-50b7-4d6c-a881-7b9c7b8e16e2",
    "dfcad4a6-a30b-4205-9241-51c792457b5a",
    "ec367855-1232-400f-a3cb-5b2ac4e1546c",
    "f4a8fb73-1c04-4363-a608-f613f86a7c43",
    "f89ff640-339f-49b0-b170-8ae87b47e0af",
})


def _nims_uuid(url) -> str | None:
    s = str(url)
    return s[len(NIMS_MDR):].strip("/") if s.startswith(NIMS_MDR) else None


def relabel_nims_mlp(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D7: relabel the NIMS MDR ML-force-constant rows (method + source). Returns (frame, changed
    rows as they were). The tier follows from the new method string through `tier_of`."""
    if "source_url" not in df.columns or "method" not in df.columns:
        return df, df.iloc[0:0]
    d = df.copy()
    uid = d.source_url.map(_nims_uuid)
    named = d.method.astype(str).str.contains(r"pypolymlp|\bmlp\b|machine-learned force", case=False,
                                              regex=True)
    m = uid.notna() & (uid.isin(NIMS_MLP_VERIFIED) | named)
    m &= (d.method.astype(str) != MLP_METHOD)
    if "source" in d.columns:
        m |= uid.notna() & uid.isin(NIMS_MLP_VERIFIED) & (d.source.astype(str) != MLP_SOURCE)
    before = d[m].assign(rule="D7", action="relabelled",
                         reason="phono3py on pypolymlp machine-learned force constants -> tier 2")
    d.loc[m, "method"] = MLP_METHOD
    if "source" in d.columns:
        d.loc[m, "source"] = MLP_SOURCE
    return d, before


def unverified_nims(df: pd.DataFrame) -> list[str]:
    """NIMS MDR datasets in the frame that are neither verified MLP nor self-describing as MLP."""
    if "source_url" not in df.columns:
        return []
    uid = df.source_url.map(_nims_uuid)
    named = df.method.astype(str).str.contains(r"pypolymlp|\bmlp\b|machine-learned force", case=False,
                                               regex=True)
    bad = uid.notna() & ~uid.isin(NIMS_MLP_VERIFIED) & ~named
    return sorted(df.loc[bad, "source_url"].astype(str).unique())
