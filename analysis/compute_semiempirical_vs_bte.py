"""How far semi-empirical estimates (tier 3) and full Boltzmann-transport calculations (tier 1) differ.

Supplementary section "How far semi-empirical and transport values differ" quotes four numbers -- the
compounds carrying both labels, the median disagreement factor, the share within 2x and the worst
factor (93 / 1.38 / 84% / 11.5 when it was written) -- from data/exports/kappa_v2/semiempirical_vs_bte.csv,
a file with NO producer in the repository, last written 2026-09-04: before the 2026-09-13 tier
correction moved hundreds of "Slack ... NOT BTE" rows out of tier 1. This script is its producer
(FIXPASS C2, 2026-10-05) and recomputes it from the CURRENT tiers.

DEFINITION (the one the old file encodes -- reproduced exactly for the 41 of its 93 compounds whose
rows the tier correction left untouched):
  * half Heuslers (run_loco_chemistry.sites == "half"), corpus deduplicated (source_identity.dedupe);
  * full BTE = tier 1 whose method is not semi-empirical (family_calibration.NOT_BTE);
    semi-empirical = tier 3;
  * per compound and 100 K temperature bin, the median of each kind; ratio = semi-empirical / BTE;
  * a compound's ratio is the median over its matched bins, n = the number of matched bins,
    dev = max(ratio, 1/ratio).

Writes semiempirical_vs_bte.csv (compound, ratio, n, dev) and semiempirical_vs_bte.json (the four
quoted figures, read by make_paper_numbers.py).
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import source_identity as SI
from family_calibration import NOT_BTE
from run_loco_chemistry import sites

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
OUT = "data/exports/kappa_v2/semiempirical_vs_bte.csv"
OUT_JSON = "data/exports/kappa_v2/semiempirical_vs_bte.json"


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    tr = SI.dedupe(pd.read_csv(TRAIN, low_memory=False))
    tr["r"] = tr.formula.map(red)
    tr["t"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr = tr[(tr.k > 0) & tr.r.notna() & tr["T"].notna()]
    tr = tr[tr.r.map(lambda c: (sites(c) or ["?"])[0] == "half")]
    tr["Tbin"] = (tr["T"] / 100).round() * 100
    meth = tr.method.astype(str).str.lower()
    bte = tr[(tr.t == 1) & ~meth.str.contains(NOT_BTE, regex=True)].groupby(["r", "Tbin"]).k.median()
    semi = tr[tr.t == 3].groupby(["r", "Tbin"]).k.median()
    j = pd.concat([bte.rename("bte"), semi.rename("semi")], axis=1).dropna().reset_index()
    j["ratio"] = j.semi / j.bte
    g = j.groupby("r").agg(ratio=("ratio", "median"), n=("ratio", "size"))
    g["dev"] = np.maximum(g.ratio, 1.0 / g.ratio)
    g.index.name = "compound"
    g.to_csv(OUT)
    summ = dict(
        n_compounds=int(len(g)), n_matched_bins=int(g.n.sum()),
        median_factor=round(float(g.dev.median()), 2),
        within_2x_pct=round(100.0 * float((g.dev <= 2.0).mean()), 1),
        max_factor=round(float(g.dev.max()), 1), max_compound=str(g.dev.idxmax()),
        semi_higher_pct=round(100.0 * float((g.ratio > 1).mean()), 1),
        definition=("half Heuslers; per (compound, 100 K bin) median of tier-3 semi-empirical over median of "
                    "tier-1 full BTE (not NOT_BTE); per-compound median over matched bins; factor = "
                    "max(ratio, 1/ratio); deduplicated corpus, current tiers"),
        _generated_by="compute_semiempirical_vs_bte.py")
    json.dump(summ, open(OUT_JSON, "w"), indent=2)
    print(f"half Heuslers with both a semi-empirical and a full-BTE value in a matched 100 K bin: {summ['n_compounds']}")
    print(f"  median factor {summ['median_factor']}   within 2x {summ['within_2x_pct']}%   "
          f"worst {summ['max_factor']} ({summ['max_compound']})   semi-empirical higher in {summ['semi_higher_pct']}%")
    print(f"wrote {OUT}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
