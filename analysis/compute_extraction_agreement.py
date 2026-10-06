"""How accurate is the LLM extraction? Measure it against independently curated values.

WHY THIS EXISTS. The Methods section discloses that direct extraction from primary publications was
performed by a multimodal language model, and discloses its worst failure mode (fabrication from
metadata stubs, now gated). It does not report how accurate the surviving extractions ARE, and a
referee will not accept a disclosure in place of a number -- particularly since seven compounds
enter the corpus only through that route and three of them anchor a bonding family in which
predictions are issued.

THE TEST, AND A TRAP IT HAS TO AVOID. Starrydata is a human-curated digitisation of published
transport curves, built with no knowledge of this project. Where it and our own extraction have
read THE SAME PUBLICATION, the two are independent readings of one figure and their disagreement is
extraction error. Where they have read DIFFERENT publications on the same compound, the
disagreement is mostly sample-to-sample variation between laboratories and says nothing about
extraction: NbFeSb alone spans 2.5 to 28 W/m/K across the groups that have measured it, so matching
our reading of one paper against another group's sample manufactures a 3.8x "error" that is not one.

Both are reported below, separately. The same-source figure is the measure of extraction accuracy;
the cross-source figure is a measure of the literature.

This bounds agreement, not truth -- two readers can misread one figure the same way -- but a
disagreement would be decisive evidence of a problem.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pymatgen.core import Composition

import source_identity as SI
from run_loco_chemistry import sites

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
OUT = "data/exports/kappa_v2/extraction_agreement.json"
CURATED = "starrydata"
_c: dict = {}


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def ishalf(r):
    if r not in _c:
        try:
            _c[r] = (sites(r) or ["?"])[0] == "half"
        except Exception:  # noqa: BLE001
            _c[r] = False
    return _c[r]


def main() -> int:
    tr = SI.dedupe(pd.read_csv(TRAIN, low_memory=False))
    tr["red"] = tr.formula.map(red)
    tr["t"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    m = tr[(tr.t == 0) & (tr.k > 0)].dropna(subset=["red", "T"])
    m = m[m.red.map(ishalf)]
    src = m.source.astype(str).str.lower()
    cur = m[src.str.contains(CURATED)]
    ours = m[~src.str.contains(CURATED)]
    print(f"measured half-Heusler rows: {len(m)}")
    print(f"  curated (Starrydata)        : {len(cur):5d} rows, {cur.red.nunique()} compounds")
    print(f"  our own extraction / mining : {len(ours):5d} rows, {ours.red.nunique()} compounds")
    shared = sorted(set(cur.red) & set(ours.red))
    print(f"  compounds in BOTH           : {len(shared)}\n")
    if not shared:
        print("no overlap -- the extraction cannot be checked this way")
        return 1

    def pairs(same_source: bool):
        rows = []
        for c in shared:
            a, b = cur[cur.red == c], ours[ours.red == c]
            r = []
            for T, k, doi in zip(b["T"].values, b.k.values, b.source_doi.astype(str).values):
                aa = a[a.source_doi.astype(str) == doi] if same_source else a
                if not len(aa):
                    continue
                j = int(np.abs(aa["T"].values - T).argmin())
                if abs(aa["T"].values[j] - T) <= 25:
                    r.append(k / aa.k.values[j])
            if r:
                rows.append(dict(compound=c, n=len(r), ratio=float(np.median(r))))
        return pd.DataFrame(rows)

    out = {}
    for tag, same in (("same publication (extraction error)", True),
                      ("different publications (literature spread)", False)):
        R = pairs(same)
        key = "same_source" if same else "cross_source"
        if not len(R):
            print(f"{tag}: no temperature-matched pairs")
            out[key] = None
            continue
        dev = (R.ratio - 1).abs() * 100
        print(f"\n{tag.upper()}")
        print(f"  {len(R)} compounds, {int(R.n.sum())} matched points")
        print(f"  median |deviation|      : {dev.median():.1f}%")
        print(f"  within 10% / 25%        : {100 * (dev <= 10).mean():.0f}% / "
              f"{100 * (dev <= 25).mean():.0f}% of compounds")
        print(f"  median ratio ours/theirs: {R.ratio.median():.3f}")
        print(f"  worst                   : {R.loc[dev.idxmax(), 'compound']} at "
              f"{R.loc[dev.idxmax(), 'ratio']:.2f}x")
        print(R.assign(dev_pct=dev.round(1)).sort_values("dev_pct").to_string(index=False))
        out[key] = dict(n_compounds=int(len(R)), n_points=int(R.n.sum()),
                        median_abs_deviation_pct=round(float(dev.median()), 1),
                        within_10pct=round(float(100 * (dev <= 10).mean()), 1),
                        within_25pct=round(float(100 * (dev <= 25).mean()), 1),
                        median_ratio=round(float(R.ratio.median()), 3),
                        worst_compound=str(R.loc[dev.idxmax(), "compound"]),
                        worst_ratio=round(float(R.loc[dev.idxmax(), "ratio"]), 2))
    Path(OUT).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
