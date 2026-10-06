"""Is the cap right for the FAMILY constants, where it bites much earlier?

Section S8 already reports the cap for the global constant, which reaches unity at 696 K. The
family constants are larger, so their caps bind sooner -- Co-Bi at c = 0.75 caps at 430 K, Sb-Pt at
c = 0.58 near 590 K -- and a much larger share of the family arm therefore sits in the uncorrected
regime. Whether the cap is right THERE was never tested, and Sb-Pt was the specific worry, its
error rising from a few percent below the cap to tens of percent above it.

The test is the only one that matters: above each family's own cap, compare the capped transfer
function against the same power law allowed to run past unity.

Read-only. Writes data/exports/kappa_v2/cap_test.json.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import pandas as pd

import family_calibration as FC

OUT = "data/exports/kappa_v2/cap_test.json"
FAMCAL = "data/exports/kappa_v2/family_calibration.json"


def main() -> int:
    J = json.load(open(FAMCAL, encoding="utf-8"))
    fam = J["families"]
    # p is shared by construction; it is stored per family, so read it there rather than assume
    ps = sorted({v["p"] for v in fam.values()})
    assert len(ps) == 1, f"exponent is not shared across families: {ps}"
    p = ps[0]
    pub = FC.build_published()

    rows = []
    for _, r in pub.iterrows():
        f = FC.yzfam(r.compound)
        if f not in fam:
            continue
        c, T = fam[f]["c"], float(r["T"])
        un = c * (T / 300.0) ** p
        rows.append(dict(compound=r.compound, fam=f, T=T, above=un > 1.0, uncapped_factor=un,
                         ape_capped=abs(r.k_pred * min(un, 1.0) - r.k_ref) / r.k_ref * 100,
                         ape_uncapped=abs(r.k_pred * un - r.k_ref) / r.k_ref * 100))
    d = pd.DataFrame(rows)
    hi = d[d.above]

    rep = dict(
        # IN-SAMPLE and PER ROW (FIXPASS S9): the family constants were fitted on these same rows,
        # and every figure below is a median over rows, not over compounds
        in_sample=True,
        per_row=True,
        shared_exponent=p,
        n_rows=int(len(d)),
        n_rows_above_cap=int(len(hi)),
        n_compounds_above_cap=int(hi.compound.nunique()),
        T_range_above_cap=[float(hi["T"].min()), float(hi["T"].max())],
        capped_median_ape=round(float(hi.ape_capped.median()), 1),
        uncapped_median_ape=round(float(hi.ape_uncapped.median()), 1),
        uncapped_better_pct=round(float((hi.ape_uncapped < hi.ape_capped).mean() * 100), 0),
        max_uncapped_factor=round(float(hi.uncapped_factor.max()), 2),
        by_family={f: dict(n=int(len(g)),
                           capped=round(float(g.ape_capped.median()), 1),
                           uncapped=round(float(g.ape_uncapped.median()), 1))
                   for f, g in hi.groupby("fam")},
        # COMPUTED, NOT TYPED. The old verdict was a fixed "KEEP THE CAP" sentence with the numbers
        # dropped into it, and it went on printing that verdict beside numbers saying the opposite
        # once the family arm became uncapped. These rows are IN-SAMPLE and use constants fitted
        # under the adopted form, so they favour whichever form that is; the held-out comparison
        # of the two forms is test_family_form_nested.py.
        family_form=sorted({v.get("form", "SHAPE") for v in fam.values()}),
        verdict=(f"Above their own caps: capped {hi.ape_capped.median():.1f}%, uncapped "
                 f"{hi.ape_uncapped.median():.1f}%; uncapped better on "
                 f"{(hi.ape_uncapped < hi.ape_capped).mean()*100:.0f}% of rows (in-sample, constants "
                 f"fitted under the adopted form). Uncapped, the family value reaches "
                 f"{hi.uncapped_factor.max():.2f}x the calculation at the highest temperatures."),
        _meaning=("above = the family's own power law has passed unity at that temperature, so the "
                  "capped prediction is the bare calculation. Sb-Pt was the specific worry."))

    print(f"above the cap: {rep['n_rows_above_cap']} rows, {rep['n_compounds_above_cap']} compounds,"
          f" {rep['T_range_above_cap'][0]:.0f}-{rep['T_range_above_cap'][1]:.0f} K")
    print(f"  capped   {rep['capped_median_ape']}%")
    print(f"  uncapped {rep['uncapped_median_ape']}%  (better on {rep['uncapped_better_pct']:.0f}%)")
    for f, v in rep["by_family"].items():
        print(f"    {f:8} n={v['n']:3}  capped {v['capped']:6.1f}%  uncapped {v['uncapped']:6.1f}%")
    print("\n" + rep["verdict"])

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
