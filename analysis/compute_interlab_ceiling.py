"""The inter-laboratory agreement ceiling, computed rather than quoted.

WHY THIS SCRIPT EXISTS. The project repeatedly compared its accuracy to a "79.5% within 2x
inter-laboratory ceiling" and concluded it was "essentially at the limit". That figure appears in
three places in the codebase and **is computed in none of them** -- it was a hard-coded literal. The
only inter-lab numbers actually derived anywhere are a 16.7% coefficient of variation and a
"53-55% within 25%", both over the whole thermoelectric corpus rather than Heuslers, and neither
using the within-2x metric. So the comparison was unsupported.

It also has to be computed on the MATCHED population -- the same compounds, the same temperature
range, the same metric -- or it is not a ceiling for our number at all.

THE WEIGHTING IS NOT A DETAIL, IT IS THE ANSWER. A compound measured by 40 papers contributes 780
pairwise comparisons and would dominate a pooled count, so the same data supports very different
"ceilings" depending on how it is aggregated. All are reported, because a reader deserves to
see that the choice matters:

  pooled pairs          every inter-DOI pair counted once. Dominated by well-studied compounds.
  per compound, mean    each compound's own agreement rate, averaged. The like-for-like comparison
                        to our per-compound within-2x, and the strictest of the three.
  per compound, median  the typical compound.

Only pairs from DIFFERENT source DOIs count. Two rows from the same paper are one laboratory, and
Starrydata digitises many points from a single curve, so pooling within a DOI would manufacture
agreement.

SCOPE AND HYGIENE (FIXPASS C3, 2026-10-05). The earlier version ran on the 64 blind-test compounds of
the pre-pivot set -- full Heuslers and the two-phase CuAgTe composite included -- on the raw training
CSV (a preprint and its journal version counted as two laboratories) and quoted only pair-weighted
disagreement. Now:
  * PRIMARY SCOPE = the in-domain half Heuslers (1:1:1, VEC 18, a cubic C1b record that is not
    polymorphic: make_paper_predictions.vec / structure_status, the same screen as
    compute_seed_averaged.in_domain). All half Heuslers are reported as a secondary scope.
  * the corpus is deduplicated (source_identity.dedupe: one source per preprint/journal pair);
  * a data deposit (Starrydata snapshot, Zenodo, figshare, DataCite 10.17188) is not a laboratory and
    contributes no pairs;
  * disagreement is reported PER COMPOUND (median over compounds of each compound's median pairwise
    factor) beside the pooled, pair-weighted figure.
The top-level keys keep their old names and hold the PRIMARY scope. The 1.33 hard-coded as the
"marginal extrapolation" threshold in make_paper_predictions / test_range_gate / make_paper_numbers
was the old pooled median_disagreement_x; this script does not change those thresholds, it reports
the new values for the decision.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import itertools
import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
OUT = "data/exports/kappa_v2/interlab_ceiling.json"
PER_OUT = "data/exports/kappa_v2/interlab_per_compound.csv"
MIN_PAIRS = 3          # a compound needs at least this many inter-DOI pairs to get its own rate
T_RANGE = (200, 1200)
# a data deposit is not a laboratory (same pattern as reference_choice / find_gap_papers NOT_A_PAPER)
DEPOSIT = re.compile(r"^10\.(17188|5281/zenodo|6084/m9\.figshare)[./]", re.I)


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def _norm(doi):
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", str(doi).strip().lower()).rstrip(".,;)")


def ceiling(m: pd.DataFrame) -> dict | None:
    """Agreement statistics for one population of tier-0 rows (columns red, Tbin, lab, k)."""
    cell = m.groupby(["red", "Tbin", "lab"]).k.median().reset_index()
    pooled_n = pooled_w2 = 0
    ratios: list[float] = []
    per: list[dict] = []
    for c, g in cell.groupby("red"):
        n = w2 = 0
        rc: list[float] = []
        for _, gg in g.groupby("Tbin"):
            for a, b in itertools.combinations(gg.k.values, 2):
                r = max(a, b) / min(a, b)
                rc.append(r)
                n += 1
                w2 += r <= 2.0
        ratios += rc
        pooled_n += n
        pooled_w2 += w2
        if n >= MIN_PAIRS:
            per.append(dict(compound=c, pairs=n, labs=int(g.lab.nunique()), within_2x=w2 / n,
                            median_factor=float(np.median(rc))))
    P = pd.DataFrame(per)
    if not pooled_n or not len(P):
        return None
    return dict(
        population=dict(compounds=int(len(P)), compounds_with_tier0=int(m.red.nunique()),
                        compounds_with_2plus_labs=int((m.groupby("red").lab.nunique() >= 2).sum()),
                        rows=int(len(m)), T_range=list(T_RANGE), min_pairs=MIN_PAIRS),
        within_2x=dict(pooled_pairs=pooled_w2 / pooled_n, per_compound_mean=float(P.within_2x.mean()),
                       per_compound_median=float(P.within_2x.median())),
        n_pairs=int(pooled_n),
        median_disagreement_x=float(np.median(ratios)),
        p90_disagreement_x=float(np.quantile(ratios, 0.9)),
        median_disagreement_x_per_compound=float(P.median_factor.median()),
        _P=P)


def main() -> int:
    import source_identity as SI
    from make_paper_predictions import vec, structure_status
    from run_loco_chemistry import sites

    d = SI.dedupe(pd.read_csv(TRAIN, low_memory=False))
    d["t"] = pd.to_numeric(d.method_tier, errors="coerce")
    d["k"] = pd.to_numeric(d.kappa_L, errors="coerce")
    d["T"] = pd.to_numeric(d.temperature_K, errors="coerce")
    e = d[(d.t == 0) & (d.k > 0) & d["T"].between(*T_RANGE) & d.source_doi.notna()].copy()
    e["lab"] = e.source_doi.map(_norm)
    n_dep = int(e.lab.str.match(DEPOSIT).sum())
    e = e[~e.lab.str.match(DEPOSIT)]
    e["red"] = e.formula.map(red)
    e = e.dropna(subset=["red"])
    e = e[e.red.map(lambda r: (sites(r) or ["?"])[0] == "half")]
    e["Tbin"] = (e["T"] / 100).round() * 100

    def dom(c):
        try:
            s = structure_status(c)
            return vec(c) == 18 and s[0] and not s[1]
        except Exception:  # noqa: BLE001
            return False
    ok = {c: dom(c) for c in e.red.unique()}
    scopes = {"in_domain_half_heuslers": e[e.red.map(ok)], "all_half_heuslers": e}
    print(f"tier-0 half-Heusler rows {T_RANGE[0]}-{T_RANGE[1]} K after dedupe: {len(e)}  "
          f"(data-deposit rows dropped as not-a-laboratory: {n_dep})")

    res = {}
    for name, m in scopes.items():
        r = ceiling(m)
        if r is None:
            print(f"{name}: not enough multi-laboratory data to state a ceiling")
            continue
        res[name] = r
        P = r["_P"]
        print("\n" + "=" * 84)
        print(f"INTER-LABORATORY AGREEMENT -- {name}: {r['population']['compounds']} compounds with "
              f">= {MIN_PAIRS} inter-DOI pairs, {r['n_pairs']} pairs")
        print("=" * 84)
        print(f"  within 2x   pooled pairs {100 * r['within_2x']['pooled_pairs']:.1f}%   "
              f"per-compound mean {100 * r['within_2x']['per_compound_mean']:.1f}%   "
              f"per-compound median {100 * r['within_2x']['per_compound_median']:.1f}%")
        print(f"  disagreement factor   pooled median {r['median_disagreement_x']:.2f}x   "
              f"pooled p90 {r['p90_disagreement_x']:.2f}x   "
              f"per-compound median {r['median_disagreement_x_per_compound']:.2f}x")
        for _, q in P.sort_values("within_2x").head(6).iterrows():
            print(f"    {q.compound:<11}{q.pairs:>5} pairs {q.labs:>3} labs   agree {100 * q.within_2x:>5.1f}%"
                  f"   median factor {q.median_factor:.2f}x")
    if "in_domain_half_heuslers" not in res:
        return 1
    prim = res["in_domain_half_heuslers"]
    P = prim.pop("_P")
    sec = {k: {kk: vv for kk, vv in v.items() if kk != "_P"} for k, v in res.items()
           if k != "in_domain_half_heuslers"}
    out = dict(
        scope="in-domain half Heuslers (VEC 18, cubic C1b, not polymorphic); deduped; deposits excluded",
        **prim,
        per_compound=P.sort_values("within_2x").to_dict("records"),
        other_scopes=sec,
        threshold_note=("make_paper_predictions / test_range_gate / make_paper_numbers hard-code 1.33 "
                        "(the old pooled median over the pre-pivot 64-compound set). The comparable "
                        "figures are median_disagreement_x (pooled) and "
                        "median_disagreement_x_per_compound; the threshold is not changed here."),
        note="The 79.5% literal previously quoted is not computed anywhere in this repository. "
             "Use these figures and always state the weighting.",
        _generated_by="compute_interlab_ceiling.py")
    Path(OUT).write_text(json.dumps(out, indent=2))
    P.to_csv(PER_OUT, index=False)
    print(f"\nwrote {OUT}\nwrote {PER_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
