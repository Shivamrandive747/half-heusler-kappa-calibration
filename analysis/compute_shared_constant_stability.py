"""How stable is the shared constant? The stability block pre-registered for it.

paper/evidence/PREREG_shared_constant_on_calculations.md ("What will be reported") requires the
deployed (c, p) to be reported with its stability: leave-one-cluster-out, leave-one-family-out and
a compound bootstrap (10-90%) of the calculation-fitted constant. This script computes exactly that,
on the pairs and with the fit of shared_constant.py, and writes

    data/exports/kappa_v2/shared_constant_stability.json

which make_paper_numbers.py ingests under calibration.stability.

  deployed       shared_constant.shared_cp()
  LOCO / LOFO    one fit per X-site chemistry cluster / bonding family removed
                 (shared_cp_without_cluster / shared_cp_without_family): min, max, median, IQR of c
                 and p, which unit(s) give each extreme, and every fit landing on the edge of the
                 (c, p) grid -- an edge fit means the optimum may lie outside the grid
  bootstrap      200 resamples of COMPOUNDS with replacement (numpy default_rng(0)); a compound
                 drawn twice is renamed so it counts as two compounds in the per-compound
                 objective; same fit (extend_blind_test.fit_cp) as shared_constant._fit
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings
from datetime import datetime, timezone

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import extend_blind_test as E
import shared_constant as SCN

OUT = "data/exports/kappa_v2/shared_constant_stability.json"
N_BOOT = 200
SEED = 0
C_EDGE = (float(E.CS.min()), float(E.CS.max()))
P_EDGE = (float(E.PS.min()), float(E.PS.max()))


def _edge(c: float, p: float) -> list[str]:
    out = []
    if np.isclose(c, C_EDGE[0]) or np.isclose(c, C_EDGE[1]):
        out.append(f"c={c:g} at grid edge {C_EDGE}")
    if np.isclose(p, P_EDGE[0]) or np.isclose(p, P_EDGE[1]):
        out.append(f"p={p:g} at grid edge {P_EDGE}")
    return out


def _summary(fits: dict[str, tuple[float, float]]) -> dict:
    """min / max / median / IQR of c and p over held-out units, with the unit(s) at each extreme."""
    rec = {"n_units": len(fits),
           "per_unit": {u: dict(c=float(c), p=float(p)) for u, (c, p) in sorted(fits.items())}}
    for j, name in ((0, "c"), (1, "p")):
        v = np.array([f[j] for f in fits.values()], dtype=float)
        lo, hi = float(v.min()), float(v.max())
        rec[name] = dict(
            min=lo, max=hi, median=float(np.median(v)),
            iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))],
            min_from=sorted(u for u, f in fits.items() if np.isclose(f[j], lo)),
            max_from=sorted(u for u, f in fits.items() if np.isclose(f[j], hi)))
    rec["grid_edge_fits"] = {u: e for u, (c, p) in sorted(fits.items()) if (e := _edge(c, p))}
    return rec


def bootstrap(d: pd.DataFrame, n: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    comps = np.array(sorted(d.compound.unique()))
    groups = {c: g for c, g in d.groupby("compound")}
    cs, ps, edges = [], [], 0
    for _ in range(n):
        pick = rng.choice(comps, size=len(comps), replace=True)
        # rename every draw, so a compound drawn twice is two compounds to the objective
        frame = pd.concat([groups[c].assign(compound=f"{c}#{k}") for k, c in enumerate(pick)],
                          ignore_index=True)
        c, p = E.fit_cp(frame)
        cs.append(float(c))
        ps.append(float(p))
        edges += bool(_edge(c, p))
    cs, ps = np.array(cs), np.array(ps)
    return dict(n_resamples=n, rng="numpy.random.default_rng(%d)" % seed,
                unit="compound (with replacement; duplicates renamed)",
                c=dict(p10=float(np.percentile(cs, 10)), p50=float(np.median(cs)),
                       p90=float(np.percentile(cs, 90))),
                p=dict(p10=float(np.percentile(ps, 10)), p50=float(np.median(ps)),
                       p90=float(np.percentile(ps, 90))),
                n_grid_edge_fits=int(edges))


def main() -> int:
    d = SCN.pairs()
    c0, p0 = SCN.shared_cp()
    print(f"deployed shared constant c={c0:.3f} p={p0:.3f} "
          f"({d.compound.nunique()} compounds, {len(d)} rows)")
    loco = _summary({cl: SCN.shared_cp_without_cluster(cl) for cl in sorted(set(d.chem))})
    lofo = _summary({f: SCN.shared_cp_without_family(f) for f in sorted(set(d.fam.dropna()))})
    for lab, r in (("leave-one-cluster-out", loco), ("leave-one-family-out", lofo)):
        print(f"  {lab:<22} c {r['c']['min']:.2f}-{r['c']['max']:.2f} (median {r['c']['median']:.2f})"
              f"   p {r['p']['min']:.2f}-{r['p']['max']:.2f} (median {r['p']['median']:.2f})"
              f"   edge fits: {r['grid_edge_fits'] or 'none'}")
    bs = bootstrap(d)
    print(f"  bootstrap ({bs['n_resamples']})      c 10-90% {bs['c']['p10']:.2f}-{bs['c']['p90']:.2f}"
          f"   p 10-90% {bs['p']['p10']:.2f}-{bs['p']['p90']:.2f}"
          f"   edge fits {bs['n_grid_edge_fits']}")
    R = {"_generated_by": "compute_shared_constant_stability.py",
         "_generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "prereg": "paper/evidence/PREREG_shared_constant_on_calculations.md",
         "data": dict(n_compounds=int(d.compound.nunique()), n_rows=int(len(d))),
         "grid": dict(c=list(C_EDGE), p=list(P_EDGE)),
         "deployed": dict(c=float(c0), p=float(p0)),
         "deployed_at_grid_edge": _edge(c0, p0),
         "leave_one_cluster_out": loco,
         "leave_one_family_out": lofo,
         "bootstrap": bs}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
