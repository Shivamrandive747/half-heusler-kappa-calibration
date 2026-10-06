"""How much of the headline rests on individual compounds?

WHY THIS EXISTS. The in-domain validation set is 38 compounds and the separation from the nulls
sits at p ~ 0.05. A referee's first question about a result of that size is not "what is the median"
but "how many compounds would have to move for it to go away". The manuscript reports the spread
across model SEEDS and says nothing about the spread across COMPOUNDS, which is the larger of the
two on a sample this size and the one that decides whether the claim is real.

Three tests, none of which refits anything:

  JACKKNIFE   drop each compound in turn and recompute. Reports the full range of the headline and
              names the compound whose removal moves it most, in each direction.
  BOOTSTRAP   resample compounds with replacement, 10000 times, for a percentile interval on the
              median error and on the paired difference against each null.
  SIGN TEST   on how many of the 38 the calibrated prediction beats each null, which is
              distribution-free and does not care that median APE is a skewed statistic.

The per-compound errors are taken from compute_results_indomain's own nested calibration, so this
measures the published number and not a re-derivation of it.
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

import compute_results_indomain as R

OUT = "data/exports/kappa_v2/robustness.json"
B = 10000
SEED = 0


def main() -> int:
    d = R.load_domain() if hasattr(R, "load_domain") else None
    if d is None:
        # rebuild the domain frame exactly as compute_results_indomain.main does
        import extend_blind_test as E
        from make_paper_predictions import structure_status
        D = pd.read_csv("data/exports/kappa_v2/target_blind_test.csv")
        D = D[D.klass == "half"].copy()
        D["VEC"] = D.compound.map(R.vec)
        st = {c: structure_status(str(c)) for c in D.compound.unique()}
        D["ind"] = (D.VEC == 18) & D.compound.map(lambda c: st[c][0] and not st[c][1])
        d = D[D.ind].copy()
        nested = R.nested_calibrated(d)
    else:
        nested = R.nested_calibrated(d)

    # per_compound already returns a frame indexed by compound
    pc_cal = R.per_compound(nested, "k_cal").ape
    nulls = {}
    for kind in ("constant", "powerlaw"):
        nn = R.null_predictions(d, kind)
        if nn is not None:
            nulls[kind] = R.per_compound(nn, "k_null").ape
    common = pc_cal.index
    for k in nulls:
        common = common.intersection(nulls[k].index)
    pc_cal = pc_cal.loc[common]
    nulls = {k: v.loc[common] for k, v in nulls.items()}
    n = len(pc_cal)
    print(f"{n} in-domain compounds, nested calibration\n")
    print(f"headline median APE: {pc_cal.median():.1f}%")

    # ---- jackknife -----------------------------------------------------------------------------
    jk = np.array([pc_cal.drop(c).median() for c in pc_cal.index])
    lo_i, hi_i = int(np.argmin(jk)), int(np.argmax(jk))
    print("\nJACKKNIFE (drop one compound)")
    print(f"  median ranges {jk.min():.1f}% - {jk.max():.1f}%  (full sample {pc_cal.median():.1f}%)")
    print(f"  removing {pc_cal.index[lo_i]} gives the lowest ({jk[lo_i]:.1f}%), "
          f"{pc_cal.index[hi_i]} the highest ({jk[hi_i]:.1f}%)")

    # ---- bootstrap -----------------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    boot = np.median(pc_cal.values[idx], axis=1)
    ci = np.percentile(boot, [2.5, 97.5])
    print("\nBOOTSTRAP over compounds (10000 resamples)")
    print(f"  median APE 95% CI: {ci[0]:.1f}% - {ci[1]:.1f}%")

    out = dict(n=int(n), median_ape=round(float(pc_cal.median()), 1),
               jackknife=dict(lo=round(float(jk.min()), 1), hi=round(float(jk.max()), 1),
                              most_favourable_to_drop=str(pc_cal.index[lo_i]),
                              least_favourable_to_drop=str(pc_cal.index[hi_i])),
               bootstrap_ci=[round(float(ci[0]), 1), round(float(ci[1]), 1)], nulls={})

    # ---- paired comparison against each null ---------------------------------------------------
    print("\nAGAINST EACH NULL, paired per compound")
    from scipy import stats
    for k, v in nulls.items():
        diff = v.values - pc_cal.values           # positive = calibrated is better
        wins = int((diff > 0).sum())
        bd = np.median(diff[idx], axis=1)
        dci = np.percentile(bd, [2.5, 97.5])
        sign_p = float(stats.binomtest(wins, n, 0.5).pvalue)
        w_p = float(stats.wilcoxon(v.values, pc_cal.values).pvalue)
        print(f"  {k:<10} calibrated better on {wins}/{n} compounds "
              f"(sign test p={sign_p:.3f}); median paired gain "
              f"{np.median(diff):+.1f} pp, 95% CI [{dci[0]:+.1f}, {dci[1]:+.1f}]; "
              f"Wilcoxon p={w_p:.3f}")
        out["nulls"][k] = dict(better_on=wins, of=int(n), sign_test_p=round(sign_p, 4),
                               median_gain_pp=round(float(np.median(diff)), 1),
                               gain_ci=[round(float(dci[0]), 1), round(float(dci[1]), 1)],
                               wilcoxon_p=round(w_p, 4))

    # ---- the same test at the level of the holdout unit ------------------------------------------
    #
    # The compounds are not independent: the holdout unit is the chemistry cluster, and Ni-Sb
    # contributes eight compounds that share a bonding family and in places a measuring laboratory.
    # Treating them as eight independent paired observations inflates any test over compounds. The
    # honest unit is the cluster, so the same comparison is run over cluster medians.
    from run_loco_chemistry import cluster_of
    cl = pd.Series({c: (cluster_of(c) or "?") for c in pc_cal.index})
    print(f"\nSAME COMPARISON AT THE HOLDOUT UNIT ({cl.nunique()} chemistry clusters, "
          f"not {n} compounds)")
    for k, v in nulls.items():
        a = pc_cal.groupby(cl).median()
        b = v.groupby(cl).median()
        m_ = int((b > a).sum())
        sp = float(stats.binomtest(m_, len(a), 0.5).pvalue)
        wp = float(stats.wilcoxon(b.values, a.values).pvalue)
        print(f"  {k:<10} calibrated better on {m_}/{len(a)} clusters "
              f"(sign test p={sp:.3f}, Wilcoxon p={wp:.3f})")
        out["nulls"][k]["cluster_level"] = dict(better_on=m_, of=int(len(a)),
                                                sign_test_p=round(sp, 4),
                                                wilcoxon_p=round(wp, 4))

    # ---- how many compounds must flip? ----------------------------------------------------------
    print("\nHOW MANY COMPOUNDS WOULD HAVE TO MOVE")
    for k, v in nulls.items():
        diff = v.values - pc_cal.values
        wins = int((diff > 0).sum())
        need = 0
        w = wins
        while w > n / 2 and stats.binomtest(w, n, 0.5).pvalue < 0.05:
            w -= 1
            need += 1
        print(f"  {k:<10} {need} of the {wins} compounds it currently wins would have to flip "
              f"before the sign test stops rejecting at 0.05")
        out["nulls"][k]["flips_to_lose_significance"] = int(need)

    Path(OUT).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
