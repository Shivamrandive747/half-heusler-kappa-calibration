"""The shared transfer function, fitted on CALCULATIONS against MEASUREMENTS. One place, used everywhere.

kappa_exp = kappa_BTE * min[c (T/300)^p, 1]

WHY IT MOVED HERE (pre-registered: paper/evidence/PREREG_shared_constant_on_calculations.md). The
constant used to be fitted on the machine-learning model's predictions for the blind-test compounds
(`extend_blind_test.fit_cp` on target_blind_test.csv, seed 0 only). Refitted on each of the five model
seeds after the 2026-09-13 tier correction, c stayed at 0.45-0.55 but p came out 0.75, 0.95, 1.20,
1.25, 1.25: the exponent was set by the model's random seed. The paper's subject is the step from a
calculation to an experiment, so the constant is fitted on that step, with no model in it.

DATA: the published-route pairs of `family_calibration.build_published()` -- in-domain half Heuslers
with a full BTE calculation paired with the single-laboratory reference, under the existing pairing
rules (34 compounds, 434 rows on 2026-10-05). FORM AND OBJECTIVE: unchanged (`extend_blind_test.fit_cp`).

HELD-OUT USE. A scored compound must never inform the constant applied to it, so every caller that
scores compounds passes the compounds to EXCLUDE -- the held-out chemistry cluster in the blind test,
the held-out family in the unseen-family test. `shared_cp()` with no exclusion is the deployed
constant.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import contextlib
import io
from functools import lru_cache

import pandas as pd

import extend_blind_test as E


@lru_cache(maxsize=1)
def pairs() -> pd.DataFrame:
    """Published calculation / reference measurement pairs, with chemistry cluster and family."""
    import family_calibration as FC
    from make_paper_predictions import yzfam
    from run_loco_chemistry import cluster_of
    with contextlib.redirect_stdout(io.StringIO()):
        d = FC.build_published()
    d = d.copy()
    d["fam"] = d.compound.map(yzfam)
    d["chem"] = d.compound.map(lambda c: cluster_of(c) or "?")
    return d


@lru_cache(maxsize=4096)
def _fit(excluded: frozenset) -> tuple[float, float]:
    d = pairs()
    if excluded:
        d = d[~d.compound.isin(excluded)]
    return E.fit_cp(d)


def shared_cp(exclude=()) -> tuple[float, float]:
    """(c, p) fitted on every pair except those of the compounds in `exclude`.

    `exclude` is an iterable of compound formulas as they appear in the blind-test files. Reduced
    formulas are compared, so "LaPtSb" and "LaSbPt" exclude the same compound.
    """
    from pymatgen.core import Composition

    def red(f):
        try:
            return Composition(str(f)).reduced_formula
        except Exception:  # noqa: BLE001
            return str(f)
    want = {red(x) for x in exclude}
    if want:
        d = pairs()
        hit = frozenset(c for c in d.compound.unique() if red(c) in want)
    else:
        hit = frozenset()
    return _fit(hit)


def shared_p_excluding(compounds) -> float:
    """The shared EXPONENT a family form carries when it is scored on held-out compound(s).

    A family form (SHAPE / SHAPE_NOCAP) borrows the shared p. When a compound is scored leave-one-out,
    that p must not have seen the compound either, so it is shared_cp(exclude=compounds)[1] -- the same
    cache entry as the shared arm scored on that compound, so it costs nothing extra. The DEPLOYED
    family constants (json, predictions) keep the deployed p, shared_cp()[1].
    """
    return float(shared_cp(exclude=compounds)[1])


def _same_pairs(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Identical pair sets (compound, T, k_ref, k_pred), order-free."""
    if len(a) != len(b):
        return False
    cols = ["compound", "T", "k_ref", "k_pred"]
    x = a[cols].sort_values(cols).reset_index(drop=True)
    y = b[cols].sort_values(cols).reset_index(drop=True)
    return bool((x.compound.values == y.compound.values).all()
                and all(abs(float(u) - float(v)) < 1e-12
                        for c in cols[1:] for u, v in zip(x[c].values, y[c].values)))


@lru_cache(maxsize=4096)
def shared_cp_in_fold(held: str) -> tuple[float, float]:
    """(c, p) applied to `held` when it is scored LEAVE-ONE-COMPOUND-OUT inside its family.

    Fitted on the published pairs AS THEY STAND IN THAT FOLD (family_calibration.fold_pairs: the
    siblings' references re-chosen without `held`, PREREG amendment A), with `held` removed (rule 4).
    Where the fold moves no other compound's reference this is exactly shared_cp(exclude=[held]) and
    reuses its cache entry; only a fold that re-references a sibling costs a refit.
    """
    import family_calibration as FC
    f = FC.fold_pairs(held)
    f = f[f.compound != held]
    base = pairs()
    base = base[base.compound != held]
    if _same_pairs(f, base):
        return shared_cp(exclude=[held])
    return E.fit_cp(f)


def shared_p_in_fold(held: str) -> float:
    """The shared exponent a family form carries when `held` is scored in its fold."""
    return float(shared_cp_in_fold(held)[1])


def _hit(exclude) -> frozenset:
    """The pair compounds matching `exclude` by reduced formula."""
    from pymatgen.core import Composition

    def red(f):
        try:
            return Composition(str(f)).reduced_formula
        except Exception:  # noqa: BLE001
            return str(f)
    want = {red(x) for x in exclude}
    if not want:
        return frozenset()
    return frozenset(c for c in pairs().compound.unique() if red(c) in want)


@lru_cache(maxsize=4096)
def _fit_c(excluded: frozenset, fitter=None) -> float:
    if fitter is None:
        from compute_seed_averaged import fit_c_only as fitter
    d = pairs()
    if excluded:
        d = d[~d.compound.isin(excluded)]
    return fitter(d)


def shared_c_only(exclude=(), fitter=None) -> float:
    """The ONE-parameter ablation of the shared form (c only, p = 0, capped at 1), fitted on the
    same calculation pairs with the same exclusion rule as shared_cp. By default
    compute_seed_averaged.fit_c_only supplies the grid and objective (E.CS, per-compound median
    |log10 error|); a caller with its own pre-existing c-only grid passes it as `fitter` (a
    module-level function taking a frame with compound/k_pred/k_ref/T and returning c)."""
    return _fit_c(_hit(exclude), fitter)


def shared_c_only_without_cluster(cluster: str, fitter=None) -> float:
    """c-only ablation with every pair of one X-site chemistry cluster removed."""
    d = pairs()
    return _fit_c(frozenset(d.loc[d.chem == cluster, "compound"].unique()), fitter)


def shared_cp_without_cluster(cluster: str) -> tuple[float, float]:
    """(c, p) with every pair of one X-site chemistry cluster removed (blind-test holdout unit)."""
    d = pairs()
    return _fit(frozenset(d.loc[d.chem == cluster, "compound"].unique()))


def shared_cp_without_clusters(clusters) -> tuple[float, float]:
    """(c, p) with every pair of SEVERAL X-site chemistry clusters removed (a 50/50 cluster split's
    test half in nested_validation). Labels as run_loco_chemistry.cluster_of gives them; a missing
    label (None/NaN) is the "?" bucket, as in pairs()."""
    labels = {"?" if (x is None or (isinstance(x, float) and x != x)) else str(x) for x in clusters}
    d = pairs()
    return _fit(frozenset(d.loc[d.chem.isin(labels), "compound"].unique()))


def shared_cp_without_family(fam: str) -> tuple[float, float]:
    """(c, p) with every pair of one bonding family removed (unseen-family holdout unit)."""
    d = pairs()
    return _fit(frozenset(d.loc[d.fam == fam, "compound"].unique()))


def factor_without_own_cluster(d: pd.DataFrame):
    """Per-row shared correction factor min[c (T/300)^p, 1] for a frame of scored rows, each row
    using the constant fitted WITHOUT its compound's X-site chemistry cluster (cluster_of)."""
    import numpy as np
    from run_loco_chemistry import cluster_of
    lab = d.compound.map(lambda c: cluster_of(c) or "?").values
    T = d["T"].values.astype(float)
    out = np.empty(len(d))
    for cl in set(lab):
        m = lab == cl
        c, p = shared_cp_without_cluster(cl)
        out[m] = E.apply_cp(T[m], c, p)
    return out


if __name__ == "__main__":
    c, p = shared_cp()
    print(f"deployed shared constant (calculations vs measurements): c={c:.3f} p={p:.3f} "
          f"on {pairs().compound.nunique()} compounds, {len(pairs())} rows")
