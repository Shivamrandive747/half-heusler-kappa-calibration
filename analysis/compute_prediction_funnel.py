"""The prediction funnel: every unmeasured half Heusler with a calculation, and what stops each.

WHY THIS SCRIPT EXISTS. Section 3 walks the reader from 181 candidates to four predictions, and
the two outer counts (181, 102) reproduced from the corpus while the family split between them
("82 in a family with no measured member, 11 with one") did not, under any definition of "measured
member" -- and no committed script produced it. The split in the manuscript also had no bucket for
families holding exactly two members, which cannot support a fit either, since the family rule
requires three. This computes the funnel once, with the family rule's own definition of a member,
and writes it where the audit can read it.

A MEMBER is a compound carrying both a laboratory measurement and a published transport
calculation, because that is what a family calibration is fitted on (Section 2, tab:famcal).
Counting any measured compound instead gives 69/15/9; the manuscript's 82/11 matches neither.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pymatgen.core import Composition

import source_identity as SI
from make_paper_predictions import RADIO, structure_status, vec, yzfam
from run_loco_chemistry import sites

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
FAMCAL = "data/exports/kappa_v2/family_calibration.json"
OUT = "data/exports/kappa_v2/prediction_funnel.json"
NOT_BTE = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"
BTE = "bte|boltz|phono3py|shengbte|almabte|iterative|rta"
T_LO, T_HI = 150.0, 450.0
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


def in_domain(c) -> bool:
    try:
        st = structure_status(str(c))
        return vec(c) == 18 and st[0] and not st[1] and not any(e in c for e in RADIO)
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    tr = SI.dedupe(pd.read_csv(TRAIN, low_memory=False))
    tr["red"] = tr.formula.map(red)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr = tr.dropna(subset=["red"])
    tr = tr[tr.red.map(ishalf) & (tr.k > 0)]
    m = tr.method.astype(str).str.lower()
    calc = tr[(tr.tier == 1) & m.str.contains(BTE, regex=True)
              & ~m.str.contains(NOT_BTE, regex=True)]
    meas = tr[tr.tier == 0]

    measured = set(meas.red)
    cands = sorted(set(calc[calc["T"].between(T_LO, T_HI)].red) - measured)
    dom = [c for c in cands if in_domain(c)]

    adopted = {f for f, v in json.load(open(FAMCAL))["families"].items() if v.get("adopted")}
    members = pd.Series([yzfam(c) for c in measured & set(calc.red)]).value_counts()

    buckets = {"adopted": [], "two_members": [], "one_member": [], "no_member": []}
    for c in dom:
        f = yzfam(c)
        if f in adopted:
            buckets["adopted"].append(c)
        else:
            n = int(members.get(f, 0))
            buckets["no_member" if n == 0 else "one_member" if n == 1 else "two_members"].append(c)

    out = dict(candidates=len(cands), in_domain=len(dom), adopted_families=sorted(adopted),
               **{k: dict(n=len(v), compounds=sorted(v)) for k, v in buckets.items()},
               member_definition="carries both a laboratory measurement and a published transport "
                                 "calculation (the family rule's own definition)")
    print(f"candidates (calculation at {T_LO:.0f}-{T_HI:.0f} K, no measurement): {len(cands)}")
    print(f"clearing VEC=18, cubic non-polymorphic, non-radioactive     : {len(dom)}")
    for k in ("no_member", "one_member", "two_members", "adopted"):
        print(f"   family holds {k.replace('_', ' '):<12}: {len(buckets[k]):3d}")
    assert sum(len(v) for v in buckets.values()) == len(dom)
    Path(OUT).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
