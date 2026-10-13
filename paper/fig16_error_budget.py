"""Error budget on one multiplicative axis (2026-10-12 upgrade).

One question: how large is each route's typical disagreement with measurement, against the spread the
field itself shows when two laboratories measure, or two groups calculate, the same compound?

Every row is a per-compound factor 10^|log10 r| >= 1 (a factor of 2 either way plots at 2):
  - laboratory vs laboratory: interlab_per_compound.csv, each compound's median pairwise max/min
    factor over its in-domain laboratories (compute_interlab_ceiling.py);
  - calculation vs calculation: calc_spread_per_compound.csv, max/min over independent full BTE
    sources at 280-320 K (make_paper_numbers.py, source_disagreement.calculations_300K);
  - published calculation vs reference measurement, uncorrected and with the shared constant
    refitted without the compound's chemistry cluster: offset_per_compound.csv (fig02_offset.py);
  - published calculation with the family constant, held out: accuracy_per_compound.csv;
  - crystal structure to measurement, in domain, chemistry cluster held out of both steps, and the
    two compound-blind nulls: accuracy_per_compound.csv (compute_accuracy_summary.py; model rows are
    each compound's median over the five seeds).
The two spread rows compare two sources with each other; every other row compares a prediction with
ONE chosen reference. They are different measures and the caption says so. Nothing is fitted here.
Writes figures/fig16_error_budget.pdf and data/exports/kappa_v2/error_budget.json.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
from pathlib import Path

import numpy as np
import pandas as pd

import figlib as F

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "data" / "exports" / "kappa_v2"


def rows():
    il = pd.read_csv(EX / "interlab_per_compound.csv")
    cs = pd.read_csv(EX / "calc_spread_per_compound.csv")
    off = pd.read_csv(EX / "offset_per_compound.csv")
    acc = pd.read_csv(EX / "accuracy_per_compound.csv")
    fac = lambda r: 10 ** np.abs(np.log10(np.asarray(r, float)))
    A = lambda k: fac(acc[acc.route == k].ratio)
    return [
        ("Same compound, two sources", None, None, None),
        ("two laboratories", il.median_factor.values, "ref", "lab_vs_lab"),
        ("two calculations", cs.factor.values, "ref", "calc_vs_calc"),
        ("Published calculation", None, None, None),
        ("uncorrected", fac(off.ratio_uncorrected), "raw", "published_uncorrected"),
        ("shared constant", fac(off.ratio_shared_cluster_out), "ours", "published_shared"),
        ("family constant", A("published_calc_family_constant"), "ours", "published_family"),
        ("From crystal structure", None, None, None),
        ("model, uncorrected", A("ml_uncorrected_in_domain"), "raw", "model_uncorrected"),
        ("model + shared", A("ml_shared_in_domain"), "ours", "model_shared"),
        ("constant null", A("null_constant_in_domain"), "null", "null_constant"),
        ("power-law null", A("null_power_law_in_domain"), "null", "null_power"),
    ]


def main() -> int:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    R = rows()
    F.use_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    W, H = F.SINGLE / F.MM, 74.0
    fig = plt.figure(figsize=(W * F.MM, H * F.MM))
    ax = fig.add_axes([33.0 / W, 9.5 / H, (W - 33.0 - 11.0) / W, (H - 11.0) / H])
    col = {"ref": "#6E747A", "raw": F.VERMILLION, "ours": F.OURS, "null": "#A9AEB3"}
    XMAX = 5.0
    rng = np.random.default_rng(7)
    summary, ytick, ylab = {}, [], []
    y = 0.0
    for lab, vals, kind, key in R:
        if vals is None:
            y -= 0.35
            ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=F.FS_MIN, fontweight="bold", color="#1F1F1F")
            y -= 0.85
            continue
        v = np.asarray(vals, float)
        med = float(np.median(v))
        vc = np.minimum(v, XMAX * 0.985)
        jit = rng.uniform(-0.22, 0.22, size=len(v))
        ax.scatter(vc, y + jit, s=5, color=col[kind], alpha=0.55, lw=0, zorder=3)
        n_clip = int((v > XMAX).sum())
        if n_clip:
            ax.annotate(f"{n_clip} beyond", xy=(XMAX * 0.985, y), xytext=(0, -10.5),
                        textcoords="offset points", ha="right", va="center", fontsize=F.FS_MIN,
                        color="#555555")
        ax.plot([med, med], [y - 0.36, y + 0.36], color=col[kind] if kind != "null" else "#6E747A",
                lw=2.0, solid_capstyle="butt", zorder=4)
        ax.text(1.015, y, f"{med:.2f}$\\times$", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=F.FS_MIN, color="#1F1F1F")
        ytick.append(y)
        ylab.append(f"{lab} ({len(v)})")
        summary[key] = dict(n=int(len(v)), median_factor=round(med, 2))
        y -= 1.0
    ref = summary["lab_vs_lab"]["median_factor"]
    ax.axvspan(1.0, ref, color="#E9ECEF", zorder=0, lw=0)
    # say what the band is on the figure itself, in the empty strip under the last row
    ax.text(np.sqrt(ref), y - 0.05, "within lab\nspread", ha="center", va="center",
            fontsize=F.FS_MIN, color="#5F6368", linespacing=1.0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(1.0, XMAX)
    ax.xaxis.set_major_locator(mticker.FixedLocator([1, 1.5, 2, 3, 5]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["1", "1.5", "2", "3", "5"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(y - 0.5, 0.0)
    ax.set_yticks(ytick)
    ax.set_yticklabels(ylab, fontsize=F.FS_MIN)
    ax.tick_params(axis="y", which="both", length=0, pad=2, right=False, left=False)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", labelsize=F.FS_TICK, pad=1.5, top=False)
    ax.set_xlabel("disagreement factor, $10^{|\\log_{10} r|}$", labelpad=1.5)
    ax.grid(axis="x", color="#E3E3E3", lw=0.5, zorder=0)
    ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    reg = json.loads((EX / "paper_numbers.json").read_text())["error_budget"]
    assert "_unavailable" not in reg, "registry error_budget unavailable -- run make_paper_numbers.py"
    for k, v in summary.items():
        assert reg["rows"][k] == v, f"error_budget.{k}: figure {v} vs registry {reg['rows'][k]}"
    out = F.save(fig, "fig16_error_budget")
    meta = dict(_generated_by="paper/fig16_error_budget.py",
                unit="per-compound factor 10^|log10 r| (>= 1); median over compounds",
                note=("lab_vs_lab and calc_vs_calc compare two sources with each other; every other row "
                      "compares a prediction with one chosen reference measurement"),
                rows=summary)
    (EX / "error_budget.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(summary, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
