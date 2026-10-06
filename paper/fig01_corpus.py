"""Figure 1 (printed as Figure 2) -- why this paper exists: calculations are plentiful, measurements
are not.

The paper's entire premise is one ratio: many half Heuslers carry a published full
Boltzmann-transport lattice thermal conductivity, few have been measured in a laboratory. Any model
trained on the abundant quantity learns to reproduce CALCULATIONS, and inherits whatever systematic
offset the calculations carry. This figure has to make that scarcity felt in about two seconds,
because every later figure depends on the reader accepting it.

REDESIGN (2026-10-05, author: panel (a) "looks very simple").
  (a) was a four-bar proportional funnel. It drew the ratio, but its last two bars were not
      successive stages -- a measured compound need not carry a calculation -- so the funnel shape
      itself implied a nesting the data do not have, and the caption had to say so. Panel (a) is now
      an UpSet matrix of the half-Heusler corpus by EVIDENCE TYPE: one row per method tier, one
      column per combination of tiers a compound actually carries, with the number of compounds in
      each combination on top and the size of each tier on the left. It still shows the ratio (the
      tier-0 and tier-1 bars on the left), and it now shows what the funnel hid: which measured
      compounds also carry a full calculation -- the only ones on which a calculation can be
      checked against experiment -- and how much of the corpus rests on ML-potential or
      semi-empirical values only, which this paper never fits. Columns are grouped (measured, then
      not) and sorted by count within each group; the counts are all read from the training set.
  (b) How many independent publications stand behind each measured compound, now split by whether
      the compound also carries a full calculation. A first draft plotted the measured and
      calculated clouds together; 2,400 measured points buried the calculated ones and the offset
      was invisible. Figure 3 makes that argument properly on temperature-matched pairs. This panel
      carries information nothing else in the paper does: many of the measured compounds rest on a
      SINGLE publication, which is the honest limit on the evidence.

Tier 1 here is the script's own definition (tier 1 AND a transport-calculation method label, not a
semi-empirical one), exactly as before. Tier-1 rows failing that test are left out of the tier-1
row; on today's corpus they belong to compounds that carry a qualifying calculation anyway, so no
combination changes, and the script prints how many there are so that stays checked.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pymatgen.core import Composition

import figlib as F

TRAIN = "data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv"
BTE = "bte|boltz|phono3py|shengbte|almabte|iterative|rta"
NOT_BTE = "not bte|semi-empirical|slack|debye-callaway|reduced-model|reduced model|bte-approx"

# evidence rows of the matrix, top to bottom. Colour follows the paper's roles: measurement is the
# darkest ink, the full calculation is the uncorrected-DFT grey, and the two tiers this paper never
# fits recede further. Each row is also NAMED, so identity never rests on the grey level alone.
ROWS = [("M", "measured (tier 0)", F.MEASURED),
        ("B", "full DFT-BTE (tier 1)", "#7D848B"),
        ("ML", "ML-potential BTE (tier 2)", "#B3B8BD"),
        ("S", "semi-empirical (tier 3)", "#D3D6D9")]


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    from run_loco_chemistry import sites
    import source_identity as SI

    # a preprint and the journal article it became are ONE paper, not two -- without this, panel
    # (b)'s "independent publications" count could double one source.
    tr = SI.dedupe(pd.read_csv(TRAIN, low_memory=False))
    tr["red"] = tr.formula.map(red)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr = tr.dropna(subset=["red", "k", "T"])
    tr = tr[tr.k > 0]
    tr["klass"] = tr.red.map(lambda r: (sites(r) or ["?"])[0])
    half = tr[tr.klass == "half"]
    m = half.method.astype(str).str.lower()
    is_bte = m.str.contains(BTE, regex=True) & ~m.str.contains(NOT_BTE, regex=True)
    meas = half[half.tier == 0]
    calc = half[(half.tier == 1) & is_bte]
    n_all, n_half = tr.red.nunique(), half.red.nunique()
    n_calc, n_meas = calc.red.nunique(), meas.red.nunique()
    t1_other = set(half[(half.tier == 1) & ~is_bte].red) - set(calc.red)
    print(f"  Heusler compounds={n_all}   half={n_half}   calculated={n_calc}   measured={n_meas}")
    print(f"  tier-1 compounds with no qualifying calculation (left out of row B): {len(t1_other)}")
    print(f"  papers behind the measured set: {meas.source_doi.nunique()}")

    # ---- evidence sets ------------------------------------------------------------------------
    sets = {"M": set(meas.red), "B": set(calc.red),
            "ML": set(half[half.tier == 2].red), "S": set(half[half.tier == 3].red)}
    keys = [k for k, _, _ in ROWS]
    combo = {}
    for r in sorted(set(half.red)):
        sig = tuple(k for k in keys if r in sets[k])
        if sig:
            combo[sig] = combo.get(sig, 0) + 1
    n_none = n_half - sum(combo.values())
    if n_none:
        print(f"  {n_none} half Heuslers carry only tier-1 rows without a qualifying method label")
    # measured combinations first, then the rest; by count within each group
    cols = sorted(combo.items(), key=lambda kv: ("M" not in kv[0], -kv[1], kv[0]))
    n_meas_cols = sum("M" in s for s, _ in cols)
    both = sum(v for s, v in combo.items() if "M" in s and "B" in s)
    print("  combinations: " + ", ".join(f"{'+'.join(s)}={v}" for s, v in cols))
    print(f"  measured AND calculated: {both}; measured only (no tier-1): {n_meas - both}")

    F.use_style()
    fig = plt.figure(figsize=(F.DOUBLE, F.DOUBLE * 0.30))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.06)
    ga = outer[0].subgridspec(2, 2, width_ratios=[0.40, 1.0], height_ratios=[1.0, 1.0],
                              wspace=0.02, hspace=0.04)
    ax_int = fig.add_subplot(ga[0, 1])
    ax_mat = fig.add_subplot(ga[1, 1], sharex=ax_int)
    ax_set = fig.add_subplot(ga[1, 0], sharey=ax_mat)
    ax_txt = fig.add_subplot(ga[0, 0])
    axb = fig.add_subplot(outer[1])

    # gap between the two column groups, so "measured" reads as one block
    xs = np.array([i + (0.0 if i < n_meas_cols else 0.8) for i in range(len(cols))])
    ny = len(ROWS)
    yrow = {k: ny - 1 - i for i, (k, _, _) in enumerate(ROWS)}

    # -- intersection bars
    for x, (sig, v) in zip(xs, cols):
        if "M" in sig:
            col = F.MEASURED
        elif "B" in sig:
            col = "#7D848B"
        else:
            col = "#C4C8CC"
        ax_int.bar(x, v, width=0.68, color=col, zorder=3)
        ax_int.text(x, v + max(combo.values()) * 0.02, str(v), ha="center", va="bottom",
                    fontsize=6.2, color="#333333")
    ymax = max(combo.values()) * 1.36
    ax_int.set_ylim(0, ymax)
    ax_int.set_ylabel("compounds", labelpad=1.5)
    ax_int.tick_params(axis="y", labelsize=6.4, pad=1.5)
    ax_int.yaxis.set_major_locator(mticker.MultipleLocator(100))
    ax_int.grid(False)
    ax_int.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_int.tick_params(axis="y", which="minor", left=False, right=False)
    ax_int.spines[["top", "right"]].set_visible(False)
    ax_int.tick_params(axis="y", right=False)
    # group brackets
    def bracket(x0, x1, label, col):
        y = ymax * 0.91
        ax_int.plot([x0 - 0.34, x1 + 0.34], [y, y], color=col, lw=0.8, clip_on=False)
        for xe in (x0 - 0.34, x1 + 0.34):
            ax_int.plot([xe, xe], [y, y - ymax * 0.035], color=col, lw=0.8, clip_on=False)
        ax_int.text((x0 + x1) / 2, y + ymax * 0.02, label, ha="center", va="bottom",
                    fontsize=6.4, color=col, fontweight="bold")
    bracket(xs[0], xs[n_meas_cols - 1], f"measured: {n_meas}", F.MEASURED)
    bracket(xs[n_meas_cols], xs[-1], f"never measured: {n_half - n_meas}", "#6E757C")

    # -- dot matrix
    for i in range(ny):
        if i % 2 == 0:
            ax_mat.axhspan(i - 0.5, i + 0.5, color="#F3F4F5", lw=0, zorder=0)
    for x, (sig, _) in zip(xs, cols):
        on = [yrow[k] for k in sig]
        off = [yrow[k] for k in keys if k not in sig]
        ax_mat.scatter([x] * len(off), off, s=13, color="#DDE0E3", edgecolor="none", zorder=2)
        dc = F.MEASURED if "M" in sig else "#4A5056"
        if len(on) > 1:
            ax_mat.plot([x, x], [min(on), max(on)], color=dc, lw=1.1, zorder=3)
        ax_mat.scatter([x] * len(on), on, s=15, color=dc, edgecolor="none", zorder=4)
    ax_mat.set_ylim(-0.5, ny - 0.5)
    ax_mat.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax_mat.set_yticks([yrow[k] for k in keys])
    ax_mat.set_yticklabels([lab for _, lab, _ in ROWS], fontsize=6.3)
    ax_mat.tick_params(axis="y", which="both", left=False, right=False, pad=2)
    ax_mat.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_mat.grid(False)
    for sp in ax_mat.spines.values():
        sp.set_visible(False)
    ax_mat.yaxis.set_label_position("left")

    # -- set sizes, drawn leftward
    sizes = [len(sets[k]) for k in keys]
    for (k, _, col), n in zip(ROWS, sizes):
        ax_set.barh(yrow[k], n, height=0.62, color=col, zorder=3,
                    edgecolor="#9AA0A6" if col == "#D3D6D9" else "none", lw=0.4)
        ax_set.text(n + max(sizes) * 0.04, yrow[k], f"{n}", ha="right", va="center",
                    fontsize=6.6, fontweight="bold",
                    color={"M": F.MEASURED, "B": "#33383D"}.get(k, "#555555"), zorder=4)
    ax_set.set_xlim(max(sizes) * 1.38, 0)
    ax_set.set_xlabel("compounds per tier", fontsize=6.6, labelpad=1.0)
    ax_set.grid(False)
    ax_set.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax_set.tick_params(axis="x", which="both", top=False, labelsize=6.2, pad=1.2)
    ax_set.tick_params(axis="x", which="minor", bottom=False)
    ax_set.spines[["top", "left", "right"]].set_visible(False)
    ax_set.set_xticks([0, 200, 400])

    # -- the headline, in the empty corner. Computed, never typed: an earlier draft read
    # "10 to 1" from memory while the data said 5.6.
    ratio = n_calc / n_meas
    ax_txt.axis("off")
    F.panel_label(ax_txt, "a", loc="upper left")
    ax_txt.text(0.0, 0.70, f"{n_half} half Heuslers\nwith a $\\kappa_L$ record\n"
                f"(of {n_all} Heuslers)", transform=ax_txt.transAxes, fontsize=6.6, va="top",
                ha="left", color="#333333", linespacing=1.25)
    ax_txt.text(0.0, 0.18, f"calculated : measured\n= {n_calc} : {n_meas} = {ratio:.1f} : 1",
                transform=ax_txt.transAxes, fontsize=6.6, va="top", ha="left",
                color="#222222", fontweight="bold", linespacing=1.25)
    # No figure title (2026-10-06 compaction): the caption names the panel. The column-bar key sits
    # inside the intersection panel, over the measured group, where the bars are short.
    leg = ax_int.legend(handles=[Patch(fc=F.MEASURED, label="includes a measurement"),
                                 Patch(fc="#7D848B", label="tier-1 calculation, never measured"),
                                 Patch(fc="#C4C8CC", label="tiers 2-3 only")],
                        loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=6.0,
                        frameon=False, handlelength=1.0, handleheight=0.7, borderaxespad=0.2,
                        labelspacing=0.2, handletextpad=0.4)
    ax_int.text(xs[n_meas_cols - 1] + 0.3, max(v for s, v in cols if "M" in s) * 1.0
                + ymax * 0.07, f"{both} of {n_meas} measured also\ncarry a tier-1 calculation",
                ha="right", va="bottom", fontsize=6.0, color=F.MEASURED, linespacing=1.15)

    # (b) how much independent evidence stands behind each measured compound
    src = meas.groupby("red").source_doi.nunique()
    has_calc = src.index.isin(sets["B"])
    bins = [1, 2, 3, 5, 10, 10_000]
    names = ["1", "2", "3-4", "5-9", "10+"]
    inb = [(src >= lo) & (src < hi) for lo, hi in zip(bins[:-1], bins[1:])]
    c_yes = [int((b & has_calc).sum()) for b in inb]
    c_no = [int((b & ~has_calc).sum()) for b in inb]
    counts = [a + b for a, b in zip(c_yes, c_no)]
    # Evidence count is ordinal: one hue, light to dark. Vermillion stays on the single-publication
    # bar, which is the panel's point. The accent blue is reserved for "our result" and is not
    # spent on the literature. The no-calculation part of each bar is the same hue, lighter and
    # hatched, so the split survives greyscale.
    bcols = [F.VERMILLION, "#A4AAB0", "#8C939A", "#737B83", "#5E666E"]
    plt.rcParams["hatch.linewidth"] = 0.6
    x = np.arange(len(names))
    for i in range(len(names)):
        axb.bar(x[i], c_yes[i], width=0.66, color=bcols[i], zorder=3)
        axb.bar(x[i], c_no[i], bottom=c_yes[i], width=0.66, color="white", edgecolor=bcols[i],
                hatch="/////", lw=0.8, zorder=3)
        axb.text(x[i], counts[i] + max(counts) * 0.03, str(counts[i]), ha="center", fontsize=7.0,
                 fontweight="bold" if i == 0 else "normal",
                 color=F.VERMILLION if i == 0 else "#555555")
    axb.set_xticks(x)
    axb.set_xticklabels(names)
    axb.tick_params(axis="x", which="minor", bottom=False, top=False)
    axb.set_xlabel("independent papers per compound", labelpad=1.5)
    axb.set_ylabel("measured compounds", labelpad=1.5)
    axb.tick_params(labelsize=6.6, pad=1.5)
    axb.set_ylim(0, max(counts) * 1.42)
    axb.grid(False)
    axb.grid(axis="y", color="#E6E6E6", lw=0.4)
    axb.legend(handles=[Patch(fc="#5E666E", ec="none", label="also has a tier-1 calculation"),
                        Patch(fc="white", ec="#5E666E", hatch="/////",
                              label="measured only")],
               loc="upper right", fontsize=6.0, handlelength=1.4, borderaxespad=0.4)
    axb.set_title(f"{counts[0]} of the {sum(counts)} measured rest on one paper",
                  fontsize=7.2, loc="center", pad=3)
    print("  sources per measured compound (calc / no calc): " +
          ", ".join(f"{n}={a}/{b}" for n, a, b in zip(names, c_yes, c_no)))
    F.panel_label(axb, "b", loc="upper left")

    F.save(fig, "fig01_corpus")
    # the caption's counts, so they reach the registry instead of being read off the console
    import json
    Path("data/exports/kappa_v2/corpus_figure.json").write_text(json.dumps(dict(
        _generated_by="paper/fig01_corpus.py",
        n_heusler=int(n_all), n_half=int(n_half), n_calculated=int(n_calc), n_measured=int(n_meas),
        n_measured_with_tier1=int(both), n_measured_only=int(n_meas - both),
        n_measured_single_publication=int(counts[0]),
        n_single_publication_with_tier1=int(c_yes[0]), n_single_publication_no_tier1=int(c_no[0])),
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
