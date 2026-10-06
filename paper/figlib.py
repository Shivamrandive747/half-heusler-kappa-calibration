"""Shared helpers for the manuscript figures.

Two jobs. First, keep the six figures dimensionally consistent -- a journal cares about column
width in millimetres and matplotlib thinks in inches, and converting by hand every time is how
figures end up subtly different sizes. Second, always write a PNG next to the PDF, because the PDF
is what the journal gets and the PNG is what gets LOOKED at before it is called finished. A figure
that has never been viewed is not a finished figure.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
STYLE = str(ROOT / "paperstyle.mplstyle")
OUTDIR = ROOT / "figures"

# Elsevier column widths, millimetres -> inches
MM = 1 / 25.4
SINGLE = 90 * MM
ONEHALF = 140 * MM
DOUBLE = 190 * MM

# Okabe-Ito, named so figures read by intent rather than by hex
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#000000"
GREY = "#8C8C8C"
LIGHTGREY = "#D9D9D9"

# ---------------------------------------------------------------------------------------------
# SEMANTIC ROLES -- the single most important consistency decision in the paper.
#
# Colour encodes the ROLE a quantity plays, not merely which series it is, and the same role wears
# the same colour in every figure. A reader learns the scheme once in Figure 2 and can then read
# Figures 3-5 without consulting a legend at all:
#
#     measurement        vermillion, filled, solid   -- the ground truth; blue's colour-blind-safe
#                                                    partner (author, 2026-10-06: no black blocks)
#     uncorrected DFT    grey, open, DASHED          -- always visibly provisional
#     our calibration    accent blue, solid          -- the paper's contribution, one accent only
#     issued prediction  accent blue, OPEN marker    -- same colour, hollow: no measurement exists
#     refused / flagged  desaturated grey            -- present, deliberately recessive
#
# Two rules follow and must not be broken. Nothing else in the paper may use the accent blue, or it
# stops meaning "our result". And every role carries a marker or line style as well as a hue, so
# the encoding survives greyscale printing and colour-vision deficiency.
MEASURED = VERMILLION
MEASURED_LIGHT = "#FBE3D4"   # fill for measurement boxes and bars that carry text
RAW_DFT = "#9AA0A6"
OURS = BLUE
PREDICTED = BLUE
REFUSED = "#B8BCC0"
BAND = "#E8E8E8"

ROLE = {
    "measured":  dict(color=MEASURED, marker="o", ls="none", ms=4.0, mfc=MEASURED,
                      mec="white", mew=0.5, zorder=5, label="measured"),
    "raw":       dict(color=RAW_DFT, marker="none", ls="--", lw=1.1, zorder=3,
                      label="uncorrected DFT"),
    "ours":      dict(color=OURS, marker="none", ls="-", lw=1.5, zorder=4,
                      label="calibrated (this work)"),
    "predicted": dict(color=PREDICTED, marker="o", ls="none", ms=4.0, mfc="white",
                      mec=PREDICTED, mew=1.2, zorder=5, label="prediction"),
}


# Continuous ramp for temperature. Truncated `inferno`: warm reads as hot, it is perceptually
# uniform and monotonic in lightness (so it survives greyscale), and it stays clear of the accent
# blue, which means "our result" and must not be spent on a covariate. The ends are trimmed
# because pure black is confusable with the measurement ink and pure near-white disappears.
def temp_cmap(lo: float = 0.12, hi: float = 0.90):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    base = plt.get_cmap("inferno")
    return mcolors.LinearSegmentedColormap.from_list(
        "temp", base(np.linspace(lo, hi, 256)))


def use_style() -> None:
    plt.style.use(STYLE)


def nested_calibrate(d, cp_without_cluster, apply_cp, cluster_of):
    """Calibrate each compound with its OWN chemistry cluster held out of the (c, p) fit.

    Every figure that quotes an accuracy statistic must use this, not a single fit over the whole
    set. A review found the figures scoring an in-sample calibration against leave-one-out nulls
    and printing 30% beside a table that reported 31.0% -- the method held to a weaker standard
    than its own baselines, in the one panel whose title advertises that the nulls are held out.

    `cp_without_cluster(label)` returns the (c, p) fitted without that cluster -- since 2026-10-05
    shared_constant.shared_cp_without_cluster, the shared constant fitted on published calculations
    (PREREG_shared_constant_on_calculations rule 4), no longer a fit on the other clusters' model
    predictions.

    Returns the frame with a `k_cal` column and the per-fold constants in `_c`, `_p`.
    """
    import pandas as pd
    out = []
    clus = d.compound.map(lambda c: cluster_of(c) or "?")
    for cl in sorted(set(clus)):
        te, tr = d[clus == cl], d[clus != cl]
        if not len(te) or len(tr) < 4:
            continue
        ci, pi = cp_without_cluster(cl)
        if not np.isfinite(ci):
            continue
        out.append(te.assign(k_cal=te.k_pred.values * apply_cp(te["T"].values, ci, pi),
                             _c=ci, _p=pi))
    return pd.concat(out) if out else None


def figure(width: float = SINGLE, height: float | None = None, **kw):
    """A figure at a journal column width. Height defaults to a 4:3-ish block."""
    use_style()
    return plt.subplots(figsize=(width, height if height else width * 0.72), **kw)


def save(fig, name: str, also_png: bool = True) -> Path:
    """Write the PDF the journal wants, plus a PNG so the figure can actually be inspected."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTDIR / f"{name}.pdf"
    # dpi applies ONLY to artists marked rasterized=True; everything else stays vector. 600 is the
    # usual journal floor for a raster panel, and without it a rasterized layer would fall back to
    # the ~100 dpi screen default and look soft in print.
    fig.savefig(pdf, dpi=600)
    if also_png:
        fig.savefig(OUTDIR / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf}  (+ .png for inspection)")
    return pdf


def band_2x(ax, lo: float, hi: float, color: str = "#EDEDED", alpha: float = 1.0,
            label: str | None = "within 2x") -> None:
    """Shade the factor-of-two envelope around the 1:1 line on a parity plot.

    The band is the paper's headline criterion, so it belongs under the data rather than in the
    caption -- a reader should be able to count the points outside it. It therefore needs an EDGE:
    an unbounded grey wedge reads as background texture, while a bounded one reads as a threshold
    that a point is inside or outside of. Fill lightened and edges drawn for that reason.
    """
    x = np.logspace(np.log10(lo), np.log10(hi), 100)
    ax.fill_between(x, x / 2, x * 2, color=color, alpha=alpha, lw=0, zorder=0, label=label)
    for f in (0.5, 2.0):
        ax.plot([lo, hi], [lo * f, hi * f], color="#C4C4C4", lw=0.5, ls=(0, (4, 2.5)), zorder=1)
    ax.plot([lo, hi], [lo, hi], color="#6E6E6E", lw=0.8, ls="--", zorder=2)


# ---------------------------------------------------------------------------------------------
# ONE CONDUCTIVITY SCALE FOR THE WHOLE PAPER.
#
# Every figure that draws a lattice thermal conductivity used to compute its own limits from its
# own data, so the same quantity appeared at four different scales: Figure 2 spanned 2-30, Figure 4
# spanned 0.3-20, Figure 5 was hardcoded to 0.62-44 and Figure 6 fitted itself to its own bars. A
# reader comparing a 5 W/m/K prediction in Figure 5 against a 5 W/m/K measurement in Figure 2 had to
# re-read both axes to find out they were the same number. Figure 4 even carried a comment claiming
# it used "the same conductivity ticks as Figures 2, 3 and 5" -- the TICKS were harmonised while the
# LIMITS were not, which is the half of the job that does not survive being looked at.
#
# The range is the union of what the figures actually draw, not a round number: the lowest drawn
# value is the lower 90% bound of the Sb-Pd prediction near 0.6, and the highest is a calculated
# value near 32. `warn_clip` fails loudly rather than silently cropping a point, because a shared
# axis that hides data is worse than four inconsistent ones.
# 0.5-35 rather than a range wide enough to swallow every outlier. One out-of-domain compound in
# Figure 4 measures 0.24, and extending the floor to reach it cost every panel in the paper half a
# decade of empty space -- Figure 2's cloud, which filled its panel, retreated into the top corner.
# A single deliberately-recessive point is not worth that, so it is drawn ON the axis edge by
# `edge_markers` and labelled, which is what a journal figure does with an out-of-range value.
# The floor is 0.38 rather than 0.5 for one specific reason: LaSbPt's lower 90% bound in Figure 6
# sits at 0.40, and truncating the whisker of an interval plot removes the very thing the interval
# is there to show. A scatter point can be moved to the edge and flagged; the end of a whisker
# cannot. 0.12 of a decade is a cheap price for drawing every interval in full.
KAPPA_LIM = (0.38, 35.0)
KAPPA_TICKS = (0.5, 1, 2, 3, 5, 10, 20, 30)
KAPPA_LABEL = "$\\kappa_L$ (W m$^{-1}$ K$^{-1}$)"


def warn_clip(vals, tag: str) -> int:
    """Report any conductivity the shared scale would cut off. Returns how many."""
    v = np.asarray([x for x in np.ravel(np.asarray(vals, dtype=float))
                    if np.isfinite(x) and x > 0], dtype=float)
    if not len(v):
        return 0
    below = int((v < KAPPA_LIM[0]).sum())
    above = int((v > KAPPA_LIM[1]).sum())
    if below or above:
        print(f"  CLIPPED in {tag}: {below} value(s) below {KAPPA_LIM[0]}, "
              f"{above} above {KAPPA_LIM[1]} (data range "
              f"{v.min():.2f}-{v.max():.2f}) -- widen KAPPA_LIM or exclude them deliberately")
    return below + above


def edge_markers(ax, x, y, axis: str = "y", color: str = "#B8BCC0") -> int:
    """Draw points that fall below the shared scale ON the axis floor, as open triangles.

    Dropping a point silently is not an option, and stretching the axis for one outlier costs every
    other figure. The triangle says "this value is off the scale, in this direction", which is the
    convention a reader already knows, and the count is annotated so nothing is hidden.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = KAPPA_LIM[0]
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    below = ok & ((y < lo) if axis == "y" else (x < lo))
    if not below.any():
        return 0
    if axis == "y":
        xs, ys, mark = np.clip(x[below], lo * 1.05, None), np.full(int(below.sum()), lo * 1.10), "v"
    else:
        xs, ys, mark = np.full(int(below.sum()), lo * 1.10), np.clip(y[below], lo * 1.05, None), "<"
    ax.scatter(xs, ys, marker=mark, s=20, facecolors="none", edgecolors=color, lw=0.9, zorder=6)
    return int(below.sum())


def kappa_ticks(ax, both: bool = True) -> None:
    """The shared conductivity tick set, restricted to what the shared limits show."""
    log_ticks(ax, [v for v in KAPPA_TICKS if KAPPA_LIM[0] <= v <= KAPPA_LIM[1]], both=both)


def log_ticks(ax, values=(2, 3, 5, 10, 20, 30, 50), both: bool = True) -> None:
    """Label a log axis at readable values rather than at decades.

    Over one and a half decades matplotlib's default leaves a single 10^1 on the axis, and the
    reader cannot estimate any quantity from the figure. Plain integers, not powers.
    """
    import matplotlib.ticker as mticker
    for axis in ((ax.xaxis, ax.yaxis) if both else (ax.xaxis,)):
        axis.set_major_locator(mticker.FixedLocator(values))
        axis.set_major_formatter(mticker.FixedFormatter([str(v) for v in values]))
        axis.set_minor_locator(mticker.NullLocator())


def parity_axes(ax, lo: float, hi: float, xlabel: str, ylabel: str) -> None:
    """Square, log-log, equal limits -- the only honest framing for a ratio-valued comparison."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.04, *,
                outside: bool = False, loc: str = "upper left") -> None:
    """A boxed "(a)" inside the axes, as in Paliwal and Alam, Phys. Rev. Materials 9 (2025).

    Placing it inside the frame keeps the label attached to its panel when a journal reflows the
    float, and costs no margin: an outside label has to be paid for in whitespace on every panel
    of a nine-panel grid. `dx`/`dy` are honoured only with `outside=True`, which is kept for the
    rare panel whose upper corners are both occupied by data.
    """
    if outside:
        ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top", ha="left")
        return
    x, y, ha, va = {"upper left": (0.035, 0.965, "left", "top"),
                    "upper right": (0.965, 0.965, "right", "top"),
                    "lower left": (0.035, 0.035, "left", "bottom"),
                    "lower right": (0.965, 0.035, "right", "bottom")}[loc]
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va=va, ha=ha, zorder=20,
            bbox=dict(boxstyle="square,pad=0.28", fc="white", ec="#BBBBBB", lw=0.5, alpha=0.92))


def annotate_n(ax, n: int, where: str = "lower right") -> None:
    pos = {"lower right": (0.97, 0.03, "right", "bottom"),
           "lower left": (0.03, 0.03, "left", "bottom"),
           "upper left": (0.03, 0.97, "left", "top"),
           "upper right": (0.97, 0.97, "right", "top")}[where]
    ax.text(pos[0], pos[1], f"n = {n}", transform=ax.transAxes,
            ha=pos[2], va=pos[3], fontsize=7, color="#444444")


def check_cvd(colors: list) -> None:
    """Crude deuteranopia simulation, so a palette choice is checked rather than assumed.

    Not a substitute for a proper simulator, but it catches the common failure of two series that
    are obviously different to normal vision and identical without the green channel.
    """
    def sim(hexc):
        r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
        # Vienot-Brettel-Mollon deuteranope approximation in linear-ish RGB
        return (0.625 * r + 0.375 * g, 0.7 * r + 0.3 * g, b)
    out = [sim(c) for c in colors]
    print("  deuteranopia check (pairs closer than 0.10 are a problem):")
    worst = 1.0
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            d = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(out[i], out[j]))))
            worst = min(worst, d)
            if d < 0.10:
                print(f"    {colors[i]} vs {colors[j]}   separation {d:.3f}  TOO CLOSE")
    print(f"    minimum separation across all pairs: {worst:.3f}")
