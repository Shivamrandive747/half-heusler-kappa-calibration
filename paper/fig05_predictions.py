"""Figure 5 -- the issued predictions, each beside the measurements that license it.

ONE COMPACT FOREST PANEL, replacing a 2x2 grid of temperature panels that was mostly empty: every
issued compound rests on a single calculated temperature, so the temperature axis carried no curve
and the panels carried one point each.

  * One row per ISSUED compound (PAPER_PREDICTIONS.csv), at the temperature it is QUOTED at, with its
    50% and 90% conformal bands (lo50/hi50, lo90/hi90 as tabulated -- a 500 K quote carries the 500 K
    band). A 500 K quote is a diamond and says so in its row label.
  * Rows are grouped by (Y,Z) bonding family. Each group opens with its measurements:
      validated family -> every measured member's median at 250-350 K (black dots) and the shaded span
                          they define, carried down through the group's rows. The span is
                          make_paper_predictions.measured_family_range on the same training file the
                          previous version of this figure read -- the span the range gate uses -- and
                          it is checked against the fam_measured_range column.
      anchored family  -> one measured member, so no span: its reference laboratory's value
                          (reference_changes.json, the laboratory the constant was fitted on) at each
                          quoted temperature, carried down as a dashed line, with any other
                          laboratory's value for the same compound drawn faint beside it.
  * A last group holds the two structure-only predictions (STRUCTURE_ONLY_12_PREDICTIONS.csv, the
    registry's structure_only source): the model's theory-scale value (open) moved by the shared
    transfer function to the experimental scale (filled), as in fig10_ml (c). Neither family has a
    measured member.

FLAGGED and REFUSED compounds are never drawn and never named; the script asserts it.
Runs BEFORE make_paper_numbers.py, so it reads the registry's sources, not the registry; when the
registry exists its prediction count and structure-only values are cross-checked.
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import json
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
PRED = "data/Target_Materials/PAPER_PREDICTIONS.csv"
SO = "data/Target_Materials/STRUCTURE_ONLY_12_PREDICTIONS.csv"
SO_MAIN = ("TmPbAu", "HfTeOs")
EX = "data/exports/kappa_v2"
WIN = 50                       # +-K window: the 250-350 K of measured_family_range, at any T
SPAN = "#E4E4E4"
GREY = "#8A8A8A"


def red(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def yzfam(f):
    try:
        e = sorted(Composition(str(f)).elements, key=lambda x: (x.X if x.X else 99.0))
        return f"{e[1]}-{e[2]}" if len(e) == 3 else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    import shared_constant as SCN
    from make_paper_predictions import measured_family_range

    c_sh, p_sh = SCN.shared_cp()
    FAMCAL = json.load(open(f"{EX}/family_calibration.json"))["families"]
    REF = {k: v["doi"] for k, v in json.load(open(f"{EX}/reference_changes.json"))["all_choices"].items()}

    tr = pd.read_csv(TRAIN, low_memory=False)
    tr["red"] = tr.formula.map(red)
    tr["tier"] = pd.to_numeric(tr.method_tier, errors="coerce")
    tr["k"] = pd.to_numeric(tr.kappa_L, errors="coerce")
    tr["T"] = pd.to_numeric(tr.temperature_K, errors="coerce")
    tr = tr.dropna(subset=["red", "k", "T"])
    tr = tr[(tr.k > 0) & tr["T"].between(150, 1050)]
    tr["fam"] = tr.red.map(yzfam)
    meas = tr[tr.tier == 0]
    SPANS = measured_family_range(tr)                    # the range gate's own 250-350 K spans

    P = pd.read_csv(PRED)
    iss = P[P.status == "ISSUED"].copy()
    hidden = set(P[P.status != "ISSUED"].compound)       # FLAGGED + REFUSED: never drawn, never named
    assert len(iss), "no ISSUED predictions to plot"
    assert not (set(iss.compound) & hidden), "a compound is both ISSUED and not"
    S = pd.read_csv(SO).set_index("compound").loc[list(SO_MAIN)]

    # ---- honesty checks against the registry, when one exists ---------------------------------
    try:
        N = json.load(open(f"{EX}/paper_numbers.json"))
        n_reg = len(N["predictions"]["ISSUED"])
        if n_reg != len(iss):
            print(f"  !! registry lists {n_reg} ISSUED, PAPER_PREDICTIONS {len(iss)} -- registry is stale; "
                  "rerun make_paper_numbers.py")
        for cpd in SO_MAIN:
            rg = N["structure_only"]["main_text"][cpd]
            for k in ("kappa_theory_300", "kappa_exp_shared_300"):
                if abs(rg[k] - float(S.loc[cpd, k])) > 0.006:
                    print(f"  !! {cpd} {k}: registry {rg[k]} vs {SO} {S.loc[cpd, k]} -- registry stale")
    except (FileNotFoundError, KeyError) as e:
        print(f"  registry cross-check skipped ({e})")

    # ---- the family context at a temperature ----------------------------------------------------
    def members_at(fam, T):
        m = meas[(meas.fam == fam) & meas["T"].between(T - WIN, T + WIN)]
        return m.groupby("red").k.median()

    def anchored_at(fam, T):
        """(member, reference-lab value, {other doi: value}) for a one-member family."""
        mem = sorted(set(meas[meas.fam == fam].red))
        assert len(mem) == 1, f"{fam} is drawn as anchored but has {len(mem)} measured members"
        cm = mem[0]
        m = meas[(meas.red == cm) & meas["T"].between(T - WIN, T + WIN)]
        by = m.groupby(m.source_doi.astype(str).str.lower()).k.median()
        ref = str(REF[cm]).lower()
        assert ref in by.index, f"{cm}: reference laboratory has no value near {T} K"
        return cm, float(by[ref]), by.drop(ref).to_dict()

    # ---- build the rows ---------------------------------------------------------------------------
    def basis(fam):
        fc = FAMCAL.get(fam, {})
        if fc.get("loo_family_ape") is not None:
            return "validated", f"held out {fc['loo_family_ape']:.1f}%"
        if fc.get("insample_ape") is not None:
            return "anchored", f"anchored, in-sample {fc['insample_ape']:.1f}%"
        raise SystemExit(f"{fam}: issued with neither a held-out nor an anchored constant")

    fams = sorted(set(iss.family),
                  key=lambda f: (basis(f)[0] != "validated", iss[iss.family == f].kappa_pred_quoted.median()))
    rows = []          # dict(kind, y, label, ...)
    y = 0
    groups = []
    for fam in fams:
        kind, btxt = basis(fam)
        g = iss[iss.family == fam].sort_values("kappa_pred_quoted")
        Ts = sorted(set(g.quoted_at_K.astype(int)))
        y0 = y
        if kind == "validated":
            for T in Ts:
                assert T == 300, f"{fam}: a validated family quoted at {T} K -- span needs a check"
            mk = members_at(fam, 300)
            lo_, hi_, n_ = SPANS[fam]
            assert np.isclose(mk.min(), lo_) and np.isclose(mk.max(), hi_) and len(mk) == n_
            tab = set(g.fam_measured_range)
            assert tab == {f"{lo_:.1f}-{hi_:.1f} (n={n_})"}, f"{fam}: span {tab} != recomputed"
            rows.append(dict(kind="meas", y=y, fam=fam, vals=mk.values,
                             label=f"{fam}: {n_} measured", note=btxt))
        else:
            ctx = {T: anchored_at(fam, T) for T in Ts}
            cm = ctx[Ts[0]][0]
            rows.append(dict(kind="anchor", y=y, fam=fam, ctx=ctx,
                             label=f"{fam}: {cm} measured", note=btxt))
        y += 1
        for r in g.itertuples():
            T = int(r.quoted_at_K)
            k, l9, h9, l5, h5 = (float(r.kappa_pred_quoted), float(r.lo90), float(r.hi90),
                                 float(r.lo50), float(r.hi50))
            assert l9 <= l5 <= k <= h5 <= h9, f"{r.compound}: value outside its own band"
            assert int(getattr(r, "band_at_K", T)) == T, f"{r.compound}: band not at the quoted T"
            # the context drawn on THIS row is the family's measurement at THIS row's temperature
            ctx = (("span", SPANS[fam][0], SPANS[fam][1]) if kind == "validated"
                   else ("ref", anchored_at(fam, T)[1]))
            rows.append(dict(kind="iss", y=y, fam=fam, T=T, k=k, l9=l9, h9=h9, l5=l5, h5=h5, ctx=ctx,
                             label=r.compound + (f", {T} K" if T != 300 else ""),
                             note=f"{k:.1f} ({l9:.1f}–{h9:.1f})"))
            y += 1
        groups.append((y0, y - 1))
    y0 = y
    rows.append(dict(kind="sohead", y=y, label="structure only", note=f"$\\times${c_sh:.2f} at 300 K"))
    y += 1
    for cpd in sorted(SO_MAIN, key=lambda c_: float(S.loc[c_, "kappa_exp_shared_300"])):
        kt, ke = float(S.loc[cpd, "kappa_theory_300"]), float(S.loc[cpd, "kappa_exp_shared_300"])
        rows.append(dict(kind="so", y=y, kt=kt, ke=ke, label=f"{cpd} ({S.loc[cpd, 'family']})",
                         note=f"{ke:.2f} (model {kt:.2f})"))
        y += 1
    groups.append((y0, y - 1))
    nrow = y

    drawn = [r for r in rows if r["kind"] == "iss"]
    assert sorted(r["label"].split(",")[0] for r in drawn) == sorted(iss.compound), "an ISSUED row was dropped"
    F.warn_clip([v for r in drawn for v in (r["l9"], r["h9"])] +
                [v for r in rows if r["kind"] == "so" for v in (r["kt"], r["ke"])], "fig05")

    # ---- draw ---------------------------------------------------------------------------------------
    F.use_style()
    fig = plt.figure(figsize=(F.SINGLE, (nrow * 3.55 + 27) * F.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.43], wspace=0.0)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1], sharey=ax)
    H = 0.5
    for r in rows:
        yy = r["y"]
        if r["kind"] == "meas":
            # members within 4% of each other would print as one dot: step them off the row line
            v = np.sort(np.asarray(r["vals"], dtype=float))
            off = np.zeros(len(v))
            for i in range(1, len(v)):
                if np.log10(v[i] / v[i - 1]) < 0.017 and off[i - 1] == 0:
                    off[i - 1], off[i] = -0.17, 0.17
            ax.scatter(v, yy + off, s=9, c=F.MEASURED, lw=0, zorder=5)
        elif r["kind"] == "anchor":
            for T, (_, v, others) in r["ctx"].items():
                ax.scatter([v], [yy], s=13 if T == 300 else 15, c=F.MEASURED, lw=0, zorder=5,
                           marker="o" if T == 300 else "D")
                for ov in others.values():
                    ax.scatter([ov], [yy], s=11 if T == 300 else 13, facecolors="none",
                               edgecolors=GREY, lw=0.7, zorder=4, marker="o" if T == 300 else "D")
        elif r["kind"] == "iss":
            if r["ctx"][0] == "span":
                ax.fill_betweenx([yy - H, yy + H], r["ctx"][1], r["ctx"][2], color=SPAN, lw=0, zorder=1)
            else:
                ax.plot([r["ctx"][1]] * 2, [yy - H, yy + H], color=GREY, lw=0.9, ls=(0, (2.5, 1.5)),
                        zorder=2)
            ax.plot([r["l9"], r["h9"]], [yy, yy], color=F.OURS, lw=0.9, zorder=3, solid_capstyle="butt")
            ax.plot([r["l5"], r["h5"]], [yy, yy], color=F.OURS, lw=3.2, alpha=0.40, zorder=3,
                    solid_capstyle="butt")
            ax.scatter([r["k"]], [yy], s=20 if r["T"] == 300 else 19, c=F.OURS, zorder=6,
                       marker="o" if r["T"] == 300 else "D", lw=0.5, edgecolors="white")
        elif r["kind"] == "so":
            ax.annotate("", xy=(r["ke"] * 1.13, yy), xytext=(r["kt"] / 1.13, yy),
                        arrowprops=dict(arrowstyle="-|>", color=F.OURS, lw=0.9, mutation_scale=6,
                                        shrinkA=0, shrinkB=0), zorder=4)
            ax.scatter([r["kt"]], [yy], s=20, facecolors="white", edgecolors=F.OURS, lw=1.1, zorder=6)
            ax.scatter([r["ke"]], [yy], s=20, c=F.OURS, lw=0.5, edgecolors="white", zorder=6)
        # the right-hand column: the value a reader would otherwise read off a log axis
        head = r["kind"] in ("meas", "anchor", "sohead")
        axt.text(0.04, yy, r["note"], va="center", ha="left", fontsize=6.2,
                 color="#555555" if head else (F.OURS if r["kind"] in ("iss", "so") else "#333333"),
                 style="italic" if head else "normal", transform=axt.get_yaxis_transform())
    # the span of a validated family continues through the header row, so the members sit inside it
    for r in rows:
        if r["kind"] == "meas":
            lo_, hi_, _ = SPANS[r["fam"]]
            ax.fill_betweenx([r["y"] - H, r["y"] + H], lo_, hi_, color=SPAN, lw=0, zorder=1)
    for a, b in groups[1:]:
        for a_ in (ax, axt):
            a_.axhline(a - 0.5, color="#BBBBBB", lw=0.5, zorder=0)

    ax.set_xscale("log")
    ax.set_xlim(*F.KAPPA_LIM)
    F.kappa_ticks(ax, both=False)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["label"] for r in rows], fontsize=6.6)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r["kind"] in ("meas", "anchor", "sohead"):
            t.set_fontweight("bold")
            t.set_color("#333333")
    ax.tick_params(axis="y", which="both", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(F.KAPPA_LABEL)
    axt.set_axis_off()
    axt.text(0.04, -0.62, "$\\kappa_L$ (90% band)", transform=axt.get_yaxis_transform(), ha="left",
             va="bottom", fontsize=6.2, color="#333333", fontweight="bold")

    handles = [
        (Line2D([], [], color=F.MEASURED, marker="o", ls="none", ms=3.2), "measured member"),
        (Patch(fc=SPAN, ec="none"), "measured span of family"),
        (Line2D([], [], color=F.OURS, marker="o", ls="-", lw=0.9, ms=4, mec="white", mew=0.5),
         "prediction, 90% band"),
        (Line2D([], [], color=F.OURS, alpha=0.40, lw=3.2), "50% band"),
        (Line2D([], [], color=F.OURS, marker="D", ls="none", ms=3.6, mec="white", mew=0.5),
         " / ".join(f"{t} K" for t in sorted(set(iss.quoted_at_K.astype(int)) - {300})) + " (else 300 K)"),
        (Line2D([], [], color=GREY, lw=0.9, ls=(0, (2.5, 1.5))), "anchor, reference lab"),
        (Line2D([], [], color=GREY, marker="o", ls="none", ms=3.2, mfc="none", mew=0.7),
         "anchor, other lab"),
        (Line2D([], [], color=F.OURS, marker="o", ls="none", ms=4, mfc="white", mew=1.1),
         "model, theory scale"),
    ]
    fig.legend([h for h, _ in handles], [t for _, t in handles], loc="outside upper center", ncol=2,
               frameon=False, fontsize=6.2, handlelength=1.6, columnspacing=1.0, labelspacing=0.25)

    # nothing FLAGGED or REFUSED may be named anywhere on the figure
    fig.canvas.draw()
    texts = [t.get_text() for t in fig.findobj(lambda o: hasattr(o, "get_text"))]
    leak = sorted({h for h in hidden for t in texts if h in t})
    assert not leak, f"FLAGGED/REFUSED compound named on the figure: {leak}"

    F.save(fig, "fig05_predictions")
    for r in rows:
        if r["kind"] in ("iss", "so"):
            print(f"    {r['label']:<20} {r['note']}")
    print(f"  {len(drawn)} ISSUED drawn ({len(hidden)} flagged/refused withheld); shared c={c_sh}, p={p_sh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
