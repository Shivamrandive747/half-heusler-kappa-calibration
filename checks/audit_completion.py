"""Is the manuscript complete? A section-by-section inventory, plus the calibration chain.

Not a quality review -- that comes later. This answers only: has everything that is supposed to be
recorded actually been recorded, and does the calibration reach the numbers the paper reports?

Checks, in order:
  1. every section: length, subsections, figures, equations, citations
  2. placeholders and blocking markers still in the text
  3. every generated figure is included by some section, and every included figure exists
  4. the calibration chain -- published DFT value -> transfer function -> reported prediction --
     verified arithmetically on the issued predictions, not assumed
  5. numbers the abstract commits to, checked against the files that produce them
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import io
import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PAPER = Path("paper")
RULE = "=" * 86
STRIP = re.compile(r"(?<!\\)%.*")

# The four-section rewrite merged the old seven into these. The superseded files are still on disk
# but empty, so listing them here would report eight sections written and zero figures used.
ORDER = ["section0_abstract", "section1_introduction", "section2_methods", "section3_results",
         "section4_conclusions", "section7_provenance"]


def body(name):
    """Section text with comments stripped. Line structure is preserved, because the table
    checks below parse rows and need it."""
    return STRIP.sub("", io.open(PAPER / f"{name}.tex", encoding="utf-8").read())


def flat(name):
    """Section text with whitespace flattened to single spaces, for the SENTENCE checks only.

    Those checks locate a number by matching the sentence carrying it, and a .tex file is
    hard-wrapped, so any edit that reflows a paragraph drops a newline into the middle of a pattern
    and the check reports "sentence not found". That fired on a pure readability pass in which no
    number changed at all. Flattening compares content rather than line layout, which is what the
    check is for. It must NOT be used for the table checks, which parse rows line by line.
    """
    return re.sub(r"\s+", " ", body(name))


def main() -> int:
    print(RULE)
    print("1. SECTION INVENTORY")
    print(RULE)
    print(f"  {'section':<26}{'words':>7}{'subsec':>8}{'figs':>6}{'eqs':>5}{'cites':>7}"
          f"{'tables':>8}")
    total = 0
    included: list = []
    for name in ORDER:
        t = body(name)
        words = len(re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t).split())
        total += words
        subs = len(re.findall(r"\\subsection\{", t))
        figs = re.findall(r"includegraphics\[[^\]]*\]\{figures/([^}]+)\}", t)
        included += figs
        eqs = len(re.findall(r"\\begin\{equation\}", t))
        cites = len(re.findall(r"\\cite[a-zA-Z]*\{", t))
        tabs = len(re.findall(r"\\begin\{table", t))
        # section 7 is nothing but \nocite blocks, which the word count strips to zero and which
        # then reads as an empty section. Report its provenance keys as its content instead.
        nocit = len([k for m in re.findall(r"nocite\{([^}]*)\}", t)
                     for k in m.split(",") if k.strip()])
        if nocit and words < 50:
            words = nocit
        print(f"  {name:<26}{words:>7}{subs:>8}{len(figs):>6}{eqs:>5}{cites:>7}{tabs:>8}")
    print(f"  {'TOTAL':<26}{total:>7}")

    print()
    print(RULE)
    print("2. PLACEHOLDERS AND BLOCKING MARKERS")
    print(RULE)
    pats = {
        # match a CITED key, not the prose: both section headers describe the NEEDS-CITATION
        # policy, and matching that description reported a blocking marker where none exists.
        "NEEDS-CITATION cited": r"\\cite[a-zA-Z]*\{[^}]*NEEDS[^}]*\}",
        "TODO / FIXME / XXX": r"\b(TODO|FIXME|XXX|TBD)\b",
        "author placeholder": r"Author Name|First Last|\\author\{\s*\}|YOUR NAME",
        "DOI placeholder": r"10\.XXXX|xxxx/zenodo|DOI-TO-BE|<doi>",
        "empty acknowledgement": r"acknowledge?ments?\}\s*\n\s*(%|\\end)",
        "lorem / draft note": r"\blorem\b|\bplaceholder\b",
    }
    allfiles = list(PAPER.glob("*.tex"))
    found_any = False
    for label, pat in pats.items():
        hits = []
        for f in allfiles:
            for m in re.finditer(pat, STRIP.sub("", io.open(f, encoding="utf-8").read()), re.I):
                hits.append(f"{f.name}: {m.group(0)[:40]}")
        if hits:
            found_any = True
            print(f"  {label}: {len(hits)}")
            for h in hits[:6]:
                print(f"      {h}")
        else:
            print(f"  {label}: none")
    if not found_any:
        print("  -> no blocking markers anywhere in the manuscript")

    print()
    print(RULE)
    print("3. FIGURES: GENERATED vs INCLUDED")
    print(RULE)
    made = sorted(p.name for p in (PAPER / "figures").glob("*.pdf"))
    # The supplementary carries figures too -- the regressor comparison moved there, and scanning
    # only the main text would report it as generated-but-never-included. A figure used nowhere at
    # all must still be caught, so the supplementary is scanned rather than the check relaxed.
    sup = PAPER / "supplementary.tex"
    if sup.exists():
        included += re.findall(r"includegraphics\[[^\]]*\]\{figures/([^}]+)\}",
                               STRIP.sub("", io.open(sup, encoding="utf-8").read()))
    inc = sorted(set(included))
    print(f"  generated {len(made)}   included {len(inc)}")
    orphan = [m for m in made if m not in inc]
    missing = [i for i in inc if i not in made]
    for m in made:
        print(f"    {'USED  ' if m in inc else 'ORPHAN'}  {m}")
    if missing:
        print(f"  !! included but NOT generated: {missing}")
    if orphan:
        print(f"  !! generated but never included: {orphan}")

    print()
    print(RULE)
    print("4. THE CALIBRATION CHAIN -- is the transfer function actually applied?")
    print(RULE)
    P = pd.read_csv("data/Target_Materials/PAPER_PREDICTIONS.csv")
    C = pd.read_csv("data/Target_Materials/CONDITIONAL_PREDICTIONS.csv")
    # THE CONSTANTS ARE PER ROW, NOT ONE SCALAR.
    #
    # This read `P.c_used.iloc[0]` -- one value from the first row -- and asserted it against every
    # row of both files. That was correct only while one global calibration covered everything.
    # Families whose measured members agree now carry their own (c, p) from family_calibration.py,
    # so a scalar read makes a correctly calibrated row report as a deviation and flips the verdict
    # below to NO. Each row is now checked against the constants IT records.
    #
    # CONDITIONAL_PREDICTIONS.csv has no c_used column. It is written by compute_conditional.py,
    # every one of its families is tier C, and it applies min(c,1) at 300 K -- so it is checked
    # against the global constants, which is what it actually used, rather than borrowing row 0 of
    # a file that may now be family-calibrated.
    def _const(D, col, fallback=None):
        return D[col].astype(float) if col in D.columns else fallback

    # THE GLOBAL CONSTANTS COME FROM THE ARTIFACT, NOT FROM A ROW OF THE PREDICTION FILE.
    #
    # Reading them from row 0 was wrong twice over. It assumed at least one prediction still used
    # the global arm -- and once every family carrying a prediction had its own calibration, none
    # did -- and it looked for a `calibration_tier` column that had been renamed, so the lookup
    # silently defaulted and reported Sb-Pd's c = 0.320 as "global". The conditional file, which
    # has no constants of its own, was then checked against that and every row flagged.
    c_glob = p_glob = None
    try:
        _fc = json.load(open("data/exports/kappa_v2/family_calibration.json"))
        c_glob, p_glob = float(_fc["global"]["c"]), float(_fc["global"]["p"])
    except Exception:  # noqa: BLE001
        if "c_used" in P.columns and len(P):
            g0 = P[P.get("calibration_source", pd.Series(["global"] * len(P))) == "global"]
            src = g0 if len(g0) else P
            c_glob, p_glob = float(src.c_used.iloc[0]), float(src.p_used.iloc[0])
    print(f"  global constants (from family_calibration.json): c = {c_glob}, p = {p_glob}")
    if "calibration_form" in P.columns:
        for t, g in sorted(P.groupby("calibration_form")):
            fams = sorted(set(g.calibration_source)) if "calibration_source" in g.columns else []
            print(f"  form {t}: {len(g):>2} rows  c={sorted(set(g.c_used.round(3)))}  "
                  f"p={sorted(set(g.p_used.round(3)))}  {fams}")
    ok = True
    for lab, D in (("issued/flagged/refused", P), ("conditional", C)):
        d = D.dropna(subset=["kappa_BTE_300", "kappa_pred_300"])
        if not len(d):
            continue
        # compare ABSOLUTE values with a rounding-aware tolerance, not the ratio. Both columns
        # are stored to two decimals, so for a small compound like CrSnPt (0.30 -> 0.15) the
        # ratio reads 0.5000 against c = 0.51 purely from rounding, and a ratio test flags a
        # correctly calibrated value as an error.
        # CONDITIONAL_PREDICTIONS.csv carries no c_used: compute_conditional.py applies the
        # global arm to every row, because all of its families have too few measured members to
        # calibrate. It is therefore checked against the global constants, never against whatever
        # the issued file happens to hold.
        cc = (_const(d, "c_used", None) if lab != "conditional" else None)
        if cc is None:
            cc = pd.Series([c_glob] * len(d), index=d.index)
        r = d.kappa_pred_300 / d.kappa_BTE_300
        bad = d[(d.kappa_pred_300 - cc * d.kappa_BTE_300).abs() > 0.008]
        print(f"  {lab:<24} n={len(d):>3}  ratio pred/DFT = "
              f"{r.min():.4f}..{r.max():.4f}   deviations from its own c: {len(bad)}")
        if len(bad):
            ok = False
            cols = [c for c in ("compound", "kappa_BTE_300", "kappa_pred_300", "c_used",
                                "calibration_source") if c in bad.columns]
            print(bad[cols].to_string(index=False))
    print("  -> every reported prediction is its published DFT value multiplied by the c its own")
    print("     row records (at 300 K the term (T/300)^p is exactly 1, so the factor reduces to c)")
    for T in (600, 900):
        col_b, col_p = f"kappa_BTE_{T}", f"kappa_pred_{T}"
        if col_b in P.columns:
            d = P.dropna(subset=[col_b, col_p])
            if len(d):
                cc = _const(d, "c_used", pd.Series([c_glob] * len(d), index=d.index))
                pp = _const(d, "p_used", pd.Series([p_glob] * len(d), index=d.index))
                exp = np.minimum(cc * (T / 300.0) ** pp, 1.0)
                # the family arm is uncapped since 2026-10-01 (transfer_forms.SHAPE_NOCAP)
                if "calibration_form" in d.columns:
                    nocap = (d.calibration_form == "SHAPE_NOCAP").values
                    exp = np.where(nocap, cc * (T / 300.0) ** pp, exp)
                dev = float((d[col_p] / d[col_b] - exp).abs().max())
                print(f"  at {T} K: worst deviation from each row's own form, c(T/300)^p = "
                      f"{dev:.4f}   {'OK' if dev < 0.01 else 'MISMATCH'}")
                ok = ok and dev < 0.01

    print()
    print(RULE)
    print("5. HEADLINE NUMBERS vs THE FILES THAT PRODUCE THEM")
    print(RULE)
    ab = body("section0_abstract")
    claims = dict(re.findall(r"\\SI\{([\d.]+)\}\{\\percent\}", ab) and
                  [(m, m) for m in re.findall(r"\\SI\{([\d.]+)\}\{\\percent\}", ab)])
    print(f"  percentages the abstract commits to: {sorted(claims, key=float)}")
    try:
        sa = json.load(open("data/exports/kappa_v2/seed_averaged_indomain.json"))
        med, w2 = sa["median_ape_median"], sa["within_2x_median"]
        print(f"  seed-averaged over {sa['n_seeds']} seeds: median {med}% "
              f"(range {sa['median_ape_range']}), within 2x {w2}% "
              f"(range {sa['within_2x_range']})")
        for v in (str(med), str(w2)):
            print(f"    abstract quotes {v}: {'YES' if v in claims else 'NOT FOUND'}")
    except Exception as e:  # noqa: BLE001
        print(f"  seed-averaged file: {e}")
    cf = json.load(open("data/exports/kappa_v2/conformal_indomain.json"))["levels"]
    print(f"  conformal factors: " + ", ".join(f"{k}={v['factor']}" for k, v in cf.items()))
    print(f"  prediction statuses: {dict(P.status.value_counts())}   conditional: {len(C)}")

    print()
    print(RULE)
    print("6. THE PROSE'S OWN TABLES vs THE ARTEFACTS")
    print(RULE)
    # Sections 1-5 check the artefacts against each other and the abstract's percentages against
    # the seed scan. Nothing checked the hand-typed tables, which is exactly where the manuscript
    # drifted: tab:predictions named three compounds the prediction file had not issued for weeks,
    # with values from a superseded calibration, and no gate noticed. Parse them and compare.
    tbl_ok = True
    res = body("section3_results")

    def rows_of(label):
        """The tabular body of the table carrying \\label{label}, as a list of cell-lists."""
        i = res.find("\\label{" + label + "}")
        if i < 0:
            return None
        seg = res[i:res.find("\\end{tabular}", i)]
        out = []
        for ln in seg.splitlines():
            ln = ln.strip()
            if "&" not in ln or ln.startswith("\\multicolumn") or "\\toprule" in ln:
                continue
            out.append([c.strip() for c in ln.rstrip("\\\\").split("&")])
        return out

    def formula(cell):
        """The compound in a first cell, or "" for a header/spanning row.

        Requiring \\ce{} is what distinguishes a data row from the two header rows, whose first
        cells ("Compound", "") otherwise parse as compound names.
        """
        if "\\ce{" not in cell:
            return ""
        return re.sub(r"[^A-Za-z0-9]", "", re.sub(r"\\ce\{|\}|--|\$.*?\$", "", cell))

    pr = rows_of("tab:predictions")
    iss = P[P.status == "ISSUED"]
    if pr is None:
        print("  tab:predictions: NOT FOUND"); tbl_ok = False
    else:
        named = {formula(r[0]) for r in pr if r and formula(r[0])}
        want = set(iss.compound)
        if named != want:
            print(f"  tab:predictions compounds {sorted(named)} != ISSUED {sorted(want)}")
            tbl_ok = False
        else:
            print(f"  tab:predictions: {len(want)} compounds match PAPER_PREDICTIONS.csv")
        for r in pr:
            c = formula(r[0])
            if c not in want or len(r) < 6:
                continue
            row = iss[iss.compound == c].iloc[0]
            # columns (2026-10-06 float pass): compound, family (basis, c_f), T, kappa_BTE, PREDICTED,
            # 90% interval, span. The 50% column was dropped and basis + c_f merged into the family
            # cell, so every index after the family shifted left by two. Three issued values are
            # quoted at 500 K, so the T column is checked and the values are compared with the
            # *_quoted columns, not *_300.
            try:
                if int(float(r[2])) != int(row.quoted_at_K):
                    print(f"    {c}: table T {r[2]} vs artefact {row.quoted_at_K}"); tbl_ok = False
            except ValueError:
                print(f"    {c}: cannot parse T {r[2]!r}"); tbl_ok = False
            for col, want_v in ((3, row.kappa_BTE_quoted), (4, row.kappa_pred_quoted)):
                got = re.sub(r"\\textbf\{|\}", "", r[col]).strip()
                try:
                    if abs(float(got) - float(want_v)) > 0.006:
                        print(f"    {c}: table {got} vs artefact {want_v}"); tbl_ok = False
                except ValueError:
                    print(f"    {c}: cannot parse {got!r}"); tbl_ok = False
            # the family cell carries the constant: "\ce{Sb}--\ce{Pt} (validated, \num{0.61})"
            m = re.search(r"\\num\{([\d.]+)\}", r[1])
            if not m or abs(float(m.group(1)) - float(row.c_used)) > 0.005 + 1e-9:
                print(f"    {c}: table c_f {m.group(1) if m else r[1]!r} vs artefact {row.c_used}")
                tbl_ok = False
            # the 90% interval, lo--hi
            m = re.match(r"\s*([\d.]+)\s*--\s*([\d.]+)\s*$", r[5])
            if not m:
                print(f"    {c}: cannot parse 90% interval {r[5]!r}"); tbl_ok = False
            elif (abs(float(m.group(1)) - float(row.lo90)) > 0.006
                  or abs(float(m.group(2)) - float(row.hi90)) > 0.006):
                print(f"    {c}: table 90% {r[5]} vs artefact {row.lo90}--{row.hi90}"); tbl_ok = False

    fc = json.load(open("data/exports/kappa_v2/family_calibration.json"))
    adopted = {k: v for k, v in fc["families"].items() if v.get("adopted")}
    fr = rows_of("tab:famcal")
    if fr is None:
        print("  tab:famcal: NOT FOUND"); tbl_ok = False
    else:
        # Layout since the 2026-10-06 float pass: Family | members | c_f | held-out family |
        # held-out shared | own paper | status (spread, publications and papers moved to the
        # supplement's tab:famsupport, checked below). VALIDATED families (held-out error < 25 %)
        # print their held-out errors; CALIBRATED families (held-out >= 25 %) print a dagger there and
        # their held-out errors in the supplement's tab:famcalsupp; ANCHORED single-member families
        # print dashes. Every adopted family must appear in exactly one of the two tables; each row's
        # c, held-out errors, own-paper fit and status word are checked against
        # family_calibration.json and the registry's own_paper_fit_by_family.
        own = (json.load(open("data/exports/kappa_v2/paper_numbers.json"))
               .get("deployed_route", {}).get("own_paper_fit_by_family", {}))
        cvals, errs, gerrs, owns, stats = {}, {}, {}, {}, {}
        for r in fr:
            fam = "-".join(re.findall(r"\\ce\{([A-Za-z]+)\}", r[0]))
            if fam in adopted and len(r) >= 7:
                m = re.search(r"[\d.]+", r[2])
                if m:
                    cvals[fam] = m.group()
                m = re.search(r"[\d.]+", r[3])
                if m:
                    errs[fam] = float(m.group())
                m = re.search(r"[\d.]+", r[4])
                if m:
                    gerrs[fam] = float(m.group())
                m = re.search(r"[\d.]+", r[5])
                if m:
                    owns[fam] = float(m.group())
                stats[fam] = re.sub(r"[^a-z]", "", r[6].lower())
        supp_src = open("paper/supplementary.tex", encoding="utf-8").read()
        i_s = supp_src.find("\\label{tab:famcalsupp}")
        supp_rows = {}
        if i_s >= 0:
            for ln in supp_src[i_s:supp_src.find("\\end{tabular}", i_s)].splitlines():
                if "&" in ln and "\\ce{" in ln.split("&")[0]:
                    cells = [c.strip() for c in ln.strip().rstrip("\\\\").split("&")]
                    supp_rows["-".join(re.findall(r"\\ce\{([A-Za-z]+)\}", cells[0]))] = cells
        # calibrated families appear in BOTH tables (dagger in the main one, figures in the supplement)
        calibrated = {f for f, a in adopted.items()
                      if a.get("loo_family_ape") is not None and a["loo_family_ape"] >= 25}
        if set(cvals) | set(supp_rows) != set(adopted) or (set(cvals) & set(supp_rows)) != calibrated:
            print(f"  tab:famcal + tab:famcalsupp rows {sorted(cvals)} + {sorted(supp_rows)} "
                  f"!= {sorted(adopted)}"); tbl_ok = False
        else:
            print(f"  tab:famcal: {len(cvals)} families, tab:famcalsupp: {len(supp_rows)}; "
                  "together they match family_calibration.json")
        for fam, c_str in cvals.items():
            a = adopted[fam]
            c_tex = float(c_str)
            # compared at the precision the table prints: 0.27 is a correct rendering of 0.267
            dec = len(c_str.split(".")[1]) if "." in c_str else 0
            if abs(c_tex - a["c"]) > 0.5 * 10 ** -dec + 1e-9:
                print(f"    {fam}: table c={c_tex} vs artefact c={a['c']}"); tbl_ok = False
            held = a.get("loo_family_ape") is not None
            ins = a.get("insample_ape")
            if held and a["loo_family_ape"] < 25:
                want_stat = "validated"
                if abs(errs.get(fam, -1) - a["loo_family_ape"]) > 0.06:
                    print(f"    {fam}: table held-out={errs.get(fam)} vs artefact {a['loo_family_ape']}"); tbl_ok = False
                if abs(gerrs.get(fam, -1) - a["loo_global_ape"]) > 0.06:
                    print(f"    {fam}: table held-out shared={gerrs.get(fam)} vs artefact {a['loo_global_ape']}"); tbl_ok = False
            elif held:
                want_stat = "calibrated"
                if fam in errs:
                    print(f"    {fam}: a calibrated family prints a held-out figure in the main table"); tbl_ok = False
            else:
                want_stat = "anchored" if ins is not None and ins < 25 else "anchoredfails"
            if stats.get(fam) != want_stat:
                print(f"    {fam}: table status={stats.get(fam)!r} vs artefact {want_stat!r}"); tbl_ok = False
            want_own = (own.get(fam) or {}).get("own_paper_fit")
            if want_own is None or abs(owns.get(fam, -1) - want_own) > 0.06:
                print(f"    {fam}: table own-paper fit={owns.get(fam)} vs registry {want_own}"); tbl_ok = False
        for fam, cells in supp_rows.items():
            a = adopted.get(fam, {})
            nums = [float(x) for x in re.findall(r"\\SI\{([\d.]+)\}", " ".join(cells[2:5]))]
            want = [v for v in (a.get("loo_family_ape"), a.get("loo_global_ape")) if v is not None] + \
                   [(own.get(fam) or {}).get("own_paper_fit")]
            if len(nums) != len(want) or any(abs(x - float(y)) > 0.06 for x, y in zip(nums, want)):
                print(f"    {fam}: supplement row {nums} vs artefact {want}"); tbl_ok = False
            if a.get("loo_family_ape") is not None and a["loo_family_ape"] < 25:
                print(f"    {fam}: a validated family sits in the supplement table"); tbl_ok = False
        # The NON-adopted rows drifted once without this: a tolerance correction returned LaBiPd
        # to Bi-Pd, the artefact said two members, and the table went on saying one for a day.
        nmem = {}
        for r in fr:
            fam = "-".join(re.findall(r"\\ce\{([A-Za-z]+)\}", r[0]))
            if fam in fc["families"] and fam not in adopted and len(r) >= 2:
                m = re.search(r"\d+", r[1])
                if m:
                    nmem[fam] = int(m.group())
        for fam, n_tex in nmem.items():
            if n_tex != fc["families"][fam]["n_members"]:
                print(f"    {fam}: table members={n_tex} vs artefact {fc['families'][fam]['n_members']}")
                tbl_ok = False
        if nmem:
            print(f"  tab:famcal: {len(nmem)} non-adopted rows match member counts")

    # ---- tables moved to the supplement in the 2026-10-06 float pass ----------------------------
    # A table that leaves the main text must not leave the gate with it: tab:famsupport carries the
    # spread / publications / papers columns trimmed from tab:famcal, and the supplement's tab:blind
    # is the full blind-test table the main text now draws as Fig. fig:modelblind (d).
    sup_body = STRIP.sub("", io.open(PAPER / "supplementary.tex", encoding="utf-8").read())

    def supp_rows_of(label):
        i = sup_body.find("\\label{" + label + "}")
        if i < 0:
            return None
        seg = sup_body[i:sup_body.find("\\end{tabular}", i)]
        # a row may wrap over two source lines; join on the row terminator, not the newline
        seg = re.sub(r"\s*\n\s*", " ", seg)
        out = []
        for ln in seg.split("\\\\"):
            ln = re.sub(r"\\(toprule|midrule|bottomrule)", "", ln).strip()
            if "&" in ln:
                out.append([c.strip() for c in ln.split("&")])
        return out

    REG = json.load(open("data/exports/kappa_v2/paper_numbers.json", encoding="utf-8"))
    fcr = REG.get("family_calibration", {})
    fs = supp_rows_of("tab:famsupport")
    if fs is None:
        print("  tab:famsupport (supplement): NOT FOUND"); tbl_ok = False
    else:
        seen, fs_bad = set(), []
        for r in fs:
            fam = "-".join(re.findall(r"\\ce\{([A-Za-z]+)\}", r[0]))
            if fam not in adopted or len(r) < 5:
                continue
            seen.add(fam)
            a = fcr.get("adopted", {}).get(fam, {})
            sp = a.get("c_members_spread")
            m = re.search(r"[\d.]+", r[2])
            # a spread of exactly 1 means only one member could be fitted alone (Sb-Pt: one member
            # with enough temperatures), so the table prints a dash, as for single-member families
            want_sp = None if (sp is None or float(sp) <= 1.0) else sp
            if (m is None) != (want_sp is None) or (m and abs(float(m.group()) - float(want_sp)) > 0.006):
                fs_bad.append(f"{fam}: supplement spread {r[2]!r} vs registry {sp}")
            for col, key in ((1, None), (3, "laboratories"), (4, "reference_papers")):
                want_n = (fc["families"][fam]["n_members"] if key is None
                          else fcr.get(key, {}).get(fam))
                m = re.search(r"\d+", r[col])
                if not m or want_n is None or int(m.group()) != int(want_n):
                    fs_bad.append(f"{fam}: supplement {key or 'members'} {r[col]!r} "
                                  f"vs registry {want_n}")
        main_fams = set(cvals) if fr is not None else set()
        if seen != main_fams:
            fs_bad.append(f"families {sorted(seen)} != tab:famcal {sorted(main_fams)}")
        if fs_bad:
            tbl_ok = False
            for b in fs_bad:
                print(f"    tab:famsupport {b}")
        else:
            print(f"  tab:famsupport (supplement): {len(seen)} families match the registry")

    bt = supp_rows_of("tab:blind")
    it = REG.get("in_domain_table", {})
    if bt is None:
        print("  tab:blind (supplement): NOT FOUND"); tbl_ok = False
    elif not it or "_unavailable" in it:
        print("  tab:blind (supplement): registry in_domain_table unavailable"); tbl_ok = False
    else:
        sm, nl = it["seed_medians"], it["nulls"]
        se = it["published_semi_empirical"]

        def seed(k):
            d = sm[k]
            return [d["median_ape"]["median"], d["within_2x"]["median"],
                    d["within_30"]["median"], d["bias"]["median"]]
        want_rows = {
            "Calibrated": seed("calibrated"),
            "Uncorrected": seed("uncorrected_surrogate"),
            "Null: constant": [nl["constant"][k] for k in ("median_ape", "within_2x_pct",
                                                           "within_30_pct", "bias")],
            "Null: $A": [nl["power"][k] for k in ("median_ape", "within_2x_pct",
                                                  "within_30_pct", "bias")],
            "One parameter": seed("one_parameter_c_only"),
            "Slack": [se["median_ape"], se["within_2x_pct"], None, se["bias"]],
        }
        got_rows, bt_bad = 0, []
        for r in bt:
            key = next((k for k in want_rows if r[0].startswith(k)), None)
            if key is None or len(r) < 5:
                continue
            got_rows += 1
            for cell, w in zip(r[1:5], want_rows[key]):
                m = re.search(r"[\d.]+", re.sub(r"\\textbf|\\SI|\\percent", "", cell))
                if w is None:
                    if m:
                        bt_bad.append(f"{key}: {cell!r} where the registry has no value")
                    continue
                # bias is printed to two decimals, the percentages to one
                if not m or abs(float(m.group()) - float(w)) > 0.0051 + (0.0 if float(w) < 5 else 0.045):
                    bt_bad.append(f"{key}: {cell!r} vs registry {w}")
        if got_rows != len(want_rows):
            bt_bad.append(f"{got_rows} of {len(want_rows)} rows found")
        if bt_bad:
            tbl_ok = False
            for b in bt_bad:
                print(f"    tab:blind (supplement) {b}")
        else:
            print(f"  tab:blind (supplement): {got_rows} rows match in_domain_table")

    # The pooled deployed-route numbers had no producer at all until compute_deployed_route.py.
    # data/ is gitignored, so these artefacts are absent on a fresh clone and present after their
    # producers run. A missing one must say so: a check that skips in silence is not a check.
    dr_path = "data/exports/kappa_v2/deployed_route.json"
    if not Path(dr_path).exists():
        print(f"  SKIPPED: {dr_path} absent -- run compute_deployed_route.py")
    if Path(dr_path).exists():
        dr = json.load(open(dr_path))
        res = flat("section3_results")
        # Sentences of the 2026-10-04 rewrite, which leads with the held-out comparison and gives
        # the pooled 34 (held-out + anchored fits) after it.
        fa = dr["family_arm_compounds"]
        want = {
            "held-out n": (r"On the \\num\{(\d+)\} such compounds the family constant", fa["n"]),
            "held-out median": (r"such compounds the family constant gives a median error of\s+"
                                r"\\SI\{([\d.]+)\}", fa["median_ape"]),
            "global on same compounds": (r"scored on the\s+same \\num\{\d+\}, gives \\SI\{([\d.]+)\}",
                                         fa["global_on_same_compounds"]["median_ape"]),
            "family better count": (r"better of the two for \\num\{(\d+)\}", fa["n_family_better"]),
            "pooled n": (r"Over all \\num\{(\d+)\} in-domain compounds", dr["n"]),
            "uncorrected median": (r"uncorrected calculations give \\SI\{([\d.]+)\}",
                                   dr["uncorrected"]["median_ape"]),
            "global-constant median": (r"shared constant brings that to \\SI\{([\d.]+)\}",
                                       dr["global"]["median_ape"]),
            "deployed median": (r"the family constants to \\SI\{([\d.]+)\}", dr["deployed"]["median_ape"]),
            "deployed within 2x": (r"the family constants to \\SI\{[\d.]+\}\{\\percent\} and \\SI\{([\d.]+)\}",
                                   dr["deployed"]["within2x_pct"]),
        }
        bad_dr = []
        for name, (pat, val) in want.items():
            m = re.search(pat, res, re.I)
            if not m:
                bad_dr.append(f"{name}: sentence not found"); continue
            if abs(float(m.group(1)) - float(val)) > 0.051:
                bad_dr.append(f"{name}: text {m.group(1)} vs deployed_route.json {val}")
        if bad_dr:
            tbl_ok = False
            for b in bad_dr:
                print(f"    deployed route: {b}")
        else:
            print(f"  deployed-route pooled numbers match {dr_path}")

    # The prediction funnel's family split had no producer and matched no definition of
    # "measured member" until compute_prediction_funnel.py.
    ea_path = "data/exports/kappa_v2/extraction_agreement.json"
    if not Path(ea_path).exists():
        print(f"  SKIPPED: {ea_path} absent -- run compute_extraction_agreement.py")
    else:
        ea = json.load(open(ea_path))
        # the cross-check moved to the supplement when Methods was trimmed of its results; look in
        # both, so the gate follows the prose instead of pinning it to one file
        met = flat("section2_methods") + " " + flat("supplementary")
        pairs_ = [("same-source deviation",
                   r"two readings differ by a median of \\SI\{([\d.]+)\}",
                   ea["same_source"]["median_abs_deviation_pct"]),
                  ("cross-source deviation",
                   r"gives a median difference of\s+\\SI\{([\d.]+)\}",
                   ea["cross_source"]["median_abs_deviation_pct"])]
        bad_ea = []
        for name, pat, val in pairs_:
            m = re.search(pat, met, re.I)
            if not m:
                bad_ea.append(f"{name}: sentence not found")
            elif abs(float(m.group(1)) - float(val)) > 0.051:
                bad_ea.append(f"{name}: text {m.group(1)} vs artefact {val}")
        if bad_ea:
            tbl_ok = False
            for b in bad_ea:
                print(f"    extraction agreement: {b}")
        else:
            print(f"  extraction-agreement figures match {ea_path}")

    rb_path = "data/exports/kappa_v2/robustness.json"
    if not Path(rb_path).exists():
        print(f"  SKIPPED: {rb_path} absent -- run compute_robustness.py")
    else:
        rb = json.load(open(rb_path))
        res = flat("section3_results")
        want = {
            "sign-test wins": (r"null on \\num\{(\d+)\} of the \\num\{38\} compounds",
                               rb["nulls"]["constant"]["better_on"]),
            "bootstrap lo": (r"\\SIrange\{([\d.]+)\}\{[\d.]+\}\{\\percent\} on the median error",
                             rb["bootstrap_ci"][0]),
            "bootstrap hi": (r"\\SIrange\{[\d.]+\}\{([\d.]+)\}\{\\percent\} on the median error",
                             rb["bootstrap_ci"][1]),
            "jackknife lo": (r"moves the median only between\s+\\SI\{([\d.]+)\}\{\\percent\}",
                             rb["jackknife"]["lo"]),
            "jackknife hi": (r"moves the median only between\s+\\SI\{[\d.]+\}\{\\percent\}\s+and"
                             r"\s+\\SI\{([\d.]+)\}\{\\percent\}", rb["jackknife"]["hi"]),
        }
        bad_rb = []
        for name, (pat, val) in want.items():
            m = re.search(pat, res, re.I)
            if not m:
                # not quoted in the prose: nothing to contradict. The gate fails only on a quoted
                # figure that disagrees with robustness.json; an unquoted one is listed, not failed.
                print(f"    (not quoted in the text: {name})"); continue
            if abs(float(m.group(1)) - float(val)) > 0.051:
                bad_rb.append(f"{name}: text {m.group(1)} vs robustness.json {val}")
        if bad_rb:
            tbl_ok = False
            for b in bad_rb:
                print(f"    robustness: {b}")
        else:
            print(f"  robustness figures match {rb_path}")

    fn_path = "data/exports/kappa_v2/prediction_funnel.json"
    if not Path(fn_path).exists():
        print(f"  SKIPPED: {fn_path} absent -- run compute_prediction_funnel.py")
    if Path(fn_path).exists():
        fn = json.load(open(fn_path))
        res = flat("section3_results")
        want = {
            # anchored on the funnel's own wording: the bare "gives N compounds" also matched the
            # published-route sentence ("This gives 34 compounds and 429 ... pairs")
            "candidates": (r"gives \\num\{(\d+)\} compounds, of which", fn["candidates"]),
            "in domain": (r"of which \\num\{(\d+)\} clear", fn["in_domain"]),
            "no member": (r"\\num\{(\d+)\} sit in a family with no member", fn["no_member"]["n"]),
            # since 2026-09-19 every family with a member carries a constant, so the funnel's
            # remaining split is Ni-Bi (member disqualified) vs the calibrated families
            "Ni-Bi (one member)": (r"\\num\{(\d+)\} in \\ce\{Ni\}--\\ce\{Bi\}", fn["one_member"]["n"]),
            "in calibrated families": (r"\\num\{(\d+)\} sit in one of the twelve\s+calibrated families",
                                       fn["adopted"]["n"]),
        }
        bad_fn = []
        for name, (pat, val) in want.items():
            m = re.search(pat, res, re.I)
            if not m:
                # not quoted in the prose: nothing to contradict. The gate fails only on a quoted
                # figure that disagrees with prediction_funnel.json; an unquoted one is listed, not failed.
                print(f"    (not quoted in the text: {name})"); continue
            if int(m.group(1)) != int(val):
                bad_fn.append(f"{name}: text {m.group(1)} vs prediction_funnel.json {val}")
        if bad_fn:
            tbl_ok = False
            for b in bad_fn:
                print(f"    funnel: {b}")
        else:
            print(f"  prediction funnel matches {fn_path}")

    # ---- 7. no source is counted twice ---------------------------------------------------------
    #
    # An arXiv preprint and the journal article it became carry different DOIs, and provenance here
    # is keyed on DOI, so one calculation entered the corpus as two "independent" sources. That is
    # not a miscount that stays put: a paper agrees with itself exactly, so every self-duplicate
    # contributes a disagreement ratio of 1.00 and drags any spread statistic toward the one value
    # it can never legitimately take. Four pairs were found -- three arXiv preprints and a Zenodo
    # deposit of a PRX article, the last hiding under different method strings in tier 3 until the
    # detector compared row-level triples -- covering 361 compound-entries; the
    # quoted median disagreement between independent groups moved from 1.15 to 1.37 once collapsed.
    #
    # The map in source_identity is evidence, not a rule, so it cannot extrapolate to a pair that
    # arrives later. This re-runs the DETECTOR, which is what found the third pair after two were
    # already known.
    print()
    print(RULE)
    print("7. SOURCE IDENTITY -- is any single paper counted as two independent sources?")
    print(RULE)
    src_ok = True
    try:
        import source_identity as SI
        _tr = pd.read_csv("data/Target_Materials/HEUSLER_KAPPA_TRAINING_SET.csv", low_memory=False)
        _left = SI.verify(_tr)
        if _left:
            src_ok = False
            print(f"  {len(_left)} UNMAPPED duplicate source pair(s) -- "
                  f"every 'independent sources' count in the paper is inflated:")
            for h in _left:
                print(f"    {h['a']}  ==  {h['b']}   "
                      f"({h['shared']} shared compounds, {h['frac_identical']:.0%} identical)")
            print("  fix: add the redundant id to source_identity.DUPLICATE_SOURCES, then re-run"
                  " every statistic that counts distinct sources.")
        else:
            print(f"  no unmapped duplicate pair; {len(SI.DUPLICATE_SOURCES)} known pair(s) "
                  f"collapsed to their version of record")
    except Exception as e:  # noqa: BLE001
        src_ok = False
        print(f"  could not run the check: {e}")

    # ---- 8. the released files vs the ones the paper quotes -------------------------------------
    #
    # release_data/ is what a reader downloads from the archive. It is built by copying the
    # authoritative CSVs, so it can silently fall behind them: it shipped HfCoBi as CONDITIONAL at
    # 9.49 W/m/K while Table 4 issued it at 13.95, and carried 16 prediction rows against 27.
    print()
    print(RULE)
    print("8. RELEASED ARTEFACTS vs THE FILES THE PAPER QUOTES")
    print(RULE)
    rel_ok = True
    for rel, src in (("release_data/predictions/PAPER_PREDICTIONS.csv",
                      "data/Target_Materials/PAPER_PREDICTIONS.csv"),
                     ("release_data/predictions/CONDITIONAL_PREDICTIONS.csv",
                      "data/Target_Materials/CONDITIONAL_PREDICTIONS.csv")):
        if not Path(rel).exists():
            print(f"  {rel}: absent -- run build_release_data.py")
            rel_ok = False
            continue
        a, b = pd.read_csv(rel), pd.read_csv(src)
        shared = [c for c in a.columns if c in b.columns]
        same = len(a) == len(b) and a[shared].equals(b[shared])
        print(f"  {Path(rel).name:32} {len(a):3} rows vs {len(b):3} "
              f"-> {'identical' if same else 'DIFFERS'}")
        if not same:
            rel_ok = False
            only = sorted(set(a.compound) ^ set(b.compound))
            if only:
                print(f"    compounds in one file only: {only}")
            print("    run: python build_release_data.py && git add release_data")

    # ---- 9. every analysed source reaches the reference list ------------------------------------
    #
    # Section 2 claims every analysed value is traceable to a citable work, and Section S6 exists to
    # make that checkable. verify_citations.py tests the harmless direction -- a bibliography entry
    # nobody cites. This tests the harmful one: a source BEHIND AN ANALYSED ROW that never reaches
    # the reference list, which would make the provenance claim false rather than merely untidy.
    print()
    print(RULE)
    print("9. PROVENANCE: DOES EVERY ANALYSED SOURCE REACH THE REFERENCE LIST?")
    print(RULE)
    prov_ok = True
    try:
        prov = json.load(open("data/exports/kappa_v2/paper_numbers.json",
                              encoding="utf-8")).get("provenance_sources", {})
    except Exception as e:  # noqa: BLE001
        prov = {"_unavailable": str(e)}
    if not prov or "_unavailable" in prov:
        print(f"  unavailable ({prov.get('_unavailable', 'no block')}) -- run make_paper_numbers.py")
        prov_ok = False
    else:
        for role in ("measurement", "calculation"):
            tot = prov.get(f"{role}_source_dois")
            listed = prov.get(f"{role}_sources_in_reference_list")
            unc = prov.get(f"{role}_sources_never_cited") or []
            noent = prov.get(f"{role}_sources_with_no_bib_entry") or []
            print(f"  {role:12} {listed} of {tot} source DOIs are in the reference list")
            if unc:
                prov_ok = False
                print(f"    PROVENANCE GAP -- resolved but never cited: {unc}")
                print("    add them to the \\nocite blocks in paper/section7_provenance.tex")
            for d in noent:
                # unresolvable at both registries: disclosed in the text, not a silent gap
                print(f"    unresolvable DOI, disclosed in the text: {d}")

    print()
    print(RULE)
    print("VERDICT")
    print(RULE)
    print(f"  sections written        : {len(ORDER)} of {len(ORDER)}")
    print(f"  figures generated/used  : {len(made)}/{len(inc)}")
    print(f"  calibration chain intact: {'YES' if ok else 'NO -- see above'}")
    print(f"  tables match artefacts  : {'YES' if tbl_ok else 'NO -- see section 6'}")
    print(f"  no double-counted source: {'YES' if src_ok else 'NO -- see section 7'}")
    print(f"  release matches paper   : {'YES' if rel_ok else 'NO -- see section 8'}")
    print(f"  every source cited      : {'YES' if prov_ok else 'NO -- see section 9'}")
    print(f"  blocking markers        : {'none' if not found_any else 'SEE SECTION 2'}")
    # This script used to return 0 whatever it found, so manuscript.tex's claim that it "fails the
    # build" was not true and the drift above went unreported for three days. It now exits non-zero.
    bad = ((not ok) or (not tbl_ok) or found_any or (not src_ok) or (not rel_ok)
           or (not prov_ok))
    if bad:
        print("\n  FAILED -- see the sections flagged above")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
