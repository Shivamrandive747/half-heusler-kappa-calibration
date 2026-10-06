"""Assemble release_data/ -- the curated files that ship with the paper.

WHY THIS EXISTS. `.gitignore` excludes `data/` wholesale, and rightly so: it runs to several
gigabytes, most of it publisher PDFs that must never be redistributed. But the curated outputs the paper
promises to release also live under `data/`, so they would be silently absent from the public
repository -- the same trap that hid paper/evidence/ earlier in this project. Listing a file in
LICENSE-DATA does not put it in the release; copying it into a tracked directory does.

WHAT IT COPIES. Only our own curated products: the prediction lists, the fitted constants, and the
per-seed blind-test results. It deliberately does NOT copy the training set or any bulk harvest,
because those are derived from Starrydata2, AFLOW, Materials Project, JARVIS, OQMD and COD, and
redistributing them wholesale is not ours to do. The harvest scripts rebuild them from the public
APIs instead. See LICENSE-DATA.

Run:  python build_release_data.py
Then: git add release_data && git commit
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import glob
import hashlib
import io
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "release_data"

# (source, destination subdirectory, one-line description for the manifest)
WANTED: list[tuple[str, str, str]] = [
    ("data/Target_Materials/PAPER_PREDICTIONS.csv", "predictions",
     "Every candidate the method assessed: {n_issued} issued, {n_flagged} flagged, {n_refused} refused, "
     "each with its status and reason (Table 3 of the manuscript)."),
    ("data/Target_Materials/CONDITIONAL_PREDICTIONS.csv", "predictions",
     "The {n_conditional} conditional estimates and the measurement each waits on (Table S1)."),
    ("data/Target_Materials/STRUCTURE_ONLY_29_VERIFIED.csv", "predictions",
     "Compounds predicted from crystal structure alone that have no published kappa_L: category, "
     "structure source, model value, seed spread, experimental-scale value, literature status."),
    ("data/Target_Materials/STRUCTURE_ONLY_12_PREDICTIONS.csv", "predictions",
     "Structure-only predictions with full structure provenance and family held-out evidence "
     "(TmPbAu, HfTeOs and the weaker candidates), at 300, 600 and 900 K."),
    ("data/Target_Materials/CALIBRATED_NEW_PREDICTIONS.csv", "predictions",
     "New compounds inside a calibrated family (LuSbPd; SmBiPd flagged): two independent routes."),
    ("data/Target_Materials/SURROGATE_11_PREDICTIONS.csv", "predictions",
     "Compounds whose only prior kappa_L is a machine-learning surrogate value: our estimates and "
     "the literature check."),
    ("data/exports/kappa_v2/seed_averaged_indomain.json", "results",
     "Per-seed and seed-median blind-test results; the source of every headline number."),
    ("data/exports/kappa_v2/domain_headline.json", "results",
     "Fitted (c, p) and accuracy at each stage of the domain screen."),
    ("data/exports/kappa_v2/conformal_indomain.json", "results",
     "Conformal interval half-widths and their measured coverage."),
    ("data/exports/kappa_v2/paper_numbers.json", "results",
     "The numbers registry: corpus counts, nulls, and paired tests."),
    ("data/exports/kappa_v2/family_null.json", "results",
     "The family power-law baseline."),
    ("data/exports/kappa_v2/loco_chemistry.json", "results",
     "Leave-one-chemistry-out validation output."),
    ("data/exports/kappa_v2/interlab_ceiling.json", "results",
     "Inter-laboratory reproducibility of the reference measurements."),
]

# THE REMOVED-ROW RECORDS (FIXPASS C5, 2026-10-05). Every data-quality rule writes the rows it removed or
# re-tiered to an audit CSV, so a reader can see exactly what left the corpus and why. They ship in
# release_data/audit/. Matched by pattern because each rule names its own file; backup copies never match.
AUDIT_GLOBS = ["data/Target_Materials/EXCLUDED_*.csv",          # rows removed by a rule (+ the rule)
               "data/Target_Materials/TIER_MOVES_*.csv",        # rows whose method tier was re-derived
               "data/external/kappa_starrydata_heusler_duplicate_samples.csv"]  # harvest-time duplicate gate
AUDIT_DESC = "Removed-row / re-tiered-row record written by a data-quality rule (see its rule column)."
BACKUP_COPY = re.compile(r"(_backup|backup_|\.bak|\.pre_|_preaudit|\.orig\b|_before_)", re.I)

# PRIVACY. Inputs that are not redistributed must never ship: neither the files themselves, nor any
# column derived from membership of them, nor one of their identifiers appearing anywhere in a copied
# file. The names come from the local manifest (private_manifest.py), so this file names none of them;
# without the manifest only the generic "uuid" column check remains.
import private_manifest as PM  # noqa: E402

PRIVATE_FILES = PM.private_files()
PRIVATE_COLUMN = PM.column_pattern()


def _private_hits(src: Path, uuids: set) -> list:
    """Why `src` may not ship (empty list = clean)."""
    why = []
    rel = src.relative_to(ROOT).as_posix()
    if rel in PRIVATE_FILES:
        why.append("is a non-redistributed input")
    text = src.read_text(encoding="utf-8", errors="replace")
    if src.suffix == ".csv":
        head = text.splitlines()[0] if text else ""
        cols = [c for c in head.split(",") if PRIVATE_COLUMN.search(c)]
        if cols:
            why.append(f"private column(s) {cols}")
    elif PRIVATE_COLUMN.search(text):
        why.append("mentions a private column name")
    if uuids:
        hit = next((u for u in uuids if u in text), None)
        if hit:
            why.append(f"contains a private identifier ({hit[:8]}...)")
    return why


MANIFEST = """# release_data

Curated outputs accompanying the manuscript. Generated by `build_release_data.py` -- do not edit
by hand; edit the source and re-run.

These are **our own products**: the predictions, the fitted calibration, and the validation
results. Bulk copies of the upstream databases are deliberately not included; the harvest scripts
rebuild those from their public APIs. See `../LICENSE-DATA` for the full scope and terms.

The per-compound blind-test scores and the five model seeds live in `../paper/evidence/`, which is
tracked separately because the figure scripts read them from there.

| file | bytes | sha256 (first 16) | what it is |
|---|---|---|---|
"""


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    rows, missing, total = [], [], 0

    # the counts in the descriptions are READ, not typed: "five issued" and "ten conditional"
    # outlived both numbers by several revisions
    import pandas as pd
    _P = pd.read_csv(ROOT / "data/Target_Materials/PAPER_PREDICTIONS.csv")
    _C = pd.read_csv(ROOT / "data/Target_Materials/CONDITIONAL_PREDICTIONS.csv")
    counts = dict(n_issued=int((_P.status == "ISSUED").sum()),
                  n_flagged=int((_P.status == "FLAGGED").sum()),
                  n_refused=int((_P.status == "REFUSED").sum()), n_conditional=len(_C))
    uuids = PM.uuids()
    audit = sorted({Path(f).relative_to(ROOT).as_posix() for g in AUDIT_GLOBS for f in glob.glob(str(ROOT / g))
                    if not BACKUP_COPY.search(Path(f).name)})
    refused = []
    for rel, sub, desc in WANTED + [(a, "audit", AUDIT_DESC) for a in audit]:
        desc = desc.format(**counts)
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        why = _private_hits(src, uuids)
        if why:
            refused.append(f"{rel}: {'; '.join(why)}")
            continue
        dst = OUT / sub / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        data = src.read_bytes()
        total += len(data)
        digest = hashlib.sha256(data).hexdigest()[:16]
        rows.append(f"| `{sub}/{src.name}` | {len(data):,} | `{digest}` | {desc} |")
        print(f"  copied  {sub}/{src.name:<34} {len(data):>9,} bytes")

    if refused:
        print("\n  !! REFUSED -- non-redistributed data, not copied:")
        for r in refused:
            print(f"     {r}")
    if missing:
        print("\n  !! NOT FOUND -- these were promised in LICENSE-DATA but do not exist:")
        for m in missing:
            print(f"     {m}")

    OUT.mkdir(exist_ok=True)
    io.open(OUT / "README.md", "w", encoding="utf-8", newline="\n").write(
        MANIFEST + "\n".join(rows) + "\n")

    print(f"\n  {len(rows)} files, {total:,} bytes total -> release_data/")
    print("  manifest written to release_data/README.md")
    if missing or refused:
        print("\n  FAILED: fix the missing paths (or remove them from LICENSE-DATA) and strip private "
              "columns from any refused file.")
        return 1
    print("\n  OK. Next: git add release_data && git commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
