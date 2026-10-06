# Manuscript

**Predicting the experimental lattice thermal conductivity of half-Heusler compounds by machine
learning and transfer-function calibration** — targeted at *Computational Materials Science*.

## Compiling

Upload this folder to Overleaf and set `manuscript.tex` as the main document. There is no LaTeX
toolchain in the repository, so the first compile happens there. `supplementary.tex` is a
standalone document that shares `references.bib`; compile it the same way.

```
pdflatex manuscript
bibtex   manuscript
pdflatex manuscript
pdflatex manuscript
```

## Before you submit — run the gates

From the **repository root**, not this folder:

```bash
python make_paper_numbers.py                  # regenerate the registry every quoted number comes from
python verify_citations.py paper/manuscript.tex
python check_tex_structure.py                 # braces, refs, corrupted macros, blocklisted numbers
python audit_completion.py                    # every table cell and headline number against its artefact
```

The citation gate is **blocking**. It fails on an undefined `\cite` key, a DOI that no longer
resolves at Crossref or DataCite, or a citation with no supporting sentence recorded in
`claims_ledger.csv`.

## Files

| file | what it is |
|---|---|
| `manuscript.tex` | wrapper: preamble, title, front matter, data availability, back matter |
| `section0_abstract.tex` … `section4_conclusions.tex` | the body: abstract, introduction, methods, results, conclusions |
| `section7_provenance.tex` | the provenance `\nocite` blocks |
| `supplementary.tex` | the supplementary material |
| `references.bib` | **GENERATED.** Never edit by hand — `build_bibliography.py` overwrites it |
| `extra_dois.txt` | hand-added DOIs for references not in the data pipeline |
| `claims_ledger.csv` | claim ↔ source mapping, with the sentence from each source |
| `figures/*.pdf` | the figures; `.png` copies are for viewing, the PDFs go to the journal |
| `fig*.py`, `figlib.py`, `paperstyle.mplstyle` | the figure code and shared style |

### Figures

| main-text figure | script |
|---|---|
| Fig. 1 — the method | `fig08_method.py` |
| Fig. 2 — the corpus | `fig01_corpus.py` |
| Fig. 3 — the calculation–measurement offset | `fig02_offset.py` |
| Fig. 4 — family constants and held-out κ_L(T) curves | `fig09_family.py` (with `fig03_curves.py`) |
| Fig. 5 — the model step and the end-to-end blind test | `fig15_model_blind.py` |
| Fig. 6 — the issued predictions | `fig05_predictions.py` |

Supplementary figures: `fig11_shap.py`, `fig12_dataset.py`, `fig13_parity_models.py`,
`fig14_feature_corr.py`, `fig06_landscape.py`. The graphical abstract is
`fig_graphical_abstract.py`. `fig03_curves.py` and `fig04_blind.py` also write registry inputs;
`fig07_model.py` and `fig10_ml.py` draw earlier figures that are no longer in the paper.

## Outstanding before submission

- **The Zenodo DOI in `\repourl` must be re-minted** for the release that matches this version of
  the paper, then pasted into `manuscript.tex` and `CITATION.cff`.

## Rules that are enforced in code, not by memory

- No number is typed into the text. Everything traces to
  `data/exports/kappa_v2/paper_numbers.json`, which `make_paper_numbers.py` regenerates.
- No citation is written from model memory. Every entry in `references.bib` was resolved from a
  DOI through Crossref or DataCite.
- Withdrawn figures must not reappear. The enforced list lives in `check_tex_structure.py`.
- The headline is the **median over five model seeds**, quoted with its seed range, never a single
  seed. The registry's `HOW_TO_STATE` entries say how each number may be quoted.

See `.claude/skills/manuscript/SKILL.md` for the full claims ledger and blocklist, and
`.claude/skills/citations/SKILL.md` for the citation policy.
