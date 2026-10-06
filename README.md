# Predicting the experimental lattice thermal conductivity of half-Heusler compounds by machine learning and transfer-function calibration

First-principles Boltzmann-transport (BTE) calculations of the lattice thermal conductivity κ_L of
half-Heusler compounds, and machine-learned models trained on them, reproduce calculations, not
measurements. Published calculations exceed measured κ_L by a **median factor of 1.69**
(34 compounds; calculations run high in 10 of 12 bonding families).

This archive accompanies a paper that predicts the measured κ_L with two steps used as a pair:

1. **A machine-learned model** (CatBoost, 45 descriptors), trained only on full first-principles
   calculations, predicts κ_L^BTE from crystal structure. With each chemistry cluster held out it
   reproduces published calculations to a **median error of 21.9 %** (230 compounds, 88.3 % within a
   factor of two; one seed).
2. **A closed-form transfer function** maps a calculation onto the experimental scale:

$$\kappa_L^{\mathrm{exp}}(T) \;=\; \kappa_L^{\mathrm{BTE}}(T)\;\min\!\left[c\left(\tfrac{T}{300}\right)^{p},\,1\right],
\qquad c = 0.42,\quad p = 0.70$$

   fitted on published calculations against one reference laboratory per compound (34 compounds,
   429 calculation–measurement pairs). The magnitude c is well determined; the exponent p much less
   so.

**End to end**, holding each compound's chemistry cluster out of both steps and inside a predefined
domain of applicability (18 valence electrons, cubic C1_b, no known polymorph), the median error
against measurement is **33.1 %** and **87.2 % of 39 compounds** lie within a factor of two (median of
five model seeds; seed range 24.6–35.4 %). The comparisons are 41.9 % uncorrected, and 46.7 % and
45.6 % for constant and power-law null baselines (paired Wilcoxon p = 0.004; with each of the 18
chemistry clusters counted once, p < 0.05 on every seed, the largest 0.048). Without the domain
screens, all 50 measured compounds give **38.6 %** and **78.9 %**.

The two errors are separate and are stated side by side, never merged: **21.9 %** is the model
against published calculations; **33.1 %** is the full route against measurement, which already
contains the model step.

**Calibrating a published calculation directly**, the shared constant gives 32.3 % on 31 held-out
compounds. A magnitude refitted per bonding family gives **21.7 %** on the same 31, which is not
significantly better compound by compound (sign test p = 0.44). Five families are validated held
out, at 7.4–24.9 %; four more are calibrated on their own papers but not validated; three
single-member families are anchored in sample. A family no one has measured receives only the
shared constant, with a held-out median error of 33.0 %.

| | |
|---|---|
| **Paper** | `paper/manuscript.tex` and `paper/supplementary.tex` (see `paper/README.md` to build them) |
| **Model step** | 21.9 % median error against published calculations, 230 compounds, one seed |
| **End-to-end result** | 33.1 % median error, 87.2 % within 2×, Spearman ρ = 0.662, on 39 compounds in 18 chemistry clusters (five-seed medians); 38.6 % and 78.9 % over all 50 without the domain screens |
| **Corpus** | 735 half Heuslers with a κ_L record: 50 measured, from 152 publications, and 284 with a full transport calculation |
| **Predictions** | 10 issued with prediction intervals, 15 flagged and 6 refused, each with its reason; 10 conditional on a measurement in their family; TmPbAu and HfTeOs from crystal structure alone |
| **Licence** | code MIT, data CC BY 4.0 — see `LICENSE` and `LICENSE-DATA` |

---

## Reproduce the headline in three commands

```bash
pip install -e .                      # or: pip install -r requirements.txt
python analysis/run_target_blind_test.py --seed 0
python analysis/compute_seed_averaged.py
```

The last prints the per-seed table and the five-seed headline. Run everything **from the
repository root** — data paths are relative to it.

Full instructions, including how to rebuild the corpus from scratch, are in
**[`docs/REPRODUCE.md`](docs/REPRODUCE.md)**.

## What is here

```
paper/        the manuscript, the supplementary, the figures and the scripts that draw them
              paper/evidence/   per-compound scores, the five model seeds and the pre-registrations
analysis/     the calibration, validation and prediction code
corpus/       rebuilds the corpus from the public APIs (Starrydata, AFLOW, MP, JARVIS, OQMD, COD)
checks/       the pre-submission gates: citations, LaTeX structure, completeness
pipeline/     the dataset builder and model wrapper the analysis imports
data/         the curated corpus, the predictions, and the results the paper quotes;
              data/exports/kappa_v2/paper_numbers.json is the registry every quoted number comes from
release_data/ the curated outputs in one place: the prediction tables, the key results and the
              removed-row audit records (release_data/README.md lists each file with its checksum)
docs/         how to reproduce, negative results, and the submission checklist
tests/        unit tests for the unit-canonicalisation and formula-matching logic
rerun_downstream.sh   the whole analysis chain in dependency order (see docs/REPRODUCE.md)
```

### Figures and the scripts that draw them

| Main-text figure | Script |
|---|---|
| Fig. 1 — the method | `paper/fig08_method.py` |
| Fig. 2 — the corpus | `paper/fig01_corpus.py` |
| Fig. 3 — the calculation–measurement offset | `paper/fig02_offset.py` |
| Fig. 4 — family constants and held-out κ_L(T) curves | `paper/fig09_family.py` (draws its curve panels with `paper/fig03_curves.py`) |
| Fig. 5 — the model step and the end-to-end blind test | `paper/fig15_model_blind.py` |
| Fig. 6 — the issued predictions | `paper/fig05_predictions.py` |

The supplementary figures are drawn by `fig11_shap.py`, `fig12_dataset.py`,
`fig13_parity_models.py`, `fig14_feature_corr.py` and `fig06_landscape.py`. `fig03_curves.py` and
`fig04_blind.py` also write registry inputs (`curves_best3.json`, `blind_by_family.json`); neither
is a figure of the paper on its own. `fig07_model.py` and `fig10_ml.py` draw earlier figures that
are no longer in the paper. All figure scripts write to `paper/figures/`; run them from the
repository root.

## What is deliberately not here

This repository is the κ-calibration work only. Three things are excluded on purpose:

- **Bulk copies of the upstream databases.** Starrydata2, AFLOW, Materials Project, JARVIS, OQMD
  and COD are other people's resources under their own terms. `corpus/` refetches them from the
  public APIs instead. See `LICENSE-DATA`.
- **Publisher PDFs and full-text XML.** Downloaded under institutional subscription and not ours
  to redistribute.
- **The broader Heusler mining pipeline.** An earlier effort extracted magnetic, mechanical and
  structural properties too. Its only contribution here is 31 measured rows, frozen in
  `data/external/RECOVERED_from_db_rows.csv`; none of its code is on this paper's path, so it is not
  part of this release.

## How the two scales are kept apart

The single property that makes the corpus work: **measurements and calculations are never pooled.**
Every value carries a method tier — 0 experimental, 1 full Boltzmann transport, 2 approximate
first-principles (including transport calculations on machine-learned force constants), 3
semi-empirical — assigned from the *method string the source reports*, not from the repository it
came from. The model trains on tier 1 only; the two populations meet at exactly one place, the
transfer function.

That distinction is load-bearing rather than pedantic. Where a half Heusler carries both a
semi-empirical estimate and a full transport calculation in the same temperature range (161
compounds), the two differ by a median factor of 1.25, and by up to 11.4.

## Citing

See `CITATION.cff`. Please cite both the paper and the archived release. If you use the data you
are also using the upstream sources above; the supplementary material cites the publications behind
the measurements and calculations so that they can be cited too.

## Limitations worth knowing before you use it

- **A value at a stated temperature, not a curve.** One exponent serves every compound and it is
  poorly determined (bootstrap range 0.45–1.15, against 0.35–0.55 for c).
- **A domain of applicability.** The method applies to semiconducting (VEC = 18), cubic,
  non-polymorphic half Heuslers. The 11 measured compounds the screens exclude have a median error
  of 86.4 %, against 31.7 % inside the domain at the same reference seed; we report both.
- **One reference laboratory per compound.** Each measured compound is scored against one
  laboratory chosen by a fixed rule that never reads a calculation.
- **Structure-only values carry two errors**, the model step and the calibration, stated separately,
  and assume the cubic phase forms.

The "Limits of the method" subsection of the paper states each limitation with the number that
quantifies it.
