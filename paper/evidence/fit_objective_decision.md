# Decision on the pre-registered fit-objective test (2026-10-01)

**Pre-registration:** `PREREG_fit_objective.md` (commit 13f43cd).
**Result:** `fit_objective_test.json` (commit 1b6ea4c).
**Deployed:** the current objective (outer median). The candidate (outer mean) is **not deployed**.

## What the pre-registered rule said

The candidate passed both parts of the rule.
- Jackknife SD of p fell from 0.078 to 0.038, against a bar of 0.039.
- The seed-mean error fell from 35.4% to 34.8%.

The verdict under the rule was ADOPT. On the question the rule asked, the candidate is the better
fit: the headline improves from 37.2% to 34.9%, and it no longer moves when one compound is added.

## What the rule did not check, and should have

The candidate moves the global exponent from p = 0.80 to 1.05. The pre-registration scoped the test
to the global fit, but every other part of the paper inherits that exponent: the family constants
are fitted with p held at the global value, and the 300 K comparisons use the global constant
directly. None of those downstream effects was part of the test. Carried through the whole chain,
the candidate gives the following (current -> candidate):

| quantity | current | candidate |
|---|---|---|
| shape of the calibrated curve below the cap | T^-0.20 | **T^+0.05 (flat)** |
| two-parameter vs one-parameter ablation | 13.0 pp, p = 0.0013 | 5.9 pp, p = 0.046 |
| SDNNFF comparison, this work at 300 K | 32.9% | 41.6% |
| Fe-Sb family, held out | 24.8% (passes) | 41.0% (fails) |
| Sb-Pd / Sb-Pt families, held out | 43.3% / 5.6% | 58.8% / 9.3% |
| GdNiSb / TbNiSb predictions, W/m/K | 3.69 / 3.99 | 2.77 / 2.99 |

The flat curve is the decisive cost. Measured lattice conductivity falls with temperature, and the
paper already reports that the temperature shape is not predicted. A calibration that predicts no
temperature dependence below 670 K makes that limitation worse. The candidate's one gain, 2.3
points on the headline, is smaller than the seed-to-seed spread of 10 points.

## The decision

Keep the current objective. The headline is reported under it: 37.2%, the median of five seeds
ranging from 29.6% to 39.6% (seed mean 35.4%), with its sensitivity to single compounds stated in
the limitations. The candidate is not reported in the paper: the author ruled on 2026-10-01 that
the paper reports only the method it uses, not alternatives that were tried. This record stays here.

This departs from the rule's verdict, and the departure is recorded here rather than hidden. The
rule answered the question it was written to answer. It was incomplete, because it did not ask what
the change does to the rest of the paper. A future pre-registration of a change to a shared
parameter must include its downstream consumers in the test.

The candidate's full state (code and every regenerated artefact) is preserved as the git stash
"OPTION B: steadier (outer-mean) fit ...", restorable with `git stash apply`.
