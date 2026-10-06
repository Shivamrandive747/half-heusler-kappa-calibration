# Pre-registration: a steadier objective for the headline calibration fit

Written 2026-10-01 and committed **before** any result under the candidate was computed. The
commit that adds this file precedes the commit of any result, so the order can be checked.

## The problem being addressed

`extend_blind_test.fit_cp` chooses the transfer function's two parameters, c and p, on a fixed
grid by minimising the **median over compounds** of each compound's median |log10 error|. That
objective is flat: many (c, p) pairs score nearly the same, so a small change in the training data
can move the choice a long way. Measured on 2026-09-30: adding a single compound (VFeSb) to the
training data changed (c, p) by more than 0.02 in c or 0.05 in p in 2 to 14 of the 18 chemistry
folds, depending on the seed, and moved the five-seed median headline from 33.7% to 37.2% while the
mean over seeds stayed at 35.3% -> 35.4%.

## The candidate (one, fixed)

Identical to `fit_cp` in every respect except the outer statistic: the **mean over compounds**
of each compound's median |log10 error|, instead of the median over compounds.

- The per-compound median over temperatures is kept, so a single digitised point within one
  compound's curve still cannot steer the fit.
- Same grid (`extend_blind_test.CS`, `extend_blind_test.PS`), same cap at 1, same data, same folds.
- The null baselines are not fitted with this objective (the constant null is a median of the
  training conductivities, the power-law null a least-squares line), so they are unaffected.
- Scope: `fit_cp` only, i.e. the global transfer function. The per-family constants fitted in
  `family_calibration.py` are not part of this test.

The trade-off is stated in advance: an outer mean is smoother but lets one compound whose whole
curve is off pull the fit further than an outer median does. Test 1 measures exactly that influence.

## The tests, both objectives, same code otherwise

**T1, stability (jackknife).** For each of the five seed prediction files, on the current in-domain
set (39 compounds), refit (c, p) on all compounds but one, for each compound in turn.
- Primary metric: the standard deviation of the fitted p across those fits, summarised as the
  median over seeds.
- Secondary: the same for c; and the number of chemistry folds whose (c, p) moves by more than 0.02
  in c or 0.05 in p when VFeSb is added to the training data.

**T2, accuracy (the headline protocol).** `compute_seed_averaged.nested` with the fitter swapped:
every compound predicted with its whole chemistry held out, for all five seeds.
- Per-seed median error on the 39-compound domain; the median and the mean over seeds; within a
  factor of two (the ratio test, predicted/measured in [0.5, 2]).
- The same on the 38-compound domain without VFeSb, and the shift between the two.

## The decision rule (fixed now)

Adopt the candidate only if **both** hold:

1. the median-over-seeds jackknife SD of p is **at most half** that of the current objective; and
2. the mean-over-seeds median error on the 39-compound domain is **no more than 1.0 percentage
   point higher** than the current objective's.

Otherwise keep the current objective and state the instability plainly in the paper's limitations.

Either way every number above is reported. If the candidate is adopted, the whole chain is rerun
(`rerun_downstream.sh`), and the reported headline statistic stays the **median over seeds**, as the
paper already defines it. It is not switched to whichever summary looks better.

## What will not be done

No further objectives will be tried, and the grid, the thresholds and this rule will not be changed
after results are seen. If the candidate fails, that is the result.
