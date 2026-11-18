# Pre-registration: null baselines fitted the way the route is fitted

Written 2026-11-18 and committed **before** any result under it was computed. The commit that adds
this file precedes the commit of the script's output, so the order can be checked.

## The problem

The two null baselines of the blind test are fitted to rows: the constant null is the median of the
training clusters' reference rows, and the power-law null is a least-squares line through the same
rows in log-log space (`compute_seed_averaged.null`). The route's two constants are fitted per
compound: `extend_blind_test.fit_cp` minimises the median over compounds of each compound's median
|log10 error|, because every result is scored per compound. A compound measured at many
temperatures therefore weighs more in the baselines than in the route. Both baselines over-predict
by a median factor of 1.27, the route by 0.90, and the percentage error penalises over-prediction
more than under-prediction. A pre-submission review (2026-11-18) asked how much of the route's
margin over the baselines (10.3 and 11.2 percentage points, Wilcoxon p = 0.004 at the seed median)
comes from this difference in fitting. That has not been measured.

## The candidate (one, fixed)

Each baseline refitted with the route's objective, and nothing else changed.

- Objective: minimise the median over training compounds of each compound's median
  |log10(prediction / reference)| over its rows. First minimum in grid order.
- Constant null: K on a fixed grid, 401 points evenly spaced in log from 0.5 to 50 W/m/K.
- Power-law null: A (T/300)^q with A on the same grid and q from -2.00 to +0.50 in steps of 0.05
  (51 points).
- Same folds (one chemistry cluster held out at a time), same in-domain compounds, same reference
  rows, same scoring, same five seed files for the route.

## What is reported, whatever it shows

For each baseline under each fit (rows, per compound):

1. median error, share within a factor of two, median ratio (bias) and the typical factor
   10^median|log10 ratio|, with the bootstrap interval of the median error (same resampling rule as
   `compute_accuracy_summary`);
2. the paired comparison with the route: the share of compounds for which the route is closer, the
   median paired difference in percentage points with its bootstrap interval, the Wilcoxon
   signed-rank p per seed, its median over seeds, and the Holm adjustment over the two baselines
   (seed median and worst seed);
3. the same comparison with each chemistry cluster counted once (Wilcoxon and sign test per seed).

## Positive control

Run through the same code, the row-fitted baselines must reproduce the registry: median error 46.7%
and 45.6%, 84.6% within a factor of two, seed-median Wilcoxon p = 0.004 for both, the route closer
for 71.8% of compounds, and a largest cluster-level Wilcoxon p of 0.048. If they do not, nothing is
reported until the difference is explained.

## How the result will be used (decided now)

- Both fits are reported: the row fit stays in the accuracy table and in the error-budget panel, as
  specified in the earlier protocol, and the per-compound fit is given beside it in the text of the
  blind-test section and in full in the Supplement.
- If, under the per-compound fit, the Holm-adjusted seed-median p is 0.05 or larger for either
  baseline, every sentence that says the route is closer to measurement than the null baselines
  (Introduction, Results, Discussion, highlights, cover letter) will be changed to say that this
  holds for baselines fitted to rows and not for baselines fitted per compound.
- If it stays below 0.05 for both, those sentences stand, and the per-compound figures are added.
- No other choice in the paper is revisited on the strength of this test.
