# Pre-registration: the shared transfer function fitted on calculations, not on model predictions

Written 2026-10-05, BEFORE any blind-test result under this rule was computed.
Decision by the first author (2026-10-05): adopt this definition; report whatever it gives.

## Why
The shared constant (c, p) of kappa_exp = kappa_BTE * min[c (T/300)^p, 1] was fitted on the
machine-learning model's predictions for the in-domain blind-test compounds (extend_blind_test.fit_cp on
target_blind_test.csv, seed 0 only). After the 2026-09-13 tier correction was propagated
(2026-10-04 rerun), refitting it on each of the five model seeds gave c = 0.45-0.55 but
p = 0.75 / 0.95 / 1.20 / 1.25 / 1.25: the exponent is set by the model's seed, not by the physics, and
the fit's objective is nearly flat in p (forcing p = 0.80 costs 1.5-11%). The paper's subject is the
calculation-to-experiment step, so the constant is redefined on that step, with no model in it.

## The rule (fixed now)
1. DATA. The published-route pairs of `family_calibration.build_published()`: in-domain half
   Heuslers with a full BTE calculation (tier 1, not semi-empirical) paired with the single-laboratory
   reference measurement, using the existing pairing rules unchanged (34 compounds, 434 rows on
   2026-10-05).
2. FORM AND OBJECTIVE. Unchanged: capped form min[c (T/300)^p, 1]; `extend_blind_test.fit_cp`
   (median over compounds of each compound's median |log10 error|), same (c, p) grid.
3. DEPLOYED SHARED CONSTANT = that fit on all pairs. The family form keeps p = the shared p.
4. HELD-OUT USE. Wherever a compound is SCORED (blind test, nulls comparison, domain split,
   conformal residuals, ablations, published-model comparison), (c, p) is refitted on the pairs with
   every compound of the scored compound's held-out unit removed: its X-site chemistry cluster for
   the blind test, its bonding family for the unseen-family test. No scored compound ever informs
   the constant applied to it.
5. One central function (`shared_constant.py`) supplies every shared-constant fit; scripts no longer
   fit the shared constant on model predictions. Null baselines and family constants are unchanged.

## What will be reported, whatever it shows
- Headline: in-domain blind test, nested as in rule 4, median over five model seeds with the seed
  range; within 2x; paired p against both nulls; Spearman; all-compound figure beside it.
- The deployed (c, p) with its stability: leave-one-cluster-out, leave-one-family-out and a compound
  bootstrap (10-90%) of the calculation-fitted constant.
- Shared constant on unseen families (leave-one-family-out on published calculations).
- Family route held out (unchanged definition; inherits the new shared p).
No alternative definition will be tried after these numbers are seen. If the headline gets worse it
is reported as worse.
