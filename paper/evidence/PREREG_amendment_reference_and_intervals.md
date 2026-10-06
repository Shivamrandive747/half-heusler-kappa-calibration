# Pre-registration amendment: blind-test reference and interval temperature

Written 2026-10-05, BEFORE any result under these rules was computed. Decisions by the first author
(2026-10-05), after three independent audits. Amends PREREG_shared_constant_on_calculations.md.

## A. The blind-test reference is the compound's single chosen reference
Until now each blind-test compound was scored against the median of ALL tier-0 rows in each 100 K
bin, pooling laboratories. A laboratory contributing many rows -- or the same sample entered twice
(published kappa_L and our Wiedemann-Franz copy) -- outvoted the rest: NbFeSb's 500 K reference was
3.21 W/m/K against 7.82 for the median of per-source medians.

Rule: every scored compound is scored against ONE laboratory, the reference chosen by
`reference_choice.per_compound` with its existing ladder (the reference already used by the family
route and the tables, consistent with the author's 2026-09-12 ruling "one reference per compound,
never blend laboratories"). The reference is chosen without any calculation or model value (the
ladder never reads kappa_BTE). Where the family-paper rung would make the choice depend on whether
the scored compound's siblings are in the fold (the family leave-one-out leak, audit 2026-10-05),
the scored compound's reference is chosen by the ladder WITHOUT the family-paper rung. Bins, scoring
per compound, the X-site cluster holdout, the shared constant (calculation-fitted, cluster excluded)
and the nulls are otherwise unchanged; the nulls are scored against the same references.

## B. Prediction intervals describe the error at the quoted temperature
Conformal residuals were per-compound medians over temperatures but were applied to single values
quoted at one temperature. Rule: each compound's residual is taken at the temperature nearest its
quoted temperature (300 K, or 500 K where a value is quoted there), under its own calibration and
holdout as before; the ceil((m+1)alpha) rule is unchanged.

## C. Data corrections applied before the rerun (bug fixes; not choices)
Second-hand quotes out of tier 0 (secondary_quotes.py, widened detector); one copy per sample where
Starrydata carries both a published kappa_L and a total-kappa curve; the Starrydata snapshot deposit
not counted as a laboratory and the deposit regex fixed; non-Heuslers and the ZnNiSn typo removed;
measured and non-Heusler rows removed from the semi-empirical gap file; family leave-one-out
references re-chosen inside each fold; the 3807 phono3py values computed on machine-learned
(pypolymlp) force constants (NIMS MDR, Togo 2026; 47 compounds) moved from tier 1 to tier 2 as the
pipeline's own tier definition requires (author's decision 2026-10-05), so they leave model
training and the calculation-to-experiment calibration.

## What will be reported, whatever it shows
Exactly the list in the parent pre-registration, recomputed under A-C. No alternative reference rule
will be tried after the numbers are seen.
