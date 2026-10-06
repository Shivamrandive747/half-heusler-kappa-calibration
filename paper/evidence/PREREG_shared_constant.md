# Pre-registration: the best single shared constant for a family never measured (2026-10-02)

Written and committed BEFORE `test_shared_constant.py` is run. Nothing below may change after the
first run; any change is a new pre-registration.

## Question

A compound whose bonding family has no measured member cannot take a family constant. It takes one
shared transfer function, applied to its published calculation. Which single shared function is
most accurate for such a compound?

Today it is the ML route's constant (c = 0.51, p = 0.80, capped): fitted on the ML model's
predictions, applied to published calculations. Tested as if each family had never been measured,
it gives a median error of 26.8% with only half the compounds under 25% (session of 2026-10-01).

## Data

The published-route pairs used everywhere for families (`family_calibration.build_published`):
34 measured half Heuslers in 12 families, published calculation vs one laboratory's measurement.

## Protocol: leave one FAMILY out

For each family F: fit the candidate on every OTHER family, predict F's compounds. Each compound is
scored by its median absolute percentage error over its temperatures (as everywhere in the paper).

## Candidates (fixed now)

| id | form | fitted on | free |
|---|---|---|---|
| C0 | c (T/300)^p capped at 1 | ML route (blind test, in domain), family F removed | c, p |
| C1 | capped | published route | c, p |
| C2 | uncapped, c (T/300)^p | published route | c, p |
| C3 | uncapped | published route | c; p = 0.80 |
| C4 | capped | published route | c; p = 0.80 |
| C5 | uncapped | published route, chosen to MAXIMISE the share of compounds under 25% (ties: lower median log error) | c, p |

C1-C4 minimise the paper's objective: the median over compounds of each compound's median
absolute log error. Grid: c = 0.10-0.99 step 0.01; p = -1.5 to +1.5 step 0.05. Exact ties take
the geometric midpoint of the tied range in each parameter.

## Metrics

Median error over the 34 compounds; share under 25%; share within 2x; per-family median.

## Selection is itself held out

Choosing the best of six after seeing six scores is a selection. So the procedure is NESTED: inside
each outer fold (family F removed), the six candidates are scored leave-one-family-out on the
remaining 11 families, the one with the lowest inner median error is chosen, refitted on all 11,
and used to predict F. The nested result is the honest figure for "pick the best shared constant".

## Decision rule

ADOPT the nested procedure's choice only if its nested median error is LOWER than C0's AND its
nested share under 25% is NOT LOWER than C0's. Otherwise keep C0.

If adopted: the chosen candidate, fitted on all 12 families, becomes the shared constant for
compounds whose family has no measured member (the conditional tier). The ML-route headline keeps
its own constant: that is a separate estimator, and its objective was settled by
`PREREG_fit_objective.md`.

## Positive control

C0 must reproduce the 26.8% median from 2026-10-01 before anything else is read.
