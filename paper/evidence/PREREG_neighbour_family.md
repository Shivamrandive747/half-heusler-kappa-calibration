# Pre-registration: can a calibrated NEIGHBOUR family's constant stand in for an uncalibrated family?

Written 2026-10-03, before `test_neighbour_family_transfer.py` was run.

## Question
A family with no measured member (e.g. Bi-Pt, holding PrBiPt and NdBiPt) currently gets only the shared
constant (c = 0.51, p = 0.80, capped). Would the constant of a calibrated family of SIMILAR chemistry do
better?

## Definition of "neighbour"
Two families are neighbours when they share one of their two (Y,Z) elements and the other two elements
sit in the same periodic-table group (e.g. Sb-Pd and Sb-Pt: Pd and Pt are both group 10). Fixed before
the run; no other pairing is tried.

## Test
For every calibrated family F that has at least one calibrated neighbour N, every measured member m of F
is predicted three ways, from its published calculation, exactly as the family route does:
- NEIGHBOUR: N's deployed constant and form (fitted on N's members only, so m is never in the fit)
- SHARED: the global constant and form
- OWN (reference only): F's leave-one-compound-out constant
Error per compound = the family route's per-compound median absolute percentage error.
When F has several neighbours, each (F, N) pair is scored separately.

## Decision rule (fixed now)
Neighbour transfer is adopted for uncalibrated families only if BOTH:
1. the pooled median error over all scored compounds is lower for NEIGHBOUR than for SHARED, and
2. NEIGHBOUR beats SHARED in at least half of the (F, N) pairs.
Otherwise the shared constant stays, and any neighbour estimate is not reported.
The result is recorded whatever it shows.
