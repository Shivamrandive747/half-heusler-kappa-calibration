# What was tried and did not work

Six modifications to the method were implemented in full and not adopted, because they did not
improve held-out accuracy consistently. They are recorded here because the next person to work on
this problem will think of them too, and because a positive result is easier to judge beside the
alternatives that were tried.

The "Limits of the method" subsection of the paper lists them. The code is in `analysis/`; every
figure below is read from `data/exports/kappa_v2/paper_numbers.json`.

| intervention | result | script |
|---|---|---|
| **Correction without the temperature term** — κ_exp = c · κ_BTE | 38.5 % median error and 82.1 % within 2× (five-seed medians), against 33.1 % and 87.2 % with the exponent. The temperature term earns its parameter. | `analysis/compute_seed_averaged.py` |
| **Capped family form** — cap each family's correction at the bare calculation | 27.0 % on the 31 compounds of the family test, against 21.7 % uncapped; chosen inside every fold, the uncapped form wins in all 31. | — |
| **Hurdle model** — treat low- and high-conductivity compounds with separate models | Halves the low-κ error on *calculated* labels; on the measured compounds, **−1.3 pp, p = 0.84.** The gain does not survive contact with experiment. | — |
| **Magnitude term** — add a κ^β factor to the transfer function | Selected on the design half in **47 of 60** splits, then lost on the held-out half: a winner's-curse result. | `analysis/fit_magnitude_calibration.py` |
| **Add experimental values to training** | +5.5 pp at p = 0.10, and *worse* on the transition-metal subset. | `analysis/run_tier0_experiment.py` |
| **Add doped-composition data** | No gain (p ≥ 0.10). | `analysis/run_doped_experiment.py` |

## Per-family constants: adopted, with a stated limit

One constant per bonding family *is* part of the method, but its advantage is not significant
compound by compound. Calibrating published calculations, the family constants give 21.7 % on 31
compounds held out of their own family's fit, against 32.3 % for the shared constant on the same
31; the family constant is closer for 16, the shared constant for 11, with 4 ties (sign test
p = 0.44). Five families are validated held out at 7.4–24.9 %. A family with no measured member
receives only the shared constant, whose held-out error on families it never saw is 33.0 %.

## Two methodological traps this project fell into

Both cost real time, and both are easy to repeat.

**Fitting one objective and reporting another.** The calibration is fitted to minimise the median
of *per-compound* errors, which is the quantity reported. Fitting per *row* instead lets a few
heavily sampled compounds, such as `TiNiSn` and `ZrNiSn`, dominate the fit.

**Letting a compound serve as its own evidence.** An early version of the family-support rule
counted the compound being scored among its own family's supporting measurements, which selects for
the conclusion being tested. Every group statistic is now computed leave-one-out.

## One result that is easy to misread as a failure

A **family power law** — fit a curve to the measured κ(T) of the held-out compound's own bonding
family and use it, with no calculation — gives 25.4 % median error against 31.6 % for the model route
at the same reference seed, on the 34 in-domain compounds where it is defined. Neither is
significantly better (paired Wilcoxon p = 0.99; the model is closer for 50 % of compounds).

It is still not a substitute. It is undefined for 5 of the 39 in-domain compounds, whose families
hold no other measured member, and it is undefined for every family nobody has measured. And it
returns one curve per family, so it cannot order compounds *within* a family. Choosing between
members of one substitution series is exactly the task a screening user faces, and the family
baseline is uninformative for it by construction.
