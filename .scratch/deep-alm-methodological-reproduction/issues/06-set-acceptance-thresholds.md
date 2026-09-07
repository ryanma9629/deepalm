# Define methodological reproduction acceptance thresholds

Type: grilling
Status: claimed
Blocked by: 01, 02, 03, 04, 05

## Question

Using the established data inventory, convention choices, and no-swap experiment catalogue, what quantitative tolerances and qualitative criteria must pass for yield-curve reconstruction, HJM-PCA simulation, balance-sheet roll-forwards, regulatory constraints, differentiability, benchmark behavior, Deep ALM training, 5-year versus 15-year comparisons, and reproducible reporting before the project may claim a successful methodological reproduction?

## Comments

### Grilling round 1

- Define two completion levels. A passing quick profile is `development-validated`; only the complete paper-scale Corrected experiment matrix passing every required gate is `methodologically-reproduced`.
- Separate acceptance into hard numerical gates, sampling-aware statistical gates, and directional behavior gates. Hard failures cannot be excused by statistical tolerance, while stochastic quantities use predefined error bands or confidence intervals rather than exact equality.
- Divide paper behavior into core directional requirements and supporting diagnostics. Core loss/return/constraint ordering, greater long-horizon smoothness, and reduced 5-year terminal behavior for `MM(15y|5y)` are required. Exact action months and selected-scenario maturity choices are reported but do not independently block acceptance.

### Grilling round 2

- Require the full Corrected matrix: BM^E, BM^C, BM^D, and MM at 5 and 15 years plus MM(15y|5y). Require a paired Paper matrix containing BM^D and MM at both horizons plus MM(15y|5y), with common Reference Bank inputs and market innovations. Frozen-policy Reference Bank sensitivities are required; quick retraining is triggered only by a reversal of a headline conclusion or strategy ordering.
- A poor finite Paper result does not block the Corrected methodological-reproduction claim, but the paired run and disclosure are mandatory. A non-finite Paper run must identify and report the precise failure boundary.
- Separate training, checkpoint-selection, and locked final-test scenario sources. All strategies share the final test paths; 5-year paths use the first 60 months and common Brownian innovations from the 15-year paths. The locked test set is evaluated only after checkpoint selection is complete.

### Grilling round 3

- Set quick runs to 256 training paths per epoch, 128 selection paths, 128 locked test paths, 5 epochs, and batch size 32. Set paper-scale runs to 40,000 deterministically resimulated training paths per epoch, 1,600 selection paths, 1,600 locked test paths, at most 100 epochs, and batch size 32. Preserve the full network and 60/180-month horizons in both profiles.
- Train every benchmark with one registered primary seed. Train Corrected MM independently with three initialization/training-stream seeds at each horizon while sharing selection and test paths; require core MM ordering in at least two of three runs and report the median and range. Run the paired Paper matrix with the primary seed only.
- Select checkpoints by fixed-parameter mean selection total loss (`mu=4.06%` at 5 years, `mu=4.00%` at 15 years, `lambda=3.5`), breaking effectively tied results by lower penalty loss. Paper-scale runs train at least 20 epochs and then use patience 15 with a 0.1% relative-improvement threshold; quick runs always execute all five epochs.
