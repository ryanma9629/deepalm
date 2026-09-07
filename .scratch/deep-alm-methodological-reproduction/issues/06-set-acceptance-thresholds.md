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

### Grilling round 4

- Run calibration and numerical oracle tests on CPU float64. Require the spot/discount/forward round trip, PCA orthogonality and pre-fit covariance identities, deterministic HJM one-step oracle, and initial-curve match to meet maximum absolute or relative Frobenius error `1e-10`, as applicable. Any non-finite value is a hard failure.
- Require initial portfolio present values within `1e-8` relative error of targets. At every path and time require `abs(assets - liabilities - equity) <= max(1e-6 mCHF, 1e-10 * assets)`, cash reconciliation within the same tolerance, nominal cash flows and actions no lower than `-1e-8`, exact 180-bucket ladders, loan duration below 5 years, deposit duration below 3 years, and six hand-calculated constraint fixtures within `1e-10`.
- On interior CPU float64 fixtures, compare representative autograd parameters with central finite differences using `abs(g_auto - g_fd) <= 1e-6 + 1e-4 * abs(g_fd)`. Require finite policy gradients with aggregate norm above `1e-8`, zero/absent gradients across the detached state-input boundary while parameters remain trainable, finite full MPS training steps, and post-clipping global gradient norm no greater than `0.2 + 1e-6`.

### Grilling round 5

- Require the first three historical-forward PCA components to explain at least 90% of variance. Limit the aggregate cubic-loading fit error to 10% relative Frobenius norm and each component's relative error to 15%, reporting PCA truncation and polynomial approximation errors separately.
- For each convention, validate 50,000 fixed-seed one-step shocks: each standardized Brownian-factor mean must lie within three standard errors of zero and empirical curve-shock covariance must be within 5% relative Frobenius norm of `V @ V.T * dt`. Multi-step paths must remain finite and unclipped. Report short-rate and tenor-wise quantiles without imposing the paper's -1% observation as a Corrected hard gate.
- On the 1,600 common locked test paths, use 10,000 paired bootstrap resamples. MM point estimates must beat all benchmarks on mean total loss, target loss, penalty loss, and annualized return; the 95% paired interval versus BM^D must exclude zero in the expected direction for all four metrics, in at least two of three Corrected MM training seeds.

### Grilling round 6

- On each Corrected MM locked test set, limit the share of paths with any LCR or NSFR violation to 1% per constraint and CMR, Equity/RWA, or IRS violation to 2% per constraint. EYR must be the most frequently violated constraint. Require mean penalty loss and penalty ES95 below BM^D in at least two of three MM seeds.
- Define normalized action turnover as `sum_t ||a_t - a_(t-1)||_1 / (sum_t ||a_t||_1 + epsilon)`. Over the common first 60 months, require MM(15y) median turnover at least 10% below MM(5y), and require MM(15y|5y)'s final-24-month share of turnover at least 10% below MM(5y), with paired 95% bootstrap intervals in the expected direction.
- Require MM(5y) mean annualized return and CRRA evaluation to beat MM(15y|5y) with paired 95% intervals in the expected direction. Treat MM(15y|5y) penalty below BM^D(5y) as supporting. Do not require MM equity volatility, signed VaR, or signed ES to beat BM^D; report these using the paper's sign convention.

### Grilling round 7

- Require the no-swap versions of Tables 1-5 and Figures 3 and 5-17, using printed caption numbering and methodological rather than pixel equality. Also require a lightweight Hull-White extended Vasicek comparator for the Figure 4 scenario-diversity diagnostic only; it does not participate in training.
- Every result directory must contain the resolved YAML, convention/run/custom identity, Git and dependency versions, device/dtype, hashes of the CSV, source PDF, and Reference Bank snapshot, weekly dates and calibration artifacts, every seed, checkpoint history, final metrics, and a machine-readable acceptance result. CPU scenario generation is bitwise reproducible; MPS training is statistically rather than weight-bit reproducible.
- Layer automated verification into unit, property/integration, gradient, quick end-to-end, and slow paper-scale acceptance suites. Ordinary pytest does not launch paper-scale training.
- Generate a unified acceptance report with pass/fail/not-applicable, observed values, thresholds, and evidence paths. Any required hard, statistical, or core directional gate failure blocks the `methodologically-reproduced` label. Never omit adverse seeds or test paths; retain `development-validated` or report reproduction failure honestly.
