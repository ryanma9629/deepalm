# Choose the Paper and Corrected convention boundary

Type: grilling
Status: claimed
Blocked by: 02

## Question

For each material numerical ambiguity identified in the term-structure research, which behavior belongs in the Paper convention, which belongs in the Corrected convention, which convention should be the default for each run profile, and how must outputs disclose the selected convention so results cannot be compared under mismatched assumptions?

## Comments

### Grilling round 1

- Use the Corrected convention by default for both quick and paper-scale runs. Use the Paper convention only for explicit fidelity comparisons; paper-scale describes workload, not numerical conventions.
- Classify uncertainties into: material runnable formula disagreements implemented in both conventions; obvious editorial or notation errors corrected once and recorded as errata; and undisclosed numerical choices implemented as shared, configurable project defaults rather than mislabeled convention differences.
- Provide locked `paper` and `corrected` profiles. Any component override changes the run identity to `custom`. Persist the complete choice manifest, configuration hash, and code commit with every result.
- Compare conventions only as paired convention-impact experiments using the same Reference Bank, market random numbers, and experiment settings. Keep their primary result tables separate and do not interpret cross-convention outcome differences as strategy rankings.

### Grilling round 2

- On the shared 1-180 month grid, derive historical forwards from adjacent discount factors so the spot/discount/forward round trip closes numerically. Retain analytic NSS instantaneous forwards as a diagnostic, not a profile difference.
- Reconstruct valid daily curves, select the last valid curve in each `W-FRI` week, difference consecutive selected curves, and estimate a centered sample covariance with divisor `n-1`; multiply it by 52, without interpolating missing weeks or deleting outliers. Persist the selected dates.
- Scale PCA loadings by `lambda * q` in the Paper profile and by `sqrt(lambda) * q` in the Corrected profile. Report the covariance implied by `V @ V.T` under both.
- Simulate the annualized HJM diffusion directly on the monthly ALM clock with `dt=1/12`. Use 60/180 transitions and 61/181 states. Keep weekly substep simulation only as a validation tool shared by both profiles.

### Grilling round 3

- Convert SNB beta parameters to decimal rates at ingestion. Use decimal rates and volatilities, years for maturities and time steps, and mCHF for amounts throughout both profiles; percentage and basis-point units exist only at input/output boundaries.
- Stabilize each PCA eigenvector sign by making its largest-absolute-value element positive. Fit each scaled loading by unweighted cubic least squares with an intercept on normalized tenor `u = x / 15`, without a zero boundary constraint or clipping. Both profiles share the fit procedure but receive their respective scaled loadings.
- Compute the HJM volatility integral by the paper's trapezoidal rule beginning with the polynomial value at tenor zero, use the paper's forward tenor difference, and scale Gaussian innovations by `sqrt(dt)`. Use the analytic cubic integral only as a numerical validation oracle.
