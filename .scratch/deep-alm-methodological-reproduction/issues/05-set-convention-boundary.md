# Choose the Paper and Corrected convention boundary

Type: grilling
Status: resolved
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

### Grilling round 4

- Fit PCA and cubic loadings on 1-month through 15-year tenors, then initialize HJM on `N + H` monthly nodes: 240 nodes through 20 years for the 5-year run and 360 nodes through 30 years for the 15-year run. Initialize the extended curve from NSS, evaluate the cubic loadings beyond 15 years, update only the first `L - 1` nodes, discard the final node each month, and expose the first 180 nodes to ALM. Both profiles share this disclosed long-end interpretation and report extrapolation diagnostics.
- Do not clip or floor finite simulated rates in either profile. Fail on non-finite values and report the share of 1-month yields below -1% plus tenor-wise quantiles. Any artificial stress bound creates a `custom` run.
- Calculate monthly loan interest literally as `max(exp(Y + kappa_L) - 1, 0)` in the Paper profile and as `max(exp((Y + kappa_L) / 12) - 1, 0)` in the Corrected profile.

### Grilling round 5

- Treat Equation 18 as missing action quantities and summation signs. In both profiles, subtract the summed quantity-times-value of new investment bonds from cash and add the corresponding summed financing-bond proceeds, consistent with the portfolio update in Equation 17 and double-entry accounting.
- In both profiles, use `rho_SD = 4%` for non-maturity deposits, `rho_SF = 1%` for term deposits, and apply the described default rule to enterprise loans. Record the conflicting symbol and “individual loans” wording as errata without creating alternate state variables.
- Use state times 0 through H. At time zero, do not roll forward or perform an annual close; make and apply the first decision. At later decision times, roll forward before policy and restructuring, then evaluate constraints. Perform annual closes at months 12, 24, ..., H - 12. After the final decision at H - 1, roll once to H without another decision, restructuring, or dividend, and use that terminal equity in the loss.
- Keep economic/numerical conventions orthogonal to run scale. The `paper` and `corrected` profiles control formulas only; `quick` and `paper_scale` configurations control scenario counts, training samples, feature-PCA fitting subsets, seeds, optimizer settings, schedules, and early stopping. Record every undisclosed project choice without labeling it as a paper fact.

## Answer

Both quick and paper-scale runs default to the Corrected convention; the Paper convention is reserved for paired fidelity analysis using the same Reference Bank, market innovations, and experiment settings. Formula conventions are orthogonal to run-scale configurations. Locked `paper` and `corrected` profiles must emit their full choice manifest, configuration hash, and Git commit; any component override changes the convention identity to `custom`.

Only two material, runnable formula differences create profile branches: the Paper profile uses PCA loadings `lambda * q` while Corrected uses `sqrt(lambda) * q`, and Paper uses the literal loan-interest expression `max(exp(Y + kappa_L) - 1, 0)` while Corrected uses `max(exp((Y + kappa_L) / 12) - 1, 0)`. Clear editorial defects—including missing quantities and sums in the cash update, deposit-growth notation, enterprise-loan naming, and the time-zero annual-close condition—are corrected in both profiles and recorded as errata.

All other undisclosed numerical choices are explicit shared project defaults rather than invented profile differences: decimal rates, monthly discrete forwards, week-end last-valid sampling, centered `n-1` covariance annualized by 52, deterministic PCA signs, unweighted cubic fits, monthly HJM steps, `N + H` long-end grids, no rate clipping, and a 60/180-transition timeline with a final roll-forward only. Alternative exploratory choices are configurable but produce `custom` results.
