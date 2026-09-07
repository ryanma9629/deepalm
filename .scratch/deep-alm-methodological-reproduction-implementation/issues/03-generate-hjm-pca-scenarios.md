# 03: Generate HJM-PCA scenarios

**What to build:** Calibrate the paper's three-factor HJM-PCA model and generate reproducible monthly 5-year and 15-year interest-rate scenarios under both declared conventions.

**Blocked by:** 02/Reconstruct SNB term structures.

**Status:** ready-for-agent

- [ ] Each calendar week uses its last valid observation through Friday and the selected dates are retained as calibration evidence.
- [ ] Weekly forward differences are centered, use sample covariance, and are annualized by 52; eigenvector signs are deterministic.
- [ ] The first three components, explained variance, cubic volatility fits, fit errors, and implied covariance are reported for each convention.
- [ ] Paper scales loadings by eigenvalues and Corrected scales them by square-root eigenvalues, with no other convention difference introduced here.
- [ ] Monthly HJM paths contain 61 states for five years and 181 states for fifteen years, expose 180 ALM tenors, and use an internally extended shrinking tenor grid.
- [ ] Common seed identities reproduce CPU scenarios bitwise, and five-year Brownian innovations can be the prefix of fifteen-year innovations.
- [ ] Curve round-trip and deterministic one-step errors are at most `1e-10`; retained PCA and cubic-fit thresholds from the spec are enforced.
- [ ] Fixed-seed one-step statistical tests validate factor means and implied covariance, while non-finite paths fail at their source without rate clipping.
