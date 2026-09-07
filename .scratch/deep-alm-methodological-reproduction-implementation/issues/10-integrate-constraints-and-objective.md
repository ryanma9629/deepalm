# 10: Integrate constraints and the training objective

**What to build:** Make complete bank trajectories trainable by calculating the six paper constraints, terminal target loss, cumulative penalty, and evaluation-only CRRA metric.

**Blocked by:** 07/Model loan growth, interest, and impairment; 08/Model deposits, costs, and dividends; 09/Add active treasury bond actions.

**Status:** ready-for-agent

- [ ] LCR, NSFR, CMR, Equity/RWA, IRS, and annual-only EYR match independent formulas at values below, at, and above their bounds.
- [ ] Parallel plus/minus 100-basis-point revaluation drives IRS and annual close timing drives EYR without introducing hidden balancing values.
- [ ] The asymmetric terminal target loss and squared cumulative constraint penalty use the approved coefficients, sampled `mu`, and sampled `lambda`.
- [ ] Training excludes CRRA; evaluation computes CRRA with gamma 10 and the recorded equity-ratio floor.
- [ ] Constraint values and violations are returned with trajectories at their applicable states and the initial values are available to later observations.
- [ ] Smooth CPU float64 fixtures pass central finite-difference gradient checks and produce a nonzero policy gradient when terminal loss depends on action.
- [ ] Full five-year and fifteen-year deterministic fixtures retain the required accounting tolerances while calculating losses.
