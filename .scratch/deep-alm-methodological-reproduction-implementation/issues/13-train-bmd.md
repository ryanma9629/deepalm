# 13: Train and evaluate BM^D

**What to build:** Add the decision-date-specific benchmark and produce the immutable selected baseline required by the multi-period model.

**Blocked by:** 11/Train and evaluate BM^E.

**Status:** ready-for-agent

- [ ] BM^D learns independent investment and funding scales and maturity distributions for every actual decision date in both resolved horizons.
- [ ] The policy remains scenario-independent, emits non-negative common-contract actions, and uses the same simulator and optimization machinery as the other benchmarks.
- [ ] Actual parameter counts are reported and the paper's indexing mismatch is recorded rather than silently changing the 60/180-transition timeline.
- [ ] Five-year and fifteen-year quick checkpoints are selected only from their selection scenarios.
- [ ] The selected parameters can be loaded through an immutable baseline reference whose identity is validated before MM use.
- [ ] Behavioral tests show time dependence, absence of scenario dependence, and correct per-date maturity availability.
