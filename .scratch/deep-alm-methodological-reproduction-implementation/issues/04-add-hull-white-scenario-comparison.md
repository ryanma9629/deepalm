# 04: Add the Hull-White scenario comparison

**What to build:** Give the researcher a lightweight Hull-White extended Vasicek comparator that demonstrates how its terminal-curve diversity differs from HJM-PCA without entering policy training.

**Blocked by:** 03/Generate HJM-PCA scenarios.

**Status:** ready-for-agent

- [ ] The comparator starts from the same canonical initial curve and produces finite, reproducible paths on the same reporting tenors and horizons.
- [ ] Comparator assumptions not disclosed by the paper are configurable and labeled as project choices.
- [ ] A diagnostic summarizes and visualizes HJM-versus-Hull-White terminal curve diversity with model, horizon, convention, units, seed, and sample size identified.
- [ ] No training or policy configuration can accidentally select Hull-White as its market scenario provider.
- [ ] Deterministic fixtures and a stage-level integration test verify the comparator and its artifact metadata.
