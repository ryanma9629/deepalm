# 11: Train and evaluate BM^E

**What to build:** Train the equal-allocation benchmark end to end, establishing the shared optimization, checkpoint, selection, and device machinery used by later policies.

**Blocked by:** 03/Generate HJM-PCA scenarios; 10/Integrate constraints and the training objective.

**Status:** ready-for-agent

- [ ] BM^E uses equal maturity distributions and two learned time-shared scale adjustments while satisfying the common policy contract.
- [ ] The benchmark trains through the differentiable simulator with RAdam, the resolved triangular learning-rate cycle, gradient clipping, and fresh deterministic epoch paths and objective parameters.
- [ ] Quick runs complete for both five-year and fifteen-year horizons using separate training, selection, and locked-test scenario identities.
- [ ] Checkpoint selection follows selection total loss with the penalty tie-break and records configuration, seed, data, convention, policy, horizon, optimizer, and history identities.
- [ ] Resume rejects incompatible checkpoints and restarts at a completed epoch boundary without repeating or skipping an epoch seed.
- [ ] A CPU training test and an Apple MPS float32 optimizer-step test require finite loss, state, and gradients and verify clipping.
