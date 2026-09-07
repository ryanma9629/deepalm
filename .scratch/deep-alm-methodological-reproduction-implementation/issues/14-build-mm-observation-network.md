# 14: Build the MM observation and shared network

**What to build:** Give the multi-period model the paper's 145-dimensional information set and a shared policy network that returns deviations from BM^D.

**Blocked by:** 03/Generate HJM-PCA scenarios; 13/Train and evaluate BM^D.

**Status:** ready-for-agent

- [ ] Yield-curve features use a centered, non-standardized three-component PCA fitted only on the approved deterministic subset of training states.
- [ ] Four independent cash-flow encoders transform investment, funding, aggregate-loan, and aggregate-deposit ladders into 128 features.
- [ ] The observation contains exactly 145 features: portfolio encodings, curve factors, five relative balance-sheet values, six prior constraints, normalized time, `mu`, and `lambda`.
- [ ] The shared ELU residual network uses the approved widths and 64-dimensional final encoding without batch normalization.
- [ ] Investment and funding heads each produce a scale deviation and softmax maturity distribution, add the frozen BM^D baseline, and apply final non-negativity.
- [ ] Stop-gradient preserves forward values, blocks gradient into the current observation state, preserves parameter gradients, and leaves future action-to-loss gradients intact.
- [ ] Observation, action, baseline, device, dtype, and shape contracts fail early on incompatible inputs.
