# 09: Add active treasury bond actions

**What to build:** Let a treasury policy actively buy investment bonds and issue funding bonds at the paper's allowed maturities while the bank continues its passive monthly evolution.

**Blocked by:** 06/Run a passive runoff simulation.

**Status:** ready-for-agent

- [ ] A treasury action contains 13 investment and 16 funding quantities with batch-first shape and rejects negative or malformed values.
- [ ] New bonds are fractional, priced on the current curve, added to the correct ladder, held to maturity, and cannot be sold or unwound early.
- [ ] Cash settlement multiplies every instrument cash flow by its action quantity and sums across maturities, correcting the written omission in Equation 18.
- [ ] The resolved timeline performs the active decision and passive market/balance-sheet transition in the documented order for all 60 or 180 actions.
- [ ] Action-dependent future cash flows remain differentiable through subsequent simulator transitions.
- [ ] Zero-action behavior matches the passive runoff trajectory, and deterministic nonzero-action fixtures reconcile independently.
