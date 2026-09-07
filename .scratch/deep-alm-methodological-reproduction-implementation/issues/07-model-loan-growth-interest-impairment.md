# 07: Model loan growth, interest, and impairment

**What to build:** Extend monthly bank evolution so mortgage and enterprise-loan books reproduce the paper's disclosed growth, pricing, maturity replacement, and credit-loss behavior.

**Blocked by:** 06/Run a passive runoff simulation.

**Status:** ready-for-agent

- [ ] Maturing loans are replaced and the disclosed deterministic loan growth is added using the approved product mix and maturity templates.
- [ ] Loan cash flows include the approved 150-basis-point spread and retain separate mortgage and enterprise-loan ladders.
- [ ] Large annual increases in the six-month rate trigger the disclosed enterprise-loan impairment channel at the correct time.
- [ ] Paper and Corrected loan-interest calculations implement only their declared annualization difference and produce hand-verifiable results.
- [ ] Loan events reconcile to cash, asset value, and equity through the simulator for monthly and annual-boundary fixtures.
- [ ] Invalid maturity distributions, growth assumptions, and non-finite loan states fail with domain-specific diagnostics.
