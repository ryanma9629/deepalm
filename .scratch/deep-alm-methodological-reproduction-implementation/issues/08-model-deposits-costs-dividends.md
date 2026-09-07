# 08: Model deposits, costs, and dividends

**What to build:** Complete the passive bank lifecycle with deposit behavior, negative cash interest, operating costs, annual closes, and dividends.

**Blocked by:** 06/Run a passive runoff simulation.

**Status:** ready-for-agent

- [ ] Non-maturity and term deposits follow the approved tranches, reference rates, caps, growth, maturities, and interest-reinvestment rules.
- [ ] Negative cash earns or pays interest according to the paper's treatment and reconciles to the monthly cash account.
- [ ] Personnel and material costs are charged separately, with annual growth applied only to personnel cost.
- [ ] Annual closes occur at months 12, 24, and later nonterminal year ends, with no close at time zero or after terminal roll.
- [ ] Dividends flow through cash and equity and are retained as observable trajectory outputs.
- [ ] Hand-calculated monthly and annual fixtures satisfy deposit, cost, dividend, cash, and balance-sheet identities.
