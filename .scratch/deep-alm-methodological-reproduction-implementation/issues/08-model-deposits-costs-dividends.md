# 08: Model deposits, costs, and dividends

**What to build:** 存款分层滚动、利息再投资、负利率现金费用、成本和年度分红正确影响现金及权益。

**Blocked by:** 06/Run a passive runoff simulation.

**Status:** completed

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] Non-maturity and term deposits follow the approved tranches, reference rates, caps, growth, maturities, and interest-reinvestment rules.
- [x] Cash earns no positive interest. When the one-month rate is negative, cash above 30 times minimum reserves incurs the paper's negative-rate charge and reconciles to the monthly cash account.
- [x] Personnel and material costs are charged separately, with annual growth applied only to personnel cost.
- [x] Annual closes occur at months 12, 24, and later nonterminal year ends, with no close at time zero or after terminal roll.
- [x] Dividends flow through cash and equity and are retained as observable trajectory outputs.
- [x] Hand-calculated monthly and annual fixtures satisfy deposit, cost, dividend, cash, and balance-sheet identities.
- [x] 保留非终点年度事件及月度/年度手算验收；无 time-zero 或 terminal-roll 后的额外分红。

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。保留原存款、成本和分红逻辑；明确负利率现金费用，不改变公式。原编号保留；本次只更新待办，不表示本票实现已经完成。
