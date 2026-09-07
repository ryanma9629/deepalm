# 07: Model loan growth, interest, and impairment

**What to build:** 贷款到期替换、新增、计息及年度减值进入可核对的月度银行轨迹。

**Blocked by:** 06/Run a passive runoff simulation.

**Status:** completed

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] Maturing loans are replaced and the disclosed deterministic loan growth is added using the approved product mix and maturity templates.
- [x] Loan cash flows include the approved 150-basis-point spread and retain separate mortgage and enterprise-loan ladders.
- [x] Large annual increases in the six-month rate trigger the disclosed enterprise-loan impairment channel at the correct time.
- [x] Paper and Corrected loan-interest calculations implement only their declared annualization difference and produce hand-verifiable results.
- [x] Loan events reconcile to cash, asset value, and equity through the simulator for monthly and annual-boundary fixtures.
- [x] Invalid maturity distributions, growth assumptions, and non-finite loan states fail with domain-specific diagnostics.
- [x] 保留 Paper/Corrected 利息年化差异和对应独立数值测试，不通过简化现金流降低计算成本。

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。保留原贷款金融逻辑及两个锁定口径，不通过减少金融事件节省资源。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-07 — 已完成：新增可微 `loans` 模块及 ALMSimulator 的 loan-dynamics 路径。按产品分别以 2–12 年/1–3 月模板续作到期额，并把 3% annual growth/12 按 55:20 分配；使用 150bp spread；Paper/Corrected 精确实现 `max(exp(Y+kappa)-1,0)` 与 `max(exp((Y+kappa)/12)-1,0)`。在非终期第 12 月，六个月利率同比上升超过 2% 时，enterprise ladder 按 excess 进行比例减值；手算、年度边界、错误输入和现金对账测试均通过。
