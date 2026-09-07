# 06: Run a passive runoff simulation

**What to build:** 参考银行沿真实生成的市场路径完整滚动 60/180 月，逐月可重建现金与资产负债恒等式。

**Blocked by:** 03/Generate HJM-PCA scenarios; 05/Build the Reference Bank.

**Status:** completed

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] The simulation accepts the canonical snapshot and a batch of market paths and returns batch-first observable states for all 60 or 180 transitions.
- [x] Cash-flow ladders settle their first entry, shift one month, and are economically revalued on the current curve without mutating the source snapshot.
- [x] Cash, assets, liabilities, and equity are tracked in mCHF and remain finite on deterministic runoff fixtures.
- [x] Assets minus liabilities minus equity and the independently reconstructed cash movement stay within the specified accounting tolerance at every state.
- [x] The same simulator contract supports short hand-calculated fixtures and complete five-year and fifteen-year paths.
- [x] Failure diagnostics identify the first path, time, and state component that becomes inconsistent or non-finite.
- [x] 保留梯子结算、移位、重估、只读初始快照，以及独立手算和完整期限测试；用小路径数而非缩短期限控制开销。
- [x] The required accounting and independently reconstructed cash-reconciliation error is at most max(1e-6 mCHF, 1e-10 times total assets) at every path and state.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。保留原 runoff 金融行为；小路径数配合完整 60/180 月测试，不缩短期限。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-07 — 已完成：新增 PyTorch `ALMSimulator`（当前的 `PassiveRunoffSimulator` 别名），按月结算、左移六条 180 月梯子、用每期当前曲线重估，并输出 batch-first cash/assets/liabilities/equity、各组合价值、结算额及两条独立误差轨迹。实测手算 2-transition 夹具与真实 HJM-PCA 生成的 5y/15y 单路径均通过；非有限市场输入返回首个 path/time/component 诊断；会计和现金重建均按 max(1e-6 mCHF, 1e-10×assets) 容差校验。
