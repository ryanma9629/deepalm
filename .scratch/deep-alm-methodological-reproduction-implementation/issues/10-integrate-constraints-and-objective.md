# 10: Integrate constraints and the training objective

**What to build:** 完整银行轨迹输出六项约束、终值目标损失、累计罚项，以及正确的参数梯度。

**Blocked by:** 07/Model loan growth, interest, and impairment; 08/Model deposits, costs, and dividends; 09/Add active treasury bond actions.

**Status:** completed

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] LCR, NSFR, CMR, Equity/RWA, IRS, and annual-only EYR match independent formulas at values below, at, and above their bounds.
- [x] Parallel plus/minus 100-basis-point revaluation drives IRS and annual close timing drives EYR without introducing hidden balancing values.
- [x] The asymmetric terminal target loss and squared cumulative constraint penalty use the approved coefficients, sampled `mu`, and sampled `lambda`.
- [x] Training excludes CRRA; evaluation computes CRRA with gamma 10 and the recorded equity-ratio floor.
- [x] Constraint values and violations are returned with trajectories at their applicable states and the initial values are available to later observations.
- [x] Smooth CPU float64 fixtures pass central finite-difference gradient checks and produce a nonzero policy gradient when terminal loss depends on action.
- [x] Full five-year and fifteen-year deterministic fixtures retain the required accounting tolerances while calculating losses.
- [x] 保留会计容差、约束边界手算、CPU float64 中央差分及非零梯度验收；CRRA 只作评估。
- [x] CPU float64 constraint-formula error is at most 1e-10; smooth-fixture gradients satisfy |g_auto - g_fd| <= 1e-6 + 1e-4 |g_fd| and the action-dependent fixture has policy-gradient norm greater than 1e-8.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。保留原约束、损失与独立会计/梯度硬检查，小样本不放宽数值容差。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-07 — 已完成：约束模块输出 LCR、NSFR、CMR、Equity/RWA、IRS 与年度 EYR 的值和非线性违约；目标模块实现抽样/固定参数、非对称终值损失、累计罚项及仅评估的 CRRA。ALMSimulator 现在提供初始和决策时点约束轨迹、年度掩码与目标组成，并保留行动到终值的梯度。`tests/test_constraints.py` 覆盖手算边界、准备金利率分支、±100bp IRS、年度分红前 EYR、负/零上一年权益、5/15 年路径和 CPU float64 中央差分；全套 `uv run pytest -q` 为 71 passed，`uv run ruff check src tests` 通过。
