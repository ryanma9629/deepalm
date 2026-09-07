# 17: Lock test evaluation and small-sample statistics

**What to build:** 冻结模型后一次性用 32 条共同测试路径评估全部政策/期限，输出损失、收益、风险、约束和配对统计。

**Blocked by:** 12/Train BM^C with the local workflow budget; 16/Train the fifteen-year MM and verify five-year truncation.

**Status:** ready-for-agent

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [ ] Training, selection, and final-test scenario identities are disjoint, and final-test results cannot participate in checkpoint choice or sensitivity-trigger decisions.
- [ ] All policies at a horizon use common locked market paths and horizon comparisons use paired Brownian prefixes.
- [ ] Reports include total, target, and penalty losses; penalty ES95; CRRA; equity-ratio moments; signed VaR95 and ES95; annualized return; and standardized dividend yield.
- [ ] Constraint reporting includes time distributions, medians, ever-violation shares, violating counts and means, worst values, mean penalty, and penalty ES95.
- [ ] Signed tail metrics retain their declared meaning and are verified by independent fixtures.
- [ ] Model evaluation is reproducible from frozen checkpoints and fails if manifests, convention identities, paths, or required seeds are incompatible.
- [ ] 保留完整指标定义与 signed VaR/ES；默认 100 次 bootstrap，标为小样本演示；未定义指标输出 unavailable 及原因，不写成零。
- [ ] 测试集与 training/selection 隔离，不能选 checkpoint、筛 seed 或触发重训；更大评估采用有界分块与固定路径身份。
- [ ] Use 100 deterministic paired common-index bootstrap resamples locally and report point differences with 95% intervals, actual path count and resample count. A configurable larger count remains opt-in; neither interval significance nor policy superiority gates local delivery.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。默认 32 条锁定测试路径和 100 次 bootstrap，仅演示统计流程。完整统计定义与防泄漏要求不变。原编号保留；本次只更新待办，不表示本票实现已经完成。
