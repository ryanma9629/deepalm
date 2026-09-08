# 17: Lock test evaluation and small-sample statistics

**What to build:** 冻结模型后一次性用 32 条共同测试路径评估全部政策/期限，输出损失、收益、风险、约束和配对统计。

**Blocked by:** 12/Train BM^C with the local workflow budget; 16/Train the fifteen-year MM and verify five-year truncation.

**Status:** resolved

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] Training, selection, and final-test scenario identities are disjoint, and final-test results cannot participate in checkpoint choice or sensitivity-trigger decisions.
- [x] All policies at a horizon use common locked market paths and horizon comparisons use paired Brownian prefixes.
- [x] Reports include total, target, and penalty losses; penalty ES95; CRRA; equity-ratio moments; signed VaR95 and ES95; annualized return; and standardized dividend yield.
- [x] Constraint reporting includes time distributions, medians, ever-violation shares, violating counts and means, worst values, mean penalty, and penalty ES95.
- [x] Signed tail metrics retain their declared meaning and are verified by independent fixtures.
- [x] Model evaluation is reproducible from frozen checkpoints and fails if manifests, convention identities, paths, or required seeds are incompatible.
- [x] 保留完整指标定义与 signed VaR/ES；默认 100 次 bootstrap，标为小样本演示；未定义指标输出 unavailable 及原因，不写成零。
- [x] 测试集与 training/selection 隔离，不能选 checkpoint、筛 seed 或触发重训；更大评估采用有界分块与固定路径身份。
- [x] Use 100 deterministic paired common-index bootstrap resamples locally and report point differences with 95% intervals, actual path count and resample count. A configurable larger count remains opt-in; neither interval significance nor policy superiority gates local delivery.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。默认 32 条锁定测试路径和 100 次 bootstrap，仅演示统计流程。完整统计定义与防泄漏要求不变。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-08 — 已实现并验证。LockedEvaluator 仅加载冻结 checkpoint，并按 horizon 用固定 seed、`test` split、epoch 0 和共同 global path index 生成一次市场路径；不暴露任何 checkpoint 选择、重训或敏感性触发接口。manifest 锁定 convention、输入身份、test/bootstrap seed、路径身份和 checkpoint hash。输出总/目标/罚金损失、正向 penalty ES95、CRRA、权益比率、带符号下尾 VaR/ES、年化收益、标准化股利收益及完整六项约束摘要；不可定义指标保留 unavailable 和原因。配对 bootstrap 固定为 100 次，记录样本量、点差和 95% 区间，仅作小样本描述。

## Answer

已交付 `LockedEvaluator` 及其可序列化 manifest。它只在共同、固定的 final-test 路径上执行冻结 BM/MM checkpoint；对数据、约定、seed、训练/选择路径及测试路径规模不兼容的 checkpoint fail fast。报告保留不可定义指标的原因，并以 100 次确定性配对 bootstrap 输出小样本描述统计。聚焦测试 5 项、全量测试 108 项均通过。
