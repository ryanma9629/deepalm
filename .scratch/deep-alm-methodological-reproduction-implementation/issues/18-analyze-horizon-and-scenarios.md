# 18: Demonstrate horizon effects and rate scenarios

**What to build:** 利用冻结轨迹比较 turnover、终端集中度和五种利率类别，不再训练模型。

**Blocked by:** 17/Lock test evaluation and small-sample statistics.

**Status:** resolved

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] Normalized action turnover is computed from consecutive L1 action changes using a documented denominator safeguard.
- [x] Terminal concentration measures the share of turnover in the final 24 months of a five-year window.
- [x] MM(5y), MM(15y) over its first 60 months, and MM(15y|5y) are compared with paired uncertainty intervals.
- [x] Category outputs cover steep, upward, downward, inverted, and constant-steepness behavior and identify policy, path rule, sample size, horizon, and units.
- [x] Edge cases involving zero action volume, tied rankings, overlapping categories, and insufficient paths behave deterministically.
- [x] 每类默认 5 条，保留论文排名与允许重叠规则、索引及样本量；数量超过可用路径则拒绝，不能静默改变数量。
- [x] 期限平滑度方向和区间显著性只报告，不作本机过关条件。
- [x] Use the same local 100-resample paired statistics for horizon comparisons. Zero-volume safeguards, tied rankings and category overlap remain explicit; no branch may launch retraining.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。默认每类别 5 条路径；保留排名规则和期限指标，经济方向不作为本机门槛。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-08 — 已实现并验证。`HorizonScenarioAnalyzer` 只接受完成后的动作快照和市场前缀身份，不含 trainer、optimizer 或 checkpoint 选择接口。它以连续动作的 L1 差分除以总动作 L1 量加 epsilon 计算 turnover；零 turnover 的终端集中度明确标记 unavailable，而非伪造为零。MM(5y)、MM(15y) 前 60 月与 MM(15y|5y) 使用共同、内容寻址的五年前缀身份配对；截断轨迹必须复用 15 年 checkpoint 身份并声明零 optimizer update。五类论文路径规则保留稳定 tie-break、允许重叠和实际全局索引。定向测试 4 项、全量测试 120 项及 ruff 均通过；两轴代码审查无遗留问题。

## Answer

已交付冻结轨迹的期限与情景分析模块。输出路径级 normalized turnover、终端 24 月 turnover concentration、三种 MM 轨迹的 100 次 deterministic paired-bootstrap 区间，以及带策略、路径规则、全局索引、样本量、期限和单位的五类情景摘要。配对市场身份、截断来源与零换手的不可定义集中度均 fail fast 或显式标记 unavailable；分析过程不会重训模型。
