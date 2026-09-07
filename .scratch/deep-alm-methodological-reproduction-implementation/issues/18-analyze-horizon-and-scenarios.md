# 18: Demonstrate horizon effects and rate scenarios

**What to build:** 利用冻结轨迹比较 turnover、终端集中度和五种利率类别，不再训练模型。

**Blocked by:** 17/Lock test evaluation and small-sample statistics.

**Status:** ready-for-agent

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [ ] Normalized action turnover is computed from consecutive L1 action changes using a documented denominator safeguard.
- [ ] Terminal concentration measures the share of turnover in the final 24 months of a five-year window.
- [ ] MM(5y), MM(15y) over its first 60 months, and MM(15y|5y) are compared with paired uncertainty intervals.
- [ ] Category outputs cover steep, upward, downward, inverted, and constant-steepness behavior and identify policy, path rule, sample size, horizon, and units.
- [ ] Edge cases involving zero action volume, tied rankings, overlapping categories, and insufficient paths behave deterministically.
- [ ] 每类默认 5 条，保留论文排名与允许重叠规则、索引及样本量；数量超过可用路径则拒绝，不能静默改变数量。
- [ ] 期限平滑度方向和区间显著性只报告，不作本机过关条件。
- [ ] Use the same local 100-resample paired statistics for horizon comparisons. Zero-volume safeguards, tied rankings and category overlap remain explicit; no branch may launch retraining.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。默认每类别 5 条路径；保留排名规则和期限指标，经济方向不作为本机门槛。原编号保留；本次只更新待办，不表示本票实现已经完成。
