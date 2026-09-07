# 13: Train BM^D and freeze the local baselines

**What to build:** 两种期限分别得到按决策时点学习的 benchmark 及可冻结、可追溯的 MM baseline。

**Blocked by:** 11/Train BM^E locally and measure resource use.

**Status:** ready-for-agent

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [ ] BM^D learns independent investment and funding scales and maturity distributions for every actual decision date in both resolved horizons.
- [ ] The policy remains scenario-independent, emits non-negative common-contract actions, and uses the same simulator and optimization machinery as the other benchmarks.
- [ ] Actual parameter counts are reported and the paper's indexing mismatch is recorded rather than silently changing the 60/180-transition timeline.
- [ ] The selected parameters can be loaded through an immutable baseline reference whose identity is validated before MM use.
- [ ] Behavioral tests show time dependence, absence of scenario dependence, and correct per-date maturity availability.
- [ ] 保留 60/180 个实际决策日、参数量与论文索引差异记录；默认 local_flow，仅用 selection 选模。
- [ ] Both horizons complete genuine local_flow training with one seed each, and selection uses only the fixed selection scenarios. Freeze the selected baseline even if its small-run economic performance is poor; do not introduce an economic-superiority gate.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。以 local_flow 训练两期限并冻结所选 BM^D；不以经济效果决定是否可以作为 MM 依赖。原编号保留；本次只更新待办，不表示本票实现已经完成。
