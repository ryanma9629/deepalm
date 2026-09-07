# 12: Train BM^C with the local workflow budget

**What to build:** 学习固定期限分配和规模的 benchmark 在 5/15 年上使用同一 trainer 训练与选模。

**Blocked by:** 11/Train BM^E locally and measure resource use.

**Status:** ready-for-agent

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [ ] BM^C learns time-shared investment and funding maturity distributions and scale adjustments through the common policy contract.
- [ ] Benchmark scale combines the next-period maturing amount with the learned adjustment and produces non-negative 29-dimensional actions.
- [ ] Tests distinguish BM^C's learned constant allocation from BM^E's equal allocation while showing that neither depends on current market scenario.
- [ ] 默认 local_flow；验证情景无关、跨时点共享分配、非负行动，留待 17 统一锁定测试比较。
- [ ] Both full horizons complete genuine local_flow training and selection through the same trainer. Selected checkpoints expose the same core loss/action records as BM^E; final-test evaluation belongs to 17.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。以 local_flow 训练两期限，保留 BM^C 金融/策略契约；避免在 17 之前提前消费最终测试集。原编号保留；本次只更新待办，不表示本票实现已经完成。
