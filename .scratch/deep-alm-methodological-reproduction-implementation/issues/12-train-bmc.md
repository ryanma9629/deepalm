# 12: Train BM^C with the local workflow budget

**What to build:** 学习固定期限分配和规模的 benchmark 在 5/15 年上使用同一 trainer 训练与选模。

**Blocked by:** 11/Train BM^E locally and measure resource use.

**Status:** resolved

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] BM^C learns time-shared investment and funding maturity distributions and scale adjustments through the common policy contract.
- [x] Benchmark scale combines the next-period maturing amount with the learned adjustment and produces non-negative 29-dimensional actions.
- [x] Tests distinguish BM^C's learned constant allocation from BM^E's equal allocation while showing that neither depends on current market scenario.
- [x] 默认 local_flow；验证情景无关、跨时点共享分配、非负行动，留待 17 统一锁定测试比较。
- [x] Both full horizons complete genuine local_flow training and selection through the same trainer. Selected checkpoints expose the same core loss/action records as BM^E; final-test evaluation belongs to 17.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。以 local_flow 训练两期限，保留 BM^C 金融/策略契约；避免在 17 之前提前消费最终测试集。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-07 — 已实现并验证：BM^C 使用两个跨时点共享的 softmax 期限分配与两个共享规模调整量，并与 BM^E 共用 BenchmarkTrainer 的可微训练、选择、checkpoint、设备验证和资源记录流程。local_flow 实测：5 年期更新 18.57 秒、选择 17.88 秒、峰值 RSS 668.2 MB、MPS 28.0 MB；15 年期更新 51.62 秒、选择 48.29 秒、峰值 RSS 727.1 MB、MPS 53.2 MB。审计证据位于 `artifacts/quick-skeleton/BM_C_5y.pt`、`BM_C_5y.profile.json`、`BM_C_15y.pt` 与 `BM_C_15y.profile.json`；每份 checkpoint 包含每轮 selection 的损失与实际动作汇总。最终测试集未使用，留待 17。
