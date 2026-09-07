# 16: Train the fifteen-year MM and verify five-year truncation

**What to build:** 完成 compact 180 月训练和独立的 2 路径 paper-width 更新，并重用所选 15 年模型评估 MM(15y|5y)。

**Blocked by:** 15/Train the five-year MM and check paper width.

**Status:** ready-for-agent

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [ ] Time is normalized by 15 years at every decision during fifteen-year training.
- [ ] `MM(15y|5y)` loads the selected fifteen-year model, performs only its first 60 decisions, retains `t/15`, performs the five-year terminal roll, and is not retrained.
- [ ] Five-year and fifteen-year policies can consume paths derived from common Brownian innovations for their shared first 60 months.
- [ ] Truncation does not renormalize time, substitute the five-year baseline, or alter trained parameters.
- [ ] Deterministic tests distinguish the truncated policy from independently trained MM(5y) and reproduce its actions from checkpoint evidence.
- [ ] 全过程使用 15 年 BM^D 及原 t/15；截断只执行前 60 个决策与终点 roll，不重训、不改时间归一化。
- [ ] 保留共同 Brownian 前缀、checkpoint/preprocessing/baseline 身份及长路径资源测量。有限但经济表现不佳不阻断本票。
- [ ] Use the same 32/32/32, two-epoch, batch-eight local_flow defaults and compact architecture for the full 180-transition job. Add exactly one separate two-path full-horizon paper-width optimizer check; record its update/resource evidence separately.
- [ ] MM(15y|5y) reuses the frozen selected checkpoint and adds no optimizer update; policy time awareness and the fifteen-year baseline remain unchanged.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。原全宽 quick 训练改为 compact local_flow，另做一次两路径全宽检查；保留完整十五年期限和不重训的截断。原编号保留；本次只更新待办，不表示本票实现已经完成。
