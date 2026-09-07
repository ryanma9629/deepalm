# 15: Train the five-year MM and check paper width

**What to build:** compact MM 完整训练、选模和重载 60 月策略；同一实现的 paper-width MM 完成一次 2 路径完整优化器更新。

**Blocked by:** 14/Validate MM observations and configurable network widths; 21/Bound training interruptions and verify recovery.

**Status:** ready-for-agent

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [ ] Training paths, `mu`, and `lambda` are regenerated from named epoch seeds while selection and locked-test scenarios remain fixed and isolated.
- [ ] All 60 applications share network parameters and receive the correct `t/T` value and prior post-restructuring constraint features.
- [ ] The selected five-year BM^D checkpoint is frozen, identity-checked, and used by both action heads throughout training.
- [ ] Losses, state, actions, and gradients remain finite, and training changes MM actions on a deterministic smoke fixture.
- [ ] The selected checkpoint and run evidence are sufficient to repeat evaluation without rerunning training.
- [ ] 默认 32/32/32、2 轮、batch 8，使用冻结 5 年 BM^D；完整时序与梯度，不要求训练后优于 benchmark。
- [ ] 论文宽度检查不覆盖 compact checkpoint，不增加第二套训练矩阵；其更新和资源实测单列。验证参数、状态、损失和梯度有限。
- [ ] Run one primary compact training job with eight default optimizer updates and one separate two-path paper-width forward/backward/optimizer check over all 60 transitions. Record the extra update and synchronized resource measurements separately; both use the selected five-year BM^D identity.
- [ ] The selected checkpoint includes the feature transform and frozen-baseline identity required by the shared recovery contract; verify that these dependencies survive checkpoint reload.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。原全宽 quick 训练改为 compact local_flow，另做一次两路径全宽完整优化器检查。先完成 21 的恢复能力。原编号保留；本次只更新待办，不表示本票实现已经完成。
