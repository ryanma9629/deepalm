# 03: 四策略合同接入通用生命周期

**What to build:** 用户能够仅凭四策略本机配置，使用同一组通用动作执行 BM^E、BM^C、BM^D、MM 在 5 年和 15 年的初步比较，并取得八成员、共同锁定情景的诚实报告。

**Blocked by:** 01: Workflow Contract 与通用计划.

**Status:** resolved

- [x] 通用生命周期发布八个预期 policy-horizon 成员、32 个更新和 MM 的同期限冻结 BM^D baseline reference。
- [x] 评估和报告验证每个期限的共同锁定测试情景，并披露单 seed、小路径数和两 epoch 的本机验证边界。
- [x] 行为级 CLI 测试覆盖四策略合同而不依赖场景化子命令。

## Answer

`local-four-policy-comparison` 现通过配置驱动的 `plan`、`run`、`evaluate`
和 `report` 生命周期执行。四策略的八个 policy-horizon 成员、32 次更新、
MM 同期限 BM^D 冻结基线、共同锁定评估身份和本机验证边界均由生产路径验证；
不兼容来源的诊断也会使用四策略合同身份。
