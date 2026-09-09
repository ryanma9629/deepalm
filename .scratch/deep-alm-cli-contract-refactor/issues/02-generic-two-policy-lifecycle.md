# 02: 两策略合同接入通用生命周期

**What to build:** 用户能够仅凭两策略本机配置，使用通用 `run`、`evaluate` 和 `report` 执行 BM^D/MM 的 5 年与 15 年已修正本机验证，并获得包含冻结 BM^D baseline reference 与兼容性证据的产物。

**Blocked by:** 01: Workflow Contract 与通用计划.

**Status:** resolved

- [x] 通用生命周期发布四个预期 policy-horizon 成员、16 个更新和 MM 的同期限冻结基准引用。
- [x] 通用评估与报告拒绝合同、市场、Reference Bank 或锁定情景身份不兼容的证据。
- [x] 行为级 CLI 测试覆盖完整两策略生命周期而不依赖内部 Runner 路由。

## Answer

`run`、`evaluate` 与 `report` 现按 local-two-policy-validation Workflow Contract 分派到完整 BM^D/MM 生命周期。通用评估和报告在消费来源前验证 Workflow Contract、Execution Profile、市场/Reference Bank 配置、策略矩阵、期限、种子及 artifact semantics；锁定评估证据也记录合同与资源档身份。现有场景化命令保留至收缩 ticket。
