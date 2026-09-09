# 01: Workflow Contract 与通用计划

**What to build:** 用户能够用一个完整配置定义 Deep ALM 的 Workflow Contract 与 Execution Profile，并通过通用 `plan` 动作查看解析后的 TreasuryPolicy 矩阵、资源预算、合同身份和金融语义；现有场景化运行命令在扩展阶段保持可用。

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] 配置能声明并验证 Workflow Contract 和独立的 Execution Profile，且解析结果包含全部实际运行值。
- [x] `plan` 以公共 CLI seam 输出合同、TreasuryPolicy 成员、期限、资源预算和 Corrected Financial Semantics 身份。
- [x] 两策略、四策略、既有本机、方法论研究计划和银行单设备 commissioning 合同可被解析或明确报告能力边界。
- [x] 既有场景化命令和金融计算在本 ticket 中没有被删除或改变。

## Answer

Workflow Contract 与 Execution Profile 已成为每个可执行配置的必填、可审计身份。通用 `plan` 现输出二者、已修正金融语义与完整 resolved configuration；所有资源档都验证设备、数值类型、网络、路径、epoch、批大小、早停、时间和内存信封。保留了现有场景化命令，未修改金融计算。已完成 focused 与完整测试、lint 及三个本机合同的 `plan` 冒烟验证。
