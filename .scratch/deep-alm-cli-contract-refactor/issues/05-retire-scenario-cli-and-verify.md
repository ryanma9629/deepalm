# 05: 收缩场景化 CLI 并完成回归

**What to build:** 用户只看到动作导向的公共生命周期；历史场景化产物仍保留，但只有在 Workflow Contract 与已有身份检查完全兼容时才可复用。

**Blocked by:** 02: 两策略合同接入通用生命周期; 03: 四策略合同接入通用生命周期; 04: 迁移配置样例与双语文档.

**Status:** resolved

- [x] 删除场景化执行命令和重复别名，通用动作、准备、诊断和恢复动作仍可独立使用。
- [x] 已退役命令在 CLI 中被拒绝，旧目录不会仅凭文件名被推断或提升为新合同产物。
- [x] 完整测试、lint、CLI help 和全部样例 `plan` 冒烟验证通过。

## Answer

公开 CLI 现仅保留 `plan`、`run`、`evaluate`、`report` 与独立的
`preflight`、`bank`、`device-check`、`resume` 动作。旧场景命令和重复别名
会作为未知命令被拒绝；现有 artifact identity 校验继续要求显式兼容的
Workflow Contract，不会从历史目录名推断合同。命令面测试覆盖全部退役名称、
帮助输出和五份发布配置的 `plan` 解析。
