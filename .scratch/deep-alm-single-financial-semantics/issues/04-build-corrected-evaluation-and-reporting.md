# 04: 交付已修正评估与报告链路

**What to build:** 让用户可以对已完成的四成员 Corrected Local Validation Pilot 运行锁定评估和审计报告，并只获得当前金融语义、完整训练证据对应的结果。

**Blocked by:** 02 — 隔离当前金融语义的 artifact 重用; 03 — 交付已修正本机验证试跑.

**Status:** resolved

- [x] `corrected-evaluate` 仅接受完整、当前版本的四成员来源，并生成身份链接的锁定评估证据。
- [x] `corrected-report` 仅从兼容的 pilot 和评估证据生成有限、原子写入的审计报告；不再包含 paired interval、口径差异或不完整成员提升逻辑。
- [x] 退役 paired 评估/报告命令在 CLI 中不再公开，调用它们按未知命令处理。

## Answer

2026-09-09：新增 `corrected-evaluate` 与 `corrected-report`。评估严格校验完整的四成员、16-update、当前训练语义和 checkpoint 内容哈希，并写入与 pilot manifest 绑定的锁定证据；报告只接受该完整、兼容的 pilot/evaluation 链路，原子写入四成员审计报告。paired 评估/报告 CLI、Runner 与报告实现已退役，旧命令按未知命令处理。已通过聚焦回归、Ruff、完整 Pytest 与双轴 code review。
