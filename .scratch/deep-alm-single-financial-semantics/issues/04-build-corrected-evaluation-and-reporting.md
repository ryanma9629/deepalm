# 04: 交付已修正评估与报告链路

**What to build:** 让用户可以对已完成的四成员 Corrected Local Validation Pilot 运行锁定评估和审计报告，并只获得当前金融语义、完整训练证据对应的结果。

**Blocked by:** 02 — 隔离当前金融语义的 artifact 重用; 03 — 交付已修正本机验证试跑.

**Status:** ready-for-agent

- [ ] `corrected-evaluate` 仅接受完整、当前版本的四成员来源，并生成身份链接的锁定评估证据。
- [ ] `corrected-report` 仅从兼容的 pilot 和评估证据生成有限、原子写入的审计报告；不再包含 paired interval、口径差异或不完整成员提升逻辑。
- [ ] 退役 paired 评估/报告命令在 CLI 中不再公开，调用它们按未知命令处理。
