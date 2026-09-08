# 09: 执行修复后的受限配对训练

**What to build:** 用新兼容输入从头生成两口径、两期限的 BM^D/MM 八项试跑证据。

**Blocked by:** 08.

**Status:** ready-for-agent

来源：Deep ALM 论文一致性修复增量规格 Revision 1；本专项独立编号，旧专项票不变。

- [ ] 保留M5单进程MPS float32、compact64/64/32/32、train16/selection16、batch8、2epochs、1主seed、共32更新；每个MM依赖本口径本期限新冻结BM^D。
- [ ] 保留600秒协作式总预算、独立12GiB RSS/MPS及420秒预测门槛；超限停止并保留未完成证据，不擅改参数。
- [ ] 不复用旧权重、不使用locked test选择参数；记录8个新checkpoint及实际资源。资源未完成时本票仍未完成，后续可报告诊断但不算正式验收。

## Comments

2026-09-08 — 用户已确认十票拆分与阻塞关系；按本地 Markdown tracker 发布。
