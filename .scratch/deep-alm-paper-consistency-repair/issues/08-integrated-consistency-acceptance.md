# 08: 全链路集成与论文一致性验收

**What to build:** 对所有修复形成基于原文/errata与独立手算的要求—证据清单，验证完整入口及旧产物拒绝。

**Blocked by:** 02, 03, 04, 05, 07.

**Status:** ready-for-agent

来源：Deep ALM 论文一致性修复增量规格 Revision 1；本专项独立编号，旧专项票不变。

- [ ] 检查训练、selection、导入、敏感性、截断、恢复及report每条路径；同形状旧身份不可绕过，02–07已完成语义如实汇总。
- [ ] 完成CPU float64独立公式/梯度及60/180步CPU/MPS小样本检查，保留既有dtype容差与不可用硬件声明。
- [ ] 全套测试与Standards/Spec双轴审查无未解决实质问题；只验证实现，不启动配对训练或宣称收敛。

## Comments

2026-09-08 — 用户已确认十票拆分与阻塞关系；按本地 Markdown tracker 发布。
