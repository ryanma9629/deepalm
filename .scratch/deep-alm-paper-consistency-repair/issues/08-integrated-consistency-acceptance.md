# 08: 全链路集成与论文一致性验收

**What to build:** 对所有修复形成基于原文/errata与独立手算的要求—证据清单，验证完整入口及旧产物拒绝。

**Blocked by:** 02, 03, 04, 05, 07.

**Status:** resolved

来源：Deep ALM 论文一致性修复增量规格 Revision 1；本专项独立编号，旧专项票不变。

- [ ] 检查训练、selection、导入、敏感性、截断、恢复及report每条路径；同形状旧身份不可绕过，02–07已完成语义如实汇总。
- [ ] 完成CPU float64独立公式/梯度及60/180步CPU/MPS小样本检查，保留既有dtype容差与不可用硬件声明。
- [ ] 全套测试与Standards/Spec双轴审查无未解决实质问题；只验证实现，不启动配对训练或宣称收敛。

## Comments

2026-09-08 — 用户已确认十票拆分与阻塞关系；按本地 Markdown tracker 发布。

## Answer

2026-09-08 — 已完成 02–07 修复的全链路验收登记。受限 local
workflow 现在只接受 CPU float64；其 acceptance-report 记录修复—证据清单、
当前 evaluation artifact semantics、训练/选择、导入、敏感性、截断、恢复及
报告路径，并明确不宣称配对训练、收敛或银行模型批准。BM^E 与 MM 的真实
device-validation 都覆盖 5/15 年小样本 CPU/MPS/CUDA 更新、恢复及不可用状态。

验证：`uv run pytest -q` 为 243 passed（114.70s）；Ruff 定向检查通过。以
`097836b` 为实施前基线的 Standards 与 Spec 双轴复审均无未解决实质发现。
