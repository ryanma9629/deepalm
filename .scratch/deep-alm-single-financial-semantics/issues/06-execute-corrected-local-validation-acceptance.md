# 06: 执行已修正本机验证试跑验收

**What to build:** 在约定的 M5 MacBook 资源限制内实际执行完整 Corrected Local Validation Pilot，并交付能证明流程可运行、但不宣称收敛或银行模型获批的审计证据。

**Blocked by:** 03 — 交付已修正本机验证试跑; 04 — 交付已修正评估与报告链路; 05 — 收口退役表面并发布实现版勘误.

**Status:** resolved

- [x] 实际四任务 pilot、锁定评估和报告在十分钟内完成，或在既定资源 guard 触发时以失败 artifact 明确结束而不擅改预算。
- [x] 完成证据显示四个预期成员、16 次实际更新、有限损失/梯度/所需风险值、同期限冻结 baseline 依赖和当前金融语义版本。
- [x] 验收结果明确标注为本机技术流程验证，不宣称论文数值复现、经济收敛或银行生产模型批准。

## Answer

2026-09-09：在 M5/MPS、float32、既定 600 秒/12 GiB guard 下执行完整 corrected-only 生命周期。`artifacts/corrected-local-validation-pilot` 于 151.30 秒完成四成员（BM^D/MM × 5/15 年）试跑，每成员 4 次、合计 16 次实际更新，均记录有限非零优化信号；峰值 RSS 为 1,030,488,064 bytes，峰值 MPS 已分配内存为 238,845,952 bytes。MM 的 5 年和 15 年成员分别链接当次同期限 BM^D 冻结 baseline。锁定评估在 21.04 秒完成，并生成四成员的有限年化回报、风险和约束报告；最终报告成功生成。三个 artifact 均使用 `corrected-financial-semantics-v1`，并明确声明仅用于本机技术流程验证，不构成收敛、方法论数值复现或银行模型批准。`acceptance_status` 继续为 `pending`，因为该受限 corrected 生命周期不将自身升级为生产/模型批准。
