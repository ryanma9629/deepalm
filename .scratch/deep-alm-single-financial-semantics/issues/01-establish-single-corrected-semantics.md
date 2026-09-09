# 01: 建立唯一已修正金融语义契约

**What to build:** 让普通配置、市场情景和银行滚动只执行唯一的已修正金融语义：协方差一致的 PCA 缩放和按月贷款计息。用户不能通过配置或内部调用选择、覆写或间接触发已退役的金融公式。

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] 配置拒绝任何 convention 区块、退休 profile 或逐公式覆写；无该区块的标准配置可解析并完成受限 preflight。
- [x] 市场校准、情景生成和贷款状态转移只暴露唯一公式，且确定性 PCA/贷款测试证明修正后的数值行为。
- [x] 所有通用训练、模拟、选择、敏感性和评估调用迁移到无口径选择的共享契约，完整回归套件保持通过。

## Answer

2026-09-09 — 已移除运行配置、HJM PCA/情景、贷款转换、ALM rollout、训练、
评估、分析和通用报告中的金融口径选择。标准配置及版本控制的 YAML 不再携带
`convention`，遗留该区块会被 schema 拒绝；PCA 固定采用平方根特征值缩放，贷款
利息固定按月计。旧 paired 生命周期仍按其现有 8-job/32-update 公共契约准确声明，
其替换与退役由后续 03–05 票完成。

验证：`uv run ruff check src tests`、`uv run python -m compileall -q src tests` 和
`uv run pytest -q` 均通过；完整套件为 237 passed（113.04s）。以 `a9a74a2` 为
实施前基线的 Standards 与 Spec 双轴复核均无未解决的 Ticket 01 范围问题。
