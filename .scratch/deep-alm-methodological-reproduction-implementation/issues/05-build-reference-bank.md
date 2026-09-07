# 05: Build the Reference Bank

**What to build:** 构建、保存和重载标准参考银行，输出可核对的资产负债表与假设说明。

**Blocked by:** 02/Reconstruct SNB term structures.

**Status:** completed

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] The canonical bank starts with 10,000 mCHF of assets, uses the approved balance-sheet shares, and calculates equity as assets minus liabilities.
- [x] Seasoned cohorts generate six 180-month nominal cash-flow ladders using the approved loan, deposit, investment, and funding maturity assumptions.
- [x] Each portfolio is scaled to its target economic value on the canonical initial curve within `1e-8` relative error.
- [x] The snapshot records costs, product assumptions, units, provenance, initial-curve identity, targets, and a stable content hash.
- [x] Synthetic generation and saved-snapshot loading satisfy the same immutable provider contract and return equivalent economic content.
- [x] Validation rejects identity, value, shape, sign, duration, unit, and provenance failures; aggregate loan duration is below five years and deposit duration below three years.
- [x] A standalone Reference Bank stage produces an assumption-labeled Table 1 suitable for review.
- [x] 保留 10,000 mCHF、既定资产负债比例、seasoned cohorts、六条 180 月梯子、现值与久期验收及来源哈希；现金不作平衡项。
- [x] 初始快照验证是独立可执行阶段。通用输入格式和非标准银行的扩展验证归 26，不重复开发第二套银行模型。
- [x] Canonical duration and economic-value requirements remain reference-case validations; 26 adds noncanonical import without replacing the canonical construction or weakening its checks.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。保留原 Reference Bank 金融构建与验收；通用非标准输入验证在 26。依赖简化为 02，不重新开发已完成市场能力。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-07 — 已完成：`ReferenceBankProvider` 构造 six 180-month ladders、基于 canonical initial curve 缩放现值、冻结快照并以内容哈希保护；`deepalm bank` 原子写出 `reference-bank.json` 与含假设/来源标签的 `table-1.json`。真实 SNB 输入端到端运行得到 10,000 mCHF assets、9,000 mCHF liabilities、1,000 mCHF equity、loan duration 2.9231 years、deposit duration 1.5183 years；全量 lint/test 通过（41 passed）。
