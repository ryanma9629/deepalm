# 26: Validate replaceable bank and market inputs

**What to build:** 用户可导入不同于标准参考案例的合成银行及匹配市场输入，验证、保存、重载并完成被动滚动。

**Blocked by:** 06/Run a passive runoff simulation.

**Status:** ready-for-agent

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [ ] 提供有版本的快照导入/导出、schema、示例及映射说明；覆盖估值、六条梯子、成本、增长/利差、产品假设、币种/百万计量单位、估值日、曲线身份与 provenance。
- [ ] 通过既有 ReferenceBankProvider 和 MarketScenarioModel 契约接受匹配输入；非标准合成样本不得被悄悄重建为 10,000 mCHF、2022-07-15 的标准银行。
- [ ] 区分标准案例的组合/久期目标与通用的会计、有限值、形状、符号、单位及日期一致性校验。非标准样本按其声明的目标和假设验证。
- [ ] 拒绝不支持的产品、超出支持期限的现金流、混合币种与市场/快照不一致，不默默截断或转换；不实现 FX 聚合。
- [ ] 用合成数据验证保存/重载身份与滚动现金一致性；实际银行源系统连接、数据抽取和业务映射签字留给银行环境。
- [ ] Expose import/export and validation through the existing provider and runner-stage contracts, and compare round-tripped content hashes and independently reconstructed passive cash/accounting results on a matched synthetic market batch.
- [ ] An alternate single-currency snapshot declares millions of that currency consistently; canonical mCHF assumptions, valuation date, calibration history and duration targets are not imposed on all imports. Reject incomplete provenance, wrong ladder length, negative contractual ladders beyond tolerance, and incompatible curve/snapshot identities.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。新增可替换输入切片；用非标准合成样本验证导入到滚动，不连接银行真实源系统。使用追加编号；本次只更新待办，不表示本票实现已经完成。
