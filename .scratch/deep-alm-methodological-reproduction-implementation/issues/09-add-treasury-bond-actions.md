# 09: Add active treasury bond actions

**What to build:** 13 个投资期限和 16 个融资期限可买入/发行并持有至到期，交易对未来现金流和损失保持可微。

**Blocked by:** 06/Run a passive runoff simulation.

**Status:** completed

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] A treasury action contains 13 investment and 16 funding quantities with batch-first shape and rejects negative or malformed values.
- [x] New bonds are fractional, priced on the current curve, added to the correct ladder, held to maturity, and cannot be sold or unwound early.
- [x] Cash settlement multiplies every instrument cash flow by its action quantity and sums across maturities, correcting the written omission in Equation 18.
- [x] The resolved timeline performs the active decision and passive market/balance-sheet transition in the documented order for all 60 or 180 actions.
- [x] Action-dependent future cash flows remain differentiable through subsequent simulator transitions.
- [x] Zero-action behavior matches the passive runoff trajectory, and deterministic nonzero-action fixtures reconcile independently.
- [x] 保留数量加权现金结算、非负 29 维行动、禁止卖出及完整行动/转移顺序。

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。保留原 29 维无互换行动与完整可微转移。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-07 — 已完成：`TreasuryAction`、数量加权现值结算和滚动交易顺序已接入 `ALMSimulator`。`tests/test_treasury.py` 与 `tests/test_runoff.py` 覆盖非负/形状、3 个月短期点差、零行动、60/180 月末行动、独立现金对账和中央差分梯度；全套 `uv run pytest -q` 为 59 passed，`uv run ruff check src tests` 通过。
