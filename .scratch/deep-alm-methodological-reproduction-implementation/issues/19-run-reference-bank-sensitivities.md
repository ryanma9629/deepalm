# 19: Evaluate a representative Reference Bank sensitivity

**What to build:** 用冻结策略评估 5,000 mCHF 单因素规模变体，走通敏感性分析与报告。

**Blocked by:** 17/Lock test evaluation and small-sample statistics.

**Status:** resolved

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] Every approved scale, duration, spread, and cost sensitivity produces a validated snapshot that differs from the canonical bank in only its named factor.
- [x] Invalid or economically inconsistent variants fail Reference Bank validation and remain visible in the sensitivity report.
- [x] 所有已批准规模、期限权重、利差、成本变体仍可配置并通过构建校验；本机默认只评估上述代表项。
- [x] 结果变化如实展示，不触发自动重训。全部变体评估及显式重训移入 24 的延期研究范围，不能使用最终测试结果作重训决策。
- [x] Evaluate the 5,000 mCHF variant with frozen canonical Corrected policies on common market identities at the applicable horizons, recording the exact one-factor change and any invalid snapshot or undefined metric.
- [x] Do not infer generalization-versus-structural conclusions from this small frozen-policy demonstration. Wider variant evaluation and any retraining require an explicitly budgeted research plan and never use locked final-test results as a trigger.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。本机仅评估冻结策略的 5,000 mCHF 代表项；全部变体构建能力保留。移除自动重训触发，扩展研究执行范围转入 24。原编号保留；本次只更新待办，不表示本票实现已经完成。

## Answer

Implemented approved one-factor Reference Bank snapshot construction for scale, duration weights, spreads, and operating costs through one shared registry. The local Runner evaluates only the required 5,000 mCHF scale variant with canonical frozen Corrected checkpoints on common locked paths; it writes an atomic sensitivity artifact with snapshot identities, invalid-variant diagnostics, frozen evaluation evidence, and an explicit non-generalization boundary. No training dependency or retraining trigger is present. Broader variant evaluation remains deferred to Ticket 24; end-to-end CLI orchestration remains Ticket 23.

Validation: focused and related suite `61 passed`; full suite `143 passed`; Ruff and `git diff --check` passed. Final standards and specification reviews found no blocking issues.
