# 24: Assess the complete methodological reproduction

**What to build:** 未来在完整研究证据上评估 methodologically-reproduced；本次不实现或执行该大规模验收。

**Blocked by:** 20/Run the paired Paper-convention experiment; 23/Run and validate the complete local workflow.

**Status:** needs-triage

**Execution scope:** opt-in-research

**Specification revision:** 2 (2026-09-07)

本票是延期研究范围记录，不是当前可领取的本机实现票。须显式确认研究环境及资源计划后再激活；24 还需先细拆。下列勾选项是未来研究验收要求，不阻塞 local_flow。

- [ ] Final assessment requires the complete Corrected paper-scale matrix, all three MM seeds per horizon, the paired Paper matrix or disclosed numerical failure, sensitivities, and every required artifact.
- [ ] Corrected MM benchmark improvements and paired-confidence requirements are evaluated for total, target, and penalty losses and annualized return exactly as specified.
- [ ] Constraint violation-rate, EYR frequency, mean penalty, and penalty ES95 gates are evaluated per qualifying MM run and seed.
- [ ] Five-year versus fifteen-year turnover, terminal-concentration, annualized-return, and CRRA directionality gates use paired intervals and their declared thresholds.
- [ ] Every gate reports pass, fail, or not-applicable together with observed value, threshold, direction, and evidence reference.
- [ ] Adverse seeds and paths remain in the report, while equity volatility, VaR, ES, and supporting-only comparisons cannot silently become hard gates.
- [ ] `methodologically-reproduced` is emitted only when every mandatory gate passes; otherwise the completed report explains why the label was withheld.
- [ ] 集中保留从旧 19/21/22 移出的全部敏感性与显式重训、完整 Corrected paper-scale 矩阵、各期限 3 个 MM seed、10,000 次 bootstrap 和完整无互换研究图表的范围。
- [ ] 保留原经济/统计方向与阈值以及已接受的 cubic-fit 例外；不据本机小样本修改阈值或宣布方法论复现成功。
- [ ] 本项是延期范围记录，不是当前可一次领取的大实现票。启用时拆分“扩展矩阵与敏感性”“完整研究报表”“经济验收”后再执行；其完成也不等于银行真实数据模型获批。
- [ ] Preserve the historical economic protocol: MM point improvements against all benchmarks and paired 95% intervals against BM^D for total, target, penalty loss and annualized return in at least two of three MM seeds at each horizon.
- [ ] Preserve per-qualifying-run ever-violation limits of 1% for LCR/NSFR and 2% for CMR, Equity/RWA and IRS; EYR is the most frequent violation and penalty mean/ES95 improve on BM^D in at least two seeds. These are deferred paper-comparison conditions, not local or bank-approval criteria.
- [ ] Preserve the 10% horizon-turnover and terminal-concentration reductions with the specified paired directions, and MM(5y) return/CRRA direction versus MM(15y|5y). Keep supporting-only truncated-MM penalty and non-superiority equity-volatility/VaR/ES comparisons separate.
- [ ] Before activation, split the extended Corrected matrix/sensitivities, full research reporting and economic evaluator into bounded executable tickets. This needs-triage record must not be claimed as one large current implementation task.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。保留编号和原研究门槛，集中接收旧 19/21/22 的延期范围；后续明确研究环境后先拆分，再执行，不等于银行模型审批。原编号保留；本次只更新待办，不表示本票实现已经完成。
