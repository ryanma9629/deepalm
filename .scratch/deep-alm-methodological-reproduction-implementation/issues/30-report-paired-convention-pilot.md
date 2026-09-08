# 30: Report the bounded paired-convention pilot

**What to build:** 将配对口径试跑及其评估汇总为可审计研究报告，明确它只量化公式差异和数值行为。

**Blocked by:** 29/Evaluate paired-convention pilot outcomes and failures.

**Status:** ready-for-agent

**Execution scope:** opt-in-research

**Specification revision:** 2 (2026-09-08)

- [ ] Publish atomically a machine-readable paired-pilot report with separate Paper and Corrected sections for BM^D/MM 5/15 年及可用的 MM(15y|5y)。
- [ ] Identify for every paired result the two formula choices, convention, architecture, device/dtype, registered seed, common Reference Bank, market/calibration, split/path identity, horizon, checkpoint and frozen-baseline identity, actual updates, and resource use.
- [ ] Report paired differences, 100-resample intervals where applicable, constraints, numerical behavior, and every incomplete/non-finite branch with its evidence reference. Do not fabricate a pair, interval, or zero value when a member is missing.
- [ ] State prominently that this is a `paired-convention-research-pilot`: it is neither convergence evidence, paper-result replication, `methodologically-reproduced`, nor bank-model approval.
- [ ] Include a precise deferred-work inventory: paper widths, three MM seeds, sensitivity retraining, 10,000 bootstrap, full paper figures, and Ticket 24’s economic acceptance gates.

## Comments

2026-09-08 — Created from Ticket 20’s confirmed M5 pilot plan. The report is intentionally smaller than the full research reporting and economic assessment deferred to Ticket 24.
