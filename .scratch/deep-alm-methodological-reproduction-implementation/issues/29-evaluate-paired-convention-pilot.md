# 29: Evaluate paired-convention pilot outcomes and failures

**What to build:** 对已完成或中断的配对口径试跑生成锁定的、身份校验的评估证据；有限但较差的 Paper 表现和 Paper 数值失败均应被如实区分。

**Blocked by:** 28/Run the bounded paired-convention pilot matrix.

**Status:** resolved

**Execution scope:** opt-in-research

**Specification revision:** 2 (2026-09-08)

- [ ] Reuse only complete, semantically compatible paired-pilot checkpoints and frozen BM^D dependencies. Verify convention, Reference Bank, market/calibration, code/runtime, seed/path, architecture, horizon, and checkpoint identities before evaluation.
- [ ] Evaluate each available 5/15-year BM^D/MM checkpoint exactly once on its 64 locked paths, report loss, return, constraint, finite-value, and resource observations, and keep Paper and Corrected evidence separate.
- [ ] Evaluate each available MM(15y|5y) with the first 60 actions of its own 15-year MM and original `t/15` feature. Record zero optimizer updates and the source-checkpoint identity.
- [ ] Produce 100-resample paired-bootstrap comparisons only where both members have matching complete locked paths; otherwise record `not-applicable` with the incompatible or missing identity.
- [ ] A finite but worse Paper outcome is a reported result and does not fail Corrected. A non-finite Paper outcome identifies model stage, policy, convention, horizon, path, month, tensor/value, and preceding compatible evidence; it is not converted into a Corrected failure.
- [ ] The evaluator must not emit economic-superiority gates, `development-validated`, or `methodologically-reproduced`; this ticket supplies evidence for the bounded pilot only.

## Comments

2026-09-08 — Created from Ticket 20’s confirmed M5 pilot plan. This separates factual paired outcomes and numerical-failure disclosure from the later research assessment in Ticket 24.

2026-09-08 — Evaluated the completed paired pilot at `artifacts/paired-convention-pilot-evaluation`. All eight frozen checkpoints passed identity checks and were evaluated once on 64 locked paths per horizon. All 16 Paper-minus-Corrected, 100-resample paired intervals were available. Both MM(15y|5y) evaluations used their own 15-year checkpoints, ran exactly 60 action steps with the original 15-year time feature, and recorded zero optimizer updates. The evidence remains descriptive only; Ticket 30 owns its report.
