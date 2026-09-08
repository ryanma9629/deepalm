# 28: Run the bounded paired-convention pilot matrix

**What to build:** 在 M5 / 32 GiB MacBook 上，以显式 opt-in 配置运行 Paper 与 Corrected 的配对 BM^D/MM 5/15 年研究试跑；它有严格资源护栏，但不宣称论文复现。

**Blocked by:** 13/Train BM^D locally; 16/Train the fifteen-year MM and verify five-year truncation; 21/Bound training interruptions and verify recovery; 27/Validate single-device portability and bank handoff.

**Status:** resolved

**Execution scope:** opt-in-research

**Specification revision:** 2 (2026-09-08)

- [x] Add an explicit `paired_convention_pilot` configuration whose only financial difference between Paper and Corrected is the locked PCA-loading scale and loan-interest annualization; it rejects custom conventions, swaps, paper-width escalation, multiple seeds, BME/BMC, and sensitivity retraining.
- [x] On the declared M5 machine, require one MPS float32 process, compact architecture, two epochs, 16 training paths and 16 selection paths per epoch, batch size 8, 64 locked-test paths, and one registered primary seed.
- [x] Orchestrate exactly eight primary training jobs when their financial rollouts remain finite: Paper and Corrected each run BM^D and MM at 5 and 15 years, and each completed job records four optimizer updates. Record MM(15y|5y) with zero training updates and its own selected 15-year MM checkpoint; Ticket 29 performs the locked evaluation.
- [x] Keep paired runs on the same canonical Reference Bank and matching market/objective/model-initialization/data-loader seed identities and path identities for each split/epoch/global index. They never share a checkpoint or a frozen BM^D baseline across conventions.
- [x] Each MM loads the matching convention/horizon selected frozen BM^D baseline and preserves the existing identity and recovery contracts.
- [x] Apply one 600-second wall-clock budget and 12 GiB RSS/MPS guards to the pilot. After the first completed epoch of both 15-year MM jobs, estimate total pure-training time; if it exceeds 420 seconds, do not start remaining work and produce an `incomplete` bundle with measurements and resumable diagnostics.
- [x] Budget exhaustion, unavailable MPS, non-finite computation, incompatible resume, or missing required jobs return nonzero and preserve stage/path/month/value diagnostics. A successful pilot is labeled only `paired-convention-research-pilot`.

## Comments

2026-09-08 — Created from Ticket 20 after the user explicitly selected the bounded M5 pilot. The 32-update matrix is a formula/numerical-behavior experiment, not the deferred paper-scale research protocol.

2026-09-08 — Implemented with isolated Paper/Corrected checkpoint trees, shared immutable inputs and seed registry, atomic completion/failure bundles, and deterministic resource-control fixtures. A real M5 attempt stopped during Corrected MM 15-year training at the accounting hard gate (`cash_reconciliation`, path 1, month 15, observed 0.9895085096 mCHF versus tolerance 0.0913200006 mCHF). It emitted a failed diagnostic bundle rather than a misleading successful matrix; therefore no completed paired pilot is available yet for Ticket 29's checkpoint evaluation.
