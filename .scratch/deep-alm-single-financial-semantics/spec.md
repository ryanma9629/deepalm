# Deep ALM：唯一已修正金融语义增量规格

Type: spec
Status: ready-for-agent
Date: 2026-09-09
Scope: retire executable alternate financial conventions; preserve a single auditable corrected model and its bounded local validation flow

## Problem Statement

当前项目同时携带论文逐字公式分支与已修正金融语义。前者的贷款利息未按月年化，已在最小可复现实验中证明会把终值权益推至目标之上、令非对称目标和约束罚项同时为零，并使优化器没有梯度。继续把这套已知错误的公式作为运行时配置、校准缓存、训练矩阵、评估与报告的一部分，会使后续银行落地存在误选语义、混用 artifact 和误读结果的风险。

用户要在 MacBook 上完成完整技术流程的验证，并在银行 GPU 环境采用同一套金融模型进行正式训练。论文仍是方法来源，原始 errata 仍是审计依据，但项目不再需要、也不应提供可执行的 Paper convention。项目需要一份唯一、可维护的实现版勘误，清楚说明当前实现相对论文及其 errata 的修正、已知简化和能力缺口。

## Solution

将运行时代码收敛为唯一的已修正金融语义。删除所有 `paper/corrected` 金融口径选择、公式分支和 paired-convention 工作流；固定协方差一致的 PCA 缩放与按月贷款计息，以及已验收的现金守恒、存款、策略输入和风险计量修复。新旧 artifact 通过不可变的金融语义版本严格隔离。

以 Corrected Local Validation Pilot 替换双口径配对试跑：BM^D 与 MM 在 5 年和 15 年四项任务中完成受限训练，MM 各自依赖同期限、同语义、当次新生成的冻结 BM^D baseline。`corrected-pilot`、`corrected-evaluate` 和 `corrected-report` 是唯一专用入口；完成的本机证据必须在十分钟内产生有限、兼容、可审计的完整链路结果。

原论文、原始 errata、旧 paired artifact 与诊断记录保留在历史位置，只作为不可执行的审计证据。项目维护一份 Implementation Errata，将论文来源映射到唯一运行时语义，并分别披露修正、范围简化和银行 GPU 阶段的能力缺口。

## User Stories

1. As a treasury model developer, I want one executable financial semantics, so that every local and bank-side run uses the same economic rules.
2. As a reviewer, I want the erroneous unannualized loan-interest rule removed from executable code, so that a known flat-loss failure cannot be selected by configuration.
3. As a reviewer, I want covariance-consistent PCA loading scaling fixed in the market model, so that scenarios have one documented interpretation.
4. As a model developer, I want loan interest always converted to a monthly rate, so that monthly cash flows use consistent units.
5. As a user of run configuration, I want no convention block or per-formula overrides, so that configuration cannot silently change a financial definition.
6. As a maintainer, I want market calibration to expose one set of loadings and coefficients, so that downstream scenario generation cannot choose a retired branch.
7. As a maintainer, I want loan transition APIs to have one rate interpretation, so that simulation callers cannot request obsolete behavior.
8. As a researcher, I want corrected deposit rollover, dated initial rate history and cash conservation retained, so that simplification of convention handling does not undo accepted financial repairs.
9. As a researcher, I want corrected MM observations, BM^D maturity-relative actions and risk metrics retained, so that the unified model preserves the audited methodology.
10. As an artifact consumer, I want every newly written checkpoint, frozen baseline, recovery state, evaluation and report marked `corrected-financial-semantics-v1`, so that semantic compatibility is explicit.
11. As an artifact consumer, I want a new stage to reject a missing, older or different financial-semantics version, so that historic learned weights cannot reenter the current chain.
12. As a reviewer, I want old Paper, paired and pre-version artifacts retained unchanged, so that the diagnostic history remains independently auditable.
13. As a maintainer, I want the runtime never to deserialize historic paired evidence as a valid input, so that historical preservation does not become compatibility support.
14. As a local developer, I want a corrected-only pilot command, so that I can exercise the intended four-task workflow without running a meaningless comparison.
15. As a local developer, I want the pilot to train BM^D and MM for both 5-year and 15-year horizons, so that the full supported horizon matrix is exercised.
16. As a local developer, I want each MM task to consume a newly trained, frozen BM^D baseline of the same horizon and semantic version, so that the policy dependency is genuine.
17. As a local developer, I want 16 actual optimizer updates across the four pilot jobs, so that a nominally completed run cannot consist of zero-gradient steps.
18. As a local developer, I want the complete pilot to stay within the agreed ten-minute MacBook budget, so that it remains a workflow validation rather than an accidental local training campaign.
19. As a reviewer, I want finite loss, gradients, model parameters and required risk evidence for every pilot member, so that completion reflects executable numerical behavior.
20. As a reviewer, I want evaluation and reporting to require a completed four-member corrected-only source bundle, so that a partial or historic artifact cannot be promoted to validation evidence.
21. As a CLI user, I want the retired paired commands absent, so that the public interface does not advertise unsupported financial semantics.
22. As a CLI user, I want dedicated corrected pilot, evaluation and report commands, so that local validation evidence has an unambiguous lifecycle.
23. As a reviewer, I want the historical zero-gradient Paper reproduction removed from executable tests, so that the suite specifies current behavior rather than reproducing an intentionally retired defect.
24. As a reviewer, I want direct regression tests for the unique PCA and monthly-loan rules, so that these corrections remain protected after internal refactoring.
25. As a documentation reader, I want one Implementation Errata that maps source evidence to current behavior, so that I can distinguish a formula correction from a deliberate scope choice.
26. As a bank integrator, I want the Implementation Errata to disclose the no-swap scope, shared six-month new-loan pricing, compact local network and paper-width training gap, so that handoff planning is not mistaken for a claim of full production readiness.
27. As a documentation reader, I want the source paper, source PDF hash, paper-width network checks and paper coverage inventory retained, so that methodology provenance remains available without reviving an alternative formula branch.
28. As a future maintainer, I want the retirement decision recorded in the ADR and glossary, so that a later contributor does not reintroduce an obsolete Paper profile for comparison.

## Implementation Decisions

- ADR-0002 is authoritative for this change. The project has one executable Corrected Financial Semantics; alternate financial conventions are not a runtime feature.
- Configuration schema removes the convention section and all convention profile/override validation. The resolved run identity contains no user-selectable financial-formula profile.
- Market calibration, scenario generation, loan cash-flow transition, training, selection, evaluation, reporting, recovery and sensitivity paths expose only the corrected formula. APIs no longer accept a convention selector and stored calibration values are no longer indexed by a convention name.
- The fixed financial formulas include square-root-eigenvalue PCA loading scaling and monthly loan-interest annualization. Existing accepted repairs for deposit cash conservation and reference terms, dated deposit history, MM economic observations, BM^D maturity-relative action size, cash charge, loan growth, tail-risk calculation and evaluation/reporting metrics remain mandatory.
- The new immutable artifact field is `financial_semantics_version = corrected-financial-semantics-v1`. It is checked for exact equality whenever a checkpoint, frozen BM^D baseline, recovery state, evaluation source or report source is reused. No metadata-only migration or weight reinterpretation is allowed.
- All current-version artifacts omit the retired convention field. Historic artifact layouts remain files on disk only; no new command reads, upgrades or translates them.
- Replace paired pilot configuration and lifecycle with a corrected-only configuration and public commands for pilot, locked evaluation and report generation. The source bundle has exactly four members: BM^D and MM at each of 5 and 15 years.
- The corrected local validation pilot retains the agreed M5/MPS compact budget: one process, float32, compact 64/64/32/32 architecture, two epochs, 16 training paths and 16 selection paths per epoch, batch size eight, one registered master seed, 600-second cooperative wall-clock guard and the existing independent RSS/accelerator guardrails. The operational acceptance target is under ten minutes.
- Each corrected pilot member performs four actual parameter updates, making 16 in total. An update must have a finite nonzero optimization signal and result in an observable policy-parameter change. A job that exhausts the budget, emits non-finite numerical values or produces no optimization update fails the validation bundle rather than becoming a descriptive successful result.
- Each MM member uses a newly selected BM^D checkpoint from the same pilot horizon and semantic version, validates a content-addressed frozen-baseline reference, and keeps that baseline non-trainable during MM optimization.
- Corrected evaluation/reporting require the complete four-job source bundle and compatible current-version evidence. They do not calculate paired intervals, convention deltas, incomplete paired-member summaries or historical-failure promotion.
- Retire the paired commands and paired runner/reporting interfaces completely. Calling those names is handled by normal unknown-command parsing; there is no compatibility wrapper or special migration runtime API.
- Delete the paired pilot configuration, paired run-scale profile and paired tests. Remove the executable historical zero-gradient Paper diagnostic. Replace them with behavior-focused tests for the corrected financial formulas, corrected-only pilot lifecycle, strict artifact identity and complete bundle requirements.
- Consolidate the existing financial-correction ledger into one Implementation Errata. It records, for each item, the source statement or erratum, the adopted model behavior, classification as correction/simplification/capability gap, impact boundary and public verification evidence. The original errata is preserved as source material; the new document is the current implementation authority.
- Preserve source-paper provenance, source PDF validation, paper-width architecture/commissioning terminology and paper-output coverage inventory. These terms refer to a source or a network-scale/coverage concept, not a retired financial convention; documentation must say so explicitly.
- Keep known limitations explicit: swaps remain out of scope; new loans use one six-month yield across maturities; compact local execution is not convergence evidence; full paper-width training remains a bank-GPU capability gap until separately implemented and verified.

## Testing Decisions

- The primary acceptance seam is the public corrected-pilot CLI lifecycle through the Runner: it must build the four-member matrix, train BM^D and MM, resolve each frozen baseline, evaluate and report an identity-linked completed bundle. This highest existing seam is preferred over a new orchestration interface.
- Tests at the primary seam must observe public artifacts and statuses, not private helper calls or storage layout. They verify exactly four expected job identities, 16 actual updates, finite required evidence, same-horizon MM baseline linkage, current semantic-version identity and successful corrected evaluation/report completion.
- Configuration tests verify that a convention section, a retired profile or per-formula override is rejected as invalid schema input, while ordinary corrected configurations resolve without a selectable convention.
- Market and loan tests use compact deterministic oracle inputs to prove the single PCA scaling and monthly interest transformation. They must not retain an alternate formula merely to compare outputs.
- Simulator/policy/evaluation tests retain the accepted hand oracles for cash conservation, reference-term rollover, dated initial history, MM economic ratios, BM^D live-maturity adjustment, risk tails and report statistics. These are behavior tests at their established public seams.
- Compatibility tests exercise checkpoint, frozen baseline, recovery, evaluation and report reuse with current version, missing version and stale version artifacts. The observable contract is acceptance only for exact current identity; no test invokes a historic Paper runtime path.
- CLI tests prove the new corrected commands dispatch to the intended runner seams and the former paired command names are no longer public commands.
- Documentation tests or static checks ensure the Implementation Errata is the only maintained current implementation mapping, contains the required classifications, and does not present a Paper convention as runnable.
- Existing prior art includes configuration-resolution tests, scenario/loan deterministic tests, simulator rollout and policy tests, training/recovery artifact-identity tests, runner workflow tests and CLI dispatch tests. New tests should extend these highest available public seams rather than introduce mocks around formula helpers.

## Out of Scope

- Restoring a literal Paper convention, a dual-convention research comparison, or any compatibility layer for old paired artifacts.
- Recreating the paper's private bank dataset, matching its reported numeric results, or claiming methodological reproduction from the compact local pilot.
- Adding swaps or validating errata items that apply only to the excluded swap scope.
- Replacing the documented shared six-month new-loan pricing with a maturity-specific cohort-pricing model.
- Enabling or executing full paper-width formal training on the MacBook, tuning the compact pilot to economic convergence, or spending bank GPU resources.
- Deleting, rewriting, relocating or retroactively relabeling historic Paper/paired artifacts and source documents.

## Further Notes

- The old two-convention ADR is superseded by ADR-0002. The project glossary uses Corrected Financial Semantics, Implementation Errata and Corrected Local Validation Pilot as the canonical terms.
- The old paired pilot failed because the retired formula made its selected Paper batch flat, not because MPS or the optimizer was defective. That exact historical behavior belongs in the Implementation Errata and preserved artifact history, not in executable code or regression requirements.
- The existing financial-correction ledger is source material for the Implementation Errata migration. It must be updated rather than copied so the project has one current implementation mapping.
