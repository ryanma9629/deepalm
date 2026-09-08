# Deep ALM：论文一致性修复增量规格

Type: spec
Status: ready-for-agent
Revision: 1 — 2026-09-08
Parent specification revision: 3
Scope: specification only; implementation and reruns follow dependency-ordered tickets

## Problem Statement

用户希望在 MacBook 上用小规模计算跑通与《Deep treasury management for banks》一致的技术流程，再在银行 GPU 环境进行正式训练。已有程序完成了训练、评估和报告，但经原论文、用户指定的 errata 和现有规格交叉核查，仍存在七项明确实现偏差，以及评估统计遗漏和图表映射错误。执行成功、会计恒等式成立、测试通过，都不能单独证明公式语义一致。

本增量以修正后的论文为金融基准，恢复此前已要求但未正确实现的行为。它不把资源缩减解释为更换金融规则的授权，不追求作者私有数据或训练结果的数值重现。旧 Ticket 28–30 的成功记录保留为旧实现的执行证据；它们不能作为这批修正已经验收的证据。

## Solution

在既有 PyTorch 流水线上修复状态转移、策略输入与参数化、评估和报告。每项修复必须具有原文依据、独立手算验收、公开接口观察结果和产物兼容性说明。先完成所有相关确定性与梯度测试，再统一进行一次受限配对训练、锁定评估与报告重建。

本文件是总规格 Revision 3 的专项增量；下列修复要求优先于旧实现、旧测试以及旧票的完成勾选。七项错误在 Paper 和 Corrected 中共同修复，两种口径仍仅在 PCA 载荷和贷款利率年化表达式上分支。统一六个月新贷款定价继续作为单独披露的简化；正式 paper 宽度训练能力缺口另列，不自动纳入本轮实现。

## User Stories

1. As a researcher, I want every correction tied to a paper section or formula and the applicable erratum, so that a code change has an independently inspectable basis.
2. As a reviewer, I want genuine implementation errors separated from approved scope choices, so that compact training never excuses a different financial model.
3. As a model developer, I want the five MM ratios to use pre-action economic values and the initial asset scale, so that the network receives the paper's stated information.
4. As a model developer, I want constraint features shifted by their lower bounds except IRS, so that policy preprocessing matches the paper while penalties retain their corrected direction.
5. As a treasury researcher, I want BM^D to learn adjustments above maturing positions at each date, so that its parameterization matches the other benchmarks and Equation 43.
6. As a treasury researcher, I want a deposit to retain its original reference-term class through rollover, so that its maturity profile is not inadvertently redistributed.
7. As a treasury researcher, I want deposit growth and capitalized interest attributed to the correct classes, so that the liability evolution follows the disclosed scheme.
8. As a bank integrator, I want reference-term schedules carried in validated snapshots, so that aggregate ladders do not hide missing cohort information.
9. As a researcher, I want the first two deposit-rate windows to consume dated pre-simulation history, so that interest initialization is not replaced by repeated initial yields.
10. As a reviewer, I want historical sampling dates, yields and source identities recorded, so that no future data enters the initial moving average.
11. As a reviewer, I want a zero-rate rollover example to verify both cash conservation and maturity-class preservation, so that one invariant cannot conceal failure of the other.
12. As a researcher, I want dividend yield divided by the number of nonterminal dividend years, so that five- and fifteen-year reports implement Equation 52b.
13. As a researcher, I want raw violating constraint values distinguished from transformed penalties, so that Table 4 statistics retain their units and interpretation.
14. As a researcher, I want mean violation counts conditional on violating scenarios, so that counts have the same population as the paper.
15. As a researcher, I want equity skewness and excess kurtosis computed using the paper's population moments, so that Equation 51 is fully represented.
16. As a reviewer, I want undefined metrics and missing evidence explicitly labeled, so that absent observations are not reported as zero.
17. As a reviewer, I want each table and figure mapped by its printed caption and page, so that a coverage label actually identifies the intended output.
18. As a developer, I want regressions tested through existing simulator, policy, evaluation and runner interfaces, so that tests survive internal refactoring.
19. As a reviewer, I want old financial and policy semantics rejected even when tensor shapes match, so that obsolete checkpoints cannot silently reenter the repaired workflow.
20. As a maintainer, I want report-only corrections distinguished from training-rule corrections, so that valid immutable inputs can be reused without relabeling old learned weights.
21. As a local developer, I want one bounded post-fix training matrix with the existing 600-second budget, so that fixing correctness does not trigger prolonged local tuning.
22. As a researcher, I want failed or incomplete branches retained with their evidence references, so that reports do not select only successful outcomes.
23. As a bank integrator, I want the paper-width training guard listed as an outstanding capability gap, so that the handoff does not promise configuration-only scaling that is unavailable.
24. As a reviewer, I want source-based consistency acceptance kept distinct from economic convergence, so that a successful small run has a precise meaning.

## Implementation Decisions

### Baseline and unchanged scope

- Source precedence is: explicit user decisions and the accepted convention ADR; user-specified errata where it changes a formula; original paper where not superseded; documented project choices for undisclosed details. A code path or passing test is never its own source of financial truth.
- Correction IDs retain the review's numbering: C-1, C-5, C-6, C-3, C-2, C-7 and C-8. R-1 and R-2 identify report omissions and caption mapping. Number 4 remains the approved compact resource choice; it is not a missing correction.
- Preserve continuous-compounding units, the E-03 nonnegative cash cost, E-04 quantity-weighted transaction PV, E-05 penalty direction, E-10 raw-tail selection followed by centering, fixed-rate loan cohorts, and the previously repaired deposit cash ledger.
- Preserve no swaps, 5/15-year horizons, 60/180 decisions and 61/181 states, no initial or terminal dividend, and no terminal action. E-07, N-01, N-02 and C-03 remain inapplicable to the excluded swap extension, not verified implementations.
- The bond cash settlement continues to follow E-04. The inconsistency between the paper's par/spread narrative and that formula is a separate baseline issue and is not changed in this repair.
- No new top-level service is introduced. Reuse ReferenceBankProvider, MarketScenarioModel, TreasuryPolicy/MMPolicy observation construction, ALMSimulator.rollout, LockedEvaluator and ReproductionRunner. Internal contract extensions must propagate through training, selection, evaluation, truncation, sensitivity and imported-bank paths.

### C-1 — Economic-value MM ratios

- Basis: Section 3.2.1.3, Equations 45a–e, PDF page 19; existing relative-balance-sheet requirement. No erratum changes these features.
- At each decision, use the current pre-restructuring values: A = cash + investment PV + mortgage PV + enterprise-loan PV; E = A − deposit PV − funding PV. Calculate PV with the current curve and the same valuation time as the simulator.
- The five features, in order, are A/A0, E/A, C/A, investment PV/A, funding PV/A. A0 is the actual initial economic asset value of the supplied bank, not a hardcoded canonical amount and not the current A.
- Retain nominal 180-month ladders divided by 100 for the four learned encoders. Economic ratios and nominal portfolio encodings serve different roles; replacing either with the other is prohibited. Observation dimension remains 145.
- Extend policy state with the initial-asset identity and sufficient pre-action valuation information. Whether values are passed from the simulator or calculated through a shared valuation function is internal; tests must prove both use the same discounts/time. Finite negative equity is retained. Undefined ratios at A=0 or invalid A0 fail with path/time diagnostics; do not silently epsilon-clamp the economic denominator or clip the balance sheet.
- Preserve the existing stop-gradient contract: current observation-state history is detached, encoder and policy parameters still learn, and action-dependent future transitions remain differentiable. MM(15y|5y) retains initial A0 and the original 15-year time normalization.
- Hand oracle: initial A0=800; current C=100, investment PV=200, loan PV=700, deposit PV=500, funding PV=300. Then A=1000, E=200 and the five inputs equal (1.25, 0.20, 0.10, 0.20, 0.30). Construct discounted ladders with nominal totals different from these PVs; a nominal-sum implementation must fail. Changing only A0 to 1000 makes only the first ratio 1.00.

### C-5 — Deposit reference-term preservation

- Basis: Section 2.2.3, PDF pages 8–9, the assigned-reference-term rollover rule and the interest-reinvestment rule following Equation 12. The aggregate cash-conservation fix does not satisfy this requirement by itself.
- Retain separate reference classes of 1, 2, 12 and 120 months within each of non-maturity and term deposits. Each class owns its remaining nominal monthly schedule. A practical bounded state is four 180-month schedules per deposit product per path; no individual customer simulation is required.
- On a roll, shift each class's remaining schedule. Its matured amount is split evenly across months 1 through its own reference term and credited back to that same class. The reference term is the original behavioral category, not the remaining tenor of an arbitrary aggregate bucket.
- Only exogenous new deposit growth is allocated across reference classes using the declared initial weights. Calculate growth on the same pre-roll product balance as the existing rule. Interest is calculated on each class's pre-roll balance at that product's rate and capitalized within its original class, spread using the same reference-term mechanism. Preserve the existing rate sign/cap/floor convention; no new nonnegative-interest clamp is introduced.
- Sum the class schedules to produce the existing aggregate deposit ladders used for valuation, encoders and constraints. Positive cash inflow from renewed principal offsets its maturity payment once; growth enters cash once; capitalized interest changes liabilities without a net cash payment. Operating costs and E-03 remain separate.
- Extend the snapshot schema to require reference-term schedules, terms, weights and provenance, and validate that their sum reproduces each aggregate ladder. Canonical and sensitivity construction preserve their declared PV targets and units while adding provenance for the class decomposition. Export/import and noncanonical synthetic fixtures exercise the same contract.
- An old aggregate-only deposit snapshot cannot silently infer class ownership from global weights. Rebuild canonical inputs from their declared generator; imported banks must supply class schedules or an explicit, separately approved mapping. Aggregate-only v2 inputs are incompatible with the new transition; legacy loan-cohort data must be preserved during any rebuild.
- Hand oracles at zero interest/growth: (a) a 1-unit, one-month-reference deposit maturing now returns entirely to month 1, with zero in months 2–180; (b) a 2-unit, two-month-reference deposit returns 1 to month 1 and 1 to month 2; (c) a mixed book conserves total nominal balance and each class balance through repeated rolls. Separately, 100 units of new NMD growth assigns class totals 40, 30, 25 and 5. Interest of 2 on a two-month class adds 1 to each of its first two buckets, not to other classes.

### C-6 — Dated history for deposit initialization

- Basis: Equation 11c, PDF page 8. At monthly state k, the deposit reference uses the previous three six-month yields. Month 1 needs (Y0, Y−1, Y−2); month 2 needs (Y1, Y0, Y−1); month 3 needs (Y2, Y1, Y0).
- Connect Deposit Reference History to the actual simulator input. The initial historical prefix must be explicit, immutable and shared across paths; after initialization the rolling window uses simulated path values. Existing historical loading without consumption is insufficient.
- Monthly sampling is a project convention, not a paper fact: derive target dates by subtracting one and two calendar months from the snapshot valuation date, clipping the day to month-end if necessary. For each target choose the last complete observed curve on or before the target; record target date, selected observation date, six-month yield and source identity. Y0 remains the scenario's exact initial curve. Do not use a future observation, interpolate, or fabricate missing history by repeating Y0.
- Apply the same rule to canonical, imported and sensitivity banks. Missing historical coverage fails before training. Deterministic fixtures may provide an explicit dated history; every such value is declared as fixture input. New initial-history identity must participate in snapshot/scenario/run compatibility without changing the HJM calibration sample or random streams.
- Hand oracle: Y0=3%, Y−1=2%, Y−2=1% gives first-month NMD reference 2% and annual deposit rate 1.2%, below its cap. On 1000 units the interest is 1000 × (exp(0.012/12)−1), approximately 1.00050017. The old repeated-Y0 calculation gives approximately 1.50112556. If Y1=4%, the second-month reference is 3%; no Y−2 term remains. A holiday fixture verifies last-observation-on-or-before selection and a missing-history fixture verifies failure.

### C-3 — BM^D maturity-relative scale

- Basis: Equation 43 and its financing analogue, PDF page 18; it applies to all three benchmarks. E-09/N-03 affect the timeline and parameter count, not the scale formula.
- For investments and financing independently, BM^D uses the positive part of the pre-action first ladder bucket plus the date-specific learned adjustment, multiplied by that date's softmax maturity allocation. Preserve the same meaning of first-bucket maturity amount used by BM^E/BM^C; do not introduce a separate principal/coupon convention here.
- Keep 60/180 rows of parameters and the existing default numeric adjustment initialization. Reinterpretation of saved absolute scales as adjustments is forbidden. Tests may set adjustments through the public model state serialization contract.
- Scenario independence means no direct dependence on the market, deposits, loans or constraints for a fixed date and the same investment/funding schedules. It does not mean ignoring different maturity amounts supplied by different banks/states. A common initial bond book evolving under the standalone benchmark still produces common actions across rate scenarios.
- In MM, evaluate the frozen BM^D parameterization on the same detached current bond schedules required by the established baseline contract. Never replace it with a saved absolute action sequence from an earlier run.
- Hand oracle: with first-bucket investment maturity 100, funding maturity 40, adjustments +10 and −5, summed actions are 110 and 35. Zero adjustments give 100 and 40. Investment adjustment −120 gives a zero investment action. Changing unrelated market features leaves these benchmark outputs unchanged; changing the maturity bucket changes the corresponding scale.

### C-2 — Constraint feature centering

- Basis: Section 3.2.1.4, PDF page 19. Use the previous post-restructuring raw constraint values, with the existing initial-state and annual-mask conventions. E-05 governs the penalty path separately.
- In LCR, NSFR, CMR, Equity/RWA, IRS, EYR order, subtract (1.05, 1.05, 1.00, 0.17, 0, 0) for the policy observation. IRS remains its raw value; EYR has lower bound zero. Bounds must come from the authoritative constraint definition, not an independent divergent constant set.
- Keep both signs of x−β. Do not use ReLU, absolute value, shortfall or nonlinear penalty as this feature. Do not mutate the raw trajectory values needed by loss computation and reporting.
- Hand oracle: raw (1.10, 1.00, 0.94, 0.20, 0.09, −0.01) becomes (0.05, −0.05, −0.06, 0.03, 0.09, −0.01). E-05's penalties for the same input are unchanged by feature construction. Fixed shifts being absorbable by a bias do not justify silently changing the paper's preprocessing.

### C-7, C-8 and R-1 — Evaluation metrics

- C-7 basis: Equation 52b, PDF page 23, and nonterminal dividend events. For each path divide total dividends by E0 × (T−1), then average over paths; use T in years. MM(15y|5y) uses evaluation T=5, not its trained horizon 15. Validate 4 and 14 eligible dividend years for the two full horizons, including zero-payout years. If a generic input has T≤1, the dividend-yield metric is unavailable with a reason. Geometric return retains its own 1/T exponent.
- C-7 hand oracle: E0=1000 and four annual payouts of 10 in a five-year run yield 0.01 (1%); fourteen payouts of 10 in a fifteen-year run also yield 1%. Reporting as 0.8% or 14/15% fails.
- C-8 basis: Table 4, PDF page 26. Use the existing applicable-time mask and violation predicate. Report the mean of raw constraint values at violating observations; retain nonlinear penalty statistics under separately named fields. Report total violating observations and mean count conditional on paths with at least one violation as different quantities. Their names, denominators, units and EYR time mask must be explicit. No-violation cases have zero counts/share and unavailable conditional means, not an invented zero constraint value.
- Preserve the most-adverse raw value and label whether it is over all eligible observations or violating observations. Report the actual last applicable constraint state with its state/month index; do not invent a terminal observation. Keep capital ratios dimensionless internally and any percent conversion at the presentation edge.
- C-8 hand oracle: CMR=0.94 alone has raw violating mean 0.94, whereas its transformed penalty is 0.1236. With per-path counts (2,0,1), total count=3, ever-violating share=2/3 and conditional mean count=1.5. With no violating paths, that conditional mean is unavailable.
- R-1 basis: Equations 51a–e, PDF page 22, Table 3. On finite equity ratios compute population central moments m2, m3 and m4 with divisor n; standard deviation is sqrt(m2), skewness is m3/m2^(3/2), and excess kurtosis is m4/m2²−3. Do not use an unrelated bias-corrected library estimator or Pearson kurtosis without subtracting 3.
- R-1 hand oracle: ratios (1,1,1,5) have mean 2, m2=3, m3=6, m4=21, skewness 2/sqrt(3) and excess kurtosis −2/3. A constant sample has zero standard deviation but undefined skewness/kurtosis, each emitted as unavailable with a reason. Finite zero/negative equity ratios remain included. Non-finite raw inputs produce explicit invalid evidence without silently filtering paths.
- Apply these definitions to LockedEvaluator, frozen-policy sensitivities, horizon/truncation evaluation, paired comparisons where reported, and report serialization. Preserve E-10's centered tail definitions. Reports must use strict JSON-safe availability records; undefined values may not leak as NaN/Infinity numeric literals.

### R-2 — Correct printed table and figure coverage

- Basis: printed caption and visual subject in the exact source PDF, rather than inconsistent cross-references in its prose. Coverage records include printed identifier, page, subject, supported policy scope, source-PDF hash and matching output evidence.
- Required inventory is fixed below. Full figure generation remains deferred where no corresponding local output exists. A PCA statistic cannot mark an architecture diagram or equity histogram generated. A metric summary may be labeled related evidence but not completion of a missing figure.

| Printed item | PDF page | Subject |
| --- | --- | --- |
| Table 1 | 7 | Economic balance sheet |
| Table 2 | 16 | Hyperparameters |
| Table 3 | 21 | Main results: losses, equity distribution, returns and dividends |
| Table 4 | 26 | Constraint statistics |
| Table 5 | 34 | Category statistics; disclose covered policy columns |
| Figure 3 | 17 | Simulated one-month yields under HJM-PCA |
| Figure 4 | 17 | Terminal five-year yield curves, HJM-PCA versus Hull-White |
| Figure 5 | 20 | Decision-network architecture |
| Figure 6 | 24 | Equity-ratio histograms |
| Figure 7 | 24 | Constant benchmark strategies |
| Figure 8 | 25 | Investment and financing volume |
| Figure 9 | 26 | LCR and CMR |
| Figure 10 | 27 | Equity/RWA |
| Figure 11 | 28 | Interest-rate sensitivity and portfolio durations |
| Figure 12 | 29 | Five-year yield-curve scenarios |
| Figure 13 | 30 | Five-year decisions |
| Figure 14 | 30 | Five-year sensitivity gaps |
| Figure 15 | 31 | Fifteen-year yield-curve scenarios |
| Figure 16 | 32 | Fifteen-year decisions |
| Figure 17 | 32 | Fifteen-year sensitivity gaps |

- Table 3 swap columns and Figures 18–19 remain excluded. Table 5's mixed policy scope must be disclosed; missing swap results are not inferred from no-swap category data. Existing mean-penalty/risk outputs must not be mapped to unrelated printed subjects.
- Report construction consumes evidence and never initiates training. A partial or failed branch must retain source identity, stage, convention, policy/horizon, diagnostic path/month/value where available, and an evidence reference. Missing paired members yield not-applicable intervals with reasons. A report of incomplete evidence is permissible, but cannot upgrade the experiment to completed or evidence to available. If the upstream evaluator cannot yet emit a partial report, preserve its failure bundle and expose the missing branch explicitly rather than invent metrics. This also closes the known incomplete-branch limitation in the existing paired reporter.

### Schema, semantics and old-artifact invalidation

- Define a new financial semantics version for reference-term-preserving deposits and history initialization; a new policy/observation semantics version for corrected MM inputs and BM^D scale; a new metric/report semantics version for dividends, conditional constraint statistics, moments and coverage; and a new snapshot schema version for deposit classes/history linkage. Identifiers must differ from the legacy versions and be recorded centrally. Exact spelling may be chosen in implementation, but absence/mismatch must be rejected explicitly.
- Schema changes are not cosmetic: a loader must validate class-to-aggregate sums, curve/history as-of identity, convention, horizon, architecture, seeds, calibration and baseline/checkpoint hashes. Shape compatibility alone does not establish semantic compatibility. Every consumer, including CLI reuse/resume, baseline materialization, truncation, sensitivity and report generation, must honor these versions.

| Artifact | Reuse after this repair | Required handling |
| --- | --- | --- |
| Original PDF, errata, SNB observations | Yes | Preserve and hash; do not rewrite source evidence |
| Unchanged NSS/HJM calibration and market innovations | Conditional | Reuse only when source, calibration, convention, date/grid and seed/path identities match; add separately validated history linkage |
| Training-only curve PCA | Conditional | Same exact registered curve sample/algorithm/identity; reattach only through a validated interface, otherwise refit cheaply |
| Aggregate-only deposit snapshots including legacy schema v2 | No for new simulation | Rebuild canonical class schedules or obtain explicit imported-bank mapping; assign a new content hash |
| Old BM^E/BM^C weights and recovery | No for new financial results | Deposit/history transition changed the training problem; retrain rather than resume or warm-start |
| Old BM^D weights and frozen baseline references | No | Both training environment and scale parameter meaning change; train anew and freeze new references |
| Old MM weights/recovery, including both convention pilot checkpoints | No | Observation, environment and baseline semantics change; reject even with 145 inputs and the same widths |
| Old evaluation, sensitivity, truncation and bootstrap results | No as repaired evidence | Retain historical provenance; regenerate after compatible training |
| Old coverage and reports | No as repaired evidence | Regenerate from compatible evaluation; keep old artifacts separately identified |

- Metric-only changes, considered independently, do not invalidate learned weights. They can be recomputed from complete, identity-checked raw trajectories without an optimizer step. Aggregate summaries alone are insufficient to reconstruct missing moments or conditional statistics. Because this increment also changes financial and policy semantics, its final accepted matrix requires new training regardless of any historical metric recalculation.
- Never repair compatibility by editing metadata on old checkpoints. Preserve old run directories and their original successful/failed status as historical facts, and record supersession in new evidence instead of deleting them. No automatic migration can recover lost reference-term ownership or turn absolute BM^D scales into validated learned adjustments.

### Dependency order and completion gates

1. Establish the source/hand-oracle register and versioned input/semantic contracts. This is prerequisite to publishing any new compatible checkpoint.
2. Implement C-5/C-6 through snapshot/history and simulator contracts. Implement C-1/C-2 through the observation seam and C-3 through TreasuryPolicy. These code/test slices can proceed independently where contracts permit, but no corrected full matrix starts until all pass. C-1 has high priority alongside C-5 even though integration waits for shared contracts.
3. Implement C-7/C-8/R-1 through evaluation behavior and R-2 through coverage/report behavior. Report evidence depends on corrected metrics; deterministic fixtures do not depend on model training.
4. Integrate every training, selection, import, sensitivity, truncation, restore and reporting path. Audit stale-artifact rejection and complete full-horizon CPU/MPS smoke/gradient checks. Review the diff against both the repair spec and the cited primary formulas; unresolved material findings keep the repair open.
5. Run one fresh bounded paired BM^D→MM matrix per convention/horizon, then locked evaluation including MM(15y|5y), followed by the report. Bootstrap only compatible common-path members. Publish a requirement-to-evidence ledger for every correction. Do not mark the repair resolved solely because the CLI returned zero.
- The next to-tickets phase must retain existing ticket history and add blocker-linked repair work. This specification does not renumber tickets, claim repairs implemented, or launch training.
- This repair's eight-job pilot is not a replacement for the broader four-policy local_flow acceptance matrix. BM^E/BM^C receive regression and integration coverage, but their old trained results remain historical unless separately retrained within an explicitly approved plan. A report must show this limitation; completing the paired repair pilot alone cannot newly certify the entire development-validated matrix.

### Resource budget and migration boundary

- Preserve the approved M5 MacBook Pro with 32 GiB memory as the local target. The final rerun uses the existing explicit paired pilot: one MPS float32 process, compact widths 64/64/32/32, both full horizons, one registered primary seed, 2 epochs, 16 training paths and 16 selection paths, batch size 8, and 64 locked evaluation paths. Eight primary jobs imply 32 optimizer updates; truncation adds zero updates.
- The pilot invocation keeps its 600-second wall-clock guard, 12 GiB RSS and separate 12 GiB MPS guards, and the first-epoch 15-year MM throughput projection threshold of 420 seconds. These counters overlap on unified memory and must not be added. Record elapsed overshoot of a cooperative check honestly. Do not increase budgets or weaken accounting tolerances to pass the repair.
- Evaluation/reporting remain separate bounded invocations, as in Tickets 29/30, and record their time separately. The 600-second training-pilot guard is not a promise that development tests plus training plus reporting finish within ten minutes. Existing generic local_flow defaults remain recorded but do not override this repair's explicit rerun profile.
- Deterministic CPU float64 cases and small MPS integration fixtures precede training; avoid retraining after each individual patch. If corrected states cost more than the guard allows, stop with incomplete evidence and measured causes. Any changed execution profile is a separate decision; full financial horizons, state rules and gradient paths are retained.
- The final repair may be recorded as semantically verified by tests while its bounded pilot is resource-incomplete; these statuses must be separate and the planned rerun must remain open. Poor finite model performance is not a correctness failure or a reason to tune against the locked test set.

### Separately disclosed model choices and capability gaps

| Item | Classification | This increment's treatment | Later acceptance required |
| --- | --- | --- | --- |
| One six-month yield for new loans of all maturities | Documented model simplification relative to Equation 9 | Preserve and disclose in manifests/reports; do not label full maturity-specific pricing complete | Separately decide and implement origination-tenor coupons, then validate cohort pricing and resource impact |
| MM fit currently rejects paper widths | Formal-training capability gap | Document actual support: compact selection training plus bounded paper-width update checks; no promise of configuration-only formal training | Exercise the complete fit/selection/save/resume/evaluate path with paper width on a bounded fixture before substantive training; target CUDA evidence remains separate |
| Compact network, small samples, synthetic Reference Bank, no swaps | Approved resource/data/scope choices | Preserve | Bank data mapping, substantive training and validation remain bank-side work |
| Residual-block internal details not fully disclosed | Source uncertainty | Keep current topology with an explicit implementation-choice label | Further author evidence would be needed for a layer-identical claim |

## Testing Decisions

- Reuse the test seams already agreed for this project: ALMSimulator.rollout for economic trajectories, TreasuryPolicy/MMPolicy.build_observation for actions/features, LockedEvaluator for metrics, and ReproductionRunner/CLI for immutable artifacts and compatibility. ReferenceBankProvider and MarketScenarioModel supply validated fixtures. Do not create a new service or test private helpers merely to simplify assertions.
- Tests compare publicly observable values with the hand oracles above. Each repaired behavior first has a regression that demonstrably fails the pre-repair implementation, then passes the repair. Existing shape, finite-value and ledger tests remain useful but are not substitutes for the formula examples.
- Source-derived expected values must not call the implementation helper being tested. Use small nonflat discount fixtures, mixed deposit classes and dates with differing historical yields, so that previously wrong implementations cannot accidentally pass.
- CPU float64 formula fixtures use absolute/relative tolerances no weaker than the existing applicable 1e-10 convention, with rounded numbers above treated as explanations rather than exact constants. Gradient fixtures use smooth interior cases and the retained central-difference bound of 1e-6 + 1e-4 times the reference gradient. Boundary ReLU tests assert outputs without requiring a unique derivative at zero.
- Retain current dtype-aware ledger checks: float64 max(1e-6 mCHF, 1e-10 × absolute assets) and float32 max(1e-3 mCHF, 1e-5 × absolute assets). Record dtype and tolerances; conservation must also be tested against independent cash/class oracles. No tolerance relaxation is part of this increment.
- Simulator tests cover zero-rate conservation, class-specific interest/growth, old-maturity retention, historical-rate splice, annual boundaries and complete 60/180-step smoke runs. Contract tests cover canonical and imported snapshots, sensitivities, missing history and inconsistent class aggregates. Negative deposit interest must not cause silent clipping or artificial cash creation.
- Policy tests cover exact ratio order/values, economic versus nominal valuation, initial scale, constraint centering, maturity-relative BM^D sizes, matching standalone benchmark path behavior, frozen-baseline use, zero-update 60-step truncation and preservation of the original time feature.
- Evaluation tests cover both horizons, conditional counts with zero and multiple violating paths, raw versus penalty units, skew/kurt population moments, degenerate samples, zero/negative equity ratios and strict JSON availability semantics. Test through the highest public evaluation seam; do not assert private helper identities.
- Reporter tests cover every caption mapping above, honest generated/related/deferred status, missing members, partial/non-finite branch evidence, incompatible identities, and no fabricated pair/interval/value. A simulated write failure must leave no completed final bundle. CLI tests verify adaptation to the same runner contract without triggering training.
- Before/after load tests prove that matching shape with stale financial/policy/metric semantics is insufficient. Compatible new snapshots/checkpoints round-trip; old references, resume states and report reuse fail with a specific reason. Changing only runtime/output does not bypass semantic validation.
- Implement each ticket using a red-green slice, applicable static/type checks and its targeted test file. After integration, run the full suite once and perform Standards and Spec review; the Spec axis must include original formulas/errata and independent hand expectations rather than only the ticket wording. Fix material findings before claiming completion.
- Real rerun acceptance records all eight new checkpoint identities, baseline relationships, training/selection/test separation, actual optimizer updates, resources, available locked-path comparisons and report disclosures. MPS/CUDA availability is reported truthfully; absent target CUDA cannot be marked validated. No final-test outcome may select hyperparameters, seeds or a favorable subset of scenarios.

## Out of Scope

- Implementing maturity-specific new-loan pricing or full paper-width selection training in this repair. Both are explicitly open items above, not automatically excused deviations or completed bank capabilities.
- Swaps, real-bank source-system connections, multi-GPU implementation, bank economic approval, regulator certification, or reproduction of private author weights/data.
- Large local training, more epochs/seeds to match the paper, a full paper figure suite, 10,000-resample research bootstrap, or Ticket 24 economic acceptance execution.
- Changing E-04 transaction accounting, existing loan growth/impairment rules, loss family, stop-gradient semantics, PCA discretization, or accepted cubic-fit diagnostics beyond what is necessary for the listed corrections.
- Rewriting old reviewer reports, silently updating old artifact metadata, deleting historical runs, or treating a specification publication as an implementation completion.

## Further Notes

- Parent specification: [Deep ALM total specification](../deep-alm-methodological-reproduction/spec.md). This repair is linked from its Revision 3; unaffected requirements and the accepted convention ADR remain in force.
- Confirmed assessment: [errata-based assessment](../deep-alm-methodological-reproduction/research/errata-based-code-assessment-2026-09-08.md). Supplementary evidence: [paper/code audit](../deep-alm-methodological-reproduction/research/paper-code-audit-2026-09-08.md) and [financial audit](../deep-alm-methodological-reproduction/research/financial-formula-audit.md). Their historical findings are preserved.
- Primary sources: [source paper](../../docs/Deep%20treasury%20management%20for%20banks.pdf), [user-specified errata](../../docs/errata.pdf) and its [text source](../../docs/errata.tex). They are available locally; this spec does not assert that the local errata is an author-published corrigendum. Implementation evidence records their hashes.
- Published after the user's to-spec request; synthesis reuses existing accepted seams and introduces no new top-level test boundary. The initial-history calendar sampling, grouped deposit representation and precise unavailable-metric behavior are explicit engineering decisions in this spec, distinguishable from quoted paper rules.
- The next phase is to-tickets, using this file as the self-contained repair source. Only documentation changes are authorized by this phase; no source code, tests, checkpoints, training or report artifacts are modified here.

## Comments

2026-09-08 — 根据已确认的审查结论发布增量规格。七项明确错误、两类报告遗漏、依赖顺序、独立手算验收、版本失效边界与 M5 资源预算已写明。修复工作尚未实施，旧 Ticket 28–30 保留历史完成记录。
