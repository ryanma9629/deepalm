# Deep ALM methodological reproduction

Type: spec
Status: ready-for-agent

## Problem Statement

The user wants to reproduce the complete no-swap methodology in *Deep treasury management for banks* as a maintainable Python and PyTorch project on an Apple Silicon MacBook Pro. The current repository contains the source paper, the public Swiss National Bank Nelson-Siegel-Svensson parameter export, a uv environment, and a minimal Python package, but no market calibration, bank model, differentiable balance-sheet simulator, policy implementation, training pipeline, analysis, or acceptance machinery.

The paper's method is implementable, but its collaborating bank's initial cash-flow ladders, absolute scale, product allocations, loan spread, operating costs, trained parameters, and random paths are unavailable. Several numerical details are unstated, and two written formulas have material alternative interpretations. The project must therefore produce a methodological reproduction based on real public market data and an explicitly synthetic Reference Bank, not claim exact numerical recovery of the private experiment.

The implementation must make every assumption and convention inspectable, keep the Paper convention distinct from the Corrected convention, support both 5-year and 15-year horizons, exclude interest-rate swaps, run efficiently enough for iterative local development, and generate objective evidence before it may claim a successful methodological reproduction.

## Solution

Build a package-first reproduction pipeline whose single external seam accepts a resolved YAML run configuration and produces a complete run bundle. The pipeline will:

1. ingest and validate the SNB NSS parameter export;
2. reconstruct historical continuously compounded spot, discount, and monthly forward curves;
3. calibrate three-factor HJM-PCA volatility functions and generate reproducible 5-year and 15-year market scenarios;
4. construct a transparent, internally consistent Reference Bank from versioned product templates;
5. roll the bank forward through a PyTorch-differentiable monthly ALM simulator;
6. train the BM^E, BM^C, BM^D, and MM treasury policies without swaps;
7. evaluate losses, constraints, equity outcomes, dividends, risk statistics, sensitivities, horizon effects, and scenario categories;
8. generate the paper's no-swap tables and figures plus a lightweight Hull-White scenario comparator;
9. run deterministic, statistical, gradient, accounting, and behavioral acceptance gates; and
10. publish a fully traceable acceptance report and artifact manifest.

The full Corrected experiment matrix is the primary result. A smaller paired Paper matrix isolates the effects of the paper's literal PCA scaling and loan-interest formula. A quick profile establishes development validity; only the complete paper-scale Corrected matrix can earn the `methodologically-reproduced` result.

## User Stories

1. As a researcher, I want to run the entire reproduction from one resolved configuration, so that the result does not depend on undocumented manual steps.
2. As a researcher, I want the project to call its result a methodological reproduction, so that unavailable private bank data is not misrepresented as recovered fact.
3. As a developer, I want dependency management and command execution to use uv, so that the Python environment is reproducible.
4. As a model developer, I want training to use PyTorch with Apple MPS support and CPU fallback, so that the project runs on the user's MacBook Pro.
5. As a reviewer, I want the source CSV and paper hashes recorded, so that every run identifies its exact primary inputs.
6. As a researcher, I want the SNB export parser to understand its metadata lines, semicolon delimiter, long-form parameter codes, missing observations, and units, so that calibration begins from validated data.
7. As a researcher, I want calibration restricted to 1 January 2005 through 15 July 2022, so that later observations in the export do not leak into the paper experiment.
8. As a researcher, I want NSS beta parameters converted from percentage points to decimal rates at ingestion, so that discounting and HJM drift use consistent units.
9. As a researcher, I want daily NSS spot curves reconstructed on monthly tenors, so that the public parameters become the term structures required by ALM.
10. As a reviewer, I want spot, discount, and forward curves to round-trip within a strict tolerance, so that unit or compounding errors are detected immediately.
11. As a researcher, I want each calendar week represented by its last valid observation through Friday, so that holidays do not silently remove entire weeks.
12. As a reviewer, I want the selected weekly dates persisted, so that the covariance sample can be reproduced exactly.
13. As a researcher, I want weekly forward-curve differences centered and annualized by 52, so that PCA follows the decided calibration convention.
14. As a reviewer, I want the first three PCA components and their explained variance reported, so that the retained market-risk structure is visible.
15. As a reviewer, I want eigenvector signs stabilized, so that repeated calibration produces stable plots and polynomial coefficients.
16. As a researcher, I want each retained loading fitted by an unweighted cubic tenor function, so that HJM uses the smooth volatility representation described in the paper.
17. As a researcher, I want the Paper convention to scale PCA vectors by eigenvalues, so that the paper's literal formula can be evaluated.
18. As a researcher, I want the Corrected convention to scale PCA vectors by square-root eigenvalues, so that simulated shocks reproduce the retained covariance.
19. As a reviewer, I want each convention's implied covariance reported, so that the difference between variance and volatility scaling is explicit.
20. As a researcher, I want annualized HJM coefficients simulated directly on the monthly ALM clock, so that the market and decision timelines align.
21. As a researcher, I want 60 transitions for 5 years and 180 transitions for 15 years, so that paths contain 61 and 181 observable states respectively.
22. As a researcher, I want the HJM tenor grid extended by the full simulation horizon and shortened one node per month, so that the forward-tenor derivative never needs an invented final node.
23. As a reviewer, I want long-end NSS and cubic-loading extrapolation diagnostics, so that unstable values beyond 15 years are visible.
24. As a researcher, I want finite negative and extreme rates retained without clipping, so that scenario statistics are not silently distorted.
25. As a reviewer, I want non-finite market paths to fail at their point of origin, so that downstream training cannot conceal numerical instability.
26. As a researcher, I want a lightweight Hull-White extended Vasicek simulator for scenario comparison only, so that the paper's Figure 4 diagnostic can be reproduced without creating another training pipeline.
27. As a researcher, I want one canonical Reference Bank shared by both horizons, so that horizon comparisons do not confound initial-bank differences.
28. As a reviewer, I want every Reference Bank assumption labeled and persisted, so that synthetic inputs cannot be confused with private paper data.
29. As a researcher, I want the canonical bank to begin with 10,000 mCHF of assets and the paper-anchored balance-sheet shares, so that absolute constraints and proportions have an explicit scale.
30. As a reviewer, I want equity calculated as assets minus liabilities, so that no cash or liability position silently acts as a balancing plug.
31. As a researcher, I want initial loan, deposit, investment, and funding positions constructed from seasoned product cohorts, so that the starting ladders contain realistic remaining maturities.
32. As a reviewer, I want each generated cash-flow ladder scaled to its target economic value on the initial curve, so that the Reference Bank matches its stated balance-sheet shares.
33. As a researcher, I want the initial market curve dated 15 July 2022, so that valuation and HJM simulation share a deterministic as-of date.
34. As a researcher, I want mortgage, enterprise-loan, deposit, and bond maturity distributions to match the confirmed Reference Bank policy, so that missing bank inputs are replaced consistently.
35. As a researcher, I want personnel and material costs modeled separately, so that only personnel cost receives the paper's annual growth assumption.
36. As a researcher, I want named one-factor Reference Bank sensitivities, so that conclusions can be tested against scale, duration, spread, and cost assumptions.
37. As a model integrator, I want synthetic generation and saved-snapshot loading to produce the same ReferenceBankSnapshot, so that ALM is independent of the data source.
38. As a future model integrator, I want that snapshot contract to be suitable for a later real-bank adapter, so that real data can replace synthetic data without changing the simulator.
39. As a reviewer, I want Reference Bank construction to fail on identity, present-value, shape, sign, duration, or provenance violations, so that invalid inputs never enter training.
40. As a treasury researcher, I want the bank state to track cash and six 180-month nominal cash-flow ladders, so that economic valuation and roll-forward follow the paper.
41. As a treasury researcher, I want investment and funding actions to contain the 13 and 16 permitted no-swap maturities, so that the action space is exactly 29-dimensional.
42. As a treasury researcher, I want bonds to be fractional, non-negative, held to maturity, and unavailable for sale, so that policy constraints match the paper.
43. As a treasury researcher, I want monthly loan originations to replace maturities and add the disclosed annual growth, so that the loan book evolves independently of treasury actions.
44. As a treasury researcher, I want enterprise-loan impairments driven by large annual six-month-rate increases, so that the disclosed credit channel is represented.
45. As a treasury researcher, I want deposit tranches, reference rates, caps, growth, and interest reinvestment implemented, so that liability cash flows follow the paper's rules.
46. As a treasury researcher, I want negative cash interest, operating costs, bond settlements, and dividends included in cash, so that equity changes reconcile to observable cash flows.
47. As a reviewer, I want the cash impact of every bond action multiplied by its quantity and summed across maturities, so that Equation 18's written omission cannot break accounting.
48. As a researcher, I want annual closes at months 12, 24, and subsequent nonterminal year ends, so that no dividend is paid at time zero or after the terminal roll.
49. As a reviewer, I want LCR, NSFR, CMR, Equity/RWA, IRS, and EYR calculated from transparent formulas, so that every penalty can be reconciled independently.
50. As a researcher, I want an asymmetric terminal target loss plus the paper's cumulative constraint penalty, so that training optimizes the reported objective rather than CRRA utility.
51. As a researcher, I want CRRA retained as an evaluation metric, so that training and evaluation objectives are not conflated.
52. As a model developer, I want policy observations to contain the 145 decided features, so that the MM has the same information structure as the paper.
53. As a model developer, I want all four cash-flow encoders separate, so that investment, funding, loan, and deposit structures are not forced through one shared representation.
54. As a model developer, I want stop-gradient applied to policy observations, so that current policy parameters learn without sending policy-input gradients back through earlier states.
55. As a model developer, I want the MM network weights shared across decision dates with time-to-horizon as a feature, so that the multi-period model follows the updated paper architecture.
56. As a model developer, I want MM actions expressed as deviations from a pretrained BM^D, so that scale and maturity learning begin from the strongest scenario-independent benchmark.
57. As a researcher, I want BM^E, BM^C, and BM^D trained through the same differentiable simulator, so that comparisons differ by policy flexibility rather than optimization machinery.
58. As a model developer, I want training paths and sampled target and penalty parameters refreshed deterministically by epoch, so that paper-scale training is diverse and reproducible.
59. As a reviewer, I want training, checkpoint-selection, and final-test scenarios isolated, so that the final 1,600 paths are not used for model selection.
60. As a researcher, I want all policies evaluated on common market paths, so that paired comparisons remove avoidable scenario noise.
61. As a researcher, I want 5-year paths to share the first 60 months of 15-year Brownian innovations, so that horizon comparisons are paired.
62. As a model developer, I want best-checkpoint selection based on selection total loss with a penalty tie-break, so that final model choice is deterministic and constraint-aware.
63. As a model developer, I want resumable checkpoints and seed streams, so that long paper-scale runs can recover from interruption.
64. As a researcher, I want three Corrected MM training seeds per horizon, so that the main conclusion is not based on one initialization.
65. As a reviewer, I want paired bootstrap confidence intervals on core policy differences, so that sampling noise is distinguished from economic effects.
66. As a reviewer, I want signed equity VaR and ES preserved, so that reported tail metrics retain the paper's meaning.
67. As a researcher, I want MM(15y|5y) evaluated by truncating the 15-year policy without renormalizing time, so that finite-horizon effects are tested correctly.
68. As a researcher, I want normalized action turnover and terminal concentration reported, so that the claim that long-horizon behavior is smoother is measurable.
69. As a researcher, I want the five paper-defined scenario categories reproduced, so that policy behavior can be inspected under steep, up, down, inversion, and constant-steepness paths.
70. As a reviewer, I want Reference Bank sensitivities evaluated with frozen canonical policies first, so that data sensitivity is not confused with retraining.
71. As a reviewer, I want quick retraining triggered when a sensitivity reverses a headline conclusion, so that distribution shift can be separated from structural economics.
72. As a researcher, I want the no-swap versions of Tables 1-5 and Figures 3-17 generated with corrected caption numbering, so that the analysis covers the paper without swap-only material.
73. As a reviewer, I want every result bundle to contain resolved configuration, provenance, seeds, checkpoints, metrics, and acceptance evidence, so that no chart is detached from its run.
74. As a developer, I want fast unit, integration, and gradient suites separate from slow paper-scale acceptance, so that routine development remains practical.
75. As a developer, I want a quick end-to-end profile that retains full horizons and network structure, so that it exercises the real interfaces rather than a different toy architecture.
76. As a reviewer, I want an automatically generated acceptance report with observed values and thresholds, so that the reproduction label follows declared rules.
77. As a reviewer, I want adverse seeds and paths retained in reports, so that conclusions cannot be improved by selective omission.
78. As a researcher, I want poor but finite Paper-convention results reported rather than treated as Corrected failures, so that formula fidelity and preferred methodology remain distinct.
79. As a researcher, I want a machine-readable `development-validated` or `methodologically-reproduced` outcome, so that project status is objective.
80. As a maintainer, I want implementation details hidden behind a small number of deep module interfaces, so that financial logic can evolve without forcing changes through every caller and test.

## Implementation Decisions

### Architecture and seams

- The reproduction has one external seam: `ReproductionRunner`. Its interface accepts a fully resolved run configuration and returns a `RunBundle` describing status, artifacts, metrics, checkpoints, and acceptance evidence. The CLI is an adapter over this interface and contains no financial logic.
- `MarketScenarioModel` is an internal seam with HJM-PCA and Hull-White extended Vasicek adapters. HJM-PCA supplies training and evaluation scenarios; Hull-White supplies only the Figure 4 diagnostic.
- `ReferenceBankProvider` is an internal seam returning an immutable `ReferenceBankSnapshot`. The first adapters are the synthetic seasoned-book generator and a saved-snapshot loader. A real-bank adapter is intentionally deferred.
- `TreasuryPolicy` is an internal seam whose interface maps a batch of `PolicyObservation` values to a batch of `TreasuryAction` values. BM^E, BM^C, BM^D, and MM are adapters at this seam.
- `ALMSimulator.rollout` is the primary economic and differentiability seam. It accepts a Reference Bank snapshot, a batch of market paths, a policy, objective parameters, and a resolved convention, and returns a `TrajectoryBatch` without writing files.
- `AcceptanceEvaluator` consumes completed experiment results and returns an immutable `AcceptanceReport`. Artifact rendering and serialization consume returned values rather than reaching into model internals.
- Formula-level helpers remain implementation details inside these deep modules. Tests use constructed inputs and observable results through the highest applicable seam.

### Core data contracts and tensor layout

- All public data contracts are typed, immutable where practical, and validate units, shape, dtype, device, horizon, and provenance at construction.
- `ReferenceBankSnapshot` contains the as-of date, initial curve identity, cash, equity, target economic values, six nominal cash-flow ladders for mortgages, enterprise loans, investments, funding bonds, non-maturity deposits, and term deposits, operating costs, product assumptions, units, provenance, and a content hash.
- Each snapshot cash-flow ladder has exactly 180 monthly entries. The snapshot itself has no path dimension.
- `MarketScenarioBatch` exposes ALM spot yields, discounts, and forwards with batch-first shape `[paths, H+1, 180]`, plus Brownian innovations, convention identity, seed identity, and calibration identity. Extended HJM tenors are internal to scenario generation and are not exposed as bank state.
- `PolicyObservation` has shape `[paths, 145]`. `TreasuryAction` contains investments with shape `[paths, 13]` and financing with shape `[paths, 16]`; its concatenated representation is `[paths, 29]` and must be non-negative.
- `TrajectoryBatch` contains observable states at `[paths, H+1, ...]`, actions at `[paths, H, 29]`, six constraint values and violations at each applicable decision state, annual masks and dividends, loss components, and reconciliation diagnostics.
- Rates and volatilities are decimal annual quantities, maturities and time steps are in years, and monetary quantities are in mCHF. Percentage and basis-point conversions occur only at configuration and reporting edges.

### Configuration and profile resolution

- YAML configuration is layered into source data, convention, run scale, Reference Bank, horizon/experiment, policy, optimization, seeds, output, and acceptance sections.
- Unknown keys, incompatible combinations, invalid weights, inconsistent horizons, and unit-bearing values without an expected unit fail configuration resolution.
- `paper` and `corrected` are locked convention profiles. Only PCA loading scale and loan-interest annualization differ. Any per-choice override marks the resolved convention `custom`.
- `quick` and `paper_scale` are independent run-scale profiles. Selecting a run scale never changes the convention.
- The fully resolved configuration is serialized and hashed before work begins. The hash participates in run identity and checkpoint compatibility.
- The default combination is Corrected plus quick for development commands and Corrected plus paper-scale for a full reproduction request.

### SNB ingestion and curve reconstruction

- The SNB loader accepts the long-form `rendopar` export with UTF-8 BOM, semicolon delimiter, metadata/header lines, and parameter codes `b0`, `b1`, `b2`, `b3`, `t1`, and `t2`.
- Parameter rows are pivoted by date only after duplicate-key, date, numeric, and all-six-parameters completeness checks. Fully blank parameter dates are excluded; partial dates are an error.
- HJM calibration uses complete observations from the inclusive paper window 1 January 2005 through 15 July 2022. Data after the end date is ignored for calibration. Earlier data needed for the initial deposit reference-rate history remains available.
- NSS spot curves are reconstructed with stable small-tenor evaluation. Beta values are converted from percentage points to decimals exactly once; tau values and tenors remain in years.
- Monthly tenors are 1/12 through 180/12 years. Discount factors use continuous compounding. Discrete monthly forwards are derived from adjacent log discount factors and must round-trip to spot rates and discounts.
- Analytic NSS instantaneous forwards are produced only as a diagnostic and do not alter the convention profile.

### Weekly PCA calibration and cubic volatility functions

- Daily reconstructed forward curves are grouped on a Friday-ending weekly calendar and represented by the last valid observation in each week. No parameter interpolation, curve interpolation, outlier deletion, Friday-only filtering, or every-fifth-observation shortcut is used.
- Consecutive weekly forward curves are differenced arithmetically. Their centered sample covariance uses divisor `n-1` and is multiplied by 52 before eigendecomposition.
- Eigenpairs are ordered by descending eigenvalue. Each eigenvector sign is stabilized by requiring its largest-absolute-value element to be positive.
- Three components are retained. Paper loadings are eigenvalue times eigenvector; Corrected loadings are square-root eigenvalue times eigenvector.
- Each scaled loading is fitted by ordinary unweighted cubic least squares with an intercept on normalized tenor `u=x/15`. No zero-tenor constraint, tenor weighting, or clipping is applied.
- Calibration artifacts include selected dates, historical curve matrices, differences, covariance, eigenpairs, explained variance, signed and scaled loadings, cubic coefficients, PCA truncation error, polynomial fit error, and each convention's implied covariance.

### HJM-PCA and Hull-White scenarios

- HJM coefficients are time-homogeneous tenor functions. The drift uses the HJM restriction with the decided volatility loading for the active convention.
- The volatility integral starts from the fitted polynomial value at tenor zero and uses the trapezoidal rule on the monthly grid. The cubic analytic integral is a validation oracle only.
- The Musiela tenor derivative uses the paper's forward difference. Euler-Maruyama uses `dt=1/12`, independent standard-normal factors, and an explicit square-root-time multiplier.
- The initial extended grid contains `N+H` monthly nodes: 240 for a 5-year run and 360 for a 15-year run. NSS and cubic functions are evaluated through 20 or 30 years. Each step updates the first `L-1` nodes and discards the old final node; ALM receives the first 180 nodes.
- A 5-year market batch uses the first 60 innovations of its paired 15-year seed stream. Shared initial nodes and factors must therefore reproduce the same first-60-month market path.
- Finite rates are never clipped. Non-finite forwards, discounts, yields, drift, or volatility terminate the affected run with a typed numerical error and diagnostic context.
- The Hull-White extended Vasicek adapter implements only the paper's scenario-diversity comparator and exposes the same observable curve-batch contract. It is excluded from policy training, selection, acceptance ordering, and Reference Bank sensitivities.
- Training scenarios are generated deterministically by epoch and batch seed without materializing the entire paper-scale training tensor on MPS. Selection and locked test paths are stable CPU artifacts shared across policies.

### Reference Bank construction

- The canonical Reference Bank has 10,000 mCHF total assets. Economic-value targets are cash 20%, investments 5%, mortgages 55%, and enterprise loans 20%; liabilities are non-maturity deposits 10%, term deposits 40%, and funding bonds 40%. Equity is the residual and must equal 10% within construction tolerance.
- Cash is fixed at its target during initial construction and never used as a silent balancing plug. Each non-cash seasoned ladder is scaled to its target present value using the 15 July 2022 curve.
- Initial portfolios are constructed from stationary historical monthly cohorts using the same contractual templates available during simulation, retaining only cash flows outstanding at the as-of date.
- Mortgage originations span 2 through 12 years. Ten-year mortgages receive 40% and every other eligible original term receives 6%. Enterprise-loan originations split equally across 1, 2, and 3 months.
- Each month's product-level matured principal is replaced. The additional total-loan growth amount equals 3% per year divided by 12 and is allocated to mortgages and enterprise loans in the 55:20 initial balance ratio.
- Deposit reference terms are 1 month, 2 months, 1 year, and 10 years. Non-maturity weights are 40%, 30%, 25%, and 5%; term-deposit weights are 10%, 10%, 50%, and 30%.
- Initial investment-bond cohorts use equal issuance across original terms 3 through 15 years. Initial funding-bond cohorts use equal issuance across 3 months and 1 through 15 years. Synthetic legacy pricing uses the initial curve and is labeled as a project assumption.
- The canonical loan customer spread is 150 basis points. Initial personnel cost is 3 mCHF per month and grows 2% at each annual close. Material cost is 1 mCHF per month and remains constant.
- Named one-factor sensitivity variants are: total assets 5,000/10,000/20,000 mCHF; ten-year mortgage weight 20%/40%/60% with residual weight spread equally; ten-year reference-term weight 0%/5%/15% for non-maturity deposits and 15%/30%/45% for term deposits with the offset in the one-year bucket; loan spread 100/150/200 basis points; and operating costs at 75%/100%/125% of canonical values.
- Snapshot construction records disclosed, derived, private-replacement, and unresolved-project assumptions separately. It fails rather than repairing any target-value, accounting, sign, ladder-length, duration, unit, or provenance violation.

### Financial transition rules

- Investments have 13 annual original maturities from 3 through 15 years. Funding bonds have 16 original maturities: 3 months and 1 through 15 years. Bonds are issued at par, pay semiannual coupons, may be fractional, remain non-negative, cannot be sold, and are held to maturity.
- Unit bond cash flows and coupons are derived from the prevailing curve using the paper's pricing rule. Annualized transaction spreads are -15 basis points for investments and +15 basis points for funding.
- Mortgages are default-free. Enterprise loans are impaired at an annual close when the six-month rate increased by more than 2 percentage points over the year. The impairment factor is the excess increase over 2 percentage points, allocated proportionally across outstanding enterprise-loan cash flows; impaired principal stops producing interest.
- Loan interest is profile-dependent. Paper uses the positive part of the annual effective expression without division by 12. Corrected uses the positive part of the same continuously compounded annual rate divided by 12 before exponentiation. Legacy loan interest uses the initial curve as described by the paper.
- Deposits grow independently of treasury actions: non-maturity deposits at 4% per year and term deposits at 1% per year, applied monthly. Maturities and reinvested interest are rolled in equal monthly tranches up to their assigned reference term.
- The deposit reference rate is the three-month moving average of the six-month yield. Non-maturity and term-deposit rates use the paper's pass-through, cap, and floor formulas. Monthly interest uses the exponential monthly conversion and is reinvested through the same maturity mechanism.
- Decision-independent monthly cash flow combines matured loans, new loan outflow, loan interest, new deposits, matured deposits, and operating costs.
- Cash earns no positive interest. When the one-month rate is negative, cash above 30 times minimum reserves incurs the paper's negative-rate penalty.
- Roll-forward settles the first cash-flow bucket, applies decision-independent cash flow and cash penalties, shifts every 180-month ladder left, appends zero, and revalues outstanding positions at the new curve.
- Restructuring adds action-quantity-weighted unit cash flows to investment and funding ladders. Cash subtracts the sum of investment quantity times value and adds the sum of funding quantity times value. This complete update applies in both conventions.
- At time zero there is no settlement roll or annual close; the first policy decision and restructuring occur. For later decision dates, roll-forward precedes policy and restructuring. Constraints are evaluated after restructuring. Annual closes occur at months 12, 24, and subsequent multiples through `H-12`, after constraint evaluation.
- Annual dividends equal 50% of positive annual profit and reduce cash and equity. The final action occurs at `H-1`; the simulator then performs one roll to state `H` without another policy, restructuring, constraint evaluation, annual dividend, or action.

### Balance sheet, constraints, and losses

- Assets equal cash plus economic values of loans and investments. Liabilities equal economic values of deposits and funding. Equity is always the residual.
- Risk-weighted assets equal 10% of investments plus 35% of mortgages plus 100% of enterprise loans. Equity/RWA must be at least 17%.
- HQLA equals 71% of cash plus 89% of investments. Thirty-day net outflow equals 17.6% of non-maturity deposits plus 13% of term deposits plus 1% of funding. LCR must be at least 105%.
- Available stable funding equals 95% of non-maturity deposits, 90% of term deposits, 60% of funding, and 100% of equity. Required stable funding equals 12% of investments plus 71% of loans. NSFR must be at least 105%.
- Minimum reserves follow the paper's sign-dependent treatment of deposits and equal 2.5% of the eligible base. CMR is cash divided by minimum reserves and must be at least 100%.
- IRS revalues the economic balance sheet under parallel plus and minus 100-basis-point curve shifts and uses the worst absolute equity sensitivity relative to current equity. It must not exceed 8.5%.
- EYR is evaluated only at annual constraint dates and equals annual pre-dividend equity change less 6 mCHF, divided by prior-year equity. It must be non-negative.
- Constraint values used in the next policy observation are the most recently computed post-restructuring values. At time zero, the observation uses constraint values computed on the initial pre-decision snapshot.
- Constraint breach transforms, per-constraint coefficients, time accumulation, and the outer squared penalty follow the paper. Coefficients in LCR, NSFR, CMR, Equity/RWA, IRS, EYR order are 1.0, 0.2, 1.0, 2.5, 2.0, and 0.002.
- Training uses only the asymmetric downside target loss plus `lambda` times the constraint penalty. For each training path and epoch, `mu` is sampled uniformly from 2% to 7% and `lambda` uniformly from 0.05 to 25; both are policy features.
- Evaluation fixes `lambda=3.5`, `mu=4.06%` for 5-year policies, and `mu=4.00%` for 15-year policies. CRRA with `gamma=10` is an evaluation metric only. Its equity-ratio floor defaults to `1e-8`, remains configurable as a numerical safeguard, and is recorded in the manifest.

### Policies and feature construction

- All policies produce separate investment and financing actions through the TreasuryPolicy seam and use the same simulator, objective, scenarios, and optimizer family.
- BM^E uses equal maturity distributions and two learned, time-shared scale adjustments. BM^C learns time-shared investment and financing allocations and scales. BM^D learns independent allocations and scales for every actual decision date in the resolved 60/180-transition timeline.
- Benchmark scale equals the amount maturing in the next period plus a learned adjustment. Final actions are non-negative. The implementation reports actual parameter counts and records the paper's indexing mismatch rather than silently changing the resolved timeline.
- The current yield curve is centered and projected to three dimensions by a second PCA fitted only on training scenario states. The fit uses every state when no more than 50,000 are available; otherwise it takes a deterministic, uniformly stratified sample across path and time indices. Curves are centered but not standardized tenor by tenor. Selection and locked test states are never used. The selected indices and fitted transform are persisted as run artifacts.
- Investment, funding, aggregate loan, and aggregate deposit 180-month ladders are divided by 100 and passed through four independent fully connected encoders, each producing 32 features.
- The 145-dimensional policy observation concatenates 128 portfolio-encoding features, 3 yield-curve PCA features, 5 relative balance-sheet features, 6 previous-constraint features, normalized time `t/T`, sampled or fixed `mu`, and sampled or fixed `lambda`.
- Relative balance-sheet features are total assets relative to initial assets, equity relative to assets, cash relative to assets, investments relative to assets, and funding relative to assets.
- MM uses ELU activations, residual fully connected blocks, the paper-scale hidden widths 512, 512, 256, and 128, and a 64-dimensional final encoding before its action heads. Batch normalization is not used.
- MM weights are shared across time. The same network receives `t/T`; for MM(15y|5y), time remains normalized by 15 years when evaluation stops after 60 months.
- The MM action heads produce one scalar deviation and one softmax maturity distribution for investments and for funding. Each deviation is added to the pretrained BM^D action and the final action is passed through ReLU.
- Stop-gradient is applied to the complete 145-dimensional observation immediately before policy evaluation. It is identity in the forward pass, blocks gradients into the state/feature history through the current policy input, and preserves gradients to current policy parameters and through action-dependent future transitions.

### Training, model selection, and reproducibility

- RAdam is used for every learned policy with PyTorch defaults for betas and epsilon and zero weight decay. The learning rate follows a per-update triangular cycle from 5e-4 to 5e-3 and back over four epochs, with two epochs per half-cycle and momentum cycling disabled. Global gradient clipping is 0.2. These undisclosed optimizer and cycle details, along with initialization and other numerical settings, live in resolved run configuration and are labeled as project choices rather than paper facts.
- Quick uses 256 fresh training paths per epoch, 128 selection paths, 128 locked test paths, 5 epochs, and batch size 32. It retains the complete network and 60/180-month horizons and never early-stops.
- Paper-scale uses 40,000 deterministically resimulated training paths per epoch, 1,600 selection paths, 1,600 locked test paths, at most 100 epochs, and batch size 32.
- Paper-scale checkpoint selection begins only after 20 epochs. Training stops after 15 epochs without at least 0.1% relative improvement in mean selection total loss. The selected checkpoint minimizes fixed-parameter selection total loss; effectively tied checkpoints prefer lower penalty loss.
- BM^D is trained before MM at each horizon and convention required by the experiment matrix. MM receives an immutable reference to the selected BM^D baseline parameters.
- Benchmarks use one registered primary training seed. Corrected MM uses three independent initialization and epoch-stream seeds at each horizon while sharing selection and locked test paths. Paper paired runs use the primary seed only.
- Scenario seeds, parameter-sampling seeds, model initialization seeds, data-loader order, bootstrap seeds, and sensitivity seeds are named independently in a seed registry.
- Scenario generation is CPU deterministic and must be bitwise reproducible for an unchanged environment and resolved configuration. MPS training is judged by recorded seeds and statistical acceptance, not bitwise checkpoint equality.
- Checkpoints include resolved configuration identity, convention, run scale, policy, horizon, seed, epoch, optimizer state, selection history, baseline dependency, and code/data hashes. Incompatible checkpoints fail on resume.
- Long runs support interruption and resume at completed epoch boundaries without reusing or skipping an epoch seed.

### Experiment matrix and evaluation

- The complete Corrected matrix contains BM^E, BM^C, BM^D, and MM at 5 and 15 years plus MM(15y|5y). It is required for the methodological-reproduction claim.
- The paired Paper matrix contains BM^D and MM at 5 and 15 years plus MM(15y|5y). It reuses the same Reference Bank and Brownian innovations. Poor finite Paper results are reportable findings; a non-finite result must disclose its exact failure boundary.
- MM(15y|5y) loads the selected 15-year MM, uses its first 60 decisions and its original `t/15` feature, performs the 5-year terminal roll, and is not retrained.
- Final evaluation occurs once after checkpoint freeze on locked test paths. No final-test metric participates in early stopping, checkpoint choice, hyperparameter adjustment, sensitivity-trigger decisions, or seed omission.
- Metrics include mean total, target, and penalty losses; penalty ES95; CRRA loss; equity-ratio mean, standard deviation, skewness, excess kurtosis, signed lower-tail VaR95 and ES95; annualized geometric return excluding dividends; and approximate annual dividend yield standardized by initial equity.
- Paired bootstrap comparisons use 10,000 resamples of common path indices and report point differences and 95% intervals.
- Constraint reports include per-time distributions, terminal/preterminal medians where applicable, share of paths ever violating each constraint, counts among violating paths, mean violating values, worst values, mean penalty, and penalty ES95.
- Normalized action turnover is the sum of L1 changes between consecutive actions divided by total L1 action volume plus a documented epsilon. Terminal concentration is the share of turnover occurring in the final 24 months of a 5-year window.
- The five 5-year scenario categories each select 50 paths by the paper's terminal-curve ranking rules. Categories may overlap because the paper does not disclose an exclusivity rule. Category tables are supporting diagnostics rather than independent acceptance gates.
- Frozen canonical Corrected policies are evaluated on every one-factor Reference Bank variant with common market paths. A reversal in a headline conclusion or strategy ordering triggers quick retraining for that variant and an explicit generalization-versus-structure diagnosis.

### Analysis and artifacts

- The analysis layer produces the no-swap content of Tables 1-5 and Figures 3-17, using printed caption numbering as canonical. It excludes the swap column, MM^S, and swap-only figures.
- Table 1 reports Reference Bank values and labels every synthetic assumption. Table 2 reports resolved calibration, architecture, loss, optimizer, scenario, and run settings. Table 3 contains the full required policy matrix. Table 4 reports Corrected MM constraints. Table 5 reports the five 5-year scenario categories.
- Figures report scenario bands, HJM-versus-Hull-White terminal diversity, the decision architecture, equity distributions, benchmark and MM actions, constraints, sensitivity gaps, durations, and representative 5-year and 15-year paths. Visual similarity is not an acceptance criterion; labels, semantics, sample sizes, and source runs are.
- Every run bundle records the resolved YAML, convention and run-scale identity, custom status, Git commit, Python and dependency versions, device and dtype, source CSV and PDF hashes, Reference Bank hash, weekly observation dates, calibration artifacts, full seed registry, checkpoint history, metrics, tables, figures, and acceptance evidence.
- Artifact writing is atomic at the run level. A partial or failed run preserves diagnostics but cannot masquerade as a completed bundle.
- The CLI provides one full reproduction command plus stage commands for market calibration, Reference Bank construction, policy training, evaluation/reporting, and acceptance. Stage commands use the same resolved configuration and artifact contracts as the full runner.
- CLI commands return nonzero status for configuration errors, failed hard gates, failed requested stages, incompatible resume attempts, and non-finite computations. A completed experiment with failed methodological acceptance still emits its report and returns a distinct acceptance-failure status.

### Acceptance status

- Passing the complete quick pipeline and all quick-applicable hard checks yields `development-validated`. Quick results never yield the methodological-reproduction label.
- `methodologically-reproduced` requires the complete paper-scale Corrected matrix, mandatory paired Paper runs or a precisely disclosed Paper numerical failure, all required artifacts, and every hard, statistical, and core directional gate.
- Paper-scale Corrected MM point estimates must beat all three benchmarks on mean total loss, target loss, penalty loss, and annualized return. Paired 95% intervals versus BM^D must exclude zero in the expected direction for all four metrics in at least two of three MM seeds at each horizon.
- On each qualifying Corrected MM test run, paths with any LCR or NSFR violation are limited to 1% per constraint; CMR, Equity/RWA, and IRS are limited to 2% per constraint. EYR must be the most frequently violated constraint. Mean penalty and penalty ES95 must beat BM^D in at least two MM seeds.
- Over the common first 60 months, MM(15y) median normalized turnover must be at least 10% below MM(5y). MM(15y|5y)'s final-24-month turnover share must be at least 10% below MM(5y). Paired 95% intervals must point in the expected direction.
- MM(5y) mean annualized return and CRRA evaluation must beat MM(15y|5y) with paired 95% intervals in the expected direction. MM(15y|5y) penalty below BM^D(5y) is supporting only.
- MM equity volatility, signed VaR, and signed ES are reported but are not required to beat BM^D.
- Every gate emits pass, fail, or not-applicable, its observed value, threshold, and evidence reference. Any required failure blocks `methodologically-reproduced`. Adverse seeds and paths remain included.

## Testing Decisions

- Tests assert external behavior through the highest applicable seam. They do not assert private helper calls, internal class layout, optimizer call order, or plotting implementation.
- The repository has no prior business-logic tests. The source paper, resolved Wayfinder decisions, research reports, hand-calculated financial fixtures, and independent analytic formulas are the prior art and validation oracles.
- Unit tests cover SNB parsing, NSS curve values, continuous discounting, discrete forward round-trips, bond cash-flow pricing, cash-flow shifts, deposit rates and reinvestment, loan growth and impairment, operating costs, dividends, all six constraints, penalties, losses, metrics, signed VaR/ES, and configuration resolution.
- Market-model tests cross the MarketScenarioModel seam. CPU float64 spot/discount/forward round-trip error, PCA orthogonality, retained-covariance identities, initial-curve equality, and deterministic HJM one-step equality must not exceed `1e-10` in the applicable absolute or relative Frobenius measure.
- PCA acceptance requires three-component explained variance of at least 90%, aggregate cubic-loading relative Frobenius error no more than 10%, and individual component error no more than 15%.
- HJM statistical tests use 50,000 fixed-seed one-step shocks for each convention. Each standardized factor mean must be within three standard errors of zero; empirical curve-shock covariance must be within 5% relative Frobenius error of the convention's `V V^T dt` target.
- ReferenceBankProvider contract tests run against both synthetic construction and saved-snapshot loading. Each portfolio present value must match its target within `1e-8` relative error; ladders must have length 180; nominal cash flows must be at least `-1e-8`; loan duration must be below 5 years; deposit duration must be below 3 years; units and provenance must be complete.
- ALMSimulator property and integration tests use short deterministic fixtures and full 60/180-step generated fixtures. At every path and state, absolute assets-minus-liabilities-minus-equity and cash-reconciliation error must not exceed the larger of `1e-6` mCHF and `1e-10` times total assets.
- Constraint tests include values just below, at, and just above every bound, plus shifted-curve IRS and annual-only EYR fixtures. Formula error must not exceed `1e-10`.
- Gradient tests run smooth, interior CPU float64 fixtures through ALMSimulator.rollout. Representative autograd gradients must satisfy `|g_auto-g_fd| <= 1e-6 + 1e-4 |g_fd|` against central finite differences. Policy gradient norm must exceed `1e-8` in a fixture whose terminal loss depends on action.
- Stop-gradient tests verify identical forward values with and without detachment, zero or absent gradient into the observation state through the current policy call, finite nonzero gradient into policy parameters, and gradient propagation from current actions through later active and passive transitions to terminal loss.
- MPS integration tests execute at least one complete float32 optimizer step, require finite state/loss/gradients, and require clipped global gradient norm no more than `0.2 + 1e-6`.
- Convention tests prove that locked Paper and Corrected profiles differ only in PCA scaling and loan-interest annualization. Any override must resolve to custom, and mismatched manifests must be rejected for paired comparison.
- Policy contract tests exercise BM^E, BM^C, BM^D, and MM through the same TreasuryPolicy interface and assert nonnegative `[paths,29]` output, correct scenario/time dependence, correct maturity availability, and BM^D baseline use by MM.
- Quick end-to-end tests run the complete 5-year and 15-year chain with the quick profile, including calibration, bank generation, every policy, MM(15y|5y), evaluation, artifacts, and development acceptance. These tests are explicitly invoked and are not part of the fastest unit-test command.
- Paper-scale acceptance is a slow suite invoked through the reproduction runner. It verifies the experiment matrix, locked test isolation, three MM seeds, 10,000-resample paired bootstrap intervals, constraint-rate gates, horizon-turnover gates, sensitivity evaluations, required tables/figures, and manifest completeness.
- Artifact tests validate schemas and semantic content rather than image pixels. Every table and figure must identify its policy, horizon, convention, sample size, metric units, and source run.
- Reproducibility tests rerun CPU scenario generation and saved-snapshot loading from the manifest and require identical content hashes. MPS checkpoints are not required to be bitwise identical.
- Failure-path tests cover partial SNB dates, duplicate rows, invalid units, unknown configuration keys, non-finite HJM paths, imbalanced snapshots, negative actions, missing artifacts, incompatible resume checkpoints, attempted final-test model selection, and omitted required seeds.
- Tests are grouped into unit, property/integration, gradient, quick end-to-end, and slow paper-scale acceptance layers. The ordinary fast test command excludes paper-scale work.

## Out of Scope

- Interest-rate swaps, swap cash flows, swap valuation, swap constraints, MM^S, the swap column in Table 3, the swap-volume figure, and all conclusions from the paper's swap extension.
- A 15-year swap experiment.
- Exact reconstruction of the collaborating bank, its private cash-flow ladders, its trained weights, its random paths, or its unpublished model-selection process.
- Integration with real bank source systems in this phase. Only the replaceable ReferenceBankSnapshot seam is included.
- Basel or Swiss regulatory certification. The implementation reproduces the paper's simplified constraint formulas.
- A web interface, hosted service, production deployment, distributed training platform, or cloud orchestration.
- Pixel-identical figures, numerically identical paper tables, or identical learned actions.
- Stochastic or interest-rate-sensitive customer demand beyond the paper's deterministic loan and deposit assumptions.
- A calibrated market price of interest-rate risk or a distinction between physical and risk-neutral measures beyond the paper's `P=Q` assumption.
- Trading existing bonds, short selling, early unwind, derivative hedging, transaction types other than the no-swap action set, or accounting regimes beyond economic-value balance-sheet treatment.
- Using final-test results for hyperparameter tuning, Reference Bank reverse engineering, or post-hoc acceptance-threshold changes.

## Further Notes

- The source paper and SNB CSV are already present in the repository. Their content hashes are treated as run inputs rather than copied into configuration.
- The resolved Wayfinder map and its six linked decisions are the authority for terminology, Reference Bank assumptions, convention choices, experiment scope, and acceptance thresholds.
- The public SNB data plus paper formulas are sufficient for this methodological reproduction; the Reference Bank is deliberately synthetic and must be named as such in every report.
- The implementation should benchmark real wall-clock and memory behavior on the user's Apple Silicon machine. Runtime is an engineering measurement, not a methodological acceptance gate. Streaming and resumability are required to make paper-scale execution practical.
- After approval, this spec should be passed to `/to-tickets` and split into blocker-aware tracer-bullet implementation tickets. Implementation should not proceed directly from the Wayfinder map.
