# No-swap experiments and validation catalogue

## Scope and source

This report defines the empirical reproduction target for the basic Deep ALM setting in which the bank may buy investment bonds and issue financing bonds, but may not trade interest-rate swaps. It covers the 5-year and 15-year horizons, including evaluation of the 15-year policy after 5 years. The sole primary source is Englisch et al., *Deep treasury management for banks* ([paper, especially Sections 2.4-2.5 and 3-4.4][paper]).

The paper uses confidential data from a collaborating Swiss retail bank. It explicitly says that some results are shown only in aggregated or relative form, that the exact loan and deposit portfolio compositions are withheld, and that the underlying datasets are not readily available because they belong to the bank ([paper, pp. 2, 6, 36][paper]). Consequently, the targets below support a **methodological reproduction and directional validation**, not exact recovery of the published numbers.

## Required experimental protocol

- Use a monthly ALM grid. The no-swap action at every decision date has 29 non-negative entries: 13 investment-bond maturities from 3y through 15y, and 16 financing-bond maturities comprising 3m and 1y through 15y. Bonds trade at par, may be fractional, cannot be sold or shorted, and are held to maturity ([paper, Sections 2.1.3-2.1.4, pp. 5-7][paper]).
- Run separate optimizations over 5y and 15y, corresponding to 60 and 180 monthly transition/decision periods in the paper's experimental terminology. The same initial balance sheet and initial yield curve underpin both settings ([paper, Sections 2.1.1 and 4.1, pp. 5, 23][paper]).
- Evaluate each trained strategy on 1,600 held-out HJM-PCA yield-curve scenarios ([paper, Section 4.1, p. 23][paper]).
- Include `MM(15y|5y)`: train the shared-weight main model for 15y, then stop its forward simulation after 60 rather than 180 months. Keep the time feature on its training clock, `t/15y`, rather than renormalizing it to `t/5y` ([paper, Section 4.4.1, pp. 32-33][paper]).
- For the published result models, use the Table 2 training profile: 40,000 training scenarios, batch size 32, nominally 100 epochs, RAdam, a cyclic learning-rate schedule over `[5e-4, 5e-3]`, and global gradient clipping at `0.2`. The decision network uses ELU, 64-dimensional hidden portfolio encoders, a 32-dimensional combined encoding, and hidden widths `[512, 512, 256, 128]` ([paper, Table 2, p. 16][paper]).
- Treat this training profile as incompletely specified rather than as a deterministic recipe. The prose says most development runs stopped before 100 epochs, while result models were fully trained or fine-tuned with an entirely resimulated set of paths each epoch. No seeds, exact early-stopping rule, epoch counts per reported model, or validation path identities are disclosed ([paper, Section 3.2.3.3, p. 21][paper]).

## Strategies and model variants

All three benchmarks are optimized endogenously by gradient descent under the same ALM simulator. They are scenario-independent: at a given model time they take the same action on every yield-curve path. Their scale is anchored to the amount maturing in the next period, plus a learned adjustment, and all final decisions are non-negative ([paper, Section 3.1, pp. 17-18][paper]).

| ID | Required definition | Time dependence | Scenario dependence |
| --- | --- | --- | --- |
| `BM^E` | Equal `1/N` maturity allocation. Only one investment scale and one financing scale are learned (two parameters total), shared over time. | No | No |
| `BM^C` | Investment and financing maturity allocations and scales are learned, but shared over all decision dates; the paper counts `b_B + b_K` parameters. | No | No |
| `BM^D` | The allocation and scale are learned separately at every decision date; the paper counts `(b_B + b_K)(H-1)` parameters. | Yes | No |
| `MM` | Main Deep ALM decision network. It starts from a pretrained `BM^D` action and learns state-conditional investment and financing deviations. | Shared network with `t/T` input | Yes |
| `MM(15y|5y)` | The 15y `MM` evaluated by truncating its simulation at 5y; it is not retrained. | As trained for 15y | Yes |

Constant-amount variants introduced before these three benchmarks performed poorly because legacy maturities are lumpy and are not part of the reported comparison. The required benchmark set is therefore exactly `BM^E`, `BM^C`, and `BM^D` ([paper, Section 3.1, pp. 17-18][paper]).

The required `MM` implementation details that materially affect the experiment are:

- Inputs include a 3-dimensional PCA projection of the current yield curve; separate learned encodings of the investment, financing, loan, and deposit cash-flow vectors; relative balance-sheet size/liquidity/leverage features; previous-period constraint features; and `t/T` ([paper, Section 3.2.1, pp. 18-19][paper]).
- The output head has separate scale and softmax maturity-distribution heads for investments and financing. Each combines a pretrained `BM^D` decision with a learned deviation and applies a final non-negativity clamp/ReLU ([paper, Equation 46 and Section 3.2.2, pp. 19-20][paper]).
- Network weights are shared across all decision dates, reducing parameters by roughly `H-1`; residual connections are used in fully connected layers, and the input features are detached with stop-gradient before the decision network ([paper, Equations 48-49 and Sections 3.2.3.2-3.2.3.3, p. 21][paper]).

## Loss settings

### Training objective

The reported models are **not trained on CRRA utility**. They use an asymmetric terminal target loss which penalizes only falling below the target equity:

`loss_target = negative_part(E_T - (1 + mu)^T E_0)^2`.

This replaces the CRRA component in the total loss, giving `loss = loss_target + lambda * loss_penalty`. CRRA with `gamma = 10` is retained only as an evaluation metric ([paper, Equation 47 and Section 3.2.3, pp. 19-20][paper]).

For every training path and epoch, sample `mu ~ Uniform(2%, 7%)` and `lambda ~ Uniform(0.05, 25.0)`, and pass both sampled values as model features. The authors report that sampling improves the metrics, but **does not produce meaningful goal-conditioned changes in the policy at inference**. At validation time use fixed `lambda_eval = 3.5`; use `mu_eval = 4.06%` for the four 5y no-swap columns and `mu_eval = 4.00%` for the 15y columns ([paper, Table 2 and Sections 3.2.3.1-3.2.3.2, pp. 16, 20-21; Table 3 note, p. 21][paper]).

`MM(15y|5y)` has no reported total loss and its target loss should not be compared directly with the 5y models. More generally, Table 3 warns that total and target loss cannot be compared across model categories evaluated with different return targets ([paper, Table 3 note, p. 21][paper]).

### Constraint penalty

At every applicable date, transform each breach nonlinearly, weight and sum all breaches over time, then apply the outer penalty `loss_penalty = (1 + p)^2 - 1`. This nested construction makes large individual breaches and large cumulative breach totals more than linearly costly ([paper, Equations 30-31, p. 12][paper]).

| Constrained quantity | Bound | Frequency / interpretation | Penalty coefficient `sigma_i` |
| --- | ---: | --- | ---: |
| LCR | `>= 105%` | 30-day liquidity buffer | 1.0 |
| NSFR | `>= 105%` | long-run stable funding | 0.2 |
| CMR | `>= 100%` | cash to SNB minimum reserve | 1.0 |
| Equity / RWA | `>= 17%` | leverage/capital constraint | 2.5 |
| IRS | `<= 8.5%` | absolute equity sensitivity to a parallel `+/-100 bp` curve shift | 2.0 |
| EYR | `>= 0` | annual excess return after operational cost plus mCHF 6 buffer | 0.002 |

The coefficient order above follows Table 2, not the order in which constraints appear in Section 2.4. LCR, NSFR, CMR, Equity/RWA, and IRS are treated as the five regulatory constraints; EYR is the additional bank objective. EYR is checked only at annual closing dates, while the other constraints are monitored after balance-sheet restructuring ([paper, Sections 2.4-2.5, pp. 9-12; Table 2, p. 16][paper]).

## Evaluation metrics and sign conventions

Table 3 reports averages over the 1,600 validation paths ([paper, Sections 3.3 and 4.1, pp. 22-23][paper]):

- Mean total loss, mean target loss, mean penalty loss, and the 95% expected shortfall of penalty loss.
- Mean CRRA utility loss at `gamma = 10` even though CRRA is not the training objective.
- Final equity ratio `ER = E_T / E_0`: mean, standard deviation, skewness, excess kurtosis, VaR, and expected shortfall.
- Annualized geometric return on equity `mu_bar`, excluding dividends, and approximate annual dividend yield `delta_bar`.

Two conventions need to be preserved in code and tests:

1. Equity VaR and ES are reported as **lower-tail deviations from the mean**, so the table contains negative values. A value closer to zero means a smaller downside gap; retain the signed published convention rather than silently converting it to a positive loss magnitude.
2. `delta_bar` is deliberately approximate: dividends are summed without time-value adjustment and standardized by initial equity `E_0`, not by equity at the start of each year ([paper, Equations 50-52 and discussion, pp. 22-23][paper]).

## Published aggregate results to use as reference targets

### Table 3: 5y no-swap results

| Metric | `BM^E` | `BM^C` | `BM^D` | `MM` |
| --- | ---: | ---: | ---: | ---: |
| Mean total loss (`lambda=3.5`) | 1.357 | 0.999 | 0.753 | 0.467 |
| Mean target loss (`mu=4.06%`) | 1.428 | 0.997 | 0.824 | 0.558 |
| Mean penalty loss | 0.011 | 0.032 | 0.012 | 0.003 |
| ES95 of penalty loss | 0.107 | 0.518 | 0.144 | 0.036 |
| Mean CRRA utility loss (`gamma=10`) | -0.064 | -0.071 | -0.074 | -0.079 |
| Mean final equity ratio | 1.107 | 1.124 | 1.133 | 1.153 |
| Equity-ratio standard deviation | 0.040 | 0.026 | 0.027 | 0.036 |
| Equity-ratio skewness | -0.419 | 0.047 | 0.286 | 1.682 |
| Equity-ratio excess kurtosis | 2.551 | 0.883 | 1.088 | 8.101 |
| Equity-ratio VaR95 (signed) | -0.071 | -0.043 | -0.043 | -0.049 |
| Equity-ratio ES95 (signed) | -0.101 | -0.055 | -0.053 | -0.060 |
| Annualized return, % | 2.048 | 2.353 | 2.529 | 2.883 |
| Approx. annual dividend yield, % | 1.432 | 1.503 | 1.560 | 1.855 |

Source: [paper, Table 3, p. 21][paper].

### Table 3: 15y no-swap results

| Metric | `BM^E` | `BM^C` | `BM^D` | `MM` |
| --- | ---: | ---: | ---: | ---: |
| Mean total loss (`lambda=3.5`) | 45.907 | 32.151 | 3.123 | 1.587 |
| Mean target loss (`mu=4.00%`) | 10.829 | 10.625 | 3.059 | 1.712 |
| Mean penalty loss | 10.066 | 6.193 | 0.067 | 0.015 |
| ES95 of penalty loss | 185.857 | 114.694 | 0.836 | 0.124 |
| Mean CRRA utility loss (`gamma=10`) | -0.106 | -0.107 | -0.110 | -0.110 |
| Mean final equity ratio | 1.518 | 1.514 | 1.716 | 1.812 |
| Equity-ratio standard deviation | 0.178 | 0.164 | 0.172 | 0.202 |
| Equity-ratio skewness | 0.707 | 0.946 | 0.033 | 0.398 |
| Equity-ratio excess kurtosis | 4.079 | 4.913 | 1.127 | 0.657 |
| Equity-ratio VaR95 (signed) | -0.273 | -0.242 | -0.302 | -0.320 |
| Equity-ratio ES95 (signed) | -0.356 | -0.308 | -0.368 | -0.368 |
| Annualized return, % | 2.781 | 2.768 | 3.631 | 4.002 |
| Approx. annual dividend yield, % | 3.042 | 2.982 | 3.967 | 4.515 |

Source: [paper, Table 3, p. 21][paper].

### Table 3: `MM(15y|5y)` intermediate result

The paper reports no total loss for this cross-horizon evaluation. Its other values are: target loss `45.290`, penalty loss `0.007`, penalty ES95 `0.062`, CRRA utility loss `-0.071`, mean ER `1.128`, ER standard deviation `0.043`, skewness `1.263`, excess kurtosis `5.465`, signed equity VaR95 `-0.064`, signed equity ES95 `-0.082`, annualized return `2.434%`, and approximate annual dividend yield `1.613%` ([paper, Table 3, p. 21][paper]).

### What Table 3 does and does not establish

- On both horizons, `MM` has the lowest total, target, mean penalty, and penalty-tail losses, and the highest mean final equity, annualized return, and approximate dividend yield among the no-swap strategies. The improvement over simple benchmarks is particularly large at 15y because `BM^E` and `BM^C` accumulate very large penalties ([paper, Table 3 and Section 4.1, pp. 21-24][paper]).
- The paper's prose says performance improves with more trainable parameters and that `MM` is better “across all metrics.” The raw equity-risk rows require a narrower reading: versus `BM^D`, `MM` has higher equity-ratio volatility and more negative signed VaR on both horizons; 5y ES is also more negative, while 15y ES is equal after rounding. Reproduction acceptance should therefore require the clear loss/return/constraint direction, not the literal claim that every risk statistic improves ([paper, Table 3 and Section 4.1, pp. 21-23][paper]).
- Higher positive skew under `MM` and the equity histograms support the authors' interpretation that the asymmetric target objective preserves upside while penalizing failure to reach the target. The 5y `MM` also has high excess kurtosis, so “not fat-tailed on the left” is a qualitative histogram claim rather than a conclusion from kurtosis alone ([paper, Figure 6 and Section 4.1, pp. 23-24][paper]).

## Constraint validation targets

The percentages below are percentages of the 1,600 validation scenarios, not proportions. For example, `0.12` in the paper's row means `0.12%` (approximately two scenarios) ([paper, Table 4, p. 26][paper]).

| Horizon | Constraint | Median at `T-1` | Scenarios with violation | Mean count among violating scenarios | Mean violating value | Largest violation value |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5y | CMR | 1.76 | 0.12% | 1.00 | 0.94 | 0.92 |
| 5y | LCR | 1.10 | 0.00% | - | - | - |
| 5y | NSFR | 1.33 | 0.00% | - | - | - |
| 5y | Equity/RWA | 0.20 | 0.00% | - | - | - |
| 5y | IRS | -0.03 | 0.06% | 1.00 | 0.09 | 0.09 |
| 5y | EYR | 21.17 | 20.06% | 1.08 | -2.99 | -23.80 |
| 15y | CMR | 3.66 | 0.12% | 1.00 | 0.91 | 0.85 |
| 15y | LCR | 1.23 | 0.00% | - | - | - |
| 15y | NSFR | 1.34 | 0.00% | - | - | - |
| 15y | Equity/RWA | 0.24 | 0.12% | 2.50 | 0.17 | 0.16 |
| 15y | IRS | -0.03 | 0.12% | 4.00 | 0.09 | 0.09 |
| 15y | EYR | 63.43 | 45.94% | 1.35 | -5.54 | -53.68 |

Required qualitative checks are:

- LCR and NSFR should have no violations in the published `MM` validation sets; CMR, Equity/RWA, and IRS violations should be extremely rare. EYR is by far the most frequently breached constraint despite its small coefficient because it is intrinsically difficult to satisfy under adverse annual scenarios ([paper, Section 4.2 and Table 4, pp. 25-28][paper]).
- The 5y policy runs LCR and CMR down near their lower bounds by the end; the 15y policy maintains a larger cash/liquidity cushion. NSFR remains comfortably slack and was above 130% in at least 90% of scenarios in the authors' inspection ([paper, Section 4.2 and Figure 9, pp. 25-26][paper]).
- IRS stays near zero for much of 15y and becomes negative near the terminal date in both horizons. Investment and financing durations evolve similarly when plotted against relative time `t/T`, which the authors interpret as a finite-horizon artifact rather than classical train/validation overfitting ([paper, Section 4.2 and Figure 11, pp. 27-28][paper]).

## Strategy-behaviour validation targets

These are directional checks, not hard numerical assertions:

### Benchmarks and aggregate `MM` behaviour

- All reported benchmarks invest and borrow more than the amount maturing; financing is scaled up less than investment. `BM^E(5y)` reduces cash materially, whereas the other plotted benchmark/horizon combinations slightly increase it. `BM^C(5y)` invests at shorter maturities than it finances, while `BM^C(15y)` invests at slightly longer maturities than it finances ([paper, Section 4.1.1 and Figure 7, pp. 23-24][paper]).
- `MM` actions are path-dependent, although medians and interquartile bands can still be similar across many related scenarios. Both 5y and 15y models initially raise roughly mCHF 1,000 across all scenarios because the initial state is shared; holding the proceeds as cash protects valuation if rates rise as the simulator tends to project ([paper, Section 4.1.2, pp. 24-25][paper]).
- The 5y policy often makes no investments and then spikes into 3y bonds at the 2y mark - exactly three years before the terminal date. The 15y policy invests and finances more continuously and reduces leverage over time. The authors treat the 5y spike and terminal switch toward short-term financing as finite-horizon artifacts ([paper, Section 4.1.2 and Figure 8, pp. 24-25][paper]).

### Selected 5y scenarios

- `steep`: the curve steepens. `MM(5y)` makes negligible investments, initially finances at 3m and 10y, and later uses very large short-term financing. Earlier 10y borrowing creates positive 7y-10y sensitivity, so steepening raises equity ([paper, Section 4.3.1 and Figures 12-14, pp. 28-30][paper]).
- `inversion`: the curve inverts in the latter half. `MM(5y)` makes the characteristic large 3y investment around month 24, borrows mainly at 10y for roughly the first 3.5 years, then switches to shorter funding; the short-funding volume is lower than in `steep` because the inverted short end is expensive ([paper, Section 4.3.1 and Figures 12-14, pp. 28-30][paper]).

### Selected 15y scenarios

- `incr`: the curve generally rises and steepens, with temporary declines around years 6 and 8. The model first builds slightly positive IRS through low investment and higher financing, then mainly invests at 3y and finances at 10y, reducing investment when yields temporarily fall ([paper, Section 4.3.2 and Figures 15-17, pp. 29-32][paper]).
- `inv_and_back`: the curve flattens, inverts and rises, then falls/flattens and finally steepens. Investment remains mostly short term; financing is long term early, mainly 10y from years 5-13, and short term near the end ([paper, Section 4.3.2 and Figures 15-17, pp. 29-32][paper]).
- About 40% of new mortgages are assumed to be 10y. The model frequently offsets their 10y sensitivity with 10y financing, while keeping controlled maturity-transformation exposure at other tenors. Sensitivity gaps are generally small and net IRS is close to zero ([paper, Section 4.3.2, pp. 30-32][paper]).
- An additional, un-tabulated experiment reduced the EYR penalty coefficient. The model then selected more long-term investments, achieved higher mean terminal equity, and generated higher equity volatility. This validates sensitivity to loss engineering but cannot be reproduced numerically from disclosed information ([paper, Section 4.3.2, p. 31][paper]).

## Long-horizon policy evaluated at 5y

The required directional findings for `MM(15y|5y)` are:

- It underperforms `MM(5y)` at 5y, showing that the conspicuous 5y terminal behaviour is an objective-horizon effect rather than simply failed optimization.
- Its overall 5y performance is comparable to `BM^D(5y)` and it has lower constraint penalties, but its signed equity VaR/ES and equity volatility are worse than `BM^D(5y)`.
- Its first five years resemble the beginnings of the 15y example paths: steady monthly investment/financing, slightly more funding and less investment early, mainly 3y investment, and no 2y investment spike or terminal burst of short financing.
- The paper recommends long-horizon training with analysis restricted to an earlier window as a practical mitigation for terminal-date distortion, not as a complete solution to the going-concern problem ([paper, Section 4.4.1 and Table 3, pp. 21, 32-33; Section 5, pp. 35-36][paper]).

## Scenario-category validation

From the 1,600 5y validation paths, select five non-exclusive or unspecified-overlap subsets of 50 paths each using the paper's ranking rules ([paper, Section 4.4.2, pp. 33-34][paper]):

| Category | Selection rule |
| --- | --- |
| Steep | Maximize terminal `15y yield - 1m yield`. |
| Up | Require terminal 1m yield above 2%, then maximize terminal steepness to avoid selecting only flat high-rate curves. |
| Down | Minimize terminal average yield across all maturities. |
| Inversion | Minimize terminal steepness. |
| Constant steepness | Minimize the time-series standard deviation of curve steepness. |

Table 5 reports only CRRA utility loss and penalty loss, averaged over 50 scenarios in each category:

| Metric / model | Steep | Up | Down | Inversion | Constant steepness |
| --- | ---: | ---: | ---: | ---: | ---: |
| CRRA loss, `BM^D(5y)` | -0.071 | -0.063 | -0.086 | -0.074 | -0.074 |
| CRRA loss, `MM(5y)` | -0.085 | -0.066 | -0.096 | -0.079 | -0.079 |
| CRRA loss, `MM(15y|5y)` | -0.079 | -0.048 | -0.093 | -0.064 | -0.075 |
| Penalty loss, `BM^D(5y)` | 0.050 | 0.011 | 0.113 | 0.022 | 0.006 |
| Penalty loss, `MM(5y)` | 0.001 | 0.008 | 0.005 | 0.010 | 0.000 |
| Penalty loss, `MM(15y|5y)` | 0.001 | 0.026 | 0.003 | 0.033 | 0.002 |

Across all three policies, utility is best in `Down` and worst in `Up`; the authors relate this to negative IRS near the terminal date. The two main models sharply reduce benchmark penalties, while their remaining penalties concentrate in `Up` and `Inversion`, likely through EYR breaches. Category conclusions are illustrative: selection relies mostly on the terminal curve, paths within a category can differ substantially, and 5y revaluation effects can dominate cash-flow effects ([paper, Table 5 and Section 4.4.2, pp. 33-34][paper]).

## Figure and table inventory for the no-swap reproduction

Use the **printed captions** as the canonical numbering. The prose cross-references are systematically one number higher for much of the paper (for example, prose “Figure 9” points to captioned Figure 8). This publication inconsistency must not turn into missing or duplicated reproduction plots.

| Item | Required reproduction content |
| --- | --- |
| Table 1 | Approximate initial economic balance-sheet weights and private-data caveat; input calibration context, not a result target. |
| Table 2 | HJM-PCA, network, loss, training, and optimizer hyperparameters. |
| Table 3 | Aggregate benchmark/main-model results for 5y, `15y|5y`, and 15y; omit the swap column. |
| Table 4 | Detailed 5y/15y `MM` constraint statistics. |
| Table 5 | Five-category 5y CRRA and penalty statistics. |
| Figure 3 | Simulated 1m-yield median and 5/25/75/95% bands; scenario sanity check. |
| Figure 4 | Sample 5y terminal curves from HJM-PCA versus Hull-White-extended Vasicek; scenario-diversity comparison. |
| Figure 5 | Decision-network architecture. |
| Figure 6 | 5y and 15y final-equity histograms versus the strongest benchmark (`BM^D`). |
| Figure 7 | Learned benchmark scales relative to maturing positions plus investment/financing durations. |
| Figure 8 | `MM` investment and financing volumes for 5y and 15y, with `BM^D` overlay and quantile bands. |
| Figure 9 | LCR and CMR paths for 5y and 15y with constraint bounds. |
| Figure 10 | Equity/RWA paths and annual dividend-related jumps. |
| Figure 11 | IRS plus investment and financing durations for both horizons. |
| Figures 12-14 | The two selected 5y yield paths, corresponding decisions, and sensitivity gaps. |
| Figures 15-17 | The two selected 15y yield paths, corresponding decisions, and sensitivity gaps. |

The source PDF's no-swap captions end at Figure 17. The following swap analysis is captioned Figure 18 even though its prose calls it Figure 19 ([paper, pp. 16-34][paper]).

## Aggregated, relative, or unavailable outputs

The following prevent strict numerical replication and must be called out in downstream specifications:

- Table 1 gives only approximate relative initial balance-sheet weights: assets 20% cash, 5% investments, 75% loans (55% mortgages and 20% enterprise loans); liabilities/equity 50% deposits, 40% borrowings, 10% equity. Exact cash-flow ladders and legacy portfolio compositions are private ([paper, Table 1, p. 6][paper]).
- Result metrics are aggregate statistics over 1,600 scenarios; most policy plots show medians and 5/25/75/95% bands. Single-scenario analysis uses hand-selected examples, but scenario identifiers and random seeds are not disclosed ([paper, Sections 4.1-4.4, pp. 23-34][paper]).
- Benchmark volume charts normalize investment/financing scale by the amount maturing; sensitivity gaps are aggregated into yearly tenor tranches; dividend yield is standardized by initial equity ([paper, Figures 7 and 8 and Sections 3.3-4.1, pp. 23-25][paper]).
- Although some action volumes are shown in mCHF, the initial state needed to reproduce them is unavailable. The authors say only a small portion of all generated outputs was included in the article ([paper, pp. 2, 6, 35-36][paper]).
- Model selection details, learned weights, checkpoints, exact simulated paths, category membership, and the bank's internal preference/tuning process are unavailable. These published numbers are therefore comparison landmarks, not pass/fail equality targets.

## Explicit exclusions

Exclude all of the following from this reproduction phase:

- Section 2.6 swap state, actions, cash flows, valuation, spreads, and swap-volume constraints.
- `MM^S`, the superscript-`S` 5y column in Table 3, including its `mu_eval = 5.39%` and all corresponding metrics.
- Section 4.5, its swap-only strategy claims, and the swap-volume figure captioned Figure 18 (called Figure 19 in the prose).
- Any conclusion that depends on comparing `MM^S` with `MM`, including claims of higher return or improved constraint compliance from swap access.
- A 15y swap experiment: the paper does not report one and states that its swap-volume restriction was designed for 5y and is too simplistic for a longer horizon ([paper, Sections 2.6 and 4.5, pp. 12-14, 33-35][paper]).

## Minimal acceptance implications for later specification

A faithful no-swap methodological reproduction should, at minimum:

1. Run all four policies (`BM^E`, `BM^C`, `BM^D`, `MM`) at 5y and 15y, plus `MM(15y|5y)`, on the same validation paths per horizon.
2. Reproduce all six constraint calculations, asymmetric target-plus-penalty training, fixed evaluation parameters, and the paper's signed risk metrics.
3. Verify accounting identities and constraint formulas exactly before comparing learned policies.
4. Expect `MM` to improve losses, returns, and constraint compliance over benchmarks; do not require exact published values or improvement in every equity-dispersion statistic.
5. Check for the horizon-dependent strategy signatures above, especially the 5y terminal artifact and the smoother/more cautious 15y policy.
6. Publish the aggregate tables and figures listed above with deterministic experiment metadata, while labelling all deviations caused by synthetic bank data or resolved implementation ambiguities.

[paper]: ../../../docs/Deep%20treasury%20management%20for%20banks.pdf
