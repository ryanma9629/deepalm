# Financial corrections, 2026-09-08

These corrections apply to both locked formula profiles. The existing ADR remains
unchanged: only PCA scaling and loan-interest annualization branch between Paper
and Corrected. No production/pilot training is required to validate these fixes.

## Implemented

- Deposit rollover: settle the old principal, then credit the same principal for
  renewal. The liability is renewed once. Growth is an additional inflow;
  capitalized interest adds a liability but no net cash flow. C-5 further keeps
  the original 1, 2, 12, or 120 month reference-term class through renewal;
  only new external growth is allocated using the product's global weights.
- C-6 deposit initialization: the first rate window uses scenario Y0 plus two
  dated, pre-valuation six-month yields; the second uses Y1, Y0 and Y−1. The
  calendar-month history is content-addressed in the Reference Bank snapshot
  and must match the market batch, rather than being fabricated by repeating Y0.
- C-1/C-2 MM observation: the five balance-sheet features use current
  pre-action economic values, in Equation 45 order `A/A0, E/A, C/A,
  investment PV/A, funding PV/A`; the six raw constraint features are shifted
  by their paper bounds `(1.05, 1.05, 1.00, 0.17, 0, 0)`. Nominal ladders stay
  in the four encoders. Zero A or A0 fails explicitly, while finite negative
  equity remains an input value.
- C-3 BM^D scale: at each of the fixed 60 or 180 decision dates, the action
  total is the current first-bucket maturing notional plus that date's learned
  adjustment, floored at zero; its learned softmax still allocates the total
  across the 13 investment or 16 funding maturities. A frozen BM^D inside MM
  evaluates this same parameterization on the current detached ladders.
- E-03: the negative-rate cash charge is non-negative and subtracted from cash.
  Exempt cash and non-negative short rates produce zero charge, not interest income.
- Equation 8: loan growth uses the current pre-roll nominal balance rather than
  the initial economic value. Product allocation remains 55:20.
- E-10: use raw quantiles to select tails, then subtract the full-sample mean.
  Equity risk uses `E_T/E_0`, including finite non-positive values, not annualized
  returns. Penalty risk uses the upper tail; equity risk uses the lower tail.
  Quantiles use linear interpolation and inclusive masks. Small samples and ties
  need not select exactly 5% of observations. Annualized returns remain separate.
- C-7/C-8/R-1 evaluation: standardized dividend yield divides each path's total
  payout by initial equity times its nonterminal dividend years (`T - 1`).
  Constraint reports separately identify raw values at violating observations,
  transformed violation penalties, observation counts, and path-conditional
  counts. Equity distribution statistics use Equation 51 population central
  moments, including skewness and excess kurtosis; undefined metrics are
  explicit JSON-safe availability records rather than NaN or Infinity.
- R-2 reporting: every Table 1–5 and Figure 3–17 entry is identified by its
  printed page and subject. Local diagnostic artifacts are explicitly related
  evidence, never a substitute for the printed output. Incomplete paired
  experiments retain their member evidence and diagnostics, while their paired
  intervals are marked unavailable. Report JSON rejects non-finite values and
  is written atomically.

The numerical tests include a zero-rate/zero-growth rollover oracle: monthly
equity changes by -4 mCHF operating costs only. Previously the first month lost
an additional 1,341.431906 mCHF despite zero internal cash-reconciliation error.
An active-funded bank verifies that a cash charge actually lowers simulated cash
and preserves the action gradient against a central finite difference.

## Artifact compatibility

Financial semantics are identified by
`dated-deposit-history-v6`; evaluation metrics by
`population-moments-constraints-and-report-coverage-v4`. Selected checkpoints, frozen baselines,
and epoch recovery identities must match the financial semantics version.
Missing/older versions require fresh training, not a metadata-only upgrade.
Historical checkpoints and reports are retained, not rewritten or deleted.
The policy identity is `maturity-relative-bmd-v3`, so prior absolute-scale
BM^D and nominal-observation MM checkpoints are rejected and require fresh
training. BM^D parameter state now records date adjustments rather than
absolute scales; old state keys are not migrated or reinterpreted.

## Fixed-rate loan cohorts

Snapshot schema v2 adds explicit mortgage and enterprise-loan cohorts. Each
cohort records a 180-month remaining-principal schedule and a fixed monthly
coupon. The aggregate loan ladder is validated against the sum of principal plus
coupon cash flows. At each monthly roll, legacy coupons remain unchanged; only
new originations receive a rate from the current six-month yield plus spread.
Their rate is then fixed for all later periods. Loan interest is settled through
the reconstructed loan ladder once, rather than being separately added to cash.

Synthetic legacy cohorts use the initial curve and the customer spread; their
principal schedules are rescaled so the mortgage and enterprise economic-value
targets remain unchanged. Schema v1 aggregate-only snapshots are rejected: a
bank adapter must provide explicit cohort/contract data instead of reverse
engineering coupon rates from aggregate cash flows.

Snapshot schema v3 stores the four reference-term schedules for
both non-maturity and term deposits, together with their allocation weights.
The schedules must sum exactly to each aggregate deposit ladder. Aggregate v2
deposit ladders cannot reveal a maturing balance's original reference-term
class, so they are rejected rather than reconstructed heuristically.

Snapshot schema v5 stores the dated Y−1/Y−2 deposit-rate prefix, including
calendar-month target dates, selected completed observation dates, yields, and
the dated six-month-yield source used to prove that each selected observation
is the latest complete curve on or before its target. The simulator combines
this prefix with scenario Y0/Y1 only for the first two deposit-rate windows; it
does not repeat Y0 as fabricated history. A scenario batch must declare the
same initial-history identity, including the historical-source evidence.

The new-loan rate remains a single six-month-yield rate for all maturities. This
is an explicit shared simplification; a later maturity-specific origination-curve
extension would create several new cohorts per product/month and requires its own
resource budget and validation.

## Integrated consistency acceptance

The `acceptance-report.json` written by the bounded local workflow contains a
`financial_correction_ledger`. It is an implementation and evidence register,
not an economic-performance claim. Its source basis is the paper, the
user-specified independent errata review, and the worked examples below.

| Requirement | Source / independent expectation | Public execution evidence |
| --- | --- | --- |
| C-1/C-2 MM observation | Paper Equation 45: pre-action economic ratios in the stated order; constraints are shifted by their raw bounds. | Selected checkpoints and `locked-evaluation.json`; `tests/test_policies.py`, `tests/test_mm.py`. |
| C-3 BM^D allocation | At each of 60/180 decision dates, total transaction notional is the live first maturing bucket plus the learned date adjustment. | Selected checkpoints and frozen BM^D references; `tests/test_policies.py`, `tests/test_mm_training.py`. |
| C-5/C-6 deposits | Original reference-term ownership survives rollover; Equation 11c uses dated historical six-month yields rather than repeated Y0. | `reference-bank.json`, workflow replay; `tests/test_deposits.py`, `tests/test_reference_bank.py`. |
| E-03, Equation 8, E-10 | Errata: negative-rate cash charge is a non-negative cost; loan growth starts from the pre-roll balance; tail selection uses raw quantiles before centering. | Financial rollout diagnostics and `locked-evaluation.json`; `tests/test_deposits.py`, `tests/test_loans.py`, `tests/test_evaluation.py`. |
| C-7/C-8/R-1 metrics | Dividend denominator is `E0 × (T − 1)`; constraint counts distinguish raw and penalty units; moments are population moments with explicit undefined values. | `locked-evaluation.json`, `mm-truncation.json`; `tests/test_evaluation.py`. |
| R-2 reporting | Printed Table 1–5/Figure 3–17 mapping is explicit; missing output, failed branches and unavailable paired intervals remain non-promotional. | `compact-no-swap-report.json` and `paper-coverage-inventory.json`; `tests/test_reporting.py`. |

The workflow covers training/selection, Reference Bank import, frozen-policy
sensitivity, zero-update MM(15y|5y) truncation, recovery and reporting through
one immutable bundle. `tests/test_workflow.py` exercises both 5- and 15-year
CPU float64 paths (60 and 180 steps) with the existing dtype-aware accounting
tolerances. Device commissioning tests exercise the same public seam and report
MPS/CUDA as `not-run` when unavailable; a missing accelerator is never treated
as validation.

This acceptance does not start the paired pilot, certify convergence, recreate
the paper's private results, or approve a bank model. BM^E/BM^C historical
weights remain historical unless separately retrained under an approved plan.
