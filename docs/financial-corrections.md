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
- E-03: the negative-rate cash charge is non-negative and subtracted from cash.
  Exempt cash and non-negative short rates produce zero charge, not interest income.
- Equation 8: loan growth uses the current pre-roll nominal balance rather than
  the initial economic value. Product allocation remains 55:20.
- E-10: use raw quantiles to select tails, then subtract the full-sample mean.
  Equity risk uses `E_T/E_0`, including finite non-positive values, not annualized
  returns. Penalty risk uses the upper tail; equity risk uses the lower tail.
  Quantiles use linear interpolation and inclusive masks. Small samples and ties
  need not select exactly 5% of observations. Annualized returns remain separate.

The numerical tests include a zero-rate/zero-growth rollover oracle: monthly
equity changes by -4 mCHF operating costs only. Previously the first month lost
an additional 1,341.431906 mCHF despite zero internal cash-reconciliation error.
An active-funded bank verifies that a cash charge actually lowers simulated cash
and preserves the action gradient against a central finite difference.

## Artifact compatibility

Financial semantics are identified by
`reference-term-deposits-v4`; evaluation metrics by
`centered-equity-ratio-and-penalty-v2`. Selected checkpoints, frozen baselines,
and epoch recovery identities must match the financial semantics version.
Missing/older versions require fresh training, not a metadata-only upgrade.
Historical checkpoints and reports are retained, not rewritten or deleted.

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

Snapshot schema v3 additionally stores the four reference-term schedules for
both non-maturity and term deposits, together with their allocation weights.
The schedules must sum exactly to each aggregate deposit ladder. Aggregate v2
deposit ladders cannot reveal a maturing balance's original reference-term
class, so they are rejected rather than reconstructed heuristically.

The new-loan rate remains a single six-month-yield rate for all maturities. This
is an explicit shared simplification; a later maturity-specific origination-curve
extension would create several new cohorts per product/month and requires its own
resource budget and validation.
