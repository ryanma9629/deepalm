Status: ready-for-agent

## Problem Statement

The current Corrected Financial Semantics still differs from the paper's loan dynamics in three material ways. The terminal roll skips an otherwise applicable annual enterprise-loan impairment; new mortgage and enterprise-loan cohorts share one six-month coupon even though the paper prices each origination by its own maturity; and synthetic legacy cohorts can calculate a negative coupon despite the paper's non-negative loan-coupon rule. These differences alter terminal equity, fixed future loan cash flows, and the behavior of Reference Bank construction under low-rate curves.

## Solution

Align the loan part of the Corrected Financial Semantics with the paper as amended by E-02. Every annual enterprise-loan impairment date, including a terminal year-end, applies the disclosed rate-shock rule. Each new loan maturity locks its own non-negative monthly coupon from the prevailing same-maturity curve yield and loan spread. Synthetic legacy cohorts use the same non-negative monthly coupon rule. The TreasuryPolicy decision horizon, terminal no-action rule, terminal dividend exclusion, and terminal EYR exclusion remain unchanged.

## User Stories

1. As a methodological-reproduction researcher, I want enterprise loans to impair whenever the disclosed annual rate-shock condition is met, so that terminal equity includes terminal credit losses.
2. As a methodological-reproduction researcher, I want terminal loan impairment to be independent of the dividend and EYR calendars, so that ending an experiment does not remove an economic loss from the same date.
3. As a TreasuryPolicy trainer, I want terminal target loss to consume loan values after any applicable terminal impairment, so that the loss optimizes the stated financial transition.
4. As a Reference Bank user, I want every new mortgage maturity to lock its coupon from its own current curve tenor, so that a sloped Term Structure produces the paper's maturity-dependent loan interest.
5. As a Reference Bank user, I want each short enterprise-loan maturity to lock its own coupon, so that short loan cash flows do not inherit a long or six-month rate.
6. As a model-risk reviewer, I want fixed coupons to remain fixed after issuance, so that later curve changes do not reprice existing loans.
7. As a model-risk reviewer, I want enterprise-loan impairment to reduce both remaining principal and future interest consistently, so that impaired loans stop earning interest.
8. As a user running a low-rate historical or imported scenario, I want synthetic legacy loan coupons floored at zero, so that the Reference Bank honors the paper's no-negative-loan-rate assumption.
9. As a user importing real loan contracts, I want their explicitly supplied fixed coupons validated as before, so that the synthetic-coupon correction does not rewrite bank-source contractual data.
10. As an evaluator, I want the updated loan dynamics to preserve the existing cash-reconciliation and balance-sheet identities, so that a more faithful coupon model remains auditable.
11. As a maintainer, I want the implementation errata to stop describing shared six-month new-loan pricing as a retained simplification, so that the executable semantic record matches the code.
12. As a researcher comparing existing results, I want the financial-semantic change recorded clearly, so that checkpoints and reports trained under the older transition are not interpreted as results from the repaired model.

## Implementation Decisions

- The existing ALM simulator rollout is the highest behavioral seam. It receives the full current Term Structure for loan originations, while annual impairment continues to use the six-month rate and its value one year earlier.
- Loan dynamics distinguish the annual credit-loss calendar from the annual-closing calendar. A terminal roll remains passive for TreasuryPolicy actions, dividends, EYR evaluation, and post-roll constraints, but it still settles and updates decision-independent loan dynamics, including an applicable impairment.
- New loan originations retain the current product maturity distributions. The allocation for each maturity becomes a separately fixed-rate portion of the new cohort, with a coupon calculated from the matching current curve tenor plus the configured customer spread, converted to a monthly effective rate according to E-02, and floored at zero.
- The loan state continues to represent fixed-rate contracts and must reconstruct principal and interest cash flows without any double settlement. Its representation may group identical fixed-rate portions only when that grouping preserves cash flows exactly.
- The public single-rate helper remains available for callers that need to price one maturity, while loan-transition logic uses a full curve and a maturity-specific rate vector.
- Synthetic Reference Bank cohort construction uses the same non-negative monthly-coupon rule as new loans. Imported cohort coupons remain source values subject to existing non-negative validation.
- The implementation errata records the removal of the shared-six-month pricing simplification and the terminal impairment rule as adopted financial semantics.
- Formal training keeps the existing objective-parameter sampling and optimizer configuration. A benchmark optimizer step may be a valid no-op only when its clipped gradient norm is exactly zero; a non-zero gradient that produces no parameter change remains an operational failure.
- The isolated MM paper-width feasibility probe uses a fixed upper-range return target so that it continues to exercise backward propagation and a real parameter update after the corrected loan economics. This diagnostic fixture does not alter formal training draws or model selection.

## Testing Decisions

- Tests should assert observable loan cash flows, loan economic values, impairments, terminal equity, and cash reconciliation; they should not assert a particular internal cohort count or tensor layout.
- The primary integration seam is a deterministic `ALMSimulator.rollout` with loan dynamics enabled. It will prove that an identical annual rate shock produces the same impairment at a date whether or not that date is the terminal state.
- Loan-dynamics tests will use a non-flat curve and verify that multiple newly originated maturities receive their matching fixed coupons, remain unaffected by later curve moves, and generate the expected interest flows.
- Reference Bank tests will construct a sufficiently negative synthetic curve and verify zero rather than negative coupons, while preserving validation of supplied imported coupons.
- Existing loan, runoff, Reference Bank, training, constraint, and evaluation tests are prior art. The relevant regression suite must preserve cash reconciliation, fixed-coupon behavior, and the no-terminal-dividend/EYR rules.
- Training regressions must distinguish a mathematically valid zero-loss/zero-gradient batch from an optimizer failure, while the MM paper-width probe must still demonstrate a non-zero update.

## Out of Scope

- Swaps and MM^S.
- The E-04 bond transaction price versus spread-cost ambiguity.
- Reference Bank replacement with private bank data.
- Network width, epochs, scenario count, optimizer selection, gradient-clip convention, formal-training objective sampling, HJM discretization, and other execution-profile changes. The two diagnostic compatibility rules above do not alter these settings.
- Re-training checkpoints, frozen BM^D references, locked evaluations, or published reports. They will need regeneration after this semantic change but are not part of the code repair.

## Further Notes

The selected behavioral seam is the existing ALM simulator rollout, supplemented by focused loan and Reference Bank tests. This matches the user-authorized repair scope and avoids introducing a second simulation entry point. The spec follows the repository's single Corrected Financial Semantics model and does not reintroduce a runtime paper-convention switch.
