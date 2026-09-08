# Reference Bank input contract

`ReferenceBankProvider.load()` imports a versioned JSON snapshot and
`ReferenceBankProvider.save()` emits the same contract. The in-code schema is
available through `ReferenceBankProvider.schema()`; the current
`schema_version` is `2`. Each snapshot also carries a required
`artifact_semantics` identity. It identifies the financial behavior actually
implemented and the repair items still pending; callers cannot upgrade an old
aggregate-deposit snapshot merely by changing metadata.

## Required mapping

| Input field | Meaning | Validation |
| --- | --- | --- |
| `as_of_date`, `initial_curve_identity` | Valuation date and initial market-curve identity | Must match the market batch supplied to `ALMSimulator.rollout` when the batch declares them. |
| `artifact_semantics` | Implemented snapshot behavior and correction status | Must exactly match the runtime's supported snapshot semantics; missing, unknown, or self-declared future repairs are rejected. |
| `cash`, `equity`, `target_economic_values` | Balance-sheet valuation in the declared unit | Seven targets: cash plus six product ladders; assets less liabilities must equal equity. |
| `ladders` | Monthly contractual nominal cash flows | Exactly six named arrays, each exactly 180 entries; finite and non-negative within tolerance. |
| `loan_cohorts` | Fixed-rate mortgage and enterprise-loan contracts | Each product has one or more cohorts. A cohort provides 180 remaining principal cash flows and one non-negative, already fixed monthly coupon rate. Reconstructing principal plus coupon cash flows from all cohorts must equal its aggregate loan ladder exactly. |
| `target_value_errors` | Declared relative valuation tolerance by ladder | Used against initial market discounts; values are never silently rescaled. |
| `product_assumptions` | Imported product mapping and assumptions | Must declare non-negative `loan_duration_years` and `deposit_duration_years`; imported values are not constrained by canonical duration targets. |
| `personnel_cost`, `material_cost` | Operating-cost inputs | Finite, non-negative values in the declared unit. |
| `currency`, `unit` | One balance-sheet currency and monetary unit | `currency` is an uppercase ISO-style code; `unit` must be `m` plus that code, such as `mUSD`. No FX aggregation occurs. |
| `provenance` | Source, mapping, assumptions, valuation, and market lineage | Requires non-empty `source_system`, `product_mapping`, `assumptions`, `valuation`, and `market` entries. `market` must be exactly `<initial_curve_identity>@<as_of_date>`. |

## Imported synthetic example

An imported USD-million example declares `profile: "imported"`,
`currency: "USD"`, and `unit: "mUSD"`. Its `target_economic_values` contains
`cash`, `investments`, `mortgages`, `enterprise_loans`,
`non_maturity_deposits`, `term_deposits`, and `funding`. Each corresponding
non-cash ladder is a 180-month list. The executable example and round-trip
checks are in `tests/test_replaceable_inputs.py`.

`profile: "canonical"` is reserved for the project's CHF reference case; only
that profile is checked against the 10,000 mCHF, 2022-07-15, and duration
targets. Bank-source extraction, FX conversion, and business-mapping approval
remain outside this contract.

Schema version 1 snapshots do not contain `loan_cohorts` and are rejected; they
cannot be upgraded by inferring coupons from aggregate loan cash flows. Imported
bank data must map contract-level principal schedules and locked rates explicitly.

Every market object supplied to `ALMSimulator.rollout` must also declare the
same `initial_curve_identity` and `as_of_date`. The HJM-PCA and Hull-White
market batches carry these fields automatically. The simulator rejects an
absent or mismatched lineage before rolling cash flows.
