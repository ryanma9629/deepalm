# Define methodological reproduction acceptance thresholds

Type: grilling
Status: claimed
Blocked by: 01, 02, 03, 04, 05

## Question

Using the established data inventory, convention choices, and no-swap experiment catalogue, what quantitative tolerances and qualitative criteria must pass for yield-curve reconstruction, HJM-PCA simulation, balance-sheet roll-forwards, regulatory constraints, differentiability, benchmark behavior, Deep ALM training, 5-year versus 15-year comparisons, and reproducible reporting before the project may claim a successful methodological reproduction?

## Comments

### Grilling round 1

- Define two completion levels. A passing quick profile is `development-validated`; only the complete paper-scale Corrected experiment matrix passing every required gate is `methodologically-reproduced`.
- Separate acceptance into hard numerical gates, sampling-aware statistical gates, and directional behavior gates. Hard failures cannot be excused by statistical tolerance, while stochastic quantities use predefined error bands or confidence intervals rather than exact equality.
- Divide paper behavior into core directional requirements and supporting diagnostics. Core loss/return/constraint ordering, greater long-horizon smoothness, and reduced 5-year terminal behavior for `MM(15y|5y)` are required. Exact action months and selected-scenario maturity choices are reported but do not independently block acceptance.
