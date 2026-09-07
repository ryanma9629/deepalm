# Choose the Paper and Corrected convention boundary

Type: grilling
Status: claimed
Blocked by: 02

## Question

For each material numerical ambiguity identified in the term-structure research, which behavior belongs in the Paper convention, which belongs in the Corrected convention, which convention should be the default for each run profile, and how must outputs disclose the selected convention so results cannot be compared under mismatched assumptions?

## Comments

### Grilling round 1

- Use the Corrected convention by default for both quick and paper-scale runs. Use the Paper convention only for explicit fidelity comparisons; paper-scale describes workload, not numerical conventions.
- Classify uncertainties into: material runnable formula disagreements implemented in both conventions; obvious editorial or notation errors corrected once and recorded as errata; and undisclosed numerical choices implemented as shared, configurable project defaults rather than mislabeled convention differences.
- Provide locked `paper` and `corrected` profiles. Any component override changes the run identity to `custom`. Persist the complete choice manifest, configuration hash, and code commit with every result.
- Compare conventions only as paired convention-impact experiments using the same Reference Bank, market random numbers, and experiment settings. Keep their primary result tables separate and do not interpret cross-convention outcome differences as strategy rankings.
