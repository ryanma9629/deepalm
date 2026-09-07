# Define the Reference Bank construction policy

Type: grilling
Status: claimed
Blocked by: 01

## Question

Given the disclosed inputs and proprietary gaps, what evidence hierarchy, calibration principles, accounting identities, permissible assumptions, and sensitivity ranges should govern the Reference Bank so that it is transparent, internally consistent, useful for both model horizons, and replaceable by real bank data without being presented as the paper's original bank?

## Comments

### Grilling round 1

- Use a paper-anchored evidence hierarchy: disclosed values are hard constraints, derived values are calculated, and private gaps receive the simplest transparent synthetic assumptions. External industry data is a reasonableness check, not a calibration target, and paper outputs are not reverse-engineered.
- Use the same initial Reference Bank for the 5-year and 15-year experiments.
- Generate 180-month cash-flow ladders from versioned product templates and retain both the generator parameters and generated ladders.
- Maintain one canonical Reference Bank for primary experiments plus named sensitivity variants; do not randomize the bank structure independently on every market path.

### Grilling round 2

- Set canonical initial total assets to 10,000 mCHF, with 5,000 and 20,000 mCHF reserved as scale sensitivities.
- Treat the Table 1 shares as canonical initial economic-value targets. Scale each generated cash-flow portfolio to its target present value, keep cash at its 20% target, calculate equity as assets minus liabilities, and validate that the resulting equity share is 10% within numerical tolerance.
- Construct initial portfolios as synthetic seasoned books made from prior product cohorts, then retain only cash flows outstanding at the initial date and scale them to target present values.
- Use the complete 2022-07-15 SNB curve as the canonical initial curve and calculate the deposit reference-rate history from preceding SNB observations. Record this as a project assumption, not a disclosed paper fact.
