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
