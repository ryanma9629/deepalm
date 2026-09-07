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

### Grilling round 3

- Allocate new mortgages across original terms 2-12 years with 40% at 10 years and 6% at every other eligible term. Allocate enterprise loans equally across 1, 2, and 3 months. Replace each product's matured principal and split the 3% annual growth increment in the initial 55:20 mortgage-to-enterprise ratio.
- Allocate deposit reference terms `[1m, 2m, 1y, 10y]` as `[40%, 30%, 25%, 5%]` for non-maturity deposits and `[10%, 10%, 50%, 30%]` for term deposits.
- Construct the seasoned investment-bond book from equal historical issuance across 3-15 year terms and the funding-bond book from equal issuance across 3 months and 1-15 years. Use the initial curve for synthetic legacy pricing and do not favor a benchmark allocation.
- Set the canonical loan customer spread to 150 bp and initial monthly personnel and material costs to 3 mCHF and 1 mCHF respectively. Apply the paper's 2% annual growth only to personnel costs.

### Grilling round 4

- Use one-factor-at-a-time low/base/high sensitivities: total assets 5,000/10,000/20,000 mCHF; 10-year mortgage weight 20%/40%/60%; 10-year reference-term weights of 0%/5%/15% for non-maturity deposits and 15%/30%/45% for term deposits; loan spread 100/150/200 bp; and operating costs at 75%/100%/125% of canonical levels.
- Evaluate canonical-trained policies first on every sensitivity variant with common market paths and seeds. If a headline conclusion or strategy ordering reverses, retrain that variant with the quick profile to separate policy generalization from an economic structural change.
- Require synthetic and future real-data loaders to produce the same `ReferenceBankSnapshot` contract, containing initial balances, six 180-month cash-flow ladders, product assumptions, costs, initial-curve date, units, and provenance. The ALM simulator depends only on this contract.
- Fail Reference Bank construction rather than silently repair it when balance-sheet identity, target present values, nonnegative cash flows, ladder dimensions, duration bounds, or provenance checks fail. Exact numerical tolerances belong to the later acceptance-threshold decision.

### Grilling round 3

- Allocate new mortgages across original terms 2-12 years with 40% at 10 years and 6% at every other eligible term. Allocate enterprise loans equally across 1, 2, and 3 months. Replace each product's matured principal separately, then split the total 3% annual growth increment in the initial 55:20 mortgage-to-enterprise ratio.
- Allocate deposit reference terms `[1m, 2m, 1y, 10y]` as `[40%, 30%, 25%, 5%]` for non-maturity deposits and `[10%, 10%, 50%, 30%]` for term deposits.
- Build the initial investment-bond book from equal historical monthly issuance across original terms 3-15 years, and the funding-bond book from equal issuance across 3 months and 1-15 years. Retain outstanding cohorts at the initial date and scale their present values to the targets. Use the initial curve for synthetic legacy pricing rather than implying recovery of historical coupons.
- Set the canonical loan customer spread to 150 bp. Set initial monthly personnel and material costs to 3 mCHF and 1 mCHF respectively; apply the paper's 2% annual personnel-cost growth and constant material cost.
