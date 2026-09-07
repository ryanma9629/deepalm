# 05: Build the Reference Bank

**What to build:** Construct, validate, save, and reload the transparent synthetic Reference Bank used by both reproduction horizons.

**Blocked by:** 01/Establish an auditable run skeleton; 02/Reconstruct SNB term structures.

**Status:** ready-for-agent

- [ ] The canonical bank starts with 10,000 mCHF of assets, uses the approved balance-sheet shares, and calculates equity as assets minus liabilities.
- [ ] Seasoned cohorts generate six 180-month nominal cash-flow ladders using the approved loan, deposit, investment, and funding maturity assumptions.
- [ ] Each portfolio is scaled to its target economic value on the canonical initial curve within `1e-8` relative error.
- [ ] The snapshot records costs, product assumptions, units, provenance, initial-curve identity, targets, and a stable content hash.
- [ ] Synthetic generation and saved-snapshot loading satisfy the same immutable provider contract and return equivalent economic content.
- [ ] Validation rejects identity, value, shape, sign, duration, unit, and provenance failures; aggregate loan duration is below five years and deposit duration below three years.
- [ ] A standalone Reference Bank stage produces an assumption-labeled Table 1 suitable for review.
