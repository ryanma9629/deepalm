# 17: Lock test evaluation and paired statistics

**What to build:** Evaluate every trained policy once on untouched common paths and publish comparable performance, risk, constraint, and uncertainty statistics.

**Blocked by:** 12/Train and evaluate BM^C; 13/Train and evaluate BM^D; 15/Train the five-year MM; 16/Train the fifteen-year MM and its truncation.

**Status:** ready-for-agent

- [ ] Training, selection, and final-test scenario identities are disjoint, and final-test results cannot participate in checkpoint choice or sensitivity-trigger decisions.
- [ ] All policies at a horizon use common locked market paths and horizon comparisons use paired Brownian prefixes.
- [ ] Reports include total, target, and penalty losses; penalty ES95; CRRA; equity-ratio moments; signed VaR95 and ES95; annualized return; and standardized dividend yield.
- [ ] Constraint reporting includes time distributions, medians, ever-violation shares, violating counts and means, worst values, mean penalty, and penalty ES95.
- [ ] Paired comparisons perform 10,000 deterministic common-index bootstrap resamples and report point differences with 95% intervals.
- [ ] Signed tail metrics retain their declared meaning and are verified by independent fixtures.
- [ ] Model evaluation is reproducible from frozen checkpoints and fails if manifests, convention identities, paths, or required seeds are incompatible.
