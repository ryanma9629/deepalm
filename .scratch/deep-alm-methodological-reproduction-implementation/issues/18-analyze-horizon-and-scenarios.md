# 18: Analyze horizon effects and rate scenarios

**What to build:** Turn locked policy trajectories into interpretable evidence about five-year terminal behavior, fifteen-year smoothness, and responses to representative yield-curve movements.

**Blocked by:** 17/Lock test evaluation and paired statistics.

**Status:** ready-for-agent

- [ ] Normalized action turnover is computed from consecutive L1 action changes using a documented denominator safeguard.
- [ ] Terminal concentration measures the share of turnover in the final 24 months of a five-year window.
- [ ] MM(5y), MM(15y) over its first 60 months, and MM(15y|5y) are compared with paired uncertainty intervals.
- [ ] Five scenario categories each select 50 five-year paths using the paper's terminal-curve ranking rules, with overlap permitted and disclosed.
- [ ] Category outputs cover steep, upward, downward, inverted, and constant-steepness behavior and identify policy, path rule, sample size, horizon, and units.
- [ ] Edge cases involving zero action volume, tied rankings, overlapping categories, and insufficient paths behave deterministically.
