# 12: Train and evaluate BM^C

**What to build:** Add the constant-allocation benchmark and make its five-year and fifteen-year results directly comparable with BM^E.

**Blocked by:** 11/Train and evaluate BM^E.

**Status:** ready-for-agent

- [ ] BM^C learns time-shared investment and funding maturity distributions and scale adjustments through the common policy contract.
- [ ] Benchmark scale combines the next-period maturing amount with the learned adjustment and produces non-negative 29-dimensional actions.
- [ ] Five-year and fifteen-year quick runs reuse the established trainer, scenario isolation, objective, checkpoint, and device behavior.
- [ ] Tests distinguish BM^C's learned constant allocation from BM^E's equal allocation while showing that neither depends on current market scenario.
- [ ] Evaluation on common locked paths emits the same core loss and action records as BM^E without policy-specific reporting shortcuts.
