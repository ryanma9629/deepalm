# 20: Run the paired Paper-convention experiment

**What to build:** Quantify the effect of following the paper's literal PCA scaling and loan-interest formula using a deliberately paired benchmark and MM experiment.

**Blocked by:** 16/Train the fifteen-year MM and its truncation; 17/Lock test evaluation and paired statistics.

**Status:** ready-for-agent

- [ ] The Paper matrix contains BM^D and MM at five and fifteen years plus `MM(15y|5y)` using the primary training seed.
- [ ] Paper and Corrected runs reuse the same canonical Reference Bank and Brownian innovations wherever comparison requires pairing.
- [ ] Convention contract tests prove that only PCA scaling and loan-interest annualization differ; any override resolves to custom and is excluded from locked pairing.
- [ ] The experiment reports performance, constraints, and numerical behavior without treating poor finite Paper outcomes as Corrected acceptance failures.
- [ ] A non-finite Paper result preserves diagnostics and identifies the exact model stage, path, time, and value that failed.
- [ ] Paired outputs clearly label convention, formula choices, seed, paths, horizon, and baseline dependency.
