# 06: Run a passive runoff simulation

**What to build:** Roll the Reference Bank through a market path with no active treasury trades and expose every monthly balance-sheet and cash movement as a verifiable trajectory.

**Blocked by:** 03/Generate HJM-PCA scenarios; 05/Build the Reference Bank.

**Status:** ready-for-agent

- [ ] The simulation accepts the canonical snapshot and a batch of market paths and returns batch-first observable states for all 60 or 180 transitions.
- [ ] Cash-flow ladders settle their first entry, shift one month, and are economically revalued on the current curve without mutating the source snapshot.
- [ ] Cash, assets, liabilities, and equity are tracked in mCHF and remain finite on deterministic runoff fixtures.
- [ ] Assets minus liabilities minus equity and the independently reconstructed cash movement stay within the specified accounting tolerance at every state.
- [ ] The same simulator contract supports short hand-calculated fixtures and complete five-year and fifteen-year paths.
- [ ] Failure diagnostics identify the first path, time, and state component that becomes inconsistent or non-finite.
