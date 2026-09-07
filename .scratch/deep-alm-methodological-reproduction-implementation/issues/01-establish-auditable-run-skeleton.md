# 01: Establish an auditable run skeleton

**What to build:** Let a researcher start a Deep ALM run from resolved configuration and receive an auditable run bundle before any financial stages are added.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Layered configuration resolves source data, convention, run scale, Reference Bank, experiment, policy, optimization, seeds, output, and acceptance settings.
- [ ] Locked Paper and Corrected profiles differ only in their two declared convention choices; overriding either choice marks the run as custom.
- [ ] Unknown keys, inconsistent horizons, invalid weights, incompatible profiles, and invalid unit-bearing values fail with actionable errors.
- [ ] A minimal run records resolved configuration, Git revision, runtime and dependency versions, device and dtype, input hashes, and a named seed registry.
- [ ] Run output distinguishes completed, acceptance-failed, and operationally failed states and cannot present a partial bundle as complete.
- [ ] The public runner contract and command-line adapter are covered through behavior-focused tests.
