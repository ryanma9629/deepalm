# 22: Generate no-swap tables, figures, and manifests

**What to build:** Produce the paper-aligned no-swap analysis package so every table and figure is reviewable and traceable to the run that generated it.

**Blocked by:** 04/Add the Hull-White scenario comparison; 18/Analyze horizon effects and rate scenarios; 19/Run Reference Bank sensitivities; 20/Run the paired Paper-convention experiment.

**Status:** ready-for-agent

- [ ] The analysis emits the no-swap content of Tables 1 through 5 and Figures 3 through 17 using printed caption numbering as canonical.
- [ ] Swap columns, swap-only figures, swap cash flows, and MM^S are absent from generated outputs.
- [ ] Each table and figure identifies policy, horizon, convention, sample size, units, and source run; semantic artifact tests avoid pixel-level assertions.
- [ ] The run manifest includes resolved configuration, software and device identity, data and paper hashes, bank and calibration hashes, weekly dates, seeds, checkpoints, metrics, and acceptance evidence.
- [ ] Long-end extrapolation diagnostics, PCA evidence, HJM-versus-Hull-White comparison, policy actions, equity, constraints, durations, sensitivities, and representative paths are included.
- [ ] Artifact publication is atomic at run level; failed or incomplete runs retain diagnostics but cannot appear as complete bundles.
- [ ] Caption-number discrepancies in the paper are disclosed and visual similarity is not used as an acceptance test.
