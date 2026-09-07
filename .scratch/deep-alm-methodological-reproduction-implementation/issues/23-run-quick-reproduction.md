# 23: Run the complete quick reproduction

**What to build:** Give the researcher one command that exercises the real five-year and fifteen-year architecture from source data through policies, analysis, and development acceptance.

**Blocked by:** 22/Generate no-swap tables, figures, and manifests.

**Status:** ready-for-agent

- [ ] A single resolved quick run executes calibration, Reference Bank construction, all four policies at both horizons, `MM(15y|5y)`, evaluation, sensitivities, paired Paper diagnostics, and reporting.
- [ ] Quick uses the approved 256/128/128 scenario counts, five epochs, batch size 32, full network, and full 60/180-transition horizons.
- [ ] Stage commands expose calibration, bank construction, training, evaluation/reporting, and acceptance through the same contracts as the full runner.
- [ ] Passing all quick-applicable hard checks yields machine-readable `development-validated`; quick can never yield `methodologically-reproduced`.
- [ ] Operational failures, hard-check failures, and completed acceptance failures return distinct command statuses and retain useful diagnostics.
- [ ] Unit, integration/property, gradient, quick end-to-end, and slow paper-scale test groups are separately invocable, with ordinary fast tests excluding paper-scale work.
- [ ] Replaying CPU scenarios and the saved Reference Bank from the manifest reproduces their content hashes.
