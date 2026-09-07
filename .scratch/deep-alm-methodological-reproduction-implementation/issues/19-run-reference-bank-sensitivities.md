# 19: Run Reference Bank sensitivities

**What to build:** Show whether headline policy conclusions survive the approved one-factor changes to the synthetic Reference Bank and distinguish generalization failure from structural economic change.

**Blocked by:** 05/Build the Reference Bank; 17/Lock test evaluation and paired statistics.

**Status:** ready-for-agent

- [ ] Every approved scale, duration, spread, and cost sensitivity produces a validated snapshot that differs from the canonical bank in only its named factor.
- [ ] Frozen canonical Corrected policies are evaluated first on common market paths for every sensitivity.
- [ ] Headline conclusion and policy-ordering reversals are detected using declared metrics rather than chart inspection.
- [ ] A detected reversal triggers the approved quick retraining flow for that variant without consulting locked final-test results.
- [ ] The report identifies whether frozen-policy degradation persists after quick retraining and labels the result as generalization or structural evidence.
- [ ] Invalid or economically inconsistent variants fail Reference Bank validation and remain visible in the sensitivity report.
