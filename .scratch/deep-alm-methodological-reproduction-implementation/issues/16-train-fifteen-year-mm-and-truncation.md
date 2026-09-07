# 16: Train the fifteen-year MM and its truncation

**What to build:** Train MM over 180 monthly decisions and evaluate the same learned policy over a truncated five-year window without changing its horizon awareness.

**Blocked by:** 15/Train the five-year MM.

**Status:** ready-for-agent

- [ ] A quick fifteen-year MM run uses the full 180-transition simulator, shared network, fifteen-year BM^D baseline, and the approved quick profile.
- [ ] Time is normalized by 15 years at every decision during fifteen-year training.
- [ ] `MM(15y|5y)` loads the selected fifteen-year model, performs only its first 60 decisions, retains `t/15`, performs the five-year terminal roll, and is not retrained.
- [ ] Five-year and fifteen-year policies can consume paths derived from common Brownian innovations for their shared first 60 months.
- [ ] Truncation does not renormalize time, substitute the five-year baseline, or alter trained parameters.
- [ ] Deterministic tests distinguish the truncated policy from independently trained MM(5y) and reproduce its actions from checkpoint evidence.
