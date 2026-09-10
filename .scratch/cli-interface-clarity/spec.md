# CLI interface clarity and capability safety

Status: ready-for-agent

## Problem Statement

Deep ALM has moved to configuration-defined user actions, which correctly keeps
the Workflow Contract and Execution Profile out of command names. The public
CLI is now compact, but several parts of its interface can still cause a user
to draw the wrong conclusion about an experiment:

- `resume` validates recovery evidence; it does not resume a Workflow Contract
  training matrix.
- `bank` is ambiguous about whether it creates, validates, or trains the
  Reference Bank.
- `run` is syntactically available for every configuration, while a supplied
  configuration can be executable, planning-only, or commissioning-only. In
  particular, paper-oriented and bank configurations can otherwise create a
  minimal audit bundle rather than the formal training a reader may infer from
  `run`. The former bounded local workflow is an internal integration harness,
  not a supported user workflow.
- Top-level and per-action help explain syntax, but not the required evidence,
  produced artifact, supported Workflow Contract capability, or an ordinary
  end-to-end flow.
- `plan` exposes the complete machine-oriented representation by default,
  which is useful to automation but forces a human user to inspect unrelated
  resolved configuration before learning whether it is safe and meaningful to
  proceed.
- Progress output shares stdout with the final artifact path. A caller must
  suppress useful progress to consume the path in a script.

The result is a CLI whose underlying Corrected Financial Semantics and audit
identity are sound, but whose public interface does not always make the actual
execution capability and evidence chain obvious before resources are used.

## Solution

Keep a small, action-oriented public interface. The configuration remains the
sole authority for the Workflow Contract, Execution Profile, TreasuryPolicy
matrix, Reference Bank identity, market identity, resource guard and artifact
destination. The CLI must instead make three facts explicit before execution:
what the selected configuration can execute today, what evidence each action
requires and produces, and what an action deliberately does not do.

The public lifecycle remains `plan`, `run`, `evaluate`, and `report`.
Independent diagnostic actions remain available for market preflight, Reference
Bank construction/validation, and single-device commissioning. A central
capability adapter maps the resolved Workflow Contract to its executable
actions and guidance. `plan` presents that adapter's result; action dispatch
uses the same result, preventing documentation and behavior from drifting.

The sole external test seam is invocation of the `deepalm` executable with
arguments, observed exit status, stdout/stderr and declared artifact effects.
This is intentionally above individual runners: it is the interface a bank
user, an automation script and the README all share.

## User Stories

1. As a local researcher, I want `deepalm --help` to show the normal lifecycle and short runnable examples, so that I can start a Corrected Local Validation Pilot without learning internal modules.
2. As a local researcher, I want each action's help to say what its configuration and source artifacts represent, so that I do not mix incompatible Workflow Contracts.
3. As a local researcher, I want `plan` to show a concise human-readable work, resource and capability summary by default, so that I can decide whether to run before creating market scenarios or models.
4. As an automation author, I want `plan --format json` to retain a stable, complete machine-readable execution plan, so that tooling can consume the resolved configuration without parsing prose.
5. As a user of a paper-oriented configuration, I want `plan` and `run` to state that formal paper-scale training is not yet delivered, so that a minimal audit bundle is never mistaken for a methodological reproduction.
6. As a bank user, I want a commissioning configuration to state that it supports Reference Bank validation and single-device checks rather than full policy training, so that GPU resources are not allocated under a false expectation.
7. As an ordinary user, I want the retired bounded local harness to have no shipped YAML, public Workflow Contract name, or CLI route, so that I only choose supported workflows.
8. As a user of a configuration for which a requested action is unavailable, I want a pre-execution error naming the supported next actions, so that I can correct the command without producing misleading artifacts.
9. As a user who needs a Reference Bank action, I want the public command to be named `reference-bank`, so that the command describes the audited financial object it builds or validates.
10. As an existing user of `bank`, I want a temporary clearly warned compatibility path, so that a documented rename does not silently change my command's meaning.
11. As a user reviewing recovery evidence, I want the command to be called `verify-recovery`, so that I do not believe it will continue interrupted training.
12. As a future formal-training user, I want `resume` to remain unavailable until it genuinely continues a compatible Workflow Contract matrix, so that the command name retains a precise promise.
13. As a training user, I want default progress to remain visible, so that I can observe policy/horizon/epoch advancement and failures on a long local or bank run.
14. As a script author, I want progress and diagnostics on stderr while the completed artifact path is the only normal stdout result, so that I can capture the path without disabling observability.
15. As a user considering `--overwrite`, I want help to state that replacement occurs only after a successful staged run is ready to publish, so that I understand the preservation and failure behavior of prior evidence.
16. As a user of `evaluate`, I want help and early validation to state that the source run must be a compatible completed workflow, so that locked evaluation is not applied to an unrelated checkpoint.
17. As a user of `report`, I want help and early validation to state when source-run and locked-evaluation evidence are required, so that I receive an input error rather than a late operational failure.
18. As a bank model reviewer, I want diagnostic actions to say that successful execution is not economic validation, regulatory approval, or multi-GPU support, so that artifact claims remain proportionate.
19. As a maintainer, I want a single capability adapter used by help, planning and dispatch, so that adding a future formal training or DDP Execution Profile changes the behavior in one local place.
20. As a maintainer, I want retired or compatibility commands absent from the normal command inventory, so that the visible interface remains small and action-oriented.
21. As a reviewer, I want English and Chinese documentation to use the same action names, capability statements and output conventions, so that a bilingual handoff cannot prescribe different evidence chains.
22. As a test author, I want CLI behavior checked through arguments, messages, exit status and artifact effects, so that implementation refactoring does not weaken the public contract.

## Implementation Decisions

- Preserve the configuration-defined CLI established by ADR-0003. No command-line parameter may override TreasuryPolicy members, horizons, corrected financial semantics, Reference Bank/market identity, network width, paths, epochs, device, resource guards or artifact destination. Those remain the resolved configuration's responsibility.
- Retain the public lifecycle actions `plan`, `run`, `evaluate` and `report`. Do not reintroduce scenario-bearing commands for individual pilots, policies, horizons or hardware profiles.
- Add one capability adapter at the CLI/runner seam. It accepts a resolved Workflow Contract and returns, for every public action, a state of executable, diagnostic-only, planning-only, or unavailable plus concise next-step guidance. `plan`, pre-execution validation and contextual help use this one adapter.
- Retire the bounded local workflow from the public interface: delete its shipped YAML, reject its former `bounded-local-workflow` contract name, remove it from documentation and capability summaries, and do not dispatch any public CLI action to its runner. Retain the large `run_local_workflow` implementation and direct integration tests under an explicitly internal configuration identity until a later cleanup decision; this is test infrastructure, not a supported Workflow Contract.
- Mark paper-oriented formal training as planning-only until a formal research-training implementation exists. A `run` request for it must fail before artifact publication with an actionable message; it must not create a minimum bundle that resembles completed training evidence.
- Mark the bank commissioning Workflow Contract as commissioning-only until imported Reference Bank full policy training is implemented. Its supported actions must guide a user to planning, Reference Bank validation and device commissioning rather than promise training.
- Rename the public `bank` action to `reference-bank`. Keep `bank` for one documented compatibility period as an undisplayed alias that emits a deprecation warning on stderr and otherwise preserves behavior. Update all user documentation to the new name.
- Replace the public `resume` action with `verify-recovery`. It validates compatible recovery evidence only and must say so in its summary and detailed help. `resume` is not accepted as a public alias, reserving the word for a later genuine workflow-resume implementation.
- Keep `preflight` and `device-check` as public independent diagnostic actions. Their help must state that they create bounded diagnostic evidence and do not train a full policy matrix or grant economic, production, regulatory or multi-GPU acceptance.
- Treat `device-check --policy` and `--horizon` only as selectors within the matrix already declared by configuration. Reject a value outside that matrix; the flags never add or replace a TreasuryPolicy member or horizon.
- Make human-readable text the default `plan` format. It must include the Workflow Contract, Execution Profile, policy/horizon matrix, resource envelope, target artifact directory, supported-action capability table, expected evidence and clear warnings. Add an explicit JSON format that preserves the complete current plan representation for scripts.
- Expand root and action help using concise usage examples. Root help covers a small local lifecycle and points to the README for four-policy, paper-oriented and bank paths; action help describes required configuration, expected source evidence, artifact result, relevant incompatibility condition and safety semantics.
- Continue to default training and evaluation progress to enabled. Send progress, warnings and errors to stderr; on completed mutating actions, emit only the final artifact directory on stdout. `--no-verbose` remains the explicit way to suppress progress, not a requirement for script-safe stdout.
- Preserve `--overwrite` as an explicit opt-in on `run`; do not add `--force`. Its help and error messages must say that publication is staged and an existing run directory is replaced only after successful completion.
- Validate configuration-specific requirements before expensive work. In particular, evaluation must reject incompatible/incomplete source evidence, and report must diagnose absent required locked evaluation evidence as a user-input error rather than deferring it to an opaque late failure.
- Retire any obsolete documentation, command tests and help wording in the same change. Update English and Chinese documentation together and ensure generated artifact messages use the same terminology as the domain glossary.

## Testing Decisions

- The primary test is an external CLI contract test: provide argv and a controlled configuration/artifact fixture, then assert exit status, stdout, stderr and whether the declared artifact path is created, preserved or replaced. Do not test parser implementation details or private capability-adapter internals directly.
- Existing public CLI surface tests are the prior art for command inventory, help and rejection of retired scenario commands. Extend that style to assert the visible lifecycle, `reference-bank`, `verify-recovery`, and the absence of `resume` and the compatibility-only `bank` alias from root help.
- Existing workflow, run-skeleton, corrected-pilot, four-policy-pilot, evaluation and reporting tests are the prior art for running real bounded workflows. Add public-command tests proving that each shipped Workflow Contract receives the capability claimed by `plan` and dispatches to the corresponding implementation or fails before publishing an ambiguous artifact. Keep the retired harness covered only through direct integration calls.
- Test text and JSON planning at the CLI seam. Text must contain the selected Workflow Contract, Execution Profile, capability state, artifact destination and resource guard; JSON must remain parseable and retain the established complete plan data.
- Test stream separation at the CLI seam: visible training/evaluation progress and deprecation warnings appear on stderr, while a successful action's stdout is exactly the final artifact directory plus its newline. Quiet mode preserves the same stdout contract.
- Test safety behavior for existing artifacts: default `run` preserves an existing artifact directory and returns an actionable failure; `--overwrite` atomically replaces it only after a successful staged run. Retain the existing sentinel-artifact and pilot coverage as prior art.
- Test that paper-oriented and bank commissioning `run` requests return an actionable non-success status without creating a new misleading run artifact. Test that the former bounded contract name is rejected, its YAML is absent, and the internal harness has no public CLI capability.
- Test source/evaluation validation with missing, incompatible and complete evidence chains, including the same-horizon frozen BM^D baseline reference required by MM. Assertions describe user-visible failure and artifact effect rather than helper calls.
- Run documentation consistency tests and verify English/Chinese examples use public action names and valid argument order. Run the full existing test suite and static checks after the interface migration.

## Out of Scope

- Implementing a true workflow-level `resume` command or changing Recoverable Training semantics.
- Formal paper-scale training, imported-bank full policy training, DDP/multi-GPU execution, cluster scheduling, mixed precision or production/regulatory approval.
- Altering Corrected Financial Semantics, Reference Bank assumptions, HJM calibration, TreasuryPolicy behavior, MM Curve Feature Preprocessing, frozen BM^D baseline references, optimization, selection or risk metrics.
- Adding CLI overrides for experiment/resource parameters, an output-directory override, a `--force` synonym, or a second scenario-bearing command surface.
- Rewriting artifact schemas except where a capability status or clearer diagnostic needs auditable recording.

## Further Notes

This work follows ADR-0003 rather than revising it: a command denotes an action;
the complete configuration declares the financial experiment and its resource
envelope. The capability adapter makes that separation more legible and gives
the CLI depth: users learn a small action set while the implementation handles
Workflow Contract-specific execution safely behind one seam.

The user has already accepted the direction in this specification. The chosen
test seam is the public CLI process contract; no further design interview is
required before ticket decomposition.

After reviewing the cost and role of the legacy bounded flow, the user chose
to remove it from the public contract while retaining its implementation as an
internal integration harness. Its configuration identity is resolved only by a
private test seam; the public configuration loader rejects both the former
bounded contract name and the internal test identity. That later decision
supersedes the earlier proposal to route it through public `run`.
