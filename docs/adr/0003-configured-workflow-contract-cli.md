# Configuration-defined Workflow Contract CLI

Status: accepted

Deep ALM public commands denote user actions. A complete configuration selects
the Workflow Contract, TreasuryPolicy matrix, Reference Bank and market inputs,
and Execution Profile. The resolved configuration is persisted as audit
evidence, rather than relying on a scenario-bearing command name.

Workflow Contract validates scenario-specific invariants such as policy members,
horizons, no-swap scope and acceptance purpose. Execution Profile validates the
independent device, numeric and resource envelope. This separation prevents a
hardware profile from silently selecting a TreasuryPolicy scenario.

This decision preserves ADR-0002: Corrected Financial Semantics is the sole
executable financial semantics. It does not change any financial formula or
reintroduce a Paper convention. The configuration schema migration is explicit:
every executable YAML must declare both identities.

## Public command contract

The public lifecycle is `plan`, `run`, `evaluate`, and `report`. Independent
diagnostic actions are `preflight`, `reference-bank`, `device-check`, and
`verify-recovery`. The last action validates recovery evidence; it does not
continue an interrupted policy matrix. `resume` is therefore not a public
command. The former `bank` spelling is hidden from help and retained only as a
temporarily warned compatibility alias for `reference-bank`.

`plan` is human-readable by default and reports the selected Workflow Contract,
Execution Profile, work matrix, resource guard, artifact destination, and action
capabilities. `plan --format json` provides the same capability information with
the full resolved execution plan for automation.

The capability table is also enforced before execution. A planning-only or
commissioning-only configuration cannot enter unsupported `run`, `evaluate`, or
`report` paths and cannot publish an artifact that resembles completed training.
Missing, incomplete, or incompatible evidence is reported as an input error
before a public runner can create a failure bundle.

Progress, warnings, and diagnostics are written to stderr. A successful mutating
action writes only its final artifact directory to stdout. `run --overwrite` is
the sole replacement option and publishes over an existing directory only after
the replacement run has completed successfully in staging.

## Retired bounded workflow

The old bounded CPU development workflow is not a public Workflow Contract. Its
shipped YAML and public CLI route have been removed, and the public configuration
loader rejects both its former name and the internal test identity. The large
local workflow implementation remains solely as an integration-test harness,
resolved through a private test seam. It is not documented or advertised as a
user-selectable execution path.
