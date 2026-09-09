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
every executable YAML must declare both identities. During the expand phase,
scenario-bearing commands and historic artifacts remain available with migrated
complete configurations; a later contract ticket retires the duplicate command
surface.
