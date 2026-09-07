# Domain Docs

This repository uses a single-context domain-documentation layout.

## Before exploring, read these

- `CONTEXT.md` at the repository root
- Relevant ADRs under `docs/adr/`

If these files do not exist, proceed silently. The domain-modeling workflow creates them when terminology or architectural decisions are resolved.

## File structure

```
/
├── CONTEXT.md
├── docs/
│   └── adr/
└── src/
```

## Use the glossary's vocabulary

When naming a domain concept in code, tests, issues or design documents, use the terminology defined in `CONTEXT.md`.

If the required concept is missing, reconsider whether new terminology is necessary or record the gap for domain modeling.

## Flag ADR conflicts

If proposed work contradicts an existing ADR, identify the conflict explicitly instead of silently overriding the recorded decision.
