# Plan the Deep ALM methodological reproduction

Type: wayfinder:map
Status: open

## Destination

Produce an implementation-ready decision set for a Python and PyTorch methodological reproduction of the no-swap model in *Deep treasury management for banks*, covering both 5-year and 15-year horizons and ready to collapse into a specification with `/to-spec`.

## Notes

- Use `uv` for the Python environment and PyTorch for differentiable modeling and optimization.
- Deliver a package-first implementation with a CLI, YAML configuration, automated tests, and reproducible analysis scripts; notebooks may be explanatory but do not own core logic.
- Use the public SNB yield-curve parameter data and a transparent Reference Bank in place of unavailable proprietary bank data.
- Support explicit Paper convention and Corrected convention choices where material ambiguities exist.
- Provide a quick run profile for local development and a paper-scale profile for formal experiments on the user's Apple Silicon MacBook Pro.
- Validate financial identities, constraints, gradients, scenario statistics, benchmarks, and qualitative strategy behavior rather than exact paper numbers.
- Consult `CONTEXT.md`, the source paper under `docs/`, and `docs/agents/domain.md` in every session.
- Wayfinding produces decisions, not implementation deliverables.

## Decisions so far

## Not yet specified

- The interfaces and tensor representations for balance-sheet state and monthly cash-flow ladders.
- The module seams and the boundary of PyTorch-differentiable computation.
- Concrete parameter values and runtime targets for quick and paper-scale profiles.
- The training, validation, random-seed, and checkpoint-selection protocol.
- The set of figures and result tables produced by the analysis layer.
- The concrete file schema and calibration rules for Reference Bank data.
- The layering of automated tests and financial reconciliation checks.

## Out of scope

- The interest-rate-swap extension.
- Integration with real bank data.
- A web user interface or production deployment.
- Regulatory certification.
- Pixel-identical reproduction of paper figures.
- Identical random paths, trained weights, or final numerical results.
