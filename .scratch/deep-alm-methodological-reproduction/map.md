# Plan the Deep ALM methodological reproduction

Type: wayfinder:map
Status: resolved

Scope update (2026-09-07): this map records the original Wayfinder decisions. [Specification revision 2](spec.md) now governs local workflow delivery, compute budgets and bank/GPU portability, superseding the local paper-scale execution and acceptance scope below. Financial decisions remain applicable except for explicitly recorded later user changes. Update implementation tickets from that specification before resuming implementation.

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

- [Inventory the paper's disclosed inputs and proprietary gaps](issues/01-inventory-disclosed-inputs-and-private-gaps.md): Public inputs support methodological reproduction, while proprietary cash-flow ladders, scale, product allocations, spreads, and cost bases must be replaced transparently in the Reference Bank.
- [Resolve SNB, Svensson, and HJM-PCA numerical conventions](issues/02-resolve-term-structure-conventions.md): SNB units and the high-level HJM-PCA pipeline are recoverable, but several numerical conventions are undisclosed and the paper's `lambda * q` factor conflicts with the standard `sqrt(lambda) * q` covariance factor.
- [Catalogue no-swap benchmarks, experiments, and validation claims](issues/03-catalog-no-swap-experiments.md): The reproduction must cover BM^E, BM^C, BM^D, and MM at 5-year and 15-year horizons plus MM(15y|5y), using published aggregate results as directional landmarks rather than exact targets.
- [Define the Reference Bank construction policy](issues/04-define-reference-bank-policy.md): Use one 10,000 mCHF paper-anchored canonical bank for both horizons, generated from transparent seasoned product templates behind a replaceable snapshot contract, with fail-fast financial checks and named one-factor sensitivities.
- [Choose the Paper and Corrected convention boundary](issues/05-set-convention-boundary.md): Default to Corrected and reserve Paper for paired fidelity analysis; only PCA loading scale and loan monthly-interest annualization branch, while clear errata are fixed in both and undisclosed numerical choices remain explicit shared defaults.
- [Define methodological reproduction acceptance thresholds](issues/06-set-acceptance-thresholds.md): Separate quick development validation from paper-scale methodological reproduction, with hard financial and gradient gates, sampling-aware HJM and paired-policy statistics, quantitative horizon-behavior checks, and complete reproducibility evidence.

## Ready for specification

- Collapse the linked decisions into concrete Python interfaces, tensor shapes, YAML schemas, package modules, CLI commands, artifact paths, and test modules with `/to-spec`.
- Preserve the `ReferenceBankSnapshot` seam and make the differentiable ALM transition boundary explicit in that specification.
- Use the fixed quick and paper-scale sample/epoch settings from the acceptance decision. Measure wall-clock performance on the user's Apple Silicon machine during implementation and treat it as an engineering benchmark, not a methodological pass/fail gate.
- After the specification is approved, use `/to-tickets` to split implementation into blocker-aware tracer-bullet tickets; do not implement directly from this map.

## Implementation context

- [17 Lock final-test evaluation and small-sample statistics](issues/17-lock-test-evaluation.md): Resolved with frozen-checkpoint final-test evaluation, locked common paths, auditable manifest, constraint/risk reports, and deterministic paired bootstrap statistics.

## Out of scope

- The interest-rate-swap extension.
- Integration with real bank data.
- A web user interface or production deployment.
- Regulatory certification.
- Pixel-identical reproduction of paper figures.
- Identical random paths, trained weights, or final numerical results.
