# Deep ALM

Deep ALM is a methodological reproduction of *Deep treasury management for
banks*. It uses public Swiss National Bank term-structure data and a transparent
Reference Bank in place of the paper's private bank inputs.

## Local workflow plan and market preflight

The default example is the bounded `local_flow` profile: 32 training, 32
selection and 32 locked-test paths per epoch; two epochs; batches of eight;
compact architecture; and both 5- and 15-year horizons. It preserves the
financial problem while deferring substantive training to a bank GPU environment.

Inspect its planned work before starting any market or training stage:

```bash
uv run deepalm plan --config configs/quick-skeleton.yaml
```

Run the bounded real market preflight separately:

```bash
uv run deepalm preflight --config configs/quick-skeleton.yaml
```

The preflight loads the SNB source, calibrates HJM-PCA, and creates only bounded
scenario batches. It records time/RSS/accelerator measurements and remains
`pending`: it has not constructed the Reference Bank or trained a policy.

Build the canonical 10,000 mCHF Reference Bank as a separate review stage:

```bash
uv run deepalm bank --config configs/quick-skeleton.yaml
```

This creates an immutable `reference-bank.json` and an assumption-labeled
`table-1.json` alongside its manifest. It does not train a policy, so its
acceptance status also remains `pending`.

The audit-only skeleton command remains available:

```bash
uv run deepalm run --config configs/quick-skeleton.yaml
```

All commands write an auditable manifest containing resolved configuration,
source hashes, Git revision, runtime/device identity, named seed registry, and
the execution plan. `run` does not yet run market calibration, Reference Bank
construction, or policy training; its acceptance status is therefore `pending`.

## Complete local workflow

Run the complete bounded no-swap workflow with one command:

```bash
uv run deepalm workflow --config configs/quick-skeleton.yaml
```

It calibrates the market model, saves and reloads the canonical Reference Bank,
performs the 50,000-path one-step market diagnostic, trains all four policies at
both 5 and 15 years, verifies a short recovery, runs both MM paper-width checks,
then creates locked evaluation, MM(15y|5y), representative sensitivity, horizon
analysis, and compact-report artifacts. All work shares the configured resource
budget and is published by one final atomic rename. The successful bundle is
`development-validated` only: it demonstrates the technical workflow on a small
local sample; it does not claim convergence, paper-result replication, CUDA
validation, multi-GPU readiness, or bank-model approval.

`plan` (also `profile`), `preflight` (also `calibrate`), and `bank` are bounded
preparation stages. After a completed workflow, `train --source-run`, `resume
--source-run`, `evaluate --source-run`, and `accept --source-run` reuse its
atomic evidence instead of rerunning the policy matrix. Reuse rejects a source
whose requested configuration differs in data, convention, seeds, or model
semantics. `report` remains available for explicitly combining completed source
run directories.

## Paired-convention research pilot

On an Apple Silicon Mac with MPS available, the explicitly opt-in pilot runs
Paper and Corrected BM^D/MM at both horizons on matching immutable inputs and
named scenario streams:

```bash
uv run deepalm paired-pilot --config configs/paired-convention-pilot.yaml
```

It uses the locked compact MPS/float32 32-update matrix and a 600-second total
budget. After both 15-year MM first epochs, it stops remaining work when its
measured projection exceeds 420 seconds. A numerical failure writes the precise
diagnostic and returns nonzero; a finite but weaker Paper result is evidence,
not a Corrected failure. This is a `paired-convention-research-pilot`, not a
convergence result, paper-result replication, or bank-model approval.

## Compact no-swap report

Create a compact report only from one or more completed run bundles. The report
adds a bounded five-year (at most 32 paths) HJM-versus-Hull-White diagnostic,
canonical Reference Bank summary, calibration/PCA evidence, and a Tables 1-5 /
Figures 3-17 coverage inventory. It labels missing policy, recovery, truncation,
sensitivity, scenario, and bootstrap evidence as deferred rather than inferring
results from unrelated runs.

```bash
uv run deepalm report --config configs/quick-skeleton.yaml \
  --source-run artifacts/a-completed-run
```

Source bundles must be `completed` and have the same market/paper input hashes
and financial convention as the report configuration. Generated JSON artifacts
identify policy, horizon, convention, sample size, units, source run,
architecture, and checkpoint parameter counts where evidence exists. They are a
local workflow demonstration, not paper-result replication or bank approval.

## Single-device commissioning

Run a bounded, actual device check without starting the full policy matrix:

```bash
uv run deepalm device-check --config configs/quick-skeleton.yaml --horizon 5
```

It performs one small synthetic BM^E update on CPU and every available local
accelerator. Unavailable MPS/CUDA runtimes are recorded as `not-run`; they are
not simulated. The resulting run bundle explicitly does not promise
cross-device bitwise equality or multi-GPU readiness. For the Linux/CUDA
handoff sequence and the target-side configuration, see
[the bank commissioning guide](docs/bank-single-gpu-commissioning.md).
