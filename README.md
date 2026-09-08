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
