# Deep ALM

[中文说明](README.zh-CN.md)

Deep ALM is a PyTorch implementation and methodological reproduction project
for *Deep treasury management for banks*. It uses public Swiss National Bank
term-structure data and a transparent Reference Bank in place of the paper's
private bank inputs.

The project has one executable financial convention:
`corrected-financial-semantics-v1`. It incorporates the accepted paper errata
and the implementation fixes recorded in the
[implementation errata](docs/implementation-errata.md). It intentionally does
not preserve an alternative “paper convention” runtime.

## Choose the right path

| Goal | Starting point | What a successful run proves | What it does **not** prove |
| --- | --- | --- | --- |
| 1. Exercise the complete flow on a MacBook | `configs/corrected-pilot.yaml` | Calibration, scenario generation, Reference Bank, BM^D/MM training at 5/15 years, frozen-baseline use, evaluation, and reporting all connect. | Convergence, economic superiority, paper-result replication, or bank approval. |
| 2. Get as close as possible to the paper | `paper_scale` + `paper` architecture; public data first, bank data later | The configuration can express the paper-scale resource and network target. | A finished paper-scale reproduction today: the public end-to-end runner is deliberately limited to the local validation profile. |
| 3. Move to a bank environment | `configs/bank-single-gpu-commissioning.yaml` and an imported Reference Bank snapshot | The same code path can be commissioned on one CUDA device and the bank input contract can be validated. | Production readiness, multi-GPU support, or permission to train on bank data. |

The three paths build on one another. Do not use a compact MacBook checkpoint
as a starting weight for a paper-width or bank model: a changed width, data
identity, market identity, or financial setting requires training afresh.

## Prerequisites

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/). The repository includes
the public market CSV and the paper PDF referenced by the supplied
configurations.

```bash
uv sync --locked
uv run deepalm --help
```

All commands write an auditable run bundle under `artifacts/`. The manifest
records resolved configuration, input hashes, Git revision, device/runtime,
seed registry, and measured resource use.

## 1. MacBook: run the whole technical flow with small resources

Use this path when the aim is to prove that the pipeline is wired correctly,
not to obtain an economically meaningful policy. It is the recommended first
run on the specified Apple Silicon MacBook.

The locked `corrected_pilot` configuration uses MPS with `float32`, compact
widths `64/64/32/32`, two epochs, 16 training and 16 selection paths per
epoch, 64 locked test paths, batch size 8, a 600-second wall-clock budget, and
12 GiB RSS/MPS guards. It trains four members:

```
BM^D at 5 years     MM at 5 years
BM^D at 15 years    MM at 15 years
```

Each MM member consumes the frozen BM^D baseline from the same horizon. Swaps
and `MM^S` are out of scope.

```bash
uv run deepalm corrected-pilot --config configs/corrected-pilot.yaml

uv run deepalm corrected-evaluate \
  --config configs/corrected-pilot.yaml \
  --source-run artifacts/corrected-local-validation-pilot

uv run deepalm corrected-report \
  --config configs/corrected-pilot.yaml \
  --source-run artifacts/corrected-local-validation-pilot \
  --evaluation-run artifacts/corrected-local-validation-pilot-evaluation
```

Inspect the `manifest.json` files and the generated
`corrected-pilot-report.json`. A valid local result is a complete,
identity-linked four-member evidence chain with finite updates and evaluation
artifacts. It is deliberately not an economic acceptance result. A measured
pilot on the target M5 MacBook took 151.30 seconds, followed by 21.04 seconds
of locked evaluation; treat these as a planning observation, not a performance
promise.

For the older, broader CPU/float64 development workflow, use:

```bash
uv run deepalm workflow --config configs/quick-skeleton.yaml
```

It includes BM^E, BM^C, BM^D, and MM, but remains a bounded no-swap
development-validation workflow rather than a paper-result reproduction.

### Initial four-policy comparison on the same MacBook

Use the separate four-policy pilot when you need an initial, like-for-like
comparison of BM^E, BM^C, BM^D, and MM. It keeps the same MPS/`float32`,
compact-network, 600-second and 12-GiB guards, but performs eight tasks and
32 updates. It is expected to take roughly 5--7 minutes on the target M5
MacBook; the guard remains 10 minutes because the result is still a bounded
technical comparison, not a convergence experiment.

```bash
uv run deepalm four-policy-pilot --config configs/four-policy-corrected-pilot.yaml

uv run deepalm four-policy-evaluate \
  --config configs/four-policy-corrected-pilot.yaml \
  --source-run artifacts/four-policy-corrected-local-validation-pilot

uv run deepalm four-policy-report \
  --config configs/four-policy-corrected-pilot.yaml \
  --source-run artifacts/four-policy-corrected-local-validation-pilot \
  --evaluation-run artifacts/four-policy-corrected-local-validation-pilot-evaluation
```

The report contains all eight policy/term members on the same locked test
scenarios. Compare annualized return, aggregate constraint penalty, and equity
risk as point estimates only. One seed, 64 test paths, and two epochs cannot
establish that one TreasuryPolicy is economically better.

## 2. Research path: configure a paper-oriented reproduction

The closest supported configuration vocabulary is:

| Setting | Paper-oriented value |
| --- | --- |
| `run_scale.profile` | `paper_scale`: 100 epochs, 40,000 training paths/epoch, 1,600 selection paths, 1,600 test paths, batch size 32; selection begins at epoch 20. |
| `architecture.profile` | `paper`: widths `512/512/256/128`. |
| `experiment.horizons_years` | `[5, 15]`. |
| `policy.names` | `[BM^E, BM^C, BM^D, MM]`. |
| `acceptance` | `purpose: research`, `required_status: methodologically-reproduced`. |

Start by copying `configs/quick-skeleton.yaml` outside version control and
changing those fields, together with device, output directory, resource limits,
and a new run name. Run `plan` first; it resolves the configuration without
generating market scenarios or starting training.

```bash
uv run deepalm plan --config path/to/paper-oriented.yaml
```

Important current boundary: the public `workflow`/`train` runner is restricted
to the bounded `development-validation` workflow. It will reject a
`methodologically-reproduced` paper-scale configuration rather than silently
run it under local assumptions. This is intentional evidence discipline, not a
command-line mistake. Completing a true paper-oriented run is a follow-on
engineering task: expose a research runner, then train afresh on the chosen
compute environment.

Even after that runner exists, the public inputs cannot reproduce the paper's
numeric results exactly: the paper uses private bank data, this project omits
swaps/`MM^S`, and new-loan pricing currently uses a common six-month yield
rather than a full maturity-specific origination curve. The useful research
claim is therefore *methodological comparison under disclosed substitutions*,
not identity with the paper's reported figures.

## 3. Bank path: real data and GPU configuration

Begin with a one-GPU engineering check, never with a large training job:

```bash
uv run deepalm plan --config configs/bank-single-gpu-commissioning.yaml
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --horizon 5
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --horizon 15
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --policy MM --horizon 5
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --policy MM --horizon 15
```

This configuration requests CUDA/`float32`, but intentionally uses two paths
and one epoch. It proves a portable PyTorch code path and checkpoint recovery;
it is not training.

### Data and configuration changes to make before bank training

1. **Create an imported Reference Bank snapshot.** Follow the
   [Reference Bank input contract](docs/reference-bank-inputs.md). It requires
   a valuation date and market identity, six 180-month contractual ladders,
   fixed-rate loan cohorts, four deposit reference-term classes, dated initial
   deposit history, product assumptions, units, and provenance. Validate it
   before training:

   ```bash
   uv run deepalm bank \
     --config configs/bank-single-gpu-commissioning.yaml \
     --snapshot /secure/path/reference-bank.json
   ```

2. **Replace market inputs as a matched pair.** The imported snapshot's
   `as_of_date` and `initial_curve_identity` must match the historical market
   data and every generated scenario batch. A changed curve or pre-processing
   identity invalidates prior checkpoints.

3. **Create a bank training configuration.** Use `run_scale.profile:
   bank_training` and explicitly set `epochs`, training/selection/test paths,
   batch size, selection start, early-stopping patience, and minimum relative
   improvement. Set `optimization.device: cuda`, keep `float32` initially,
   give the run a bank-owned output location, and set wall-clock/RSS/GPU-memory
   limits from an actual profiling run—not from the MacBook values.

4. **Scale only after profiling and approval.** Train from newly initialized
   weights after changing architecture width, path count, batch size, seed
   count, data, or market. Preserve the corrected financial semantics and the
   same-horizon frozen BM^D dependency for MM. Recreate checkpoints, baselines,
   locked evaluation, and reports for every changed experiment identity.

5. **Keep bank operations outside the public repository.** Put private source
   extracts, credentials, and bank artifacts in bank-controlled storage. Add
   data mapping sign-off, accounting and economic-value reconciliation,
   GPU/driver/PyTorch recording, model-risk review, and acceptance criteria to
   the bank's own delivery process.

The current repository validates imported snapshots through `deepalm bank` and
commissions one CUDA device. Connecting an imported snapshot to the full policy
training runner, multi-GPU/DDP execution, mixed-precision tuning, cluster
scheduling, and bank/regulatory acceptance are explicitly future delivery
items. See the [single-GPU commissioning guide](docs/bank-single-gpu-commissioning.md)
for the detailed handoff sequence.

## Further reading

- [Implementation errata and implementation boundaries](docs/implementation-errata.md)
- [Reference Bank input contract](docs/reference-bank-inputs.md)
- [Bank single-GPU commissioning guide](docs/bank-single-gpu-commissioning.md)
- [Paper PDF](docs/Deep%20treasury%20management%20for%20banks.pdf)
