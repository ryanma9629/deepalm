# Bank single-GPU commissioning guide

This guide verifies that the same PyTorch code path can move from the local
MacBook to one Linux CUDA device. It is an engineering handoff, not bank-model
approval, economic validation, or a multi-GPU deployment procedure.

## 1. Prepare the target runtime

Install Python 3.11 or newer and `uv` on the Linux host, then clone the
repository at the intended revision:

```bash
uv sync --locked
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Proceed only when the final command reports `True` and the expected single GPU.
The PyTorch wheel, NVIDIA driver, CUDA runtime and host policy must be approved
by the bank; this repository does not install drivers or silently fall back
from a requested CUDA device.

## 2. Verify synthetic inputs and the device path

Review `configs/bank-single-gpu-commissioning.yaml`. It selects `cuda` and
`float32`, has explicit `bank_training` bounds, and deliberately uses the
synthetic canonical Reference Bank. Run:

```bash
uv run deepalm plan --config configs/bank-single-gpu-commissioning.yaml
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --horizon 5
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --horizon 15
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --policy MM --horizon 5
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --policy MM --horizon 15
```

Each command performs a small real update on CPU and each available single-device
backend. The MM commands first establish their frozen BM^D dependency. Inspect
`single-device-validation.json` in the printed run directory. Each completed
device must report finite loss, an updated policy, clipped gradients, CPU-recovery
resume, and an explicit checkpoint reload. A backend lacking a PyTorch runtime is explicitly
`not-run` with a reason; it is not a successful test. CUDA checks are therefore
`not-run` on a MacBook and must be rerun on the target host.

The JSON also states that cross-device bitwise equality is not promised. CPU
loading is used for portable checkpoint deserialization before the requested
device receives the policy tensors. Device and dtype remapping are explicit
PyTorch state-dict conversions: source and target resolved configurations remain
in the artifacts, but they are not semantic checkpoint mismatches. Checkpoints
still reject incompatible data, feature preprocessing, widths, conventions,
horizons, Reference Bank identity, or frozen BM^D dependencies, and record the
Git revision plus checkpoint schema version in the checkpoint itself.

## 3. Exercise checkpoint recovery

Run the short interrupted/resumed training fixture in the target CI or checkout:

```bash
uv run pytest tests/test_training.py -q
uv run pytest tests/test_mm_training.py -q
```

The recovery artifacts are CPU-readable and record optimizer/scheduler state,
completed epoch, architecture, resolved configuration, seed streams, market and
Reference Bank identities, MM curve-feature PCA, and the frozen BM^D reference.
Only device, output location, and increased resource limits are permitted
runtime resume overrides; the resume lineage records those changes. Do not use a
compact checkpoint as weights for a different-width bank model.

## 4. Map and reconcile real bank inputs

Before any bank training, follow [the Reference Bank input contract](reference-bank-inputs.md)
to map the valuation date, curve identity, six 180-month contractual ladders,
balances, currency/unit, costs, spreads/growth assumptions, and provenance.
Import and validate the snapshot, reconcile accounting and discounted values,
and obtain the bank's mapping sign-off. Also supply market data/scenarios that
are compatible with the imported snapshot; a changed market, source identity or
preprocessing identity cannot reuse a local checkpoint.

Synthetic commissioning artifacts contain no private bank data. Bank-source
extraction, data connections, mapping approval, and financial reconciliation
are bank-owned acceptance steps.

## 5. Measure before scaling

Record GPU/driver/PyTorch versions, device-validation JSON, mapped-data and
snapshot identities, finite financial/gradient checks, recovery evidence, and
measured resource profiles. Only after those records are accepted should the
bank explicitly increase the `bank_training` configuration's paths, epochs,
architecture, or resource limits and train afresh on real data.

Multi-GPU/DDP, scheduling, mixed-precision tuning, cluster throughput
optimization, and a production or regulatory approval are intentionally outside
this handoff. They require separate bank-side scope and acceptance.
