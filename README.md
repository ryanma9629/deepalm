# Deep ALM

Deep ALM is a methodological reproduction of *Deep treasury management for
banks*. It uses public Swiss National Bank term-structure data and a transparent
Reference Bank in place of the paper's private bank inputs.

## Run skeleton

The first implementation slice resolves a run configuration and writes an
atomic audit bundle. From the repository root, run:

```bash
uv run deepalm run --config configs/quick-skeleton.yaml
```

The command writes a manifest containing the resolved configuration, source
hashes, Git revision, runtime/device identity, and named seed registry. It does
not yet run market calibration, Reference Bank construction, or policy training;
therefore its acceptance status is `pending`.
