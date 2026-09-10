# Distributed training design (planned)

## Status and purpose

This is a design for a future bank-owned GPU implementation. It is **not** a
capability provided by the current command-line interface. Today the repository
can validate a Reference Bank snapshot and commission the PyTorch path on one
CUDA device; it does not run full policy training from an imported bank snapshot,
launch DDP workers, or schedule cluster jobs.

The aim is to scale the formal, paper-oriented experiment without changing its
financial experiment: corrected financial semantics, a fixed initial market and
Reference Bank identity, the same 5/15-year horizons, and MM's same-horizon
frozen BM^D baseline remain the workflow contract. Distributed execution is an
execution-side module behind that contract, rather than a second financial
model.

## Recommended sequence of parallelism

Use the following layers in order. They are deliberately separate so that a
failed or changed experiment is attributable to one layer.

1. **Job-level parallelism.** Run independent jobs for horizons and baselines:
   BM^E/BM^C/BM^D at 5 and 15 years. Start MM only after the matching frozen
   BM^D artifact has passed its acceptance checks. A job scheduler can place
   these independent jobs on different GPU allocations.
2. **Data-parallel path batches within one policy job.** Use PyTorch
   `DistributedDataParallel` (DDP), with one process per GPU and NCCL on CUDA.
   Each rank receives a disjoint shard of the globally indexed scenario paths;
   DDP synchronizes the policy gradients after each local backward pass.
3. **Distributed selection and locked evaluation.** Shard selection/test paths
   by their global indices, aggregate metrics deterministically, and let rank 0
   write the sole official selection record, checkpoint and report input.

DDP is the initial adapter because the intended paper-width MM network is only
about 1.4 million parameters; rollout calculation and scenario volume, rather
than parameter storage, dominate the expected work. FSDP, tensor parallelism
and pipeline parallelism are therefore out of scope until profiling demonstrates
that DDP is inadequate. PyTorch documents the one-process-per-GPU DDP model,
the application's responsibility for input sharding, and NCCL as the preferred
GPU backend: [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

## Reproducibility rules

The primary invariant is that a distributed run represents the same global
experiment as its single-GPU counterpart. It must not silently become a
different experiment merely because more GPUs are available.

- Configuration specifies a **global** training batch size and global counts
  for training, selection and test paths. A world size of four means each rank
  handles one quarter of a divisible global batch; it does not multiply the
  global batch by four. Any intentional batch-size or learning-rate change is a
  separately identified experiment.
- Generate scenario randomness from stable global path indices and named seed
  streams, then shard those indices by rank. Do not seed independent per-rank
  Monte Carlo universes. The merged paths must be equivalent to the single-rank
  global index set, apart from unavoidable floating-point reduction order.
- Rank-local loss sums and observation counts are all-reduced before reporting
  a global loss. Rank 0 makes the best-epoch/early-stopping decision from that
  global metric, then broadcasts the decision and selected checkpoint identity.
- Validation and reporting merge results by global path index, not by whichever
  rank finishes first. Numerical parity uses stated tolerances, not bitwise
  equality, because collective reductions can change floating-point order.
- A checkpoint is published by rank 0 only after all ranks reach a barrier.
  It stores the resolved financial configuration, snapshot/market identities,
  global seed streams, world size, PyTorch/CUDA/NCCL versions and the completed
  global epoch. Resume either restores an identical distributed layout or
  explicitly starts a newly identified run; it must never partially reuse an
  incompatible optimizer state.

## Proposed interfaces and code changes

The implementation should preserve the current policy, scenario, training,
evaluation and artifact interfaces as much as possible. Add a narrow
`DistributedExecutionContext` adapter that owns rank, world size, device,
collectives, barriers and rank-0 publication. The financial modules receive a
path-index shard and remain unaware of cluster topology.

1. Extend the resolved execution profile with a non-default distributed mode
   (initially `none` or `ddp`), backend, global batch size, launch identity and
   resource limits. Hostnames, scheduler account/queue and credentials remain
   outside versioned configuration; record their approved non-sensitive
   identity in the artifact manifest.
2. Add a scenario-shard interface that derives its local indices from the
   global count and rank. Preserve current single-device behaviour when the
   context is `none` or world size is one.
3. Wrap only the trainable policy in DDP. The state-transition and balance-sheet
   rollout stay in the normal autograd graph on every rank, so each rank retains
   correct gradients through active decisions and passive market evolution.
4. Replace rank-local selection/test summaries with reducible
   sum/count/statistic objects. Rank 0 owns early-stopping, checkpoint
   selection, immutable artifact publication and concise progress output;
   other ranks log rank-scoped diagnostics.
5. Extend artifact validation and reporting to prove shard coverage, missing or
   duplicate global path indices, aggregate metric inputs and the frozen BM^D
   dependency for MM. An incomplete multi-rank artifact must not be evaluable.

Launch should use PyTorch's `torchrun` interface under the bank scheduler, for
example `torchrun --nproc-per-node=4 -m deepalm ...`; the exact Slurm/Kubernetes
wrapper belongs to the bank platform. See [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html).

## Rollout and acceptance plan

1. Finish and accept the formal single-GPU paper-width training entry point on
   a bank CUDA host. It is the semantic reference for every later comparison.
2. Implement two-GPU DDP with the same global batch and globally indexed
   scenarios. Verify gradient/loss, selected epoch, checkpoint reload,
   selection coverage and locked-evaluation metrics against the single-GPU
   reference within declared tolerances.
3. Add scheduler-managed policy/horizon job dependencies and multi-node DDP.
   Demonstrate restart/recovery and rank-failure behaviour with bank-approved
   storage.
4. Only after semantic parity is accepted, test optional throughput changes
   such as mixed precision, larger global batches, activation compilation, or
   different network widths. Each is a new experiment with its own baseline,
   checkpoint lineage and approval.

This ordering makes the high-leverage seam explicit: financial methodology is
stable; execution topology can evolve behind a small adapter and a verifiable
artifact boundary.
