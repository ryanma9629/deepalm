# 21: Enable paper-scale training and recovery

**What to build:** Run the full Corrected policy matrix at the approved research scale on the user's machine with deterministic streams, early stopping, and reliable recovery.

**Blocked by:** 17/Lock test evaluation and paired statistics.

**Status:** ready-for-agent

- [ ] Paper-scale uses 40,000 freshly simulated training paths per epoch, 1,600 selection paths, 1,600 locked test paths, batch size 32, and at most 100 epochs.
- [ ] Checkpoint selection starts after epoch 20 and stops after 15 epochs without at least 0.1% relative selection-loss improvement.
- [ ] BM^E, BM^C, and BM^D use the registered primary seed; Corrected MM uses three independent training streams per horizon while sharing selection and test paths.
- [ ] Path generation and training are streamed so the full 5-year and 15-year matrix is feasible on Apple Silicon without requiring all trajectories in memory.
- [ ] An interrupted run resumes exactly at a completed epoch boundary with optimizer, scheduler, selection history, seed stream, and baseline dependency restored.
- [ ] CPU market generation is reproducible bitwise; MPS training records reproducibility evidence without claiming bitwise checkpoint equality.
- [ ] Final locked-test evaluation runs only after all checkpoint and early-stopping choices are frozen.
