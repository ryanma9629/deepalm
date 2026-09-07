# 15: Train the five-year MM

**What to build:** Train and select the shared multi-period model on the complete 60-transition five-year problem using the selected BM^D baseline.

**Blocked by:** 14/Build the MM observation and shared network.

**Status:** ready-for-agent

- [ ] A quick five-year MM run uses the full network and horizon, 256 fresh training paths per epoch, 128 selection paths, 128 locked test paths, five epochs, and batch size 32.
- [ ] Training paths, `mu`, and `lambda` are regenerated from named epoch seeds while selection and locked-test scenarios remain fixed and isolated.
- [ ] All 60 applications share network parameters and receive the correct `t/T` value and prior post-restructuring constraint features.
- [ ] The selected five-year BM^D checkpoint is frozen, identity-checked, and used by both action heads throughout training.
- [ ] Losses, state, actions, and gradients remain finite, and training changes MM actions on a deterministic smoke fixture.
- [ ] The selected checkpoint and run evidence are sufficient to repeat evaluation without rerunning training.
