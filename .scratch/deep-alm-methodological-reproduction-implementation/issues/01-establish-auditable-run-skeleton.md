# 01: Establish an auditable run skeleton

**What to build:** Let a researcher start a Deep ALM run from resolved configuration and receive an auditable run bundle before any financial stages are added.

**Blocked by:** None (can start immediately).

**Status:** completed

- [x] Layered configuration resolves source data, convention, run scale, Reference Bank, experiment, policy, optimization, seeds, output, and acceptance settings.
- [x] Locked Paper and Corrected profiles differ only in their two declared convention choices; overriding either choice marks the run as custom.
- [x] Unknown keys, inconsistent horizons, invalid weights, incompatible profiles, and invalid unit-bearing values fail with actionable errors.
- [x] A minimal run records resolved configuration, Git revision, runtime and dependency versions, device and dtype, input hashes, and a named seed registry.
- [x] Run output distinguishes completed, acceptance-failed, and operationally failed states and cannot present a partial bundle as complete.
- [x] The public runner contract and command-line adapter are covered through behavior-focused tests.

## Comments

2026-09-07 — 核对既有实现后补记原运行骨架票的完成状态，未新增功能。实现依据为 3925249、ce4743a、d26ae86、c5233b2 及后续配置校验维护；本次运行现有 run-skeleton 行为测试，18 passed（0.70s），覆盖配置解析、口径锁定、输入身份、原子审计产物、失败诊断和 CLI。未实现的权重配置键在当前骨架中按未知键拒绝；产品权重与其约束在 05 及后续金融票中实现。

完成只指原票的审计骨架切片，默认 acceptance 仍为 pending，不表示完整训练或本机流程已验收。Revision 2 的 local_flow/预算/设备解析在 25、训练恢复在 21、基于实际证据的完整验收在 23；尤其当前调用者可传入的 acceptance 标记不是未来交付的验收证据。02–04 的完成记录保持不变。
