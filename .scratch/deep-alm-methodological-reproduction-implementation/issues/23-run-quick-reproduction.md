# 23: Run and validate the complete local workflow

**What to build:** 用户一次调用即可完成真正的 5/15 年端到端流程，并得到基于实测证据的 development-validated 或明确的失败/未完成原因。

**Blocked by:** 22/Generate a compact report and paper coverage inventory; 27/Validate single-device portability and bank handoff.

**Status:** resolved

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] Replaying CPU scenarios and the saved Reference Bank from the manifest reproduces their content hashes.
- [x] 串起数据、校准、银行、8 个主训练任务、两次 paper-width 检查、checkpoint 选择/重载、MM(15y|5y)、锁定评估、代表性分析、报告及验收；所有动作受同一次调用预算约束。
- [x] CLI 的 plan/profile、calibrate、银行构建/导入、train/resume、evaluate/report 和 accept 复用 ReproductionRunner；分阶段产物能重用但须验证语义、代码/数据与 seed 身份，避免重复跑矩阵。
- [x] 50,000 个 one-step 市场统计检查单列为有界 CPU 验证组，不是完整轨迹训练预算；只在代码/数据/校准/口径/seed/runtime 匹配时复用证据。保留独立数值 oracle，不能靠重复自身公式自证。
- [x] 运行真实 local_flow 集成验收及短恢复验证。会计、金融时序、梯度、有限性、隔离、回放、checkpoint 和必需产物仍是硬门槛；64 次主更新外的检查与额外工作单列，不能漏算。
- [x] 接受已记录的 cubic-fit 偏差为诊断，不放宽 PCA 方差解释率、数值或金融检查。收益、约束违反率、MM 排序、loss 下降、期限方向及显著性不是本机硬门槛。
- [x] 验收读取实际实现证据，不能接受调用者传入的成功标记。预算耗尽/遗漏任务非零退出且不得成功；不可用 CUDA 标为 not-run，不等于集群验收。
- [x] 快速数值/金融/梯度、市场统计、本机完整流程、设备移植、扩展研究测试可分组调用；默认 fast 不启动训练矩阵。quick/paper_scale、敏感性重训和多 seed 不能隐式升级运行。
- [x] Local completion requires all eight primary policy/horizon jobs, the matching frozen BM^D dependencies, both full-width checks, separated training/selection/test streams, financial/gradient evidence and all mandatory analysis branches. No requested stage may be silently omitted or replaced with a success flag.
- [x] The acceptance report distinguishes purpose, applicability, observation, threshold, status and evidence. Operational failure, budget exhaustion/cancellation, numerical failure, incompatible resume and completed-but-failed acceptance retain diagnostics and return nonzero.
- [x] Only actual completed local evidence can yield development-validated. Neither choosing paper_scale nor loading bank data can automatically yield methodologically-reproduced or bank-model approval.
- [x] Include the complete declared run-bundle identities, planned/completed job and update counts, resource measurements and overshoot, generated/deferred artifacts, and unavailable-device checks. Test groups or resumptions must not silently add research work to local_flow.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。默认由 quick 改为 local_flow，接入可替换输入及移植证据，以真实工程/金融证据判断 development-validated，不要求论文经济效果。原编号保留；本次只更新待办，不表示本票实现已经完成。

## Answer

Implemented the bounded atomic `workflow` command and compatible `train`/`resume`/`evaluate`/`accept` reuse adapters. The workflow records actual CPU replay, 50,000-path one-step diagnostics, eight primary jobs, recovery equivalence, financial rollout checks, evaluations, analysis, reporting, resource use, and an evidence-derived acceptance report. Reuse verifies configuration, current input hashes, Git/runtime identity, artifact hashes, checkpoint/evaluation/truncation semantics, and the passed acceptance report before avoiding a matrix rerun. Local completion remains `development-validated`, not paper replication or bank approval.

Validation: focused workflow tests and lint passed; complete suite `170 passed in 77.29s`; specification and standards reviews found no blocking issue.
